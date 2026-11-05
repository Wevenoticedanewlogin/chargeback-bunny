"""PayPal API clients — real sandbox client + fixture-backed mock.

Real client (PayPalClient), confirmed against the live Disputes API docs:
  - OAuth2 client-credentials: POST {base}/v1/oauth2/token, HTTP Basic
    base64(client_id:secret), body grant_type=client_credentials. The token
    is cached until shortly before ``expires_in``.
  - GET /v1/customer/disputes with dispute_state / disputed_transaction_id /
    create_time_after filters (start_time is deprecated upstream).
  - GET /v1/customer/disputes/{id} — full dispute incl. disputed_transactions
    (seller_transaction_id joins to an order capture id), messages, evidences.
  - GET /v2/checkout/orders/{id} and GET /v2/payments/captures/{id} — the
    ground truth a claim is checked against.
  - Actions: accept-claim, provide-evidence (multipart), make-offer,
    escalate, send-message, plus sandbox-only helpers (adjudicate,
    require-evidence, create_dispute_buyer — best effort, needs scopes).

MockPayPalClient implements the IDENTICAL async interface, serving
demo/sandbox_fixtures/*.json so the whole pipeline runs with zero keys.
Endpoint-specific details are isolated in small helpers so they're easy to
correct if the sandbox API drifts.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger("chargebackbunny.paypal")


class PayPalAPIError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"PayPal API error {status}: {message}")
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# ground-truth normalization (shared by both clients)


def ground_truth_from_order(order: dict, transaction_id: str = "") -> dict:
    """Normalize a v2 order (or capture-centric view of one) into the flat
    ground-truth dict the verify stage consumes."""
    units = order.get("purchase_units") or []
    unit = units[0] if units else {}
    captures = ((unit.get("payments") or {}).get("captures")) or []
    capture = next(
        (c for c in captures if c.get("id") == transaction_id),
        captures[0] if captures else {},
    )
    shipping = unit.get("shipping") or {}
    trackers = shipping.get("trackers") or []
    delivered = any(str(t.get("status", "")).upper() == "DELIVERED" for t in trackers)
    first_tracker = trackers[0] if trackers else {}
    amount = float((unit.get("amount") or {}).get("value") or 0.0)
    currency = (unit.get("amount") or {}).get("currency_code") or "USD"
    # 3-D Secure: a passed authentication shifts fraud liability to the issuer —
    # decisive for unauthorized-transaction claims.
    auth = ((order.get("payment_source") or {}).get("card") or {}).get("authentication_result") or {}
    three_ds = auth.get("three_d_secure") or {}
    liability_shift = bool(
        str(auth.get("liability_shift", "")).upper() in ("POSSIBLE", "YES")
        and str(three_ds.get("authentication_status", "")).upper() == "Y"
    )
    return {
        "transaction_exists": True,
        "order_id": order.get("id", ""),
        "order_status": order.get("status", ""),
        "capture_id": capture.get("id", ""),
        "capture_status": capture.get("status", ""),
        "order_amount": amount,
        "currency": currency,
        "seller_protection": (capture.get("seller_protection") or {}).get("status", ""),
        "tracking_present": bool(trackers),
        "delivered": delivered,
        "tracking_number": first_tracker.get("tracking_number") or first_tracker.get("id") or "",
        "carrier": first_tracker.get("carrier") or first_tracker.get("carrier_name_other") or "",
        "delivery_detail": first_tracker.get("delivery_detail", ""),
        "liability_shift": liability_shift,
    }


def empty_ground_truth(transaction_id: str = "") -> dict:
    return {
        "transaction_exists": False,
        "order_id": "",
        "order_status": "",
        "capture_id": transaction_id,
        "capture_status": "",
        "order_amount": 0.0,
        "currency": "USD",
        "seller_protection": "",
        "tracking_present": False,
        "delivered": False,
        "tracking_number": "",
        "carrier": "",
        "delivery_detail": "",
        "liability_shift": False,
    }


# ---------------------------------------------------------------------------
# real sandbox client


class PayPalClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str = "https://api-m.sandbox.paypal.com",
        timeout: float = 30.0,
    ):
        self.mode = "sandbox"
        self._client_id = client_id
        self._client_secret = client_secret
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._token: str | None = None
        self._token_expiry: float = 0.0

    # -- auth ----------------------------------------------------------------

    async def _access_token(self) -> str:
        # Cached with a 60s safety skew ahead of the server-provided expiry.
        if self._token and time.monotonic() < self._token_expiry:
            return self._token
        basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/v1/oauth2/token",
                headers={
                    "Authorization": f"Basic {basic}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials"},
            )
            if resp.status_code >= 400:
                raise PayPalAPIError(resp.status_code, resp.text[:300])
            data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.monotonic() + float(data.get("expires_in", 32400)) - 60.0
        return self._token

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        for attempt in range(2):
            token = await self._access_token()
            headers = kwargs.pop("headers", {})
            headers["Authorization"] = f"Bearer {token}"
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.request(method, f"{self._base}{path}", headers=headers, **kwargs)
            if resp.status_code == 401 and attempt == 0:
                # Cached token revoked/expired early — re-authenticate once.
                self._token = None
                self._token_expiry = 0.0
                continue
            if resp.status_code >= 400:
                raise PayPalAPIError(resp.status_code, resp.text[:300])
            return resp
        raise PayPalAPIError(401, "unauthorized after token refresh")

    # -- read ----------------------------------------------------------------

    async def list_disputes(self, **filters) -> list[dict]:
        """GET /v1/customer/disputes. Supported filters: dispute_state (comma
        list), disputed_transaction_id, create_time_after, create_time_before,
        page, page_size."""
        params = {k: v for k, v in filters.items() if v is not None}
        resp = await self._request("GET", "/v1/customer/disputes", params=params)
        return (resp.json() or {}).get("items", [])

    async def get_dispute(self, dispute_id: str) -> dict:
        resp = await self._request("GET", f"/v1/customer/disputes/{dispute_id}")
        return resp.json()

    async def get_order(self, order_id: str) -> dict:
        resp = await self._request("GET", f"/v2/checkout/orders/{order_id}")
        return resp.json()

    async def get_capture(self, capture_id: str) -> dict:
        resp = await self._request("GET", f"/v2/payments/captures/{capture_id}")
        return resp.json()

    async def get_transaction_ground_truth(self, transaction_id: str) -> dict:
        """Resolve a claim-referenced id (capture id preferred — that is what
        disputed_transactions[].seller_transaction_id carries) to a flat
        ground-truth dict."""
        if not transaction_id:
            return empty_ground_truth(transaction_id)
        # Capture first: its supplementary_data usually names the parent order.
        try:
            capture = await self.get_capture(transaction_id)
        except PayPalAPIError as e:
            if e.status != 404:
                raise
            capture = None
        if capture is not None:
            order_id = (
                (capture.get("supplementary_data") or {}).get("related_ids") or {}
            ).get("order_id", "")
            if order_id:
                try:
                    order = await self.get_order(order_id)
                    return ground_truth_from_order(order, transaction_id)
                except PayPalAPIError:
                    pass
            # Order unreachable — report what the capture alone proves.
            gt = empty_ground_truth(transaction_id)
            gt.update(
                transaction_exists=True,
                capture_id=capture.get("id", transaction_id),
                capture_status=capture.get("status", ""),
                order_amount=float((capture.get("amount") or {}).get("value") or 0.0),
                currency=(capture.get("amount") or {}).get("currency_code") or "USD",
                seller_protection=(capture.get("seller_protection") or {}).get("status", ""),
            )
            return gt
        # Maybe the claim referenced an order id directly.
        try:
            order = await self.get_order(transaction_id)
        except PayPalAPIError as e:
            if e.status == 404:
                return empty_ground_truth(transaction_id)
            raise
        return ground_truth_from_order(order, transaction_id)

    # -- actions ---------------------------------------------------------------

    async def accept_claim(self, dispute_id: str, note: str) -> dict:
        body: dict = {"note": note}
        resp = await self._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/accept-claim", json=body
        )
        return resp.json() if resp.content else {"status": "ok"}

    def _evidence_payload(self, evidence: dict) -> dict:
        """Build the multipart 'input' part for provide-evidence. Isolated so
        the shape is trivial to correct if the sandbox API drifts.

        evidence keys: tracking_number, carrier, notes, evidence_type
        (default PROOF_OF_FULFILLMENT), extra (raw dict merged in).
        """
        evidences: list[dict] = []
        tracking_number = evidence.get("tracking_number")
        if tracking_number:
            evidences.append(
                {
                    "evidence_type": evidence.get("evidence_type", "PROOF_OF_FULFILLMENT"),
                    "evidence_info": {
                        "tracking_info": [
                            {
                                "carrier_name": (evidence.get("carrier") or "OTHER").upper(),
                                "tracking_number": str(tracking_number),
                            }
                        ]
                    },
                }
            )
        if evidence.get("extra"):
            evidences.append(evidence["extra"])
        return {"evidences": evidences, "notes": evidence.get("notes", "")}

    async def provide_evidence(self, dispute_id: str, evidence: dict) -> dict:
        """POST /v1/customer/disputes/{id}/provide-evidence — multipart with a
        JSON 'input' part (the Disputes API does not take a plain JSON body
        here)."""
        payload = json.dumps(self._evidence_payload(evidence))
        resp = await self._request(
            "POST",
            f"/v1/customer/disputes/{dispute_id}/provide-evidence",
            files={"input": (None, payload, "application/json")},
        )
        return resp.json() if resp.content else {"status": "ok"}

    async def make_offer(self, dispute_id: str, amount: float, note: str, currency: str = "USD") -> dict:
        body = {
            "note": note,
            "offer_amount": {"currency_code": currency, "value": f"{amount:.2f}"},
            "offer_type": "REFUND",
        }
        resp = await self._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/make-offer", json=body
        )
        return resp.json() if resp.content else {"status": "ok"}

    async def escalate(self, dispute_id: str, note: str) -> dict:
        resp = await self._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/escalate", json={"note": note}
        )
        return resp.json() if resp.content else {"status": "ok"}

    async def send_message(self, dispute_id: str, message: str) -> dict:
        resp = await self._request(
            "POST", f"/v1/customer/disputes/{dispute_id}/send-message", json={"message": message}
        )
        return resp.json() if resp.content else {"status": "ok"}

    # -- sandbox-only helpers (seeding; best effort, may require scopes) ---------

    async def adjudicate(self, dispute_id: str, outcome: str) -> dict:
        """POST {id}/adjudicate — outcome: BUYER_FAVOR | SELLER_FAVOR.
        Sandbox only; dispute must be UNDER_REVIEW."""
        resp = await self._request(
            "POST",
            f"/v1/customer/disputes/{dispute_id}/adjudicate",
            json={"adjudication_outcome": outcome},
        )
        return resp.json() if resp.content else {"status": "ok"}

    async def require_evidence(self, dispute_id: str, action: str = "SELLER_EVIDENCE") -> dict:
        resp = await self._request(
            "POST",
            f"/v1/customer/disputes/{dispute_id}/require-evidence",
            json={"action": action},
        )
        return resp.json() if resp.content else {"status": "ok"}

    async def create_dispute_buyer(self, dispute: dict, buyer_email: str) -> dict:
        """POST /v1/customer/disputes as the BUYER. Sandbox-only, multipart,
        needs a PayPal-Auth-Assertion unsigned JWT and the DISPUTE_CREATE
        scope (which requires a PayPal account manager) — best effort; the
        primary seeding path is the sandbox buyer UI."""
        assertion = (
            base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).decode().rstrip("=")
            + "."
            + base64.urlsafe_b64encode(
                json.dumps({"iss": self._client_id, "email": buyer_email}).encode()
            ).decode().rstrip("=")
            + "."
        )
        token = await self._access_token()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/v1/customer/disputes",
                headers={
                    "Authorization": f"Bearer {token}",
                    "PayPal-Auth-Assertion": assertion,
                },
                files={"input": (None, json.dumps(dispute), "application/json")},
            )
        if resp.status_code >= 400:
            raise PayPalAPIError(resp.status_code, resp.text[:300])
        return resp.json() if resp.content else {"status": "ok"}


# ---------------------------------------------------------------------------
# fixture-backed mock — first-class: judges run the full demo offline


class MockPayPalClient:
    """Same async interface as PayPalClient, backed by JSON fixtures.

    Fixture layout (per directory): ``disputes.json`` = {"items": [dispute…]}
    in the real Disputes API shape; ``orders.json`` = {"items": [order…]} in
    the v2 checkout-orders shape (with shipping.trackers for ground truth).
    Multiple fixture dirs may be loaded; later dirs add to / override earlier
    ones by id. Mutations (accept_claim etc.) live in memory only.
    """

    def __init__(self, fixture_dirs: list[str | Path] | str | Path):
        self.mode = "mock"
        if isinstance(fixture_dirs, (str, Path)):
            fixture_dirs = [fixture_dirs]
        self._disputes: dict[str, dict] = {}
        self._orders: dict[str, dict] = {}
        self._actions: list[dict] = []
        for d in fixture_dirs:
            self.load_dir(d)

    def load_dir(self, fixture_dir: str | Path) -> None:
        path = Path(fixture_dir)
        disputes_file = path / "disputes.json"
        orders_file = path / "orders.json"
        if disputes_file.is_file():
            data = json.loads(disputes_file.read_text(encoding="utf-8"))
            items = data.get("items", data if isinstance(data, list) else [])
            for item in items:
                self._disputes[item["dispute_id"]] = item
        if orders_file.is_file():
            data = json.loads(orders_file.read_text(encoding="utf-8"))
            items = data.get("items", data if isinstance(data, list) else [])
            for item in items:
                self._orders[item["id"]] = item

    # -- read ----------------------------------------------------------------

    async def list_disputes(self, **filters) -> list[dict]:
        items = list(self._disputes.values())
        state = filters.get("dispute_state")
        if state:
            wanted = {s.strip() for s in str(state).split(",")}
            items = [
                d for d in items
                if d.get("dispute_state") in wanted or d.get("status") in wanted
            ]
        txn = filters.get("disputed_transaction_id")
        if txn:
            items = [
                d for d in items
                if txn
                in {
                    t.get("seller_transaction_id") for t in (d.get("disputed_transactions") or [])
                }
                | {d.get("transaction_id", "")}
            ]
        return items

    async def get_dispute(self, dispute_id: str) -> dict:
        if dispute_id not in self._disputes:
            raise PayPalAPIError(404, f"unknown dispute: {dispute_id}")
        return self._disputes[dispute_id]

    def _find_order(self, ref: str) -> dict | None:
        """Resolve by order id, or by any capture id inside an order (the
        seller_transaction_id join key)."""
        if ref in self._orders:
            return self._orders[ref]
        for order in self._orders.values():
            for unit in order.get("purchase_units") or []:
                for cap in (unit.get("payments") or {}).get("captures") or []:
                    if cap.get("id") == ref:
                        return order
        return None

    async def get_order(self, order_id: str) -> dict:
        order = self._find_order(order_id)
        if order is None:
            raise PayPalAPIError(404, f"unknown order: {order_id}")
        return order

    async def get_capture(self, capture_id: str) -> dict:
        for order in self._orders.values():
            for unit in order.get("purchase_units") or []:
                for cap in (unit.get("payments") or {}).get("captures") or []:
                    if cap.get("id") == capture_id:
                        out = dict(cap)
                        out["order_id"] = order.get("id", "")
                        return out
        raise PayPalAPIError(404, f"unknown capture: {capture_id}")

    async def get_transaction_ground_truth(self, transaction_id: str) -> dict:
        if not transaction_id:
            return empty_ground_truth(transaction_id)
        order = self._find_order(transaction_id)
        if order is None:
            return empty_ground_truth(transaction_id)
        return ground_truth_from_order(order, transaction_id)

    # -- actions (mutate in memory; fixtures on disk stay pristine) -------------

    def _mutate(self, dispute_id: str, action: str, detail: dict) -> dict:
        dispute = self._disputes.get(dispute_id)
        if dispute is None:
            raise PayPalAPIError(404, f"unknown dispute: {dispute_id}")
        dispute["status"] = "RESOLVED"
        dispute["dispute_state"] = "RESOLVED"
        record = {"action": action, "dispute_id": dispute_id, **detail}
        self._actions.append(record)
        return {
            "dispute_id": dispute_id,
            "status": dispute["status"],
            "action": action,
            "links": [
                {
                    "href": f"https://api-m.sandbox.paypal.com/v1/customer/disputes/{dispute_id}",
                    "rel": "self",
                    "method": "GET",
                }
            ],
        }

    async def accept_claim(self, dispute_id: str, note: str) -> dict:
        return self._mutate(dispute_id, "accept_claim", {"note": note})

    async def provide_evidence(self, dispute_id: str, evidence: dict) -> dict:
        dispute = self._disputes.get(dispute_id)
        if dispute is None:
            raise PayPalAPIError(404, f"unknown dispute: {dispute_id}")
        payload = PayPalClient._evidence_payload(self, evidence)  # same shape as real
        dispute.setdefault("evidences", []).extend(payload["evidences"])
        record = {"action": "provide_evidence", "dispute_id": dispute_id, "input": payload}
        self._actions.append(record)
        return {
            "dispute_id": dispute_id,
            "status": dispute.get("status", "UNDER_REVIEW"),
            "action": "provide_evidence",
            "evidences": payload["evidences"],
            "links": [],
        }

    async def make_offer(self, dispute_id: str, amount: float, note: str, currency: str = "USD") -> dict:
        return self._mutate(
            dispute_id,
            "make_offer",
            {
                "note": note,
                "offer": {
                    "offer_amount": {"currency_code": currency, "value": f"{amount:.2f}"},
                    "offer_type": "REFUND",
                },
            },
        )

    async def escalate(self, dispute_id: str, note: str) -> dict:
        return self._mutate(dispute_id, "escalate", {"note": note})

    async def send_message(self, dispute_id: str, message: str) -> dict:
        return self._mutate(dispute_id, "send_message", {"message": message})

    async def adjudicate(self, dispute_id: str, outcome: str) -> dict:
        return self._mutate(dispute_id, "adjudicate", {"adjudication_outcome": outcome})

    async def require_evidence(self, dispute_id: str, action: str = "SELLER_EVIDENCE") -> dict:
        return self._mutate(dispute_id, "require_evidence", {"requirement": action})

    async def create_dispute_buyer(self, dispute: dict, buyer_email: str) -> dict:
        new_id = dispute.get("dispute_id") or f"PP-D-{9000 + len(self._disputes)}"
        dispute = dict(dispute, dispute_id=new_id)
        self._disputes[new_id] = dispute
        return {"dispute_id": new_id, "status": dispute.get("status", "REQUIRED_ACTION"), "links": []}


# ---------------------------------------------------------------------------
# dispute mapping + factory


def dispute_fields_from_paypal(d: dict) -> dict:
    """Map a PayPal dispute object onto Dispute-model field values.

    Isolated so the shape assumptions live in exactly one place.
    """
    amount = d.get("dispute_amount") or {}
    txns = d.get("disputed_transactions") or []
    txn_id = txns[0].get("seller_transaction_id", "") if txns else d.get("transaction_id", "")
    buyer_obj = d.get("buyer") or {}
    if isinstance(buyer_obj, str):
        buyer = buyer_obj or "unknown"
    else:
        name = buyer_obj.get("name")
        if isinstance(name, dict):
            name = name.get("full_name")
        buyer = buyer_obj.get("email") or name or "unknown"
    messages = [m for m in (d.get("messages") or []) if str(m.get("posted_by", "")).upper() == "BUYER"]
    body = "\n".join(m.get("content", "") for m in messages) or d.get("buyer_claim", "")
    reason = d.get("reason", "")
    value = amount.get("value", "0")
    return {
        "paypal_dispute_id": d.get("dispute_id", ""),
        "buyer": buyer,
        "reason": reason,
        "title": f"{reason.replace('_', ' ').title()} — {amount.get('currency_code', 'USD')} {value}",
        "body": body,
        "amount": float(value or 0.0),
        "currency": amount.get("currency_code", "USD"),
        "transaction_id": txn_id,
    }


def get_paypal_client(settings):
    """Factory: mock when CHARGEBUNNY_PAYPAL_MOCK=1 or no credentials are set."""
    if settings.paypal_mock or not (settings.paypal_client_id and settings.paypal_client_secret):
        log.info("PayPal client: mock (fixtures: %s)", settings.paypal_fixture_dir)
        return MockPayPalClient([settings.paypal_fixture_dir])
    log.info("PayPal client: sandbox (%s)", settings.paypal_base_url)
    return PayPalClient(
        settings.paypal_client_id,
        settings.paypal_client_secret,
        base_url=settings.paypal_base_url,
        timeout=settings.request_timeout_s,
    )
