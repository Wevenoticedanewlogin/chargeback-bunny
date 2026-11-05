"""Chargeback Bunny HTTP API — FastAPI surface over the triage pipeline.

One Store, one TriagePipeline, one PayPal client per app instance. Every
endpoint returns plain JSON via pydantic .model_dump(); validation errors
surface as 422 (malformed payloads) or 404 (unknown ids) with a `detail`
message.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .db import Store
from .models import Autonomy, Decision, RiskTier, StageResult
from .paypal import dispute_fields_from_paypal, get_paypal_client
from .pipeline import TriagePipeline

log = logging.getLogger("chargebackbunny.server")

DASHBOARD_HTML = Path(__file__).resolve().parent / "dashboard" / "index.html"

_DASHBOARD_STUB = """<!doctype html>
<html><head><title>Chargeback Bunny</title></head>
<body style="background:#00132b;color:#cfe4ff;font-family:system-ui;display:grid;place-items:center;height:100vh">
<div style="text-align:center">
<h1>Chargeback Bunny</h1>
<p>Dashboard is not installed yet (chargebackbunny/dashboard/index.html missing).<br>
The API is live — see <a style="color:#0070e0" href="/docs">/docs</a> or <a style="color:#0070e0" href="/api/health">/api/health</a>.</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# request bodies


class MerchantIn(BaseModel):
    name: str
    paypal_merchant_id: str = ""
    autonomy: Autonomy = "supervised"
    auto_respond_cap: float = 100.0


class DisputeIn(BaseModel):
    paypal_dispute_id: str = ""       # pull from PayPal instead of raw fields
    buyer: str = "unknown"
    reason: str = ""
    title: str = ""
    body: str = ""
    amount: float = 0.0
    currency: str = "USD"
    transaction_id: str = ""
    attachments: list[str] = Field(default_factory=list)


class OverridesIn(BaseModel):
    decision: Decision | None = None
    risk: RiskTier | None = None


class ReviewIn(BaseModel):
    action: Literal["approve", "override", "reject"]
    overrides: OverridesIn | None = None
    note: str = ""
    reviewer: str = "human"


class RespondIn(BaseModel):
    reviewer: str = "human"           # who clicked respond (dashboard sends this)
    note: str = ""                    # optional override for the drafted response


# ---------------------------------------------------------------------------


def create_app(db_path: str | None = None) -> FastAPI:
    settings = Settings.from_env()
    if db_path is not None:
        settings = dataclasses.replace(settings, db_path=db_path)
    store = Store(settings.db_path)
    paypal = get_paypal_client(settings)
    pipeline = TriagePipeline(store, settings, paypal=paypal)

    app = FastAPI(title="Chargeback Bunny", version="0.1.0")
    app.state.settings = settings
    app.state.store = store
    app.state.pipeline = pipeline
    app.state.paypal = paypal

    async def _run_triage(dispute_id: str) -> None:
        try:
            await pipeline.triage(dispute_id)
        except Exception:  # noqa: BLE001 — background task must not die silently
            log.exception("triage failed for dispute %s", dispute_id)

    # -- dashboard -----------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def dashboard():
        if DASHBOARD_HTML.is_file():
            return FileResponse(DASHBOARD_HTML)
        return HTMLResponse(_DASHBOARD_STUB)

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "ok": True,
            "provider": settings.provider,
            "model": settings.model,
            "judge_model": settings.judge_model or settings.model,
            "paypal_mode": getattr(paypal, "mode", settings.paypal_mode),
        }

    # -- merchants -------------------------------------------------------------

    @app.post("/api/merchants", status_code=201)
    async def create_merchant(body: MerchantIn) -> dict:
        merchant = store.create_merchant(
            name=body.name,
            paypal_merchant_id=body.paypal_merchant_id,
            autonomy=body.autonomy,
            auto_respond_cap=body.auto_respond_cap,
        )
        return merchant.model_dump()

    @app.get("/api/merchants")
    async def list_merchants() -> list[dict]:
        return [m.model_dump() for m in store.list_merchants()]

    # -- disputes --------------------------------------------------------------

    @app.post("/api/merchants/{merchant_id}/disputes", status_code=202)
    async def submit_dispute(merchant_id: str, body: DisputeIn) -> dict:
        if store.get_merchant(merchant_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown merchant: {merchant_id}")
        if body.paypal_dispute_id:
            # Pull the dispute from PayPal instead of trusting caller fields.
            existing = store.get_dispute_by_paypal_id(merchant_id, body.paypal_dispute_id)
            if existing is not None:
                return {"dispute_id": existing.id, "status": existing.status, "deduplicated": True}
            try:
                record = await paypal.get_dispute(body.paypal_dispute_id)
            except Exception as exc:
                raise HTTPException(
                    status_code=422, detail=f"PayPal dispute fetch failed: {exc}"
                ) from exc
            fields = dispute_fields_from_paypal(record)
            dispute = store.create_dispute(merchant_id=merchant_id, **fields)
        else:
            if not body.title or not body.body:
                raise HTTPException(status_code=422, detail="title and body required for raw claims")
            dispute = store.create_dispute(
                merchant_id=merchant_id,
                paypal_dispute_id="",
                buyer=body.buyer,
                reason=body.reason,
                title=body.title,
                body=body.body,
                amount=body.amount,
                currency=body.currency,
                transaction_id=body.transaction_id,
                attachments=body.attachments,
            )
        asyncio.create_task(_run_triage(dispute.id))
        return {"dispute_id": dispute.id, "status": "triaging"}

    @app.post("/api/merchants/{merchant_id}/sync")
    async def sync_merchant(merchant_id: str) -> dict:
        """Pull new disputes from the PayPal client into cases (deduped by
        paypal_dispute_id) and triage them."""
        if store.get_merchant(merchant_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown merchant: {merchant_id}")
        try:
            records = await paypal.list_disputes()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"PayPal list_disputes failed: {exc}") from exc
        created: list[str] = []
        skipped: list[str] = []
        for record in records:
            pp_id = record.get("dispute_id", "")
            if not pp_id:
                continue
            if store.get_dispute_by_paypal_id(merchant_id, pp_id) is not None:
                skipped.append(pp_id)
                continue
            dispute = store.create_dispute(merchant_id=merchant_id, **dispute_fields_from_paypal(record))
            asyncio.create_task(_run_triage(dispute.id))
            created.append(dispute.id)
        return {"created": created, "skipped_existing": skipped, "paypal_disputes_seen": len(records)}

    @app.post("/api/webhooks/paypal", status_code=202)
    async def paypal_webhook(event: dict) -> dict:
        """PayPal webhook intake for CUSTOMER.DISPUTE.* events.

        CREATED: pull the full dispute record and start triage immediately —
        the case is usually triaged before the seller even sees PayPal's email.
        UPDATED/RESOLVED: acknowledged and attached to the existing case.
        (Signature verification via verify-webhook-signature is the production
        step; sandbox demo accepts the event directly.)
        """
        event_type = str(event.get("event_type") or "")
        resource = event.get("resource") or {}
        if not event_type.startswith("CUSTOMER.DISPUTE."):
            return {"accepted": False, "reason": f"unsupported event type {event_type!r}"}
        pp_id = resource.get("dispute_id") or resource.get("id") or ""
        if not pp_id:
            raise HTTPException(status_code=422, detail="webhook resource carries no dispute id")
        merchants = store.list_merchants()
        if not merchants:
            raise HTTPException(status_code=409, detail="no merchant registered; create one first")
        merchant = merchants[0]
        existing = store.get_dispute_by_paypal_id(merchant.id, pp_id)
        if event_type == "CUSTOMER.DISPUTE.CREATED":
            if existing is not None:
                return {"accepted": True, "deduplicated": True, "dispute_id": existing.id}
            try:
                record = await paypal.get_dispute(pp_id)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"PayPal dispute fetch failed: {exc}") from exc
            dispute = store.create_dispute(merchant_id=merchant.id, **dispute_fields_from_paypal(record))
            asyncio.create_task(_run_triage(dispute.id))
            return {"accepted": True, "dispute_id": dispute.id, "triage": "started"}
        # UPDATED / RESOLVED: no re-triage — the audit trail notes the event.
        if existing is not None:
            store.save_stage_result(
                existing.id,
                StageResult(
                    stage="webhook",
                    payload={"event_type": event_type, "paypal_dispute_id": pp_id},
                    confidence=1.0,
                    rationale=f"PayPal webhook {event_type} for {pp_id}",
                    llm_used=False,
                ),
            )
        return {"accepted": True, "dispute_id": existing.id if existing else None, "triage": "skipped"}

    @app.get("/api/disputes")
    async def list_disputes(
        merchant_id: str | None = Query(default=None),
        status: str | None = Query(default=None),
    ) -> list[dict]:
        return [d.model_dump() for d in store.list_disputes(merchant_id=merchant_id, status=status)]

    @app.get("/api/disputes/{dispute_id}")
    async def get_dispute(dispute_id: str) -> dict:
        dispute = store.get_dispute(dispute_id)
        if dispute is None:
            raise HTTPException(status_code=404, detail=f"unknown dispute: {dispute_id}")
        verdict = store.get_verdict(dispute_id)
        review = store.get_review(dispute_id)
        return {
            "dispute": dispute.model_dump(),
            "stage_results": [s.model_dump() for s in store.get_stage_results(dispute_id)],
            "verdict": verdict.model_dump() if verdict else None,
            "traces": [t.model_dump() for t in store.list_traces(dispute_id)],
            "review": review.model_dump() if review else None,
        }

    @app.get("/api/queue")
    async def review_queue() -> list[dict]:
        items = []
        for dispute in store.list_queue():
            verdict = store.get_verdict(dispute.id)
            items.append({
                "dispute": dispute.model_dump(),
                "verdict": verdict.model_dump() if verdict else None,
            })
        items.sort(key=lambda i: i["dispute"]["created_at"], reverse=True)
        return items

    @app.post("/api/disputes/{dispute_id}/review")
    async def review_dispute(dispute_id: str, body: ReviewIn) -> dict:
        dispute = store.get_dispute(dispute_id)
        if dispute is None:
            raise HTTPException(status_code=404, detail=f"unknown dispute: {dispute_id}")
        if dispute.status != "awaiting_review":
            # Only queued cases accept review. "responded" is immutable
            # (re-review would re-execute the PayPal action); "triaging" is
            # racy (the in-flight pipeline would clobber the override and
            # could double-execute the PayPal action after the debate stage).
            raise HTTPException(status_code=409, detail=f"case not reviewable in status {dispute.status!r}")
        overrides = body.overrides.model_dump(exclude_none=True) if body.overrides else None
        try:
            verdict = await pipeline.apply_review(
                dispute_id,
                action=body.action,
                reviewer=body.reviewer,
                overrides=overrides or None,
                note=body.note,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.action == "reject":
            # Rejected: case closed, no PayPal action, review recorded.
            store.update_dispute_status(dispute_id, "responded")
        elif body.action in ("approve", "override"):
            if verdict.decision in ("accept", "contest", "refund") and dispute.paypal_dispute_id:
                # Human approval is the gate: execute the PayPal action now.
                # On failure the case stays queued and the respond button retries.
                try:
                    await pipeline.execute_response(dispute_id, note=body.note or "")
                except Exception as exc:
                    raise HTTPException(status_code=502, detail=f"approved but PayPal respond failed: {exc}") from exc
            else:
                # escalate / scam_claim (or a case without a PayPal dispute id):
                # no automated action exists — the human's decision closes it.
                store.update_dispute_status(dispute_id, "responded")
        return verdict.model_dump()

    @app.post("/api/disputes/{dispute_id}/respond")
    async def respond_dispute(dispute_id: str, body: RespondIn | None = None) -> dict:
        """Execute the verdict's action in PayPal (accept_claim /
        provide_evidence / make_offer). Allowed only when the verdict was
        auto-finalized by policy or a human approved/overrode it via /review.
        The PayPal API response is recorded in the audit trail."""
        dispute = store.get_dispute(dispute_id)
        if dispute is None:
            raise HTTPException(status_code=404, detail=f"unknown dispute: {dispute_id}")
        verdict = store.get_verdict(dispute_id)
        if verdict is None:
            raise HTTPException(status_code=422, detail="dispute has no verdict yet; triage incomplete")
        review = store.get_review(dispute_id)
        approved = review is not None and review.action in ("approve", "override")
        if not approved:
            raise HTTPException(
                status_code=403,
                detail="respond requires an auto-finalized verdict or human approval via /review",
            )
        if dispute.status == "responded":
            raise HTTPException(status_code=422, detail="dispute already responded")
        if verdict.decision in ("escalate", "scam_claim"):
            raise HTTPException(
                status_code=422,
                detail=f"decision {verdict.decision!r} has no automated PayPal response",
            )
        try:
            return await pipeline.execute_response(dispute_id, note=(body.note if body else ""))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"PayPal respond failed: {exc}") from exc

    # -- stats -----------------------------------------------------------------

    @app.get("/api/stats")
    async def stats() -> dict:
        data = dict(store.stats())
        data["provider"] = settings.provider
        data["model"] = settings.model
        data["paypal_mode"] = getattr(paypal, "mode", settings.paypal_mode)
        return data

    return app
