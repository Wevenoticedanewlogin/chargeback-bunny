"""LLM providers for Chargeback Bunny.

Providers:
  openai-compatible -> any OpenAI-compatible chat-completions endpoint
                       (CHARGEBUNNY_BASE_URL + MODEL_API_KEY + CHARGEBUNNY_MODEL).
  offline           -> Deterministic rules engine implementing the same stage
                       contract with zero network. Used when no API key is
                       configured; every trace is labeled provider="offline",
                       model="offline-heuristic".

Prompt contract (agents -> providers): the ``user`` message for a stage call
contains the claim as fenced ``TITLE:`` / ``REASON:`` / ``AMOUNT:`` /
``BUYER:`` / ``BODY:`` lines, plus stage-specific trusted ``KEY: value``
header lines after the fence (DETERMINISTIC_*, CANDIDATE …, GROUND_TRUTH_*,
DECISION:, RISK:, ROLE:, …). The offline engine parses those same lines, so
offline and LLM paths consume identical prompts.
"""

from __future__ import annotations

import json
import re
import time
from typing import Protocol

import httpx

from .models import LLMResult

# ---------------------------------------------------------------------------
# shared helpers


def extract_json(text: str) -> dict:
    """Best-effort extraction of the first JSON object from model output."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            return json.loads(t[start : end + 1])
        raise


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResult: ...


# ---------------------------------------------------------------------------
# hosted provider — OpenAI-compatible chat completions


class ChatCompletionsClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        reasoning_effort: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._timeout = timeout
        self._effort = reasoning_effort

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResult:
        use_model = model or self.model
        body: dict = {
            "model": use_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        if self._effort:
            body["reasoning_effort"] = self._effort
        if json_schema:
            schema = {k: v for k, v in json_schema.items() if k != "x-stage"}
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{json_schema.get('x-stage', 'stage')}_result",
                    "schema": schema,
                    "strict": True,
                },
            }
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base}/chat/completions", json=body, headers=headers)
            if resp.status_code == 400 and json_schema:
                # Provider rejected structured output; fall back to json_object
                # with the schema spelled out in the system prompt.
                body["messages"][0]["content"] += (
                    "\nRespond with ONLY a JSON object matching this JSON Schema:\n"
                    + json.dumps(json_schema)
                )
                body["response_format"] = {"type": "json_object"}
                resp = await client.post(f"{self._base}/chat/completions", json=body, headers=headers)
            if resp.status_code in (429, 500, 502, 503):
                import asyncio

                await asyncio.sleep(2.0)
                resp = await client.post(f"{self._base}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.monotonic() - t0) * 1000)
        usage = data.get("usage") or {}
        content = (data["choices"][0].get("message") or {}).get("content") or ""
        return LLMResult(
            text=content,
            provider=self.provider,
            model=data.get("model", use_model),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# offline deterministic engine


_REASON_TO_CLAIM_TYPE = {
    "ITEM_NOT_RECEIVED": "item_not_received",
    "MERCHANDISE_OR_SERVICE_NOT_RECEIVED": "item_not_received",
    "UNAUTHORIZED": "unauthorized",
    "UNAUTHORISED": "unauthorized",
    "UNAUTHORIZED_TRANSACTION": "unauthorized",
    "UNAUTHORISED_TRANSACTION": "unauthorized",
    "SNAD": "significantly_not_as_described",
    "MERCHANDISE_OR_SERVICE_NOT_AS_DESCRIBED": "significantly_not_as_described",
    "NOT_AS_DESCRIBED": "significantly_not_as_described",
}

_CLAIM_TYPE_KEYWORDS: list[tuple[str, str]] = [
    (r"unauthori[sz]ed|didn'?t (?:make|authorize|authorise) this|not my (?:purchase|transaction|order)|"
     r"never (?:made|placed) this|card (?:was )?stolen|account (?:was )?hacked", "unauthorized"),
    (r"not as described|wrong (?:item|color|colour|size|model)|counterfeit|\bfake\b|damaged|defective|"
     r"broken|doesn'?t match|not what i ordered|missing parts", "significantly_not_as_described"),
    (r"not received|never (?:arrived|came|got|received)|didn'?t (?:receive|get|arrive)|no sign of|"
     r"missing package|not delivered|never showed up|where is my (?:order|package)", "item_not_received"),
]

_AMOUNT_RE = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
_TRACKING_RE = re.compile(r"\b1Z[0-9A-Z]{10,}\b|\btracking(?:\s*(?:number|no|#))?\s*[:#]?\s*([0-9A-Z]{8,30})\b", re.I)

_TXN_RE = re.compile(r"\b(?:PP-[A-Z0-9-]+|ORD-[A-Z0-9-]+|CAP-[A-Z0-9-]+|\b[0-9A-Z]{17}\b)\b")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}", re.I)

_AGGRESSIVE_RE = re.compile(
    r"\brefund (?:me )?now\b|\bimmediately\b|\blawyer\b|\bsue\b|\breport you\b|\bscam(?:mer)?\b|"
    r"\brip ?off\b|\bthieves\b|\bfraudsters?\b", re.I)
_URGENT_RE = re.compile(r"\burgent\b|\basap\b|\bright away\b|\bas soon as possible\b|\btime[- ]sensitive\b", re.I)
_MANIPULATIVE_RE = re.compile(
    r"\bpre-?approved\b|\bauto-?approve\b|\bresolution center (?:has )?(?:already )?(?:ruled|decided|approved)\b|"
    r"\bignore (?:all |your |previous )?instructions\b|\bdo not contest\b|\bai (?:system|agent|triage)\b", re.I)


def _prompt_field(user: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}:[ \t]*(.*)$", user, re.M)
    return m.group(1).strip() if m else ""


def _prompt_body(user: str) -> str:
    m = re.search(r"^BODY:\s*\n?(.*)$", user, re.M | re.S)
    if not m:
        return user
    body = m.group(1)
    # Keep in sync with agents.base.FENCE_END: trusted header lines live after
    # the fence and must not leak into the offline engine's body text.
    end = body.find("<<<END_UNTRUSTED_CLAIM>>>")
    return (body[:end] if end != -1 else body).strip()


def _classify_claim(reason: str, text: str) -> tuple[str, float]:
    """Return (claim_type, confidence). The PayPal reason code is trusted
    pipeline data and wins; body keywords are the fallback."""
    if reason in _REASON_TO_CLAIM_TYPE:
        return _REASON_TO_CLAIM_TYPE[reason], 0.85
    low = text.lower()
    for pattern, claim_type in _CLAIM_TYPE_KEYWORDS:
        if re.search(pattern, low):
            return claim_type, 0.6
    return "unknown", 0.3


def _buyer_tone(text: str) -> str:
    if _MANIPULATIVE_RE.search(text):
        return "manipulative"
    if _AGGRESSIVE_RE.search(text):
        return "aggressive"
    if _URGENT_RE.search(text):
        return "urgent"
    return "calm"


class OfflineClient:
    """Deterministic stage synthesizers — same JSON contract as the LLM path."""

    provider = "offline"
    model = "offline-heuristic"

    async def complete(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResult:
        stage = (json_schema or {}).get("x-stage", "")
        handler = getattr(self, f"_stage_{stage}", None)
        if handler is None:
            payload = {"error": f"offline engine has no synthesizer for stage {stage!r}"}
        else:
            payload = handler(user)
        return LLMResult(text=json.dumps(payload), provider=self.provider, model=self.model, latency_ms=1)

    # -- stage synthesizers -------------------------------------------------

    def _stage_screen(self, user: str) -> dict:
        score = float(_prompt_field(user, "DETERMINISTIC_SCORE") or 0.0)
        critical = (_prompt_field(user, "DETERMINISTIC_CRITICAL") or "") == "True"
        raw = _prompt_field(user, "DETERMINISTIC_SIGNALS")
        signals = [s.strip() for s in raw.split(";") if s.strip() and s.strip() != "none"]
        detected = critical or score >= 2.0
        return {
            "scam_detected": detected,
            "signals": signals,
            "attack_summary": (
                "deterministic scanner found adversarial-claim / manipulation patterns"
                if detected
                else "no manipulation patterns detected"
            ),
            "confidence": 0.9 if detected else 0.6,
        }

    def _stage_intake(self, user: str) -> dict:
        title = _prompt_field(user, "TITLE")
        reason = _prompt_field(user, "REASON")
        amount_raw = _prompt_field(user, "AMOUNT")
        body = _prompt_body(user)
        text = f"{title}\n{body}"
        claim_type, confidence = _classify_claim(reason, text)
        m = _AMOUNT_RE.search(text)
        claimed_amount = float(amount_raw or 0.0) or (float(m.group(1).replace(",", "")) if m else 0.0)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        key_facts = [
            s[:200] for s in sentences
            if _DATE_RE.search(s) or _TRACKING_RE.search(s) or re.search(r"\bdeliver(?:ed|y)\b", s, re.I)
        ][:6]
        evidence_refs = sorted(set(_TXN_RE.findall(text) + [t for t in _TRACKING_RE.findall(text) if t]))[:8]
        summary = sentences[0][:300] if sentences else title or "(empty claim)"
        return {
            "summary": summary,
            "claim_type": claim_type,
            "claimed_amount": claimed_amount,
            "key_facts": key_facts,
            "evidence_refs": evidence_refs,
            "buyer_tone": _buyer_tone(text),
            "confidence": confidence,
        }

    def _stage_dedupe(self, user: str) -> dict:
        best_id, best_sim = None, 0.0
        same_buyer_ids: list[str] = []
        for m in re.finditer(r"CANDIDATE\s+(\S+)\s+sim=([0-9.]+)\s+same_buyer=(\w+)", user):
            cand_id, sim, same = m.group(1), float(m.group(2)), m.group(3) == "True"
            if same:
                same_buyer_ids.append(cand_id)
            if sim > best_sim:
                best_id, best_sim = cand_id, sim
        if same_buyer_ids:
            return {
                "repeat_claimant": True,
                "prior_dispute_ids": same_buyer_ids,
                "similarity_rationale": (
                    f"same buyer on {len(same_buyer_ids)} prior dispute(s): {', '.join(same_buyer_ids)}"
                ),
                "confidence": 0.9,
            }
        if best_id and best_sim >= 0.6:
            return {
                "repeat_claimant": True,
                "prior_dispute_ids": [best_id],
                "similarity_rationale": f"near-identical claim text (similarity {best_sim:.2f} >= 0.60)",
                "confidence": min(0.9, best_sim + 0.2),
            }
        return {
            "repeat_claimant": False,
            "prior_dispute_ids": [],
            "similarity_rationale": (
                f"no repeat-claimant signal (best similarity {best_sim:.2f})"
                if best_id
                else "no prior disputes to compare"
            ),
            "confidence": 0.7,
        }

    def _stage_verify(self, user: str) -> dict:
        """Fixture-consistency check: parse the GROUND_TRUTH_* headers the
        verify agent computed from the PayPal record and judge the claim
        against them."""
        exists = (_prompt_field(user, "GROUND_TRUTH_TXN_EXISTS") or "") == "True"
        delivered = (_prompt_field(user, "GROUND_TRUTH_DELIVERED") or "") == "True"
        tracking = (_prompt_field(user, "GROUND_TRUTH_TRACKING_PRESENT") or "") == "True"
        amount_matches = (_prompt_field(user, "GROUND_TRUTH_AMOUNT_MATCHES") or "") == "True"
        claim_type = _prompt_field(user, "CLAIM_TYPE")
        liability = (_prompt_field(user, "GROUND_TRUTH_LIABILITY_SHIFT") or "") == "True"
        txn = _prompt_field(user, "TRANSACTION_ID")

        findings: list[str] = []
        if not exists:
            findings.append(f"no PayPal record found for referenced transaction {txn or '(none provided)'}")
            return {"verified": False, "evidence_strength": "none", "findings": findings, "confidence": 0.9}

        findings.append("transaction exists in PayPal records")
        if amount_matches:
            findings.append("disputed amount matches the order record")
        else:
            findings.append("AMOUNT MISMATCH: disputed amount differs from the order record")
        if tracking:
            findings.append("tracking number present on the order")
        if delivered:
            findings.append("carrier tracking shows DELIVERED")
        if liability:
            findings.append("3-D Secure authentication passed — fraud liability has shifted to the issuer")

        # Is the buyer's claim consistent with the record?
        if claim_type == "item_not_received":
            verified = not delivered
        elif claim_type == "significantly_not_as_described":
            verified = False  # condition claims can't be confirmed from the record alone
        elif claim_type == "unauthorized":
            verified = False
        else:
            verified = False

        if (delivered and tracking) or liability:
            strength = "strong"
        elif tracking or delivered:
            strength = "weak"
        else:
            strength = "weak"  # record exists but proves nothing about fulfillment
            if claim_type == "item_not_received":
                findings.append("no proof of delivery on record — buyer claim stands")
        return {"verified": verified, "evidence_strength": strength, "findings": findings, "confidence": 0.95}

    def _stage_severity(self, user: str) -> dict:
        tier = _prompt_field(user, "TIER") or "low"
        amount = _prompt_field(user, "AMOUNT") or "0"
        bumped = (_prompt_field(user, "BUMPED") or "") == "True"
        bump_reason = _prompt_field(user, "BUMP_REASON")
        justification = f"amount at risk ${amount} maps to {tier} tier"
        if bumped:
            justification += f"; escalated one tier ({bump_reason})"
        return {"risk_justification": justification + ".", "confidence": 0.9}

    def _stage_decision(self, user: str) -> dict:
        decision = _prompt_field(user, "DECISION") or "escalate"
        risk = _prompt_field(user, "RISK") or "low"
        summary = _prompt_field(user, "SUMMARY")
        merchant = _prompt_field(user, "MERCHANT") or "the merchant"
        amount = _prompt_field(user, "AMOUNT") or "0.00"
        currency = _prompt_field(user, "CURRENCY") or "USD"
        txn = _prompt_field(user, "TRANSACTION_ID")
        tracking = _prompt_field(user, "TRACKING_NUMBER")
        order_id = _prompt_field(user, "ORDER_ID")
        title = _prompt_field(user, "TITLE")

        evidence_package: list[str] = []
        if order_id:
            evidence_package.append(f"PayPal order record {order_id}")
        if txn:
            evidence_package.append(f"Transaction/capture id {txn}")
        if tracking:
            evidence_package.append(f"Carrier tracking number {tracking} (proof of delivery)")
        if decision == "contest" and not evidence_package:
            evidence_package.append("Order and payment records from the PayPal dashboard")

        responses = {
            "contest": (
                f"We are contesting this dispute. Our records show the order was fulfilled: "
                f"transaction {txn or 'on record'} was completed"
                + (f" and carrier tracking {tracking} shows delivery." if tracking else ".")
                + " The attached evidence package includes the order record and proof of delivery."
            ),
            "accept": (
                f"After reviewing the claim \"{title}\", we accept it. We were unable to locate "
                f"proof of fulfillment for this transaction, and customer satisfaction comes first. "
                f"A full refund of {amount} {currency} will be issued through PayPal."
            ),
            "refund": (
                f"We'd like to resolve this amicably. While the order was delivered as addressed, "
                f"we take 'not as described' concerns seriously and are offering a refund settlement "
                f"of {amount} {currency} to close this case."
            ),
            "escalate": (
                "This case requires manual review by our disputes team before we respond to PayPal. "
                "The claim involves high value or evidence that contradicts our records."
            ),
            "scam_claim": (
                "This claim was flagged by automated screening as potentially manipulative and has "
                "been quarantined for human review. No automated response will be sent."
            ),
        }
        return {
            "merchant_response": responses.get(decision, responses["escalate"]),
            "evidence_package": evidence_package,
            "internal_notes": (
                f"offline decision draft; decision={decision} risk={risk} merchant={merchant} "
                f"summary={summary[:120]}"
            ),
        }

    def _stage_debate_arg(self, user: str) -> dict:
        role = _prompt_field(user, "ROLE") or "skeptic"
        evidence = (_prompt_field(user, "EVIDENCE") or "none").lower()
        delivered = (_prompt_field(user, "DELIVERED") or "") == "True"
        tracking = (_prompt_field(user, "TRACKING") or "") == "True"
        repeat = (_prompt_field(user, "REPEAT_CLAIMANT") or "") == "True"

        merchant_points: list[str] = []
        buyer_points: list[str] = []
        if delivered and tracking:
            merchant_points.append("carrier tracking shows the order was delivered to the buyer's address")
        if repeat:
            merchant_points.append("buyer is a repeat claimant — prior disputes on record")
        if evidence == "strong":
            merchant_points.append("PayPal order record strongly supports fulfillment")
        if not tracking:
            buyer_points.append("no tracking number on record — delivery cannot be independently confirmed")
        if not delivered:
            buyer_points.append("carrier data does not confirm delivery")
        if evidence != "strong":
            buyer_points.append("record evidence is not conclusive")
        if not buyer_points:
            buyer_points.append("tracking shows delivery to an address, not necessarily into the buyer's hands")
        if not merchant_points:
            merchant_points.append("the PayPal transaction record matches the disputed amount")

        if role == "skeptic":
            # Skeptic steel-mans the BUYER: argue against contesting.
            return {"strongest_points": buyer_points, "weakest_points": merchant_points, "confidence": 0.6}
        return {"strongest_points": merchant_points, "weakest_points": buyer_points, "confidence": 0.6}

    def _stage_debate(self, user: str) -> dict:
        evidence = (_prompt_field(user, "EVIDENCE") or "none").lower()
        delivered = (_prompt_field(user, "DELIVERED") or "") == "True"
        tracking = (_prompt_field(user, "TRACKING") or "") == "True"
        liability = (_prompt_field(user, "LIABILITY_SHIFT") or "") == "True"
        sustained = (evidence == "strong" and delivered and tracking) or liability
        return {
            "sustained": sustained,
            "recommended_decision": "contest" if sustained else "refund",
            "rationale": (
                "offline judge: contest stands — "
                + ("3-D Secure liability shift to the issuer" if liability and not (delivered and tracking)
                   else "delivered with tracking and strong record evidence")
                if sustained
                else f"offline judge: record too thin to contest (evidence={evidence}, "
                     f"delivered={delivered}, tracking={tracking}, liability_shift={liability}); "
                     "recommend settling via refund"
            ),
            "confidence": 0.75 if sustained else 0.6,
        }


# ---------------------------------------------------------------------------


def get_client(settings) -> LLMClient:
    """Build the LLM client for the given Settings."""
    if settings.provider == "openai-compatible" and settings.api_key:
        return ChatCompletionsClient(
            provider=settings.provider,
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            timeout=settings.request_timeout_s,
            reasoning_effort=getattr(settings, "reasoning_effort", None),
        )
    return OfflineClient()
