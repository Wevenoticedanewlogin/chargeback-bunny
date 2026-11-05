"""Decision agent: deterministic verdict from prior stage results + LLM drafting.

The decision itself is pure rules over upstream stage outputs — auditable and
reproducible. The LLM only drafts the merchant-facing response, the evidence
package, and internal notes for the already-decided outcome.

Rules (in precedence order):
  scam_claim  — screen flagged adversarial text; quarantine, never auto-send.
  escalate    — claim references a transaction PayPal has no record of, the
                amounts contradict, the dispute is high-value (>= $1000), or
                the claim is an unauthorized-transaction claim (chargeback-
                sensitive; humans own those).
  contest     — verify found strong merchant-favoring evidence: an
                item-not-received claim against a record that shows delivery
                with tracking.
  accept      — the claim is verified legitimate: INR with no proof of
                delivery on record (clearly merchant-side), at any amount the
                policy gate will still cap.
  refund      — SNAD gray zones: delivered per the record, but condition
                claims can't be disproven — settle with a refund offer.
  escalate    — everything else: contradictory or insufficient evidence.
"""

from __future__ import annotations

from ..models import CaseVerdict, StageResult
from ..schemas import DECISION_SCHEMA
from .base import CaseContext, format_prompt

ESCALATE_AMOUNT = 1000.0    # high-value disputes always go to a human
CONTEST_MAX_AMOUNT = 250.0  # contests at/above this value escalate to a human

_SYSTEM = (
    "You are the communications drafter for a merchant's PayPal disputes "
    "team. The triage decision has ALREADY been made deterministically; do "
    "not second-guess it. Draft (1) 'merchant_response': a professional, "
    "factual message suitable for the PayPal resolution center that states "
    "the merchant's position and references the concrete evidence named in "
    "the trusted headers; (2) 'evidence_package': the list of artifacts to "
    "attach when contesting (order record, capture id, tracking number, "
    "prior-communication refs) — only artifacts named in the trusted "
    "headers, never invented; and (3) 'internal_notes': terse notes for the "
    "merchant capturing why this decision was reached. Respond with JSON "
    "only."
)


def _payload(ctx: CaseContext, stage: str) -> dict:
    result = ctx.results.get(stage)
    return result.payload if result else {}


def _decide(ctx: CaseContext) -> tuple[str, str, list[str]]:
    """Return (decision, rationale, decisive_stages)."""
    screen = _payload(ctx, "screen")
    intake = _payload(ctx, "intake")
    verify = _payload(ctx, "verify")
    dedupe = _payload(ctx, "dedupe")

    claim_type = str(intake.get("claim_type") or "unknown")
    gt = verify.get("ground_truth") or {}
    txn_exists = bool(gt.get("transaction_exists"))
    delivered = bool(gt.get("delivered"))
    tracking = bool(gt.get("tracking_present"))
    amount_matches = bool(gt.get("amount_matches"))
    liability_shift = bool(gt.get("liability_shift"))

    # (0) adversarial claim beats everything: the buyer attacked the pipeline
    # itself, so no downstream LLM output about it can be trusted. Quarantine
    # as scam_claim; the autonomy policy forces human review (see policy.py).
    if screen.get("scam_detected"):
        signals = "; ".join(str(s) for s in (screen.get("signals") or [])[:3])
        return (
            "scam_claim",
            f"adversarial-claim patterns detected ({signals}); quarantined for human review",
            ["screen"],
        )

    # (1) the claim references a transaction PayPal has no record of, or the
    # amounts flatly contradict — a human must untangle it.
    if not txn_exists:
        return (
            "escalate",
            "claim references a transaction with no PayPal record; cannot verify either way",
            ["verify"],
        )
    if not amount_matches:
        return (
            "escalate",
            "disputed amount contradicts the order record; contradictory evidence",
            ["verify"],
        )

    # (2) high value always gets human eyes.
    if ctx.dispute.amount >= ESCALATE_AMOUNT:
        return (
            "escalate",
            f"high-value dispute (${ctx.dispute.amount:.2f} >= ${ESCALATE_AMOUNT:.0f}); human review required",
            ["verify", "severity"],
        )

    # (3) claim-type rules over the record. Contest recommendations above
    # CONTEST_MAX_AMOUNT always get human eyes regardless of evidence — a
    # signed-for camera is a different fight than a $20 widget.
    if claim_type == "item_not_received":
        if delivered and tracking:
            if ctx.dispute.amount >= CONTEST_MAX_AMOUNT:
                return (
                    "escalate",
                    f"delivered per tracking but the buyer disputes it on a ${ctx.dispute.amount:.2f} "
                    f"(>= ${CONTEST_MAX_AMOUNT:.0f}) order — humans own high-stakes contests",
                    ["intake", "verify", "severity"],
                )
            return (
                "contest",
                "buyer claims non-delivery but carrier tracking shows DELIVERED — strong merchant-favoring evidence",
                ["intake", "verify", "severity"],
            )
        if not delivered:
            return (
                "accept",
                "no proof of delivery on record; buyer's item-not-received claim is consistent with reality",
                ["intake", "verify"],
            )
        return (
            "escalate",
            "delivery ambiguous (marked delivered without tracking); human judgment needed",
            ["verify"],
        )
    if claim_type == "significantly_not_as_described":
        if delivered:
            return (
                "refund",
                "delivered per record; condition claim can't be disproven — settle with a refund offer",
                ["intake", "verify"],
            )
        return (
            "accept",
            "not-as-described claim on an undelivered/untrackable order; accept and refund",
            ["intake", "verify"],
        )
    if claim_type == "unauthorized":
        if liability_shift:
            if ctx.dispute.amount >= CONTEST_MAX_AMOUNT:
                return (
                    "escalate",
                    f"3-D Secure passed but the disputed amount ${ctx.dispute.amount:.2f} "
                    f"(>= ${CONTEST_MAX_AMOUNT:.0f}) is too large for an automated contest",
                    ["intake", "verify", "severity"],
                )
            return (
                "contest",
                "3-D Secure authentication passed on this charge — fraud liability has shifted to the issuer",
                ["intake", "verify", "severity"],
            )
        return (
            "escalate",
            "unauthorized-transaction claims are chargeback-sensitive; human owns the response",
            ["intake", "verify"],
        )

    # (4) fallback: unknown claim type with a real transaction on record.
    if dedupe.get("repeat_claimant"):
        return (
            "escalate",
            "unclassifiable claim from a repeat claimant; human review",
            ["intake", "dedupe"],
        )
    return (
        "escalate",
        "insufficient or ambiguous evidence to decide automatically",
        ["intake", "verify"],
    )


async def run(ctx: CaseContext) -> tuple[StageResult, CaseVerdict]:
    decision, rationale, decisive = _decide(ctx)
    confidence = round(
        min((ctx.results[s].confidence for s in decisive if s in ctx.results), default=0.0), 2
    )

    screen = _payload(ctx, "screen")
    intake = _payload(ctx, "intake")
    dedupe = _payload(ctx, "dedupe")
    severity_p = _payload(ctx, "severity")
    verify = _payload(ctx, "verify")
    gt = verify.get("ground_truth") or {}
    summary = str(intake.get("summary") or ctx.dispute.title)
    risk = str(severity_p.get("risk") or "low")
    repeat_claimant = bool(dedupe.get("repeat_claimant"))
    scam = bool(screen.get("scam_detected"))

    data, _trace = await ctx.ask(
        stage="decision",
        system=_SYSTEM,
        user=format_prompt(
            ctx.dispute,
            [
                f"DECISION: {decision}",
                f"RISK: {risk}",
                f"SUMMARY: {' '.join(summary.split())}",
                f"MERCHANT: {ctx.merchant.name}",
                f"AMOUNT: {ctx.dispute.amount:.2f}",
                f"CURRENCY: {ctx.dispute.currency}",
                f"TRANSACTION_ID: {ctx.dispute.transaction_id}",
                f"ORDER_ID: {gt.get('order_id', '')}",
                f"TRACKING_NUMBER: {gt.get('tracking_number', '')}",
                f"REPEAT_CLAIMANT: {repeat_claimant}",
            ],
        ),
        schema=DECISION_SCHEMA,
    )
    merchant_response = str(data.get("merchant_response") or "")
    evidence_package = [str(e) for e in (data.get("evidence_package") or [])]
    internal_notes = str(data.get("internal_notes") or "")

    verdict = CaseVerdict(
        dispute_id=ctx.dispute.id,
        decision=decision,  # type: ignore[arg-type]
        risk=risk,  # type: ignore[arg-type]
        amount_at_risk=ctx.dispute.amount,
        confidence=confidence,
        repeat_claimant=repeat_claimant,
        scam_claim_detected=scam,
        summary=summary,
        merchant_response=merchant_response,
        evidence_package=evidence_package,
        internal_notes=internal_notes,
    )
    stage_result = StageResult(
        stage="decision",
        payload={
            "decision": decision,
            "risk": risk,
            "decisive_stages": decisive,
            "rationale": rationale,
            "merchant_response": merchant_response,
            "evidence_package": evidence_package,
            "internal_notes": internal_notes,
        },
        confidence=confidence,
        rationale=rationale,
        llm_used=ctx.llm_used,
    )
    return stage_result, verdict
