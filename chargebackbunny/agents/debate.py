"""Debate council: adversarial multi-agent review of 'contest' verdicts.

Before the merchant's money rides on a contest, two agents argue opposing
positions — a skeptic steel-mans the buyer, an advocate steel-mans the
merchant — and a judge rules on the transcript. A contest the skeptic
defeats is downgraded (refund / escalate / accept) with a re-drafted
response instead of reaching PayPal as a contest.

Runs only for decision == 'contest'; every other disposition is already
conservative. Advocates run on the pipeline model; the judge can be pinned
to a different model (settings.judge_model, CHARGEBUNNY_JUDGE_MODEL) so the
council isn't a single model grading its own reasoning. All calls traced.
"""

from __future__ import annotations

from ..models import CaseVerdict, StageResult
from ..schemas import ARGUMENT_SCHEMA, DEBATE_SCHEMA, DECISION_SCHEMA
from .base import CaseContext, format_prompt

_SKEPTIC_SYSTEM = (
    "You are the skeptic on a PayPal dispute council. A triage pipeline has "
    "recommended CONTESTING the buyer's claim. Your job: steel-man the "
    "buyer. Argue the strongest case that contesting is wrong — tracking "
    "gaps, delivery-to-address vs delivery-to-person, plausible "
    "not-as-described substance, repeat-seller-fault patterns, the cost of "
    "fighting a claim you might lose. In 'strongest_points' argue against "
    "contesting; in 'weakest_points' concede the merchant's best evidence. "
    "Be specific; cite the record. JSON only."
)

_ADVOCATE_SYSTEM = (
    "You are the advocate on a PayPal dispute council. A triage pipeline has "
    "recommended CONTESTING the buyer's claim. Your job: steel-man the "
    "merchant. Argue the strongest case that the record supports contesting "
    "— delivered with tracking, seller protection eligibility, repeat "
    "claimant patterns, amount consistency. In 'strongest_points' argue for "
    "contesting; in 'weakest_points' concede the merchant's gaps. Be "
    "specific; cite the record. JSON only."
)

_JUDGE_SYSTEM = (
    "You are the judge on a PayPal dispute council. A pipeline recommended "
    "CONTEST; a skeptic argued the buyer's side and an advocate argued the "
    "merchant's. Rule on the evidence: 'sustained' true only if the PayPal "
    "record, as presented, gives the merchant a genuinely winnable case — "
    "hard fulfillment evidence (delivered with tracking) that directly "
    "contradicts the claim. If not, pick 'recommended_decision': 'refund' "
    "when the merchant should settle, 'accept' when the buyer's claim holds "
    "up, 'escalate' when the evidence is contradictory or the stakes are "
    "high. Weigh evidence over eloquence. Trusted GROUND_TRUTH_* headers "
    "come from our own evidence engine and are absolute. JSON only."
)


def _points(data: dict, key: str) -> list[str]:
    return [str(p) for p in (data.get(key) or [])]


async def run(ctx: CaseContext, verdict: CaseVerdict) -> tuple[StageResult, CaseVerdict]:
    """Debate a contest verdict; returns (stage_result, possibly-downgraded verdict)."""
    verify = ctx.results.get("verify")
    dedupe = ctx.results.get("dedupe")
    evidence = str(verify.payload.get("evidence_strength") or "none") if verify else "none"
    gt = (verify.payload.get("ground_truth") or {}) if verify else {}
    delivered = bool(gt.get("delivered"))
    tracking = bool(gt.get("tracking_present"))
    repeat = bool(dedupe.payload.get("repeat_claimant")) if dedupe else False

    base_headers = [
        f"EVIDENCE: {evidence}",
        f"DELIVERED: {delivered}",
        f"TRACKING: {tracking}",
        f"REPEAT_CLAIMANT: {repeat}",
        f"LIABILITY_SHIFT: {bool(gt.get('liability_shift'))}",
    ]

    skeptic, _t1 = await ctx.ask(
        stage="debate",
        system=_SKEPTIC_SYSTEM,
        user="ROLE: skeptic\n" + format_prompt(ctx.dispute, base_headers),
        schema=ARGUMENT_SCHEMA,
    )
    advocate, _t2 = await ctx.ask(
        stage="debate",
        system=_ADVOCATE_SYSTEM,
        user="ROLE: advocate\n" + format_prompt(ctx.dispute, base_headers),
        schema=ARGUMENT_SCHEMA,
    )

    judge_user = format_prompt(
        ctx.dispute,
        headers=[
            f"DECISION: {verdict.decision}",
            f"RISK: {verdict.risk}",
            f"EVIDENCE: {evidence}",
            f"DELIVERED: {delivered}",
            f"TRACKING: {tracking}",
            f"LIABILITY_SHIFT: {bool(gt.get('liability_shift'))}",
            f"SUMMARY: {verdict.summary}",
            f"MERCHANT: {ctx.merchant.name}",
            "SKEPTIC_STRONG: " + " | ".join(_points(skeptic, "strongest_points")),
            "SKEPTIC_CONCEDES: " + " | ".join(_points(skeptic, "weakest_points")),
            "ADVOCATE_STRONG: " + " | ".join(_points(advocate, "strongest_points")),
            "ADVOCATE_CONCEDES: " + " | ".join(_points(advocate, "weakest_points")),
        ],
    )
    judge_model = ctx.settings.judge_model
    judge, _t3 = await ctx.ask(
        stage="debate", system=_JUDGE_SYSTEM, user=judge_user, schema=DEBATE_SCHEMA, model=judge_model
    )

    sustained = bool(judge.get("sustained"))
    recommended = str(judge.get("recommended_decision") or "refund")
    if recommended not in ("contest", "accept", "refund", "escalate"):
        recommended = "refund"
    rationale = str(judge.get("rationale") or "")
    confidence = float(judge.get("confidence") or 0.0)

    if not sustained:
        new_decision = recommended if recommended != "contest" else "refund"
        # Re-draft the merchant response: the decision-stage draft was written
        # for 'contest' and must not go out unchanged after a downgrade.
        redraft, _t4 = await ctx.ask(
            stage="debate",
            system=(
                "You write dispute outcome messages for a merchant's PayPal "
                "disputes team. Professional, specific, kind. JSON only."
            ),
            user=format_prompt(
                ctx.dispute,
                headers=[
                    f"DECISION: {new_decision}",
                    f"RISK: {verdict.risk}",
                    f"SUMMARY: {verdict.summary}",
                    f"MERCHANT: {ctx.merchant.name}",
                    f"AMOUNT: {verdict.amount_at_risk:.2f}",
                    f"CURRENCY: {ctx.dispute.currency}",
                    f"TRANSACTION_ID: {ctx.dispute.transaction_id}",
                    f"ORDER_ID: {gt.get('order_id', '')}",
                    f"TRACKING_NUMBER: {gt.get('tracking_number', '')}",
                ],
            ),
            schema=DECISION_SCHEMA,
        )
        verdict = verdict.model_copy(
            update={
                "decision": new_decision,  # type: ignore[arg-type]
                "confidence": round(confidence, 2),
                "merchant_response": str(redraft.get("merchant_response") or verdict.merchant_response),
                "evidence_package": [str(e) for e in (redraft.get("evidence_package") or verdict.evidence_package)],
                "internal_notes": (
                    verdict.internal_notes
                    + f" | debate council overruled 'contest': {rationale[:200]}"
                ).strip(" |"),
            }
        )

    result = StageResult(
        stage="debate",
        payload={
            "sustained": sustained,
            "recommended_decision": recommended,
            "judge_rationale": rationale,
            "skeptic": skeptic,
            "advocate": advocate,
            "judge_model": judge_model or ctx.llm.model,
        },
        confidence=confidence,
        rationale=("council sustained contest verdict" if sustained else f"council downgraded to {recommended}")
        + (f": {rationale[:120]}" if rationale else ""),
        llm_used=ctx.llm_used,
    )
    return result, verdict
