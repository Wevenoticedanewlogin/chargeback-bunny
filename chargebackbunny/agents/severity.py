"""Severity agent: amount-at-risk scoring.

The risk tier is DETERMINISTIC — computed from the dispute amount, bumped one
tier for repeat claimants or chargeback-stage lifecycle:

    low < $50 ≤ medium < $250 ≤ high < $1000 ≤ critical

The LLM only writes the justification text; the model never grades its own
homework.
"""

from __future__ import annotations

from ..models import RISK_TIERS, StageResult
from ..schemas import SEVERITY_SCHEMA
from .base import CaseContext, format_prompt

_SYSTEM = (
    "You are a risk analyst for a PayPal disputes desk. The risk tier has "
    "ALREADY been computed deterministically from the amount at risk (trusted "
    "TIER/AMOUNT headers); do not second-guess it. Write 'risk_justification': "
    "one or two sentences explaining what this tier means operationally for "
    "the merchant — mention the amount, and any aggravating factors named in "
    "the trusted headers (repeat claimant, chargeback lifecycle stage). "
    "Respond with JSON only."
)


def tier_from_amount(amount: float) -> str:
    if amount < 50.0:
        return "low"
    if amount < 250.0:
        return "medium"
    if amount < 1000.0:
        return "high"
    return "critical"


def bump_tier(tier: str) -> str:
    try:
        idx = RISK_TIERS.index(tier)  # ascending order
    except ValueError:
        idx = 0
    return RISK_TIERS[min(idx + 1, len(RISK_TIERS) - 1)]


_CHARGEBACK_STAGES = {"CHARGEBACK", "PRE_ARBITRATION", "ARBITRATION"}


def compute_risk(amount: float, repeat_claimant: bool, life_cycle_stage: str) -> tuple[str, bool, str]:
    """Return (tier, bumped, bump_reason)."""
    tier = tier_from_amount(amount)
    reasons = []
    if repeat_claimant:
        reasons.append("repeat claimant")
    if life_cycle_stage.upper() in _CHARGEBACK_STAGES:
        reasons.append(f"{life_cycle_stage.upper().replace('_', '-').title()} lifecycle stage")
    if reasons:
        return bump_tier(tier), True, " + ".join(reasons)
    return tier, False, ""


async def run(ctx: CaseContext) -> StageResult:
    dedupe = ctx.results.get("dedupe")
    verify = ctx.results.get("verify")
    repeat = bool(dedupe.payload.get("repeat_claimant")) if dedupe else False
    life_cycle_stage = ""
    if verify:
        life_cycle_stage = str((verify.payload.get("ground_truth") or {}).get("life_cycle_stage") or "")

    amount = ctx.dispute.amount
    tier, bumped, bump_reason = compute_risk(amount, repeat, life_cycle_stage)

    data, _trace = await ctx.ask(
        stage="severity",
        system=_SYSTEM,
        user=format_prompt(
            ctx.dispute,
            [
                f"AMOUNT: {amount:.2f}",
                f"TIER: {tier}",
                f"BUMPED: {bumped}",
                f"BUMP_REASON: {bump_reason}",
                f"REPEAT_CLAIMANT: {repeat}",
                f"LIFE_CYCLE_STAGE: {life_cycle_stage or 'INQUIRY'}",
            ],
        ),
        schema=SEVERITY_SCHEMA,
    )
    justification = str(data.get("risk_justification") or "")
    return StageResult(
        stage="severity",
        payload={
            "risk": tier,
            "amount_at_risk": amount,
            "bumped": bumped,
            "bump_reason": bump_reason,
            "justification": justification,
        },
        confidence=float(data.get("confidence") or 0.0),
        rationale=justification or f"${amount:.2f} at risk → {tier}",
        llm_used=ctx.llm_used,
    )
