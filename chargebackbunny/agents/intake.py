"""Intake agent: normalize a free-text dispute claim into a StructuredClaim."""

from __future__ import annotations

from ..models import StageResult, StructuredClaim
from ..schemas import INTAKE_SCHEMA
from .base import CaseContext, format_prompt

_SYSTEM = (
    "You are a senior disputes analyst normalizing an incoming PayPal dispute "
    "claim. Extract the fields required by the JSON schema: a one-paragraph "
    "summary; the claim_type — 'item_not_received', 'unauthorized', "
    "'significantly_not_as_described', 'other', or 'unknown' (the trusted "
    "REASON header is PayPal's own reason code and should win over your "
    "reading of the text); the claimed amount; key_facts — dated, checkable "
    "claims from the text ('delivered 10/02', 'tracking 1Z…'); evidence_refs — "
    "transaction ids, tracking numbers, and message refs mentioned; and an "
    "honest read of buyer_tone ('calm', 'urgent', 'aggressive', "
    "'manipulative'). Set 'confidence' to how certain you are of the overall "
    "extraction. Respond with JSON only."
)


async def run(ctx: CaseContext) -> StageResult:
    data, _trace = await ctx.ask(
        stage="intake",
        system=_SYSTEM,
        user=format_prompt(ctx.dispute),
        schema=INTAKE_SCHEMA,
    )
    structured = StructuredClaim(
        **{k: v for k, v in data.items() if k in StructuredClaim.model_fields}
    )
    return StageResult(
        stage="intake",
        payload=structured.model_dump(),
        confidence=structured.confidence,
        rationale=structured.summary,
        llm_used=ctx.llm_used,
    )
