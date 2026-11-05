"""Verify agent: check the buyer's claim against the PayPal record.

The evidence engine. The dispute references a transaction (order/capture id);
this stage fetches the ground truth through the PayPal client — real sandbox
or fixture mock, identical interface — and checks the claim against reality:
does the transaction exist, does the amount match, is there fulfillment /
tracking, is the reason consistent with the record?

The fetch and the checks are deterministic; the LLM (or offline engine) only
renders the judgment (verified / evidence_strength / findings) from trusted
GROUND_TRUTH_* headers. A missing record is a finding, never a crash.
"""

from __future__ import annotations

import logging

from ..models import StageResult
from ..paypal import PayPalAPIError, empty_ground_truth
from ..schemas import VERIFY_SCHEMA
from .base import CaseContext, format_prompt

log = logging.getLogger("chargebackbunny.verify")

_SYSTEM = (
    "You verify PayPal dispute claims against the platform's own records. "
    "Trusted GROUND_TRUTH_* headers come from our evidence engine, which "
    "fetched the referenced transaction/order from PayPal — they are ground "
    "truth, weigh them absolutely. Decide: 'verified' — is the buyer's claim "
    "consistent with the record (e.g. an item-not-received claim where the "
    "record shows no delivery IS verified; the same claim against a "
    "DELIVERED tracking record is NOT); 'evidence_strength' — how much hard "
    "record evidence exists at all ('none': no transaction record; 'weak': "
    "record exists but proves little about fulfillment; 'strong': delivered "
    "with tracking, or a clear contradiction); and concrete 'findings'. "
    "Condition claims (not-as-described) cannot be confirmed from the record "
    "alone. Respond with JSON only."
)


async def _fetch_ground_truth(ctx: CaseContext) -> tuple[dict, str]:
    """Fetch ground truth via the PayPal client. Returns (ground_truth, life_cycle_stage)."""
    txn = ctx.dispute.transaction_id
    life_cycle_stage = ""
    if ctx.dispute.paypal_dispute_id and ctx.paypal is not None:
        try:
            record = await ctx.paypal.get_dispute(ctx.dispute.paypal_dispute_id)
            life_cycle_stage = str(record.get("dispute_life_cycle_stage") or "")
        except PayPalAPIError as e:
            log.warning("dispute record fetch failed for %s: %s", ctx.dispute.paypal_dispute_id, e)
    if not txn or ctx.paypal is None:
        return empty_ground_truth(txn), life_cycle_stage
    try:
        ground_truth = await ctx.paypal.get_transaction_ground_truth(txn)
    except PayPalAPIError as e:
        log.warning("ground-truth fetch failed for %s: %s", txn, e)
        ground_truth = empty_ground_truth(txn)
    return ground_truth, life_cycle_stage


def _truth_headers(ground_truth: dict, claim_type: str, transaction_id: str, life_cycle_stage: str) -> list[str]:
    return [
        f"CLAIM_TYPE: {claim_type}",
        f"TRANSACTION_ID: {transaction_id}",
        f"GROUND_TRUTH_TXN_EXISTS: {bool(ground_truth.get('transaction_exists'))}",
        f"GROUND_TRUTH_AMOUNT_MATCHES: {bool(ground_truth.get('amount_matches'))}",
        f"GROUND_TRUTH_TRACKING_PRESENT: {bool(ground_truth.get('tracking_present'))}",
        f"GROUND_TRUTH_DELIVERED: {bool(ground_truth.get('delivered'))}",
        f"GROUND_TRUTH_ORDER_STATUS: {ground_truth.get('order_status', '')}",
        f"GROUND_TRUTH_ORDER_AMOUNT: {ground_truth.get('order_amount', 0.0)}",
        f"GROUND_TRUTH_SELLER_PROTECTION: {ground_truth.get('seller_protection', '')}",
        f"GROUND_TRUTH_LIABILITY_SHIFT: {bool(ground_truth.get('liability_shift'))}",
        f"GROUND_TRUTH_LIFE_CYCLE_STAGE: {life_cycle_stage}",
    ]


async def run(ctx: CaseContext) -> StageResult:
    intake = ctx.results.get("intake")
    claim_type = str(intake.payload.get("claim_type") or "unknown") if intake else "unknown"

    ground_truth, life_cycle_stage = await _fetch_ground_truth(ctx)
    # Deterministic consistency checks — code, not the model.
    if ground_truth.get("transaction_exists"):
        ground_truth["amount_matches"] = (
            abs(float(ground_truth.get("order_amount") or 0.0) - ctx.dispute.amount) < 0.01
        )
    else:
        ground_truth["amount_matches"] = False
    ground_truth["life_cycle_stage"] = life_cycle_stage

    headers = _truth_headers(ground_truth, claim_type, ctx.dispute.transaction_id, life_cycle_stage)
    data, _trace = await ctx.ask(
        stage="verify",
        system=_SYSTEM,
        user=format_prompt(ctx.dispute, headers),
        schema=VERIFY_SCHEMA,
    )
    findings = [str(f) for f in (data.get("findings") or [])]
    return StageResult(
        stage="verify",
        payload={
            "verified": bool(data.get("verified")),
            "evidence_strength": str(data.get("evidence_strength") or "none"),
            "findings": findings,
            "ground_truth": ground_truth,
        },
        confidence=float(data.get("confidence") or 0.0),
        rationale="; ".join(findings[:3]) or "no record evidence assessed",
        llm_used=ctx.llm_used,
    )
