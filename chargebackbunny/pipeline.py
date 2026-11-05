"""TriagePipeline: orchestrate the dispute-triage stages end to end.

screen -> intake -> dedupe -> verify -> severity -> decision -> debate council
(debate runs only for 'contest' verdicts and may downgrade them)

Each stage's StageResult is persisted as it completes, every LLM call is
traced, and a stage that raises records an error StageResult and lets the
pipeline continue — triage never half-fails silently. The autonomy policy
then decides whether the verdict auto-finalizes (and auto-responds through
PayPal) or waits for human review.
"""

from __future__ import annotations

import logging

from . import policy
from .agents import debate, decision, dedupe, intake, screen, severity, verify
from .agents.base import CaseContext
from .config import Settings
from .db import Store
from .llm import get_client
from .models import STAGES, CaseVerdict, ReviewDecision, StageResult
from .paypal import get_paypal_client

log = logging.getLogger("chargebackbunny.pipeline")

_AGENTS = {
    "screen": screen,
    "intake": intake,
    "dedupe": dedupe,
    "verify": verify,
    "severity": severity,
}

# Which PayPal action executes each finalized decision.
_RESPONDABLE = ("accept", "refund", "contest")


class TriagePipeline:
    def __init__(self, store: Store, settings: Settings, paypal=None):
        self.store = store
        self.settings = settings
        self.llm = get_client(settings)
        self.paypal = paypal if paypal is not None else get_paypal_client(settings)

    async def triage(self, dispute_id: str) -> tuple[CaseVerdict, bool]:
        """Run all stages; return (verdict, auto_finalized)."""
        dispute = self.store.get_dispute(dispute_id)
        if dispute is None:
            raise ValueError(f"unknown dispute: {dispute_id}")
        merchant = self.store.get_merchant(dispute.merchant_id)
        if merchant is None:
            raise ValueError(f"dispute {dispute_id} references unknown merchant {dispute.merchant_id}")

        self.store.update_dispute_status(dispute_id, "triaging")
        ctx = CaseContext(
            store=self.store,
            llm=self.llm,
            settings=self.settings,
            merchant=merchant,
            dispute=dispute,
            paypal=self.paypal,
        )

        verdict: CaseVerdict | None = None
        for stage in STAGES:
            try:
                if stage == "decision":
                    result, verdict = await decision.run(ctx)
                else:
                    result = await _AGENTS[stage].run(ctx)
            except Exception as e:
                log.exception("stage %s failed for dispute %s", stage, dispute_id)
                result = StageResult(
                    stage=stage,
                    payload={"error": str(e)},
                    confidence=0.0,
                    rationale="stage failed",
                    llm_used=False,
                )
            ctx.results[stage] = result
            self.store.save_stage_result(dispute_id, result)

        if verdict is None:  # decision stage itself raised
            verdict = CaseVerdict(
                dispute_id=dispute_id,
                decision="escalate",
                confidence=0.0,
                amount_at_risk=dispute.amount,
                summary=dispute.title,
                merchant_response="",
                internal_notes="decision stage failed; fallback verdict routes to a human",
            )
        self.store.save_verdict(verdict)

        if verdict.decision == "contest":
            # Adversarial council: a 'contest' verdict must survive skeptic vs
            # advocate debate before it can reach PayPal or a human queue.
            try:
                debate_result, verdict = await debate.run(ctx, verdict)
            except Exception as e:
                log.exception("debate failed for dispute %s", dispute_id)
                debate_result = StageResult(
                    stage="debate",
                    payload={"error": str(e)},
                    confidence=0.0,
                    rationale="debate stage failed; verdict stands",
                    llm_used=False,
                )
            ctx.results["debate"] = debate_result
            self.store.save_stage_result(dispute_id, debate_result)
            if debate_result.payload.get("sustained") is False:
                # Downgraded verdict replaces the stored one before policy runs.
                self.store.save_verdict(verdict)

        auto_finalized = policy.apply_policy(verdict, merchant)
        if auto_finalized:
            self.store.save_review(
                ReviewDecision(
                    dispute_id=dispute_id,
                    action="approve",
                    final_decision=verdict.decision,
                    final_risk=verdict.risk,
                    reviewer="chargebunny-policy",
                    note="auto-finalized by autonomy policy",
                )
            )
            if verdict.decision in _RESPONDABLE:
                try:
                    await self.execute_response(dispute_id)
                except Exception:
                    # Response failed — the verdict stands, but the case drops
                    # back to the queue so a human can retry /respond.
                    log.exception("auto-respond failed for dispute %s", dispute_id)
                    self.store.update_dispute_status(dispute_id, "awaiting_review")
                    return verdict, False
            else:
                # e.g. escalate under full autonomy: verdict final, no PayPal
                # action exists — the case closes without a response call.
                self.store.update_dispute_status(dispute_id, "responded")
        else:
            self.store.update_dispute_status(dispute_id, "awaiting_review")
        return verdict, auto_finalized

    async def execute_response(self, dispute_id: str, note: str = "") -> dict:
        """Execute the verdict's action in PayPal and record it in the audit trail."""
        dispute = self.store.get_dispute(dispute_id)
        verdict = self.store.get_verdict(dispute_id)
        if dispute is None or verdict is None:
            raise ValueError(f"unknown dispute or no verdict: {dispute_id}")
        if not dispute.paypal_dispute_id:
            raise ValueError("dispute has no PayPal dispute id; nothing to respond to")

        decision = verdict.decision
        note = note or verdict.merchant_response or f"Chargeback Bunny {decision} on {dispute.paypal_dispute_id}"
        if decision == "accept":
            response = await self.paypal.accept_claim(dispute.paypal_dispute_id, note)
            action = "accept_claim"
        elif decision == "contest":
            gt: dict = {}
            for s in self.store.get_stage_results(dispute_id):
                if s.stage == "verify":
                    gt = s.payload.get("ground_truth") or {}
            evidence = {
                "tracking_number": str(gt.get("tracking_number") or ""),
                "carrier": str(gt.get("carrier") or ""),
                "notes": note,
            }
            response = await self.paypal.provide_evidence(dispute.paypal_dispute_id, evidence)
            action = "provide_evidence"
        elif decision == "refund":
            response = await self.paypal.make_offer(
                dispute.paypal_dispute_id, verdict.amount_at_risk, note, currency=dispute.currency
            )
            action = "make_offer"
        else:
            raise ValueError(f"decision {decision!r} has no automated PayPal response")

        self.store.save_stage_result(
            dispute_id,
            StageResult(
                stage="respond",
                payload={
                    "action": action,
                    "paypal_dispute_id": dispute.paypal_dispute_id,
                    "paypal_response": response,
                },
                confidence=1.0,
                rationale=f"{action} executed against {dispute.paypal_dispute_id}",
                llm_used=False,
            ),
        )
        self.store.update_dispute_status(dispute_id, "responded")
        return {"action": action, "paypal_response": response}

    async def apply_review(
        self,
        dispute_id: str,
        action: str,
        reviewer: str = "human",
        overrides: dict | None = None,
        note: str = "",
    ) -> CaseVerdict:
        """Apply a human review decision; returns the final verdict."""
        verdict = self.store.get_verdict(dispute_id)
        if verdict is None:
            raise ValueError(f"no verdict for dispute: {dispute_id}")
        updates = {
            k: v
            for k, v in (overrides or {}).items()
            if k in ("decision", "risk") and v is not None
        }
        final = verdict.model_copy(update=updates)
        if action == "override" and "decision" in updates:
            # An overridden decision invalidates the drafted response text.
            final = final.model_copy(update={"merchant_response": "", "evidence_package": []})
        self.store.save_verdict(final)
        self.store.save_review(
            ReviewDecision(
                dispute_id=dispute_id,
                action=action,  # type: ignore[arg-type]
                final_decision=final.decision,
                final_risk=final.risk,
                reviewer=reviewer,
                note=note,
            )
        )
        return final
