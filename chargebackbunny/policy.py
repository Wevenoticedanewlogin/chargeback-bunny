"""Autonomy policy: decide whether a verdict may be finalized without a human.

Amount caps are the point — a merchant trusts the agent with small slam-dunks,
never with rent money.

  scam_claim_detected -> never auto-finalize, any mode: a human reviews both
                         the claim and the screening itself.
  human               -> never auto.
  supervised          -> auto-finalize accept/refund at confidence >= 0.85 and
                         amount_at_risk <= merchant.auto_respond_cap; contest
                         at confidence >= 0.9 and amount <= cap. Anything over
                         cap or uncertain queues for a human.
  full                -> any non-scam decision at confidence >= 0.7 and
                         amount <= cap.
"""

from __future__ import annotations

from .models import CaseVerdict, Merchant


def apply_policy(verdict: CaseVerdict, merchant: Merchant) -> bool:
    """Return True if the verdict may be auto-finalized under this merchant's policy."""
    if verdict.scam_claim_detected:
        return False
    autonomy = merchant.autonomy
    if autonomy == "human":
        return False
    if verdict.amount_at_risk > merchant.auto_respond_cap:
        return False
    if autonomy == "supervised":
        if verdict.decision in ("accept", "refund") and verdict.confidence >= 0.85:
            return True
        return verdict.decision == "contest" and verdict.confidence >= 0.9
    if autonomy == "full":
        return verdict.confidence >= 0.7
    return False
