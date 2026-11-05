"""Chargeback Bunny data model — the contract every module codes against.

Dispute triage domain model. Pydantic v2, UTC ISO-8601
timestamps, short prefixed ids (``mch_`` merchants, ``dsp_`` disputes,
``trc_`` traces) minted by db.py.
"""

from __future__ import annotations

import datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskTier = Literal["critical", "high", "medium", "low"]
RISK_TIERS: tuple[str, ...] = ("low", "medium", "high", "critical")  # ascending

Decision = Literal["contest", "accept", "refund", "escalate", "scam_claim"]
DECISIONS: tuple[str, ...] = ("contest", "accept", "refund", "escalate", "scam_claim")

CaseStatus = Literal["received", "triaging", "awaiting_review", "responded"]
Autonomy = Literal["human", "supervised", "full"]

STAGES: tuple[str, ...] = ("screen", "intake", "dedupe", "verify", "severity", "decision")

CLAIM_TYPES: tuple[str, ...] = (
    "item_not_received",
    "unauthorized",
    "significantly_not_as_described",
    "other",
    "unknown",
)


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Merchant(BaseModel):
    id: str
    name: str
    paypal_merchant_id: str = ""
    autonomy: Autonomy = "supervised"
    auto_respond_cap: float = 100.0  # max dispute amount the agent may auto-respond to
    created_at: str = Field(default_factory=utcnow)


class Dispute(BaseModel):
    """The incoming case."""

    id: str
    merchant_id: str
    paypal_dispute_id: str = ""      # PP-... id when sourced from sandbox
    buyer: str = "unknown"           # buyer email/name
    reason: str = ""                 # PayPal reason code: ITEM_NOT_RECEIVED | UNAUTHORIZED | SNAD | ...
    title: str
    body: str                        # buyer's claim text (UNTRUSTED)
    amount: float = 0.0
    currency: str = "USD"
    transaction_id: str = ""         # PayPal order/capture id referenced by the claim
    attachments: list[str] = Field(default_factory=list)
    status: CaseStatus = "received"
    created_at: str = Field(default_factory=utcnow)


class StructuredClaim(BaseModel):
    """Output of the intake stage: free-text claim normalized to structure."""

    summary: str
    claim_type: str = "unknown"      # item_not_received | unauthorized | significantly_not_as_described | other
    claimed_amount: float = 0.0
    key_facts: list[str] = Field(default_factory=list)     # dated, checkable claims ("delivered 10/02, tracking 1Z…")
    evidence_refs: list[str] = Field(default_factory=list) # transaction ids, tracking numbers, message refs
    buyer_tone: Literal["calm", "urgent", "aggressive", "manipulative"] = "calm"
    confidence: float = 0.0


class StageResult(BaseModel):
    stage: str                       # one of STAGES, or "debate" / "respond"
    payload: dict = Field(default_factory=dict)
    confidence: float = 0.0
    rationale: str = ""
    llm_used: bool = False
    created_at: str = Field(default_factory=utcnow)


class CaseVerdict(BaseModel):
    dispute_id: str
    decision: Decision
    risk: RiskTier = "low"
    amount_at_risk: float = 0.0
    confidence: float = 0.0
    repeat_claimant: bool = False
    scam_claim_detected: bool = False  # screen stage flag; NEVER auto-finalizes (quarantine)
    summary: str = ""
    merchant_response: str = ""        # drafted response to PayPal/buyer
    evidence_package: list[str] = Field(default_factory=list)  # what to attach when contesting
    internal_notes: str = ""
    created_at: str = Field(default_factory=utcnow)


class AgentTrace(BaseModel):
    """One LLM (or offline-engine) invocation, persisted for the audit trail."""

    id: str
    dispute_id: str
    stage: str
    provider: str
    model: str
    prompt: str = ""
    response: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    created_at: str = Field(default_factory=utcnow)


class ReviewDecision(BaseModel):
    dispute_id: str
    action: Literal["approve", "override", "reject"]
    final_decision: Decision
    final_risk: RiskTier = "low"
    reviewer: str = "human"
    note: str = ""
    created_at: str = Field(default_factory=utcnow)


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
