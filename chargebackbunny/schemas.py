"""JSON schemas for every LLM stage call.

The ``x-stage`` key is the dispatch contract: agents always send exactly one of
these schemas, and the offline heuristic engine in llm.py uses ``x-stage`` to
decide which rule-based synthesizer to run. Do not add schemas without a
corresponding offline synthesizer.
"""

from __future__ import annotations

SCREEN_SCHEMA: dict = {
    "x-stage": "screen",
    "type": "object",
    "properties": {
        "scam_detected": {"type": "boolean"},
        "signals": {"type": "array", "items": {"type": "string"}},
        "attack_summary": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["scam_detected", "signals", "attack_summary", "confidence"],
    "additionalProperties": False,
}

INTAKE_SCHEMA: dict = {
    "x-stage": "intake",
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "claim_type": {
            "type": "string",
            "description": "item_not_received | unauthorized | significantly_not_as_described | other | unknown",
        },
        "claimed_amount": {"type": "number"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "buyer_tone": {"type": "string", "enum": ["calm", "urgent", "aggressive", "manipulative"]},
        "confidence": {"type": "number"},
    },
    "required": ["summary", "claim_type", "claimed_amount", "key_facts", "evidence_refs", "buyer_tone", "confidence"],
    "additionalProperties": False,
}

DEDUPE_SCHEMA: dict = {
    "x-stage": "dedupe",
    "type": "object",
    "properties": {
        "repeat_claimant": {"type": "boolean"},
        "prior_dispute_ids": {"type": "array", "items": {"type": "string"}},
        "similarity_rationale": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["repeat_claimant", "prior_dispute_ids", "similarity_rationale", "confidence"],
    "additionalProperties": False,
}

VERIFY_SCHEMA: dict = {
    "x-stage": "verify",
    "type": "object",
    "properties": {
        "verified": {"type": "boolean", "description": "buyer's claim is consistent with the PayPal record"},
        "evidence_strength": {"type": "string", "enum": ["none", "weak", "strong"]},
        "findings": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["verified", "evidence_strength", "findings", "confidence"],
    "additionalProperties": False,
}

SEVERITY_SCHEMA: dict = {
    "x-stage": "severity",
    "type": "object",
    "properties": {
        # The risk tier itself is computed deterministically from the amount
        # at risk; the model only writes the justification.
        "risk_justification": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["risk_justification", "confidence"],
    "additionalProperties": False,
}

DECISION_SCHEMA: dict = {
    "x-stage": "decision",
    "type": "object",
    "properties": {
        "merchant_response": {"type": "string", "description": "Drafted response to PayPal/buyer"},
        "evidence_package": {"type": "array", "items": {"type": "string"}},
        "internal_notes": {"type": "string"},
    },
    "required": ["merchant_response", "evidence_package", "internal_notes"],
    "additionalProperties": False,
}

ARGUMENT_SCHEMA: dict = {
    "x-stage": "debate_arg",
    "type": "object",
    "properties": {
        "strongest_points": {"type": "array", "items": {"type": "string"}},
        "weakest_points": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["strongest_points", "weakest_points", "confidence"],
    "additionalProperties": False,
}

DEBATE_SCHEMA: dict = {
    "x-stage": "debate",
    "type": "object",
    "properties": {
        "sustained": {"type": "boolean"},
        "recommended_decision": {"type": "string", "enum": ["contest", "accept", "refund", "escalate"]},
        "rationale": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["sustained", "recommended_decision", "rationale", "confidence"],
    "additionalProperties": False,
}

STAGE_SCHEMAS: dict[str, dict] = {
    s["x-stage"]: s
    for s in (
        SCREEN_SCHEMA,
        INTAKE_SCHEMA,
        DEDUPE_SCHEMA,
        VERIFY_SCHEMA,
        SEVERITY_SCHEMA,
        DECISION_SCHEMA,
        ARGUMENT_SCHEMA,
        DEBATE_SCHEMA,
    )
}
