"""Dedupe agent: serial-refunder / repeat-claimant detection.

TF-IDF similarity shortlists candidates from the merchant's past disputes
(same buyer weighted heavily — a repeat claimant is a repeat claimant even
when the wording changes); an LLM (or the offline engine) then judges whether
the pattern holds.

This is NOT a rejection path: the result feeds severity (one-tier risk bump)
and the decision, never an automatic dismissal.
"""

from __future__ import annotations

from ..models import StageResult
from ..schemas import DEDUPE_SCHEMA
from ..similarity import top_k_similar
from .base import CaseContext, format_prompt

_SYSTEM = (
    "You are a senior disputes analyst deciding whether the buyer behind an "
    "incoming PayPal dispute is a repeat claimant. The trusted CANDIDATE "
    "headers list prior disputes from this merchant with similarity scores "
    "and same_buyer flags. Repeat-claimant means: the same buyer has disputed "
    "before, or the claim text is near-identical to a prior dispute "
    "(template-driven serial-refunder behavior). Different buyers with "
    "ordinary similar complaints are NOT repeat claimants. This flag feeds "
    "risk scoring — it never dismisses a claim on its own. Respond with JSON "
    "only."
)

_SAME_BUYER_SIM = 0.6  # same buyer is decisive even when wording differs


async def run(ctx: CaseContext) -> StageResult:
    corpus = ctx.store.dedupe_corpus(ctx.merchant.id, exclude_id=ctx.dispute.id)
    intake = ctx.results.get("intake")
    summary = str(intake.payload.get("summary") or "") if intake else ""
    # Compare normalized units: buyer + title + intake summary against the
    # corpus's buyer + title + verdict summary. Raw claim bodies diverge
    # stylistically even between serial claims and drown the signal.
    query = f"{ctx.dispute.buyer}\n{ctx.dispute.title}\n{summary}"
    candidates = top_k_similar(query, corpus, k=3)

    if not candidates:
        rationale = "no prior disputes to compare against"
        return StageResult(
            stage="dedupe",
            payload={
                "repeat_claimant": False,
                "prior_dispute_ids": [],
                "similarity_rationale": rationale,
                "candidates": [],
            },
            confidence=0.9,
            rationale=rationale,
            llm_used=False,
        )

    # Corpus text is 'buyer\ntitle\nsummary' — recover each candidate's buyer.
    def _buyer_of(doc_id: str) -> str:
        return (corpus.get(doc_id) or "").split("\n", 1)[0].strip()

    headers = []
    for doc_id, sim in candidates:
        same = _buyer_of(doc_id) == ctx.dispute.buyer
        if same:
            sim = max(sim, _SAME_BUYER_SIM)
        headers.append(
            f"CANDIDATE {doc_id} sim={sim:.4f} same_buyer={same}: "
            f"{' '.join(corpus[doc_id].split())}"
        )
    data, _trace = await ctx.ask(
        stage="dedupe",
        system=_SYSTEM,
        user=format_prompt(ctx.dispute, headers),
        schema=DEDUPE_SCHEMA,
    )
    repeat = bool(data.get("repeat_claimant"))
    prior_ids = [str(p) for p in (data.get("prior_dispute_ids") or [])]
    rationale = str(data.get("similarity_rationale") or "")
    return StageResult(
        stage="dedupe",
        payload={
            "repeat_claimant": repeat,
            "prior_dispute_ids": prior_ids,
            "similarity_rationale": rationale,
            "candidates": [[doc_id, sim] for doc_id, sim in candidates],
        },
        confidence=float(data.get("confidence") or 0.0),
        rationale=rationale,
        llm_used=ctx.llm_used,
    )
