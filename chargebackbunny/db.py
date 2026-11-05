"""SQLite persistence for Chargeback Bunny.

Single connection (check_same_thread=False) guarded by a lock around writes —
the FastAPI server serves requests from a threadpool, and the pipeline runs on
the event loop, so all access funnels through one locked connection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid

from .models import AgentTrace, CaseVerdict, Dispute, Merchant, ReviewDecision, StageResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS merchants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    paypal_merchant_id TEXT NOT NULL DEFAULT '',
    autonomy TEXT NOT NULL DEFAULT 'supervised',
    auto_respond_cap REAL NOT NULL DEFAULT 100.0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS disputes (
    id TEXT PRIMARY KEY,
    merchant_id TEXT NOT NULL,
    paypal_dispute_id TEXT NOT NULL DEFAULT '',
    buyer TEXT NOT NULL DEFAULT 'unknown',
    reason TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    amount REAL NOT NULL DEFAULT 0,
    currency TEXT NOT NULL DEFAULT 'USD',
    transaction_id TEXT NOT NULL DEFAULT '',
    attachments TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'received',
    created_at TEXT NOT NULL
);
-- One case per PayPal dispute per merchant. Raw (non-PayPal) submissions carry
-- '' here, so the index is partial. Dedup races resolve via IntegrityError.
CREATE UNIQUE INDEX IF NOT EXISTS idx_disputes_pp
    ON disputes (merchant_id, paypal_dispute_id) WHERE paypal_dispute_id != '';
CREATE TABLE IF NOT EXISTS stage_results (
    dispute_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0,
    rationale TEXT NOT NULL DEFAULT '',
    llm_used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    dispute_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    risk TEXT NOT NULL DEFAULT 'low',
    amount_at_risk REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    repeat_claimant INTEGER NOT NULL DEFAULT 0,
    scam_claim_detected INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    merchant_response TEXT NOT NULL DEFAULT '',
    evidence_package TEXT NOT NULL DEFAULT '[]',
    internal_notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    dispute_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    response TEXT NOT NULL DEFAULT '',
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dispute_id TEXT NOT NULL,
    action TEXT NOT NULL,
    final_decision TEXT NOT NULL,
    final_risk TEXT NOT NULL DEFAULT 'low',
    reviewer TEXT NOT NULL DEFAULT 'human',
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:12]


class Store:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- merchants -----------------------------------------------------------

    def create_merchant(
        self,
        name: str,
        paypal_merchant_id: str = "",
        autonomy: str = "supervised",
        auto_respond_cap: float = 100.0,
    ) -> Merchant:
        merchant = Merchant(
            id=_new_id("mch_"),
            name=name,
            paypal_merchant_id=paypal_merchant_id,
            autonomy=autonomy,  # type: ignore[arg-type]
            auto_respond_cap=auto_respond_cap,
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO merchants (id, name, paypal_merchant_id, autonomy, auto_respond_cap, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    merchant.id,
                    merchant.name,
                    merchant.paypal_merchant_id,
                    merchant.autonomy,
                    merchant.auto_respond_cap,
                    merchant.created_at,
                ),
            )
            self._conn.commit()
        return merchant

    def _merchant_from_row(self, row: sqlite3.Row) -> Merchant:
        return Merchant(
            id=row["id"],
            name=row["name"],
            paypal_merchant_id=row["paypal_merchant_id"],
            autonomy=row["autonomy"],
            auto_respond_cap=row["auto_respond_cap"],
            created_at=row["created_at"],
        )

    def get_merchant(self, id: str) -> Merchant | None:
        row = self._conn.execute("SELECT * FROM merchants WHERE id = ?", (id,)).fetchone()
        return self._merchant_from_row(row) if row else None

    def get_merchant_by_name(self, name: str) -> Merchant | None:
        row = self._conn.execute("SELECT * FROM merchants WHERE name = ?", (name,)).fetchone()
        return self._merchant_from_row(row) if row else None

    def list_merchants(self) -> list[Merchant]:
        rows = self._conn.execute("SELECT * FROM merchants ORDER BY created_at").fetchall()
        return [self._merchant_from_row(r) for r in rows]

    # -- disputes --------------------------------------------------------------

    def create_dispute(
        self,
        merchant_id: str,
        title: str,
        body: str,
        paypal_dispute_id: str = "",
        buyer: str = "unknown",
        reason: str = "",
        amount: float = 0.0,
        currency: str = "USD",
        transaction_id: str = "",
        attachments: list[str] | None = None,
    ) -> Dispute:
        dispute = Dispute(
            id=_new_id("dsp_"),
            merchant_id=merchant_id,
            paypal_dispute_id=paypal_dispute_id,
            buyer=buyer,
            reason=reason,
            title=title,
            body=body,
            amount=amount,
            currency=currency,
            transaction_id=transaction_id,
            attachments=list(attachments or []),
        )
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO disputes (id, merchant_id, paypal_dispute_id, buyer, reason, title, body,"
                    " amount, currency, transaction_id, attachments, status, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        dispute.id,
                        dispute.merchant_id,
                        dispute.paypal_dispute_id,
                        dispute.buyer,
                        dispute.reason,
                        dispute.title,
                        dispute.body,
                        dispute.amount,
                        dispute.currency,
                        dispute.transaction_id,
                        json.dumps(dispute.attachments),
                        dispute.status,
                        dispute.created_at,
                    ),
                )
                self._conn.commit()
        except sqlite3.IntegrityError:
            # Unique index on (merchant_id, paypal_dispute_id): another request
            # created this case concurrently — dedup to the existing one.
            existing = self.get_dispute_by_paypal_id(merchant_id, paypal_dispute_id)
            if existing is not None and paypal_dispute_id:
                return existing
            raise
        return dispute

    def _dispute_from_row(self, row: sqlite3.Row) -> Dispute:
        return Dispute(
            id=row["id"],
            merchant_id=row["merchant_id"],
            paypal_dispute_id=row["paypal_dispute_id"],
            buyer=row["buyer"],
            reason=row["reason"],
            title=row["title"],
            body=row["body"],
            amount=row["amount"],
            currency=row["currency"],
            transaction_id=row["transaction_id"],
            attachments=json.loads(row["attachments"]),
            status=row["status"],
            created_at=row["created_at"],
        )

    def get_dispute(self, id: str) -> Dispute | None:
        row = self._conn.execute("SELECT * FROM disputes WHERE id = ?", (id,)).fetchone()
        return self._dispute_from_row(row) if row else None

    def get_dispute_by_paypal_id(self, merchant_id: str, paypal_dispute_id: str) -> Dispute | None:
        row = self._conn.execute(
            "SELECT * FROM disputes WHERE merchant_id = ? AND paypal_dispute_id = ?",
            (merchant_id, paypal_dispute_id),
        ).fetchone()
        return self._dispute_from_row(row) if row else None

    def list_disputes(self, merchant_id: str | None = None, status: str | None = None) -> list[Dispute]:
        sql = "SELECT * FROM disputes"
        clauses: list[str] = []
        params: list[str] = []
        if merchant_id is not None:
            clauses.append("merchant_id = ?")
            params.append(merchant_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._dispute_from_row(r) for r in rows]

    def update_dispute_status(self, id: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE disputes SET status = ? WHERE id = ?", (status, id))
            self._conn.commit()

    # -- stage results ---------------------------------------------------------

    def save_stage_result(self, dispute_id: str, result: StageResult) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO stage_results (dispute_id, stage, payload, confidence, rationale, llm_used, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    dispute_id,
                    result.stage,
                    json.dumps(result.payload),
                    result.confidence,
                    result.rationale,
                    int(result.llm_used),
                    result.created_at,
                ),
            )
            self._conn.commit()

    def get_stage_results(self, dispute_id: str) -> list[StageResult]:
        rows = self._conn.execute(
            "SELECT * FROM stage_results WHERE dispute_id = ? ORDER BY rowid", (dispute_id,)
        ).fetchall()
        return [
            StageResult(
                stage=r["stage"],
                payload=json.loads(r["payload"]),
                confidence=r["confidence"],
                rationale=r["rationale"],
                llm_used=bool(r["llm_used"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- verdicts --------------------------------------------------------------

    def save_verdict(self, verdict: CaseVerdict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO verdicts (dispute_id, decision, risk, amount_at_risk, confidence,"
                " repeat_claimant, scam_claim_detected, summary, merchant_response, evidence_package,"
                " internal_notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    verdict.dispute_id,
                    verdict.decision,
                    verdict.risk,
                    verdict.amount_at_risk,
                    verdict.confidence,
                    int(verdict.repeat_claimant),
                    int(verdict.scam_claim_detected),
                    verdict.summary,
                    verdict.merchant_response,
                    json.dumps(verdict.evidence_package),
                    verdict.internal_notes,
                    verdict.created_at,
                ),
            )
            self._conn.commit()

    def get_verdict(self, dispute_id: str) -> CaseVerdict | None:
        row = self._conn.execute("SELECT * FROM verdicts WHERE dispute_id = ?", (dispute_id,)).fetchone()
        if not row:
            return None
        return CaseVerdict(
            dispute_id=row["dispute_id"],
            decision=row["decision"],
            risk=row["risk"],
            amount_at_risk=row["amount_at_risk"],
            confidence=row["confidence"],
            repeat_claimant=bool(row["repeat_claimant"]),
            scam_claim_detected=bool(row["scam_claim_detected"]),
            summary=row["summary"],
            merchant_response=row["merchant_response"],
            evidence_package=json.loads(row["evidence_package"]),
            internal_notes=row["internal_notes"],
            created_at=row["created_at"],
        )

    # -- traces ------------------------------------------------------------------

    def save_trace(self, trace: AgentTrace) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO traces (id, dispute_id, stage, provider, model, prompt, response,"
                " tokens_in, tokens_out, latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.id,
                    trace.dispute_id,
                    trace.stage,
                    trace.provider,
                    trace.model,
                    trace.prompt,
                    trace.response,
                    trace.tokens_in,
                    trace.tokens_out,
                    trace.latency_ms,
                    trace.created_at,
                ),
            )
            self._conn.commit()

    def list_traces(self, dispute_id: str) -> list[AgentTrace]:
        rows = self._conn.execute(
            "SELECT * FROM traces WHERE dispute_id = ? ORDER BY rowid", (dispute_id,)
        ).fetchall()
        return [
            AgentTrace(
                id=r["id"],
                dispute_id=r["dispute_id"],
                stage=r["stage"],
                provider=r["provider"],
                model=r["model"],
                prompt=r["prompt"],
                response=r["response"],
                tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"],
                latency_ms=r["latency_ms"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # -- reviews -----------------------------------------------------------------

    def save_review(self, review: ReviewDecision) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO reviews (dispute_id, action, final_decision, final_risk,"
                " reviewer, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    review.dispute_id,
                    review.action,
                    review.final_decision,
                    review.final_risk,
                    review.reviewer,
                    review.note,
                    review.created_at,
                ),
            )
            self._conn.commit()

    def get_review(self, dispute_id: str) -> ReviewDecision | None:
        row = self._conn.execute(
            "SELECT * FROM reviews WHERE dispute_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (dispute_id,),
        ).fetchone()
        if not row:
            return None
        return ReviewDecision(
            dispute_id=row["dispute_id"],
            action=row["action"],
            final_decision=row["final_decision"],
            final_risk=row["final_risk"],
            reviewer=row["reviewer"],
            note=row["note"],
            created_at=row["created_at"],
        )

    # -- queues / corpora / stats -------------------------------------------------

    def list_queue(self) -> list[Dispute]:
        """Cases waiting on a human (status='awaiting_review'), oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM disputes WHERE status = 'awaiting_review' ORDER BY created_at"
        ).fetchall()
        return [self._dispute_from_row(r) for r in rows]

    def dedupe_corpus(self, merchant_id: str, exclude_id: str | None = None) -> dict[str, str]:
        """{dispute_id: 'buyer\\ntitle\\nsummary'} for triaged disputes of this merchant.

        Repeat claimants matter even while their earlier cases still wait on a
        human, so awaiting_review and responded cases both count. The dispute
        currently being triaged must be excluded via exclude_id. The buyer
        line is first so the dedupe agent can weight same-buyer matches.
        """
        rows = self._conn.execute(
            "SELECT d.id, d.buyer, d.title, COALESCE(v.summary, '') AS summary"
            " FROM disputes d LEFT JOIN verdicts v ON v.dispute_id = d.id"
            " WHERE d.merchant_id = ? AND d.id != COALESCE(?, '')"
            " AND d.status IN ('responded', 'awaiting_review')",
            (merchant_id, exclude_id),
        ).fetchall()
        return {r["id"]: f"{r['buyer']}\n{r['title']}\n{r['summary']}" for r in rows}

    def stats(self) -> dict:
        by_status = {
            r["status"]: r["n"]
            for r in self._conn.execute("SELECT status, COUNT(*) AS n FROM disputes GROUP BY status")
        }
        by_decision = {
            r["decision"]: r["n"]
            for r in self._conn.execute("SELECT decision, COUNT(*) AS n FROM verdicts GROUP BY decision")
        }
        by_risk = {
            r["risk"]: r["n"]
            for r in self._conn.execute("SELECT risk, COUNT(*) AS n FROM verdicts GROUP BY risk")
        }
        queue_length = self._conn.execute(
            "SELECT COUNT(*) AS n FROM disputes WHERE status = 'awaiting_review'"
        ).fetchone()["n"]
        tokens = self._conn.execute(
            "SELECT COALESCE(SUM(tokens_in + tokens_out), 0) AS n FROM traces"
        ).fetchone()["n"]
        return {
            "disputes_by_status": by_status,
            "verdicts_by_decision": by_decision,
            "verdicts_by_risk": by_risk,
            "queue_length": queue_length,
            "tokens_total": tokens,
        }
