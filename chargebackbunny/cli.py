"""Chargeback Bunny CLI — serve the API, run the seeded demo, eval, smoke.

Usage:
    python -m chargebackbunny.cli demo
    python -m chargebackbunny.cli serve [--host 127.0.0.1] [--port 8430]
    python -m chargebackbunny.cli eval [--dir eval] [--db path]
    python -m chargebackbunny.cli smoke
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

from .config import REPO_ROOT, Settings
from .db import Store
from .paypal import MockPayPalClient, get_paypal_client
from .pipeline import TriagePipeline

DEMO_DIR = REPO_ROOT / "demo"
EVAL_DIR = REPO_ROOT / "eval"


def _open_store(settings: Settings) -> tuple[Store, TriagePipeline]:
    store = Store(settings.db_path)
    return store, TriagePipeline(store, settings)


# ---------------------------------------------------------------------------
# subcommands


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    app = create_app(db_path=args.db or None)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Seed Acme Widgets + 8 demo disputes and run the pipeline end to end.

    Runs fully offline: with no MODEL_API_KEY the offline heuristic engine
    drives every stage, and with CHARGEBUNNY_PAYPAL_MOCK=1 (or no PayPal
    credentials) the mock client serves demo/sandbox_fixtures.
    """
    settings = Settings.from_env()
    store, pipeline = _open_store(settings)

    merchant_file = DEMO_DIR / "merchant.json"
    disputes_file = DEMO_DIR / "disputes.json"
    if not merchant_file.is_file() or not disputes_file.is_file():
        print(f"demo data missing under {DEMO_DIR} (merchant.json, disputes.json)", file=sys.stderr)
        return 1

    mdata = json.loads(merchant_file.read_text(encoding="utf-8"))
    merchant = store.get_merchant_by_name(mdata["name"])
    if merchant is None:
        merchant = store.create_merchant(
            name=mdata["name"],
            paypal_merchant_id=mdata.get("paypal_merchant_id", ""),
            autonomy=mdata.get("autonomy", settings.default_autonomy),
            auto_respond_cap=float(mdata.get("auto_respond_cap", 100.0)),
        )
        print(f"merchant {merchant.name} seeded (autonomy={merchant.autonomy}, cap=${merchant.auto_respond_cap:.0f})")
    else:
        print(f"merchant {merchant.name} exists")
    print(f"paypal mode: {pipeline.paypal.mode} · llm provider: {settings.provider} "
          f"({settings.model if settings.provider != 'offline' else 'offline-heuristic'})")

    seeds = json.loads(disputes_file.read_text(encoding="utf-8"))

    async def run() -> list[dict]:
        rows = []
        for seed in seeds:
            existing = store.get_dispute_by_paypal_id(merchant.id, seed["paypal_dispute_id"])
            if existing is not None:
                rows.append({"file": seed["paypal_dispute_id"], "title": seed["title"],
                             "decision": "(already triaged)", "risk": "", "amount": "",
                             "conf": "", "routing": "skipped"})
                continue
            dispute = store.create_dispute(
                merchant_id=merchant.id,
                title=seed["title"],
                body=seed["body"],
                paypal_dispute_id=seed.get("paypal_dispute_id", ""),
                buyer=seed.get("buyer", "unknown"),
                reason=seed.get("reason", ""),
                amount=float(seed.get("amount", 0.0)),
                currency=seed.get("currency", "USD"),
                transaction_id=seed.get("transaction_id", ""),
            )
            verdict, auto_finalized = await pipeline.triage(dispute.id)
            dispute = store.get_dispute(dispute.id)
            routing = "auto-finalized+responded" if auto_finalized and dispute.status == "responded" else (
                "auto-finalized" if auto_finalized else "human-review"
            )
            rows.append({
                "file": seed["paypal_dispute_id"],
                "title": seed["title"][:38],
                "decision": verdict.decision,
                "risk": verdict.risk,
                "amount": f"${verdict.amount_at_risk:.2f}",
                "conf": f"{verdict.confidence:.2f}",
                "routing": routing,
            })
            if verdict.scam_claim_detected:
                rows[-1]["routing"] = "QUARANTINED (scam claim)"
        return rows

    rows = asyncio.run(run())

    headers = ("paypal id", "title", "decision", "risk", "amount", "conf", "routing")
    keys = ("file", "title", "decision", "risk", "amount", "conf", "routing")
    widths = [max(len(h), *(len(str(r[k])) for r in rows)) for h, k in zip(headers, keys)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(fmt.format(*[h.upper() for h in headers]))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*(str(r[k]) for k in keys)))

    triaged = [r for r in rows if r["routing"] != "skipped"]
    by_decision = Counter(r["decision"] for r in triaged)
    print(f"\n{len(triaged)} disputes triaged: "
          + ", ".join(f"{d}×{n}" for d, n in sorted(by_decision.items())))
    stats = store.stats()
    print(f"status: {stats['disputes_by_status']} · queue: {stats['queue_length']} awaiting review")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Score the pipeline against the labeled eval set (eval/labels.json).

    Fresh DB per run. Cases are triaged sequentially in filename order —
    repeat-claimant detection depends on earlier cases already being in the
    dedupe corpus, mirroring production arrival order, not gaming the metric.
    """
    settings = Settings.from_env()
    eval_dir = Path(args.dir)
    if not (eval_dir / "labels.json").is_file():
        print(f"no eval corpus at {eval_dir} (labels.json missing)", file=sys.stderr)
        return 1
    db_path = args.db or f"eval-{int(time.time())}.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
    settings = dataclasses.replace(settings, db_path=db_path)
    store = Store(settings.db_path)
    paypal = get_paypal_client(settings)
    eval_fixtures = eval_dir / "fixtures"
    if eval_fixtures.is_dir() and isinstance(paypal, MockPayPalClient):
        paypal.load_dir(eval_fixtures)
    pipeline = TriagePipeline(store, settings, paypal=paypal)

    labels: dict[str, dict] = json.loads((eval_dir / "labels.json").read_text(encoding="utf-8"))
    merchant_data = json.loads((eval_dir / "merchant.json").read_text(encoding="utf-8"))
    merchant = store.create_merchant(
        name=merchant_data.get("name", "Eval Merchant"),
        paypal_merchant_id=merchant_data.get("paypal_merchant_id", ""),
        autonomy=merchant_data.get("autonomy", "human"),
        auto_respond_cap=float(merchant_data.get("auto_respond_cap", 0.0)),
    )

    async def run() -> dict[str, tuple[str, str, float, bool]]:
        out: dict[str, tuple[str, str, float, bool]] = {}
        for fname in sorted(labels):
            # labels.json keys are slugs; corpus files are <slug>.json
            filename = fname if fname.endswith(".json") else f"{fname}.json"
            raw = json.loads((eval_dir / "disputes" / filename).read_text(encoding="utf-8"))
            dispute = store.create_dispute(
                merchant_id=merchant.id,
                title=raw.get("title", fname),
                body=raw.get("body", ""),
                paypal_dispute_id=raw.get("paypal_dispute_id", ""),
                buyer=raw.get("buyer", "unknown"),
                reason=raw.get("reason", ""),
                amount=float(raw.get("amount", 0.0)),
                currency=raw.get("currency", "USD"),
                transaction_id=raw.get("transaction_id", ""),
                attachments=raw.get("attachments") or [],
            )
            verdict, _ = await pipeline.triage(dispute.id)
            out[fname] = (verdict.decision, verdict.risk, verdict.confidence, verdict.scam_claim_detected)
        return out

    print(f"eval set: {eval_dir} · provider={settings.provider} model={settings.model}"
          f" judge={settings.judge_model or settings.model} · paypal={pipeline.paypal.mode}"
          f" · {len(labels)} labeled disputes")
    results = asyncio.run(run())

    rows, confusion = [], Counter()
    hits = risk_hits = 0
    for fname in sorted(labels):
        expected = labels[fname]["decision"]
        expected_risk = labels[fname].get("risk", "")
        got, risk, conf, scam = results[fname]
        ok = got == expected
        risk_ok = (not expected_risk) or risk == expected_risk
        hits += ok
        risk_hits += risk_ok
        confusion[(expected, got)] += 1
        rows.append((fname, expected, got, "✓" if ok else "✗",
                     expected_risk, risk, f"{conf:.2f}", "scam" if scam else ""))

    header = ("FILE", "EXPECTED", "GOT", "", "E.RISK", "RISK", "CONF", "")
    w = [max(len(str(c)) for c in col) for col in zip(*rows, header)]
    fmt = "  ".join(f"{{:<{x}}}" for x in w)
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*r))

    n = len(labels)
    print(f"\ndecision agreement: {hits}/{n} = {hits / n:.0%} · risk agreement: {risk_hits}/{n} = {risk_hits / n:.0%}")
    misses = {k: v for k, v in confusion.items() if k[0] != k[1]}
    if misses:
        print("disagreements (expected -> got: n):")
        for (exp, got), cnt in sorted(misses.items()):
            print(f"  {exp:13s} -> {got:13s} {cnt}")
    scam_expected = [f for f in labels if labels[f]["decision"] == "scam_claim"]
    scam_caught = sum(1 for f in scam_expected if results[f][3])
    if scam_expected:
        print(f"scam claims quarantined: {scam_caught}/{len(scam_expected)}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Smoke-test the PayPal client: list disputes (mock or sandbox)."""
    settings = Settings.from_env()
    client = get_paypal_client(settings)

    async def run() -> list[dict]:
        return await client.list_disputes()

    items = asyncio.run(run())
    print(f"paypal mode: {client.mode} · {len(items)} dispute(s)")
    for d in items:
        amount = d.get("dispute_amount") or {}
        print(f"  {d.get('dispute_id', '?'):12s} {d.get('reason', '?'):40s} "
              f"{amount.get('currency_code', '')} {amount.get('value', ''):>8s} "
              f"{d.get('dispute_life_cycle_stage', ''):12s} {d.get('status', '')}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chargebackbunny", description="Chargeback Bunny dispute triage CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the HTTP API + dashboard")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8430)
    p_serve.add_argument("--db", default="", help="sqlite path override")
    p_serve.set_defaults(func=cmd_serve)

    p_demo = sub.add_parser("demo", help="seed demo data and triage all demo disputes")
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("eval", help="score the pipeline against eval/labels.json")
    p_eval.add_argument("--dir", default=str(EVAL_DIR), help="eval set directory")
    p_eval.add_argument("--db", default="", help="sqlite path (fresh; default: eval-<ts>.db)")
    p_eval.set_defaults(func=cmd_eval)

    p_smoke = sub.add_parser("smoke", help="smoke-test the PayPal client (list disputes)")
    p_smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
