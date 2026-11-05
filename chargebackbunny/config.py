"""Chargeback Bunny configuration.

All settings come from environment variables (see .env.example). Two LLM
providers:

  openai-compatible -> any OpenAI-compatible chat-completions endpoint.
                       CHARGEBUNNY_BASE_URL + MODEL_API_KEY + CHARGEBUNNY_MODEL.
  offline           -> Deterministic rules engine. No network, no keys. Used
                       for local development and demos when no API key is
                       configured; clearly labeled "offline-heuristic" in
                       every agent trace.

PayPal settings select the real sandbox client, or the fixture-backed mock
(CHARGEBUNNY_PAYPAL_MOCK=1, or automatically when no credentials are set).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
PAYPAL_SANDBOX_BASE_URL = "https://api-m.sandbox.paypal.com"
DEFAULT_FIXTURE_DIR = REPO_ROOT / "demo" / "sandbox_fixtures"

AUTONOMY_MODES = ("human", "supervised", "full")


@dataclass(frozen=True)
class Settings:
    provider: str = "offline"           # openai-compatible | offline
    model: str = "gpt-4o-mini"
    judge_model: str | None = None      # debate-council judge override; None = same model
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    db_path: str = "chargebackbunny.db"
    default_autonomy: str = "supervised"
    request_timeout_s: float = 60.0
    max_stage_tokens: int = 16384
    reasoning_effort: str | None = None  # "low" | "medium" | "high"; None = provider default
    # PayPal
    paypal_client_id: str | None = None
    paypal_client_secret: str | None = None
    paypal_base_url: str = PAYPAL_SANDBOX_BASE_URL
    paypal_mock: bool = False
    paypal_fixture_dir: str = str(DEFAULT_FIXTURE_DIR)

    @property
    def paypal_mode(self) -> str:
        return "mock" if self.paypal_mock or not (self.paypal_client_id and self.paypal_client_secret) else "sandbox"

    @staticmethod
    def from_env() -> "Settings":
        _load_dotenv()
        provider = os.environ.get("CHARGEBUNNY_PROVIDER", "").strip().lower()
        api_key = os.environ.get("MODEL_API_KEY")
        if not provider:
            provider = "openai-compatible" if api_key else "offline"
        if provider not in ("openai-compatible", "offline"):
            provider = "offline"
        return Settings(
            provider=provider,
            model=os.environ.get("CHARGEBUNNY_MODEL", "gpt-4o-mini"),
            judge_model=os.environ.get("CHARGEBUNNY_JUDGE_MODEL") or None,
            api_key=api_key,
            base_url=os.environ.get("CHARGEBUNNY_BASE_URL", DEFAULT_BASE_URL),
            db_path=os.environ.get("CHARGEBUNNY_DB", "chargebackbunny.db"),
            default_autonomy=os.environ.get("CHARGEBUNNY_AUTONOMY", "supervised"),
            reasoning_effort=os.environ.get("CHARGEBUNNY_REASONING") or None,
            paypal_client_id=os.environ.get("PAYPAL_CLIENT_ID") or None,
            paypal_client_secret=os.environ.get("PAYPAL_CLIENT_SECRET") or None,
            paypal_base_url=os.environ.get("PAYPAL_BASE_URL", PAYPAL_SANDBOX_BASE_URL),
            paypal_mock=os.environ.get("CHARGEBUNNY_PAYPAL_MOCK", "").strip() in ("1", "true", "yes"),
            paypal_fixture_dir=os.environ.get("CHARGEBUNNY_PAYPAL_FIXTURES", str(DEFAULT_FIXTURE_DIR)),
        )


def _load_dotenv() -> None:
    """Populate os.environ from a .env file (cwd or repo root) if present.

    Existing environment variables win; the file only fills gaps. No
    dependency — ten lines instead of python-dotenv.
    """
    candidates = [Path.cwd() / ".env", REPO_ROOT / ".env"]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return
