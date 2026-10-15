"""Thin OpenAI-compatible LLM client with token accounting.

Design goals (from the case-study guardrails):
* temperature 0 + fixed seed -- no randomness in verdicts or wording,
* every call is logged (model, prompt/response chars, input/output tokens)
  so the run report can state total tokens, call count and cost,
* failures degrade gracefully: the caller falls back to deterministic
  templates, so the pipeline NEVER depends on the LLM being reachable.

Supports both a plain OpenAI-compatible endpoint and Azure OpenAI, chosen
via env vars (see .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import config


@dataclass
class CallRecord:
    purpose: str          # e.g. "explanation" or "qa"
    model: str
    prompt_chars: int
    response_chars: int
    input_tokens: int
    output_tokens: int
    ok: bool
    error: str = ""


@dataclass
class TokenLog:
    records: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.records.append(rec)

    @property
    def n_calls(self) -> int:
        return sum(1 for r in self.records if r.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.records if not r.ok)

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records if r.ok)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records if r.ok)

    def summary_lines(self) -> list[str]:
        return [
            f"- llm calls (successful / failed): {self.n_calls} / {self.n_failed}",
            f"- total input tokens: {self.input_tokens}",
            f"- total output tokens: {self.output_tokens}",
        ]


# Rough published list prices (USD per 1M tokens) for cost estimation.
_PRICE_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-35-turbo": (0.50, 1.50),
    # Groq published rates (per 1M tokens), groq.com/pricing
    "gpt-oss-20b": (0.075, 0.30),
    "gpt-oss-120b": (0.15, 0.75),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    key = next((k for k in _PRICE_PER_MTOK if k in model), None)
    if key is None:
        return None
    pin, pout = _PRICE_PER_MTOK[key]
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


class LLMClient:
    """OpenAI-compatible chat client. Returns None on any failure."""

    def __init__(self, api_type: str = None, model: str = None):
        self.api_type = (api_type or config.LLM_API_TYPE).lower()
        self.model = model or config.LLM_MODEL
        self.log = TokenLog()
        self._client = None

        try:
            from openai import OpenAI, AzureOpenAI

            if self.api_type == "azure":
                endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", config.AZURE_OPENAI_ENDPOINT)
                deployment = os.environ.get(
                    "AZURE_OPENAI_DEPLOYMENT", config.AZURE_OPENAI_DEPLOYMENT
                )
                api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
                if not (endpoint and deployment and api_key):
                    raise ValueError(
                        "azure mode needs AZURE_OPENAI_ENDPOINT, "
                        "AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_KEY"
                    )
                self._client = AzureOpenAI(
                    api_key=api_key, api_version=config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=endpoint,
                )
                self.model = deployment
            else:
                api_key = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
                base_url = os.environ.get("LLM_BASE_URL", None)
                if not api_key:
                    raise ValueError(
                        "openai mode needs LLM_API_KEY (or OPENAI_API_KEY); "
                        "set LLM_BASE_URL for other OpenAI-compatible endpoints"
                    )
                kwargs = {"api_key": api_key}
                if base_url:
                    kwargs["base_url"] = base_url
                self._client = OpenAI(**kwargs)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            self._init_error = str(exc)
            self._client = None
        else:
            self._init_error = ""

    @property
    def available(self) -> bool:
        return self._client is not None

    def status(self) -> str:
        if self.available:
            return f"llm ready: api_type={self.api_type} model={self.model}"
        return f"llm unavailable ({self._init_error}); using deterministic templates"

    def chat(self, system: str, user: str, purpose: str = "chat",
             max_tokens: int = None) -> str | None:
        """One temperature-0 completion. Returns None on failure (logged)."""
        if not self.available:
            self.log.add(CallRecord(purpose, self.model, len(system) + len(user), 0, 0, 0,
                                    False, "client unavailable"))
            return None

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=config.LLM_TEMPERATURE,
                seed=config.LLM_SEED,
                max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            self.log.add(CallRecord(
                purpose, self.model, len(system) + len(user), len(text),
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
                True, "",
            ))
            return text or None
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            self.log.add(CallRecord(purpose, self.model, len(system) + len(user), 0, 0, 0,
                                    False, str(exc)[:300]))
            return None

    def write_token_log(self, path=None, mode: str = "deterministic") -> None:
        """Write the token/cost report for one full pipeline run."""
        path = path or config.TOKEN_LOG_PATH
        cost = estimate_cost_usd(self.model, self.log.input_tokens, self.log.output_tokens)
        lines = [
            "# Token / cost log -- one full run over shipment_records.csv",
            "",
            f"- generated (utc): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- run mode: {mode}",
            f"- llm endpoint: api_type={self.api_type}, model={self.model}",
            f"- llm temperature: {config.LLM_TEMPERATURE}, seed: {config.LLM_SEED}",
            "",
            "## Totals",
            "",
            *self.log.summary_lines(),
        ]
        if cost is not None:
            lines.append(
                f"- estimated cost at published list price: ${cost:.4f} "
                f"(input ${self.log.input_tokens / 1e6 * _PRICE_PER_MTOK[next(k for k in _PRICE_PER_MTOK if k in self.model)][0]:.4f} "
                f"+ output ${self.log.output_tokens / 1e6 * _PRICE_PER_MTOK[next(k for k in _PRICE_PER_MTOK if k in self.model)][1]:.4f})"
            )
        else:
            lines.append("- estimated cost: unknown list price for this model; "
                         "see your provider's pricing page")
        if self.log.n_failed:
            lines.append(f"- failed calls (fell back to deterministic templates): {self.log.n_failed}")
        lines += ["", "## Per-call detail", "",
                  "| # | purpose | input tok | output tok | ok |",
                  "|---|---------|-----------|------------|----|"]
        for i, r in enumerate(self.log.records, 1):
            lines.append(f"| {i} | {r.purpose} | {r.input_tokens} | {r.output_tokens} | {'yes' if r.ok else 'no'} |")
        lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
