"""Shared configuration for the three LangChain/LangGraph agent systems."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_ollama import ChatOllama

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SERVER_SCRIPT = REPO_ROOT / "mcp_server" / "server.py"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "sim", "on"}


@dataclass(slots=True)
class AgentConfig:
    """Parameters shared by every system so that the comparison stays fair.

    All three systems use the *same* local Ollama model, temperature, seed and context
    window; only the orchestration (graph topology) changes between them.
    """

    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "qwen3.5:2b"))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    temperature: float = field(default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.0")))
    seed: int = field(default_factory=lambda: int(os.getenv("AGENT_SEED", "42")))
    num_ctx: int = field(default_factory=lambda: int(os.getenv("AGENT_NUM_CTX", "16384")))
    # "Thinking" mode of Qwen models; off by default to keep latency manageable on CPU.
    reasoning: bool = field(default_factory=lambda: _env_bool("AGENT_REASONING", False))
    # Hard cap on LangGraph super-steps of each ReAct loop (protects against tool loops).
    recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("AGENT_RECURSION_LIMIT", "24"))
    )
    # Wall-clock timeout for one answer(question) call.
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TIMEOUT_SECONDS", "600"))
    )
    # Generator-verifier: maximum number of rewrite rounds after a rejection.
    max_revisoes: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_REVISOES", "2")))
    # Supervisor: maximum number of delegations to specialists.
    max_delegacoes: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_DELEGACOES", "4")))
    server_script: Path = DEFAULT_SERVER_SCRIPT

    def build_llm(self) -> ChatOllama:
        """Instantiate the chat model shared by every agent/node."""
        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            seed=self.seed,
            num_ctx=self.num_ctx,
            reasoning=self.reasoning,
        )
