"""Common interface shared by the three systems: ``await system.answer(question)``."""

from __future__ import annotations

import asyncio
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.agents.config import AgentConfig
from src.agents.mcp_tools import MCPToolkit
from src.agents.prompts import build_base_system_prompt
from src.agents.telemetry import ExecutionTracker, ResourceMonitor, ResourceSnapshot, Stopwatch
from src.schema import ExecutionRecord

T = TypeVar("T")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def message_text(message: BaseMessage) -> str:
    """Plain text of a message (handles list-of-blocks content and stray <think> tags)."""
    content = message.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return _THINK_RE.sub("", str(content)).strip()


def final_ai_text(messages: list[BaseMessage]) -> str:
    """Text of the last AI message that is not a pure tool-call request."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = message_text(msg)
            if text:
                return text
    return ""


def tool_evidence(messages: list[BaseMessage], max_chars: int = 3000) -> list[dict[str, Any]]:
    """Pair each tool call with its result, for verifiers/supervisors and auditing."""
    calls: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                calls[str(tc.get("id"))] = {"ferramenta": tc["name"], "argumentos": tc["args"]}
    evidence: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            base = calls.get(str(msg.tool_call_id), {"ferramenta": msg.name, "argumentos": {}})
            text = message_text(msg)
            if len(text) > max_chars:
                text = text[:max_chars] + " …[truncado]"
            evidence.append({**base, "resultado": text})
    return evidence


@dataclass(slots=True)
class AgentResult:
    """Answer plus every execution datum required by the evaluation metrics."""

    sistema: str
    resposta: str
    ferramentas_usadas: list[str]
    chamadas_modelo: int
    latencia_segundos: float
    tokens_entrada: int | None
    tokens_saida: int | None
    modelo_agente: str
    num_agentes: int
    erros_ferramentas: int = 0
    erro_execucao: str | None = None
    recursos: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    detalhes: dict[str, Any] = field(default_factory=dict)

    def to_record(self, pergunta_id: int) -> ExecutionRecord:
        """Convert to the ``results/executions.jsonl`` contract (``src/schema.py``)."""
        return ExecutionRecord(
            sistema=self.sistema,
            pergunta_id=pergunta_id,
            resposta=self.resposta,
            ferramentas_usadas=self.ferramentas_usadas,
            chamadas_modelo=self.chamadas_modelo,
            latencia_segundos=round(self.latencia_segundos, 3),
            tokens_entrada=self.tokens_entrada,
            tokens_saida=self.tokens_saida,
            modelo_agente=self.modelo_agente,
            erro_execucao=self.erro_execucao,
            erros_ferramentas=self.erros_ferramentas,
            num_agentes=self.num_agentes,
            **asdict(self.recursos),
        )


class AgentSystem(ABC):
    """Base class: subclasses only implement the graph (``_build``) and its run (``_run``)."""

    nome: str = "base"
    num_agentes: int = 1

    def __init__(
        self,
        toolkit: MCPToolkit,
        config: AgentConfig | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.toolkit = toolkit
        self.config = config or AgentConfig()
        self.llm = llm or self.config.build_llm()
        self.esquema = ""
        self._built = False

    async def setup(self) -> None:
        """Ground the prompts on the MCP resource ``estoque://esquema`` and build the graph."""
        if self._built:
            return
        self.esquema = await self.toolkit.read_resource("estoque://esquema")
        self.system_prompt = build_base_system_prompt(self.esquema)
        self._build()
        self._built = True

    @abstractmethod
    def _build(self) -> None: ...

    @abstractmethod
    async def _run(self, question: str, config: RunnableConfig) -> tuple[str, dict[str, Any]]:
        """Run the graph and return ``(final_answer, extra_details)``."""

    def run_config(self, config: RunnableConfig, agente: str) -> RunnableConfig:
        """Child config tagging nested LLM calls with the agent name (per-agent counts)."""
        metadata = {**(config.get("metadata") or {}), "agente": agente}
        return {**config, "metadata": metadata, "recursion_limit": self.config.recursion_limit}

    async def answer(self, question: str) -> AgentResult:
        """Common interface: answer ``question`` and return the answer + execution data."""
        await self.setup()
        tracker = ExecutionTracker()
        errors_before = self.toolkit.tool_errors
        repeated_before = self.toolkit.repeated_calls
        self.toolkit.reset_cache()
        resposta = ""
        detalhes: dict[str, Any] = {}
        erro: str | None = None
        monitor = ResourceMonitor()
        with monitor, Stopwatch() as sw:
            try:
                resposta, detalhes = await asyncio.wait_for(
                    self._run(question, {"callbacks": [tracker], "metadata": {}}),
                    timeout=self.config.timeout_seconds,
                )
            except TimeoutError:
                erro = f"timeout após {self.config.timeout_seconds:.0f}s"
            except Exception as exc:
                erro = f"{type(exc).__name__}: {exc}"
        if not erro and not resposta.strip():
            erro = "resposta vazia"
        tokens_in, tokens_out = tracker.tokens
        detalhes["chamadas_por_agente"] = tracker.chamadas_por_agente
        detalhes["chamadas_repetidas_bloqueadas"] = self.toolkit.repeated_calls - repeated_before
        return AgentResult(
            sistema=self.nome,
            resposta=resposta,
            ferramentas_usadas=list(tracker.ferramentas_usadas),
            chamadas_modelo=tracker.chamadas_modelo,
            latencia_segundos=sw.elapsed,
            tokens_entrada=tokens_in,
            tokens_saida=tokens_out,
            modelo_agente=self.config.model,
            num_agentes=self.num_agentes,
            erros_ferramentas=self.toolkit.tool_errors - errors_before,
            erro_execucao=erro,
            recursos=monitor.snapshot(),
            detalhes=detalhes,
        )


class SyncAgents:
    """Synchronous facade (scripts/notebooks): owns an event loop thread + MCP session.

    Example::

        with SyncAgents() as agents:
            result = agents.answer("langgraph_unico", "Qual o preço do imóvel r001?")
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self.toolkit = MCPToolkit(self.config.server_script)
        self.systems: dict[str, AgentSystem] = {}

    def _submit(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def __enter__(self) -> SyncAgents:
        from src.agents import build_systems

        self._thread.start()
        self._submit(self.toolkit.__aenter__())
        self.systems = build_systems(self.toolkit, self.config)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._submit(self.toolkit.__aexit__(None, None, None))
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def answer(self, sistema: str, question: str) -> AgentResult:
        return self._submit(self.systems[sistema].answer(question))
