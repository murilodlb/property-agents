"""LangChain/LangGraph agent systems for the RN real-estate MCP benchmark.

Three systems share the same local Ollama model, the same MCP tools and the same
``await system.answer(question) -> AgentResult`` interface:

- ``langgraph_unico``: single ReAct agent (baseline);
- ``langgraph_verificador``: generator + verifier loop;
- ``langgraph_supervisor``: supervisor with three tool-specialized agents.
"""

from __future__ import annotations

from src.agents.base import AgentResult, AgentSystem, SyncAgents
from src.agents.config import AgentConfig
from src.agents.mcp_tools import MCPToolkit
from src.agents.single import AgenteUnico
from src.agents.supervisor import SupervisorEspecialistas
from src.agents.verifier import GeradorVerificador

SYSTEM_CLASSES: dict[str, type[AgentSystem]] = {
    cls.nome: cls for cls in (AgenteUnico, GeradorVerificador, SupervisorEspecialistas)
}


def build_systems(
    toolkit: MCPToolkit,
    config: AgentConfig | None = None,
    nomes: list[str] | None = None,
) -> dict[str, AgentSystem]:
    """Instantiate the requested systems (all three by default) sharing one MCP session."""
    cfg = config or AgentConfig()
    selected = nomes or list(SYSTEM_CLASSES)
    return {nome: SYSTEM_CLASSES[nome](toolkit, cfg) for nome in selected}


__all__ = [
    "SYSTEM_CLASSES",
    "AgentConfig",
    "AgentResult",
    "AgentSystem",
    "AgenteUnico",
    "GeradorVerificador",
    "MCPToolkit",
    "SupervisorEspecialistas",
    "SyncAgents",
    "build_systems",
]
