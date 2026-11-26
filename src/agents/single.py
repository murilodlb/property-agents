"""System 1 — single ReAct agent with all six MCP tools (baseline)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agents.base import AgentSystem, final_ai_text


class AgenteUnico(AgentSystem):
    """One LLM in a ReAct loop (reason → call tool → observe) until it writes the answer.

    Graph: ``START → model ⇄ tools → END`` (``langchain.agents.create_agent``), capped at
    ``config.max_chamadas_agente`` LLM calls.
    """

    nome = "langgraph_unico"
    num_agentes = 1

    def _build(self) -> None:
        self.graph = self.react_agent(self.toolkit.tools, self.system_prompt, "agente_unico")

    async def _run(self, question: str, config: RunnableConfig) -> tuple[str, dict[str, Any]]:
        msgs = await self.run_react(
            self.graph, [HumanMessage(question)], self.system_prompt, config, "agente_unico"
        )
        return final_ai_text(msgs), {}
