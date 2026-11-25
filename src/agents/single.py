"""System 1 — single ReAct agent with all six MCP tools (baseline)."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agents.base import AgentSystem, final_ai_text


class AgenteUnico(AgentSystem):
    """One LLM in a ReAct loop (reason → call tool → observe) until it writes the answer.

    Graph: ``START → model ⇄ tools → END`` (``langchain.agents.create_agent``).
    """

    nome = "langgraph_unico"
    num_agentes = 1

    def _build(self) -> None:
        self.graph = create_agent(
            self.llm,
            self.toolkit.tools,
            system_prompt=self.system_prompt,
            name="agente_unico",
        )

    async def _run(self, question: str, config: RunnableConfig) -> tuple[str, dict[str, Any]]:
        out = await self.graph.ainvoke(
            {"messages": [HumanMessage(question)]},
            config=self.run_config(config, "agente_unico"),
        )
        return final_ai_text(out["messages"]), {}
