"""System 3 — supervisor with specialists (hierarchical delegation) on LangGraph."""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agents.base import AgentSystem, final_ai_text, message_text
from src.agents.prompts import (
    ESPECIALISTAS,
    SINTETIZADOR_SYSTEM,
    especialista_system,
    supervisor_system,
)

NomeEspecialista = Literal["pesquisador_mercado", "analista_imoveis", "analista_financeiro"]


class Decisao(BaseModel):
    """Routing decision of the supervisor (constrained JSON via Ollama ``format``)."""

    proximo: Literal["pesquisador_mercado", "analista_imoveis", "analista_financeiro", "finalizar"]
    instrucao: str = Field(default="", description="tarefa concreta para o especialista")


class SupervisorState(TypedDict, total=False):
    pergunta: str
    relatorios: Annotated[list[dict[str, Any]], operator.add]
    delegacoes: int
    proximo: str
    instrucao: str
    resposta: str


def _relatorios_texto(relatorios: list[dict[str, Any]]) -> str:
    if not relatorios:
        return "(nenhum relatório ainda)"
    return "\n\n".join(
        f"[{i}] {r['especialista']} — tarefa: {r['instrucao']}\n{r['relatorio']}"
        for i, r in enumerate(relatorios, start=1)
    )


class SupervisorEspecialistas(AgentSystem):
    """A supervisor LLM routes sub-tasks to three tool-specialized ReAct agents.

    Graph: ``START → supervisor → {pesquisador_mercado | analista_imoveis |
    analista_financeiro} → supervisor → … → sintetizar → END``. Each specialist only sees
    the 2-3 tools of its domain, which shrinks the tool-selection space for a small model;
    the supervisor decomposes multi-step questions (e.g. *search, then compare*, or
    *verify financing, then simulate*) and writes the final answer from the reports.
    """

    nome = "langgraph_supervisor"
    num_agentes = 1 + len(ESPECIALISTAS)

    def _build(self) -> None:
        self.roteador = self.llm.with_structured_output(Decisao, method="json_schema")
        self.prompts_especialistas = {
            nome: especialista_system(nome, self.system_prompt) for nome in ESPECIALISTAS
        }
        self.especialistas = {
            nome: self.react_agent(
                self.toolkit.select(info["ferramentas"]),  # type: ignore[arg-type]
                self.prompts_especialistas[nome],
                nome,
            )
            for nome, info in ESPECIALISTAS.items()
        }

        graph = StateGraph(SupervisorState)
        graph.add_node("supervisor", self._supervisor)
        for nome in self.especialistas:
            graph.add_node(nome, self._make_especialista_node(nome))
            graph.add_edge(nome, "supervisor")
        graph.add_node("sintetizar", self._sintetizar)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda s: s["proximo"],
            {**{n: n for n in self.especialistas}, "finalizar": "sintetizar"},
        )
        graph.add_edge("sintetizar", END)
        self.graph = graph.compile(name="supervisor_especialistas")

    async def _supervisor(self, state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
        relatorios = state.get("relatorios") or []
        delegacoes = state.get("delegacoes", 0)
        if delegacoes >= self.config.max_delegacoes:
            return {"proximo": "finalizar", "instrucao": ""}
        prompt = (
            f"PERGUNTA DO CLIENTE:\n{state['pergunta']}\n\n"
            f"RELATÓRIOS DOS ESPECIALISTAS ({len(relatorios)}):\n{_relatorios_texto(relatorios)}"
            f"\n\nDelegações usadas: {delegacoes} de {self.config.max_delegacoes}."
        )
        try:
            decisao = await self.roteador.ainvoke(
                [SystemMessage(supervisor_system()), HumanMessage(prompt)],
                config=self.run_config(config, "supervisor"),
            )
            assert isinstance(decisao, Decisao)
        except Exception:
            # Malformed routing output: fall back to the generalist lookup specialist.
            decisao = Decisao(proximo="analista_imoveis", instrucao=state["pergunta"])
        if decisao.proximo == "finalizar" and not relatorios:
            decisao = Decisao(proximo="pesquisador_mercado", instrucao=state["pergunta"])
        return {"proximo": decisao.proximo, "instrucao": decisao.instrucao or state["pergunta"]}

    def _make_especialista_node(self, nome: str):  # noqa: ANN202
        agente = self.especialistas[nome]

        async def _node(state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
            anteriores = state.get("relatorios") or []
            contexto = (
                f"\n\nRelatórios anteriores da equipe:\n{_relatorios_texto(anteriores)}"
                if anteriores
                else ""
            )
            tarefa = (
                f"Pergunta original do cliente: {state['pergunta']}\n\n"
                f"Tarefa delegada pelo supervisor: {state.get('instrucao')}{contexto}"
            )
            try:
                msgs = await self.run_react(
                    agente, [HumanMessage(tarefa)], self.prompts_especialistas[nome], config, nome
                )
                relatorio = final_ai_text(msgs) or "(especialista sem relatório)"
            except GraphRecursionError:
                relatorio = (
                    "(o especialista excedeu o limite de passos sem concluir a tarefa; "
                    "considere os dados como indisponíveis)"
                )
            return {
                "relatorios": [
                    {
                        "especialista": nome,
                        "instrucao": state.get("instrucao") or "",
                        "relatorio": relatorio,
                    }
                ],
                "delegacoes": state.get("delegacoes", 0) + 1,
            }

        return _node

    async def _sintetizar(self, state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
        prompt = (
            f"PERGUNTA DO CLIENTE:\n{state['pergunta']}\n\n"
            f"RELATÓRIOS DOS ESPECIALISTAS:\n{_relatorios_texto(state.get('relatorios') or [])}"
        )
        msg = await self.llm.ainvoke(
            [SystemMessage(SINTETIZADOR_SYSTEM), HumanMessage(prompt)],
            config=self.run_config(config, "supervisor"),
        )
        return {"resposta": message_text(msg)}

    async def _run(self, question: str, config: RunnableConfig) -> tuple[str, dict[str, Any]]:
        out = await self.graph.ainvoke(
            {"pergunta": question, "relatorios": [], "delegacoes": 0},
            config=self.run_config(config, "supervisor_especialistas"),
        )
        relatorios = out.get("relatorios") or []
        return out.get("resposta") or "", {
            "delegacoes": [r["especialista"] for r in relatorios],
            "relatorios": json.loads(json.dumps(relatorios, ensure_ascii=False)),
        }
