"""System 2 — generator + verifier (evaluator-optimizer loop) on LangGraph."""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agents.base import AgentSystem, final_ai_text, tool_evidence
from src.agents.prompts import REVISAO_TEMPLATE, VERIFICADOR_SYSTEM
from src.questions import PROP_ID_RE


class Veredito(BaseModel):
    """Structured output of the verifier (constrained JSON via Ollama ``format``).

    Field order matters for small models: they fill the JSON in order, so the two
    checklist fields act as guided reasoning before the final ``aprovado`` decision.
    """

    item_ou_dado_inexistente: bool = Field(
        default=False,
        description=(
            "true se as evidências mostram que algum imóvel pedido não existe no estoque ou "
            "que o dado pedido não é registrado no catálogo"
        ),
    )
    rascunho_fiel_as_evidencias: bool = Field(
        default=True,
        description="true se todos os ids e números do rascunho constam nas evidências",
    )
    aprovado: bool = Field(description="true se o rascunho pode ser entregue ao cliente")
    problemas: list[str] = Field(
        default_factory=list, description="lista de problemas encontrados (vazia se aprovado)"
    )
    instrucoes_correcao: str = Field(
        default="", description="instruções objetivas para o gerador corrigir a resposta"
    )


class VerificadorState(TypedDict, total=False):
    pergunta: str
    mensagens: list[BaseMessage]  # full generator history (kept across revisions)
    rascunho: str
    evidencias: list[dict[str, Any]]
    revisoes: int
    vereditos: Annotated[list[dict[str, Any]], operator.add]


_INEXISTENTE_RE = re.compile(
    r"n[ãa]o (consta|existe|foi encontrad|encontr|est[áa] dispon|h[áa] registro|[ée] registrad|"
    r"possui registro|[ée] informad)|inexistente|indispon[íi]ve",
    re.IGNORECASE,
)


def _ids_sem_evidencia(rascunho: str, evidencias: list[dict[str, Any]]) -> list[str]:
    """Deterministic pre-check: ids cited in the draft that never appear in tool I/O."""
    corpus = json.dumps(evidencias, ensure_ascii=False).lower()
    citados = {m.lower() for m in PROP_ID_RE.findall(rascunho)}
    return sorted(i for i in citados if i not in corpus)


class GeradorVerificador(AgentSystem):
    """A ReAct *generator* drafts the answer; a *verifier* audits it against the evidence.

    Graph: ``START → gerar → verificar → (aprovado | limite) → END``; on rejection
    ``preparar_revisao`` appends the verifier feedback to the generator history and
    ``gerar`` runs again
    (at most ``config.max_revisoes`` times). The verifier sees the question, the exact tool
    calls/results of the generator and the draft, so it can catch hallucinated ids/values,
    false premises (pegadinhas) and incomplete answers — the main failure modes of a small
    local model in this domain.
    """

    nome = "langgraph_verificador"
    num_agentes = 2

    def _build(self) -> None:
        self.gerador = self.react_agent(self.toolkit.tools, self.system_prompt, "gerador")
        self.verificador = self.llm.with_structured_output(Veredito, method="json_schema")

        graph = StateGraph(VerificadorState)
        graph.add_node("gerar", self._gerar)
        graph.add_node("verificar", self._verificar)
        graph.add_node("preparar_revisao", self._preparar_revisao)
        graph.add_edge(START, "gerar")
        graph.add_edge("gerar", "verificar")
        graph.add_conditional_edges(
            "verificar", self._rota, {"revisar": "preparar_revisao", "fim": END}
        )
        graph.add_edge("preparar_revisao", "gerar")
        self.graph = graph.compile(name="gerador_verificador")

    async def _gerar(self, state: VerificadorState, config: RunnableConfig) -> dict[str, Any]:
        mensagens = state.get("mensagens") or [HumanMessage(state["pergunta"])]
        msgs = await self.run_react(self.gerador, mensagens, self.system_prompt, config, "gerador")
        return {
            "mensagens": msgs,
            "rascunho": final_ai_text(msgs),
            "evidencias": tool_evidence(msgs),
        }

    async def _verificar(self, state: VerificadorState, config: RunnableConfig) -> dict[str, Any]:
        evidencias = state.get("evidencias") or []
        rascunho = state.get("rascunho") or ""
        sem_evidencia = _ids_sem_evidencia(rascunho, evidencias)
        checagem = (
            f"ATENÇÃO: os ids {sem_evidencia} são citados no rascunho mas não aparecem em "
            "nenhuma evidência."
            if sem_evidencia
            else "Todos os ids citados aparecem nas evidências."
        )
        prompt = (
            f"PERGUNTA DO CLIENTE:\n{state['pergunta']}\n\n"
            f"EVIDÊNCIAS ({len(evidencias)} chamadas de ferramentas):\n"
            f"{json.dumps(evidencias, ensure_ascii=False, indent=1)}\n\n"
            f"CHECAGEM AUTOMÁTICA DE IDS: {checagem}\n\n"
            f"RASCUNHO DA RESPOSTA:\n{rascunho}"
        )
        cfg = self.run_config(config, "verificador")
        try:
            veredito = await self.verificador.ainvoke(
                [SystemMessage(VERIFICADOR_SYSTEM), HumanMessage(prompt)], config=cfg
            )
            assert isinstance(veredito, Veredito)
        except Exception as exc:  # malformed JSON from the small model: accept the draft
            veredito = Veredito(aprovado=True, problemas=[f"verificador falhou: {exc}"])
        if (
            not veredito.aprovado
            and veredito.item_ou_dado_inexistente
            and veredito.rascunho_fiel_as_evidencias
            and _INEXISTENTE_RE.search(rascunho)
        ):
            # The draft correctly reports a missing item/datum: demanding the missing values
            # would push the generator to hallucinate them (observed with 2B models).
            veredito.aprovado = True
            veredito.problemas.append("aprovado por regra: rascunho informa item/dado inexistente")
        if not evidencias or sem_evidencia:
            # Hard rules the LLM verifier must not override.
            veredito.aprovado = False
            if not evidencias:
                veredito.problemas.append("Nenhuma ferramenta foi consultada.")
            if sem_evidencia:
                veredito.problemas.append(f"Ids sem evidência nas ferramentas: {sem_evidencia}.")
        return {"vereditos": [veredito.model_dump()]}

    def _rota(self, state: VerificadorState) -> Literal["revisar", "fim"]:
        ultimo = (state.get("vereditos") or [{}])[-1]
        if ultimo.get("aprovado") or state.get("revisoes", 0) >= self.config.max_revisoes:
            return "fim"
        return "revisar"

    def _preparar_revisao(self, state: VerificadorState) -> dict[str, Any]:
        """Append the verifier feedback to the generator history (no LLM call)."""
        ultimo = (state.get("vereditos") or [{}])[-1]
        feedback = REVISAO_TEMPLATE.format(
            problemas="\n".join(f"- {p}" for p in ultimo.get("problemas") or []) or "-",
            instrucoes=ultimo.get("instrucoes_correcao") or "revise a resposta.",
        )
        return {
            "mensagens": [*state.get("mensagens", []), HumanMessage(feedback)],
            "revisoes": state.get("revisoes", 0) + 1,
        }

    async def _run(self, question: str, config: RunnableConfig) -> tuple[str, dict[str, Any]]:
        out = await self.graph.ainvoke(
            {"pergunta": question, "revisoes": 0, "vereditos": []},
            config=self.run_config(config, "gerador_verificador"),
        )
        vereditos = out.get("vereditos") or []
        return out.get("rascunho") or "", {
            "revisoes": out.get("revisoes", 0),
            "vereditos": vereditos,
        }
