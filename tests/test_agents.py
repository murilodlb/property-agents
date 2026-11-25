"""Offline tests for the agent systems (no Ollama): MCP adapter, graphs and telemetry."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.agents import AgentConfig, MCPToolkit, build_systems
from src.agents.verifier import Veredito, _ids_sem_evidencia


class ScriptedChat(GenericFakeChatModel):
    """Fake chat model that supports ``bind_tools``/structured output for graph tests."""

    structured: list[Any] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChat:  # type: ignore[override]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        from langchain_core.runnables import RunnableLambda

        queue = self.structured

        def _next(_: Any) -> Any:
            return queue.pop(0)

        return RunnableLambda(_next)


def _tool_call(name: str, args: dict[str, Any], idx: int = 0) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{idx}", "type": "tool_call"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def _final(text: str) -> AIMessage:
    return AIMessage(
        content=text, usage_metadata={"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}
    )


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_toolkit_errors_are_returned_not_raised() -> None:
    async def _go() -> None:
        async with MCPToolkit() as tk:
            names = {t.name for t in tk.tools}
            assert {"buscar_imoveis", "detalhes_imovel", "comparar_imoveis"} <= names
            tool = tk.select(["detalhes_imovel"])[0]
            ok = await tool.ainvoke({"imovel_id": "r001"})
            assert json.loads(ok)["imovel_id"] == "r001"
            missing = await tool.ainvoke({"imovel_id": "r9999"})
            assert missing.startswith("ERRO") and "r9999" in missing
            bad = await tk.select(["simular_compra_financiamento"])[0].ainvoke(
                {"imovel_id": "r001", "prazo_meses": 9999}
            )
            assert bad.startswith("ERRO")
            again = await tool.ainvoke({"imovel_id": "r9999"})
            assert again.startswith("AVISO") and tk.repeated_calls == 1
            assert tk.tool_errors == 2
            assert "Esquema do Estoque" in await tk.read_resource()

    run(_go())


def test_ids_sem_evidencia() -> None:
    ev = [{"ferramenta": "detalhes_imovel", "argumentos": {"imovel_id": "r001"}, "resultado": "{}"}]
    assert _ids_sem_evidencia("O r001 e o r777 são bons", ev) == ["r777"]


def _systems(llm: ScriptedChat) -> Any:
    async def _go(nome: str, question: str) -> Any:
        async with MCPToolkit() as tk:
            systems = build_systems(tk, AgentConfig(), [nome])
            systems[nome].llm = llm
            return await systems[nome].answer(question)

    return _go


def test_single_agent_records_trajectory_and_tokens() -> None:
    llm = ScriptedChat(
        messages=iter([_tool_call("detalhes_imovel", {"imovel_id": "r001"}), _final("r001 ok")])
    )
    res = run(_systems(llm)("langgraph_unico", "Detalhes do r001?"))
    assert res.erro_execucao is None
    assert res.resposta == "r001 ok"
    assert res.ferramentas_usadas == ["detalhes_imovel"]
    assert res.chamadas_modelo == 2
    assert (res.tokens_entrada, res.tokens_saida) == (30, 12)
    rec = res.to_record(7)
    assert rec.sistema == "langgraph_unico" and rec.num_agentes == 1
    assert rec.ram_mb_pico is not None and rec.ram_mb_pico > 0


def test_verifier_rejects_then_accepts() -> None:
    llm = ScriptedChat(
        messages=iter(
            [
                _final("O r001 custa R$ 1,00"),  # draft without tools -> hard rejection
                _tool_call("detalhes_imovel", {"imovel_id": "r001"}),
                _final("O r001 custa R$ 86.098,58"),
            ]
        )
    )
    llm.structured = [Veredito(aprovado=True), Veredito(aprovado=True)]
    res = run(_systems(llm)("langgraph_verificador", "Preço do r001?"))
    assert res.erro_execucao is None
    assert res.resposta == "O r001 custa R$ 86.098,58"
    assert res.detalhes["revisoes"] == 1
    assert res.detalhes["vereditos"][0]["aprovado"] is False
    assert res.ferramentas_usadas == ["detalhes_imovel"]


def test_supervisor_delegates_and_synthesizes() -> None:
    from src.agents.supervisor import Decisao

    llm = ScriptedChat(
        messages=iter(
            [
                _tool_call("detalhes_imovel", {"imovel_id": "r001"}),
                _final("Relatório: r001 custa R$ 86.098,58"),
                _final("Resposta final: o r001 custa R$ 86.098,58"),
            ]
        )
    )
    llm.structured = [
        Decisao(proximo="analista_imoveis", instrucao="detalhes do r001"),
        Decisao(proximo="finalizar"),
    ]
    res = run(_systems(llm)("langgraph_supervisor", "Preço do r001?"))
    assert res.erro_execucao is None
    assert res.resposta.startswith("Resposta final")
    assert res.detalhes["delegacoes"] == ["analista_imoveis"]
    assert res.num_agentes == 4


@pytest.mark.parametrize("nome", ["langgraph_unico", "langgraph_verificador"])
def test_execution_errors_are_captured(nome: str) -> None:
    llm = ScriptedChat(messages=iter([]))  # exhausted model raises inside the graph
    llm.structured = [Veredito(aprovado=True)]
    res = run(_systems(llm)(nome, "?"))
    assert res.erro_execucao
    assert res.to_record(1).erro_execucao
