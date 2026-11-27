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


def test_model_call_cap_forces_final_answer() -> None:
    cfg = AgentConfig()
    cfg.max_chamadas_agente = 2
    loop = [_tool_call("detalhes_imovel", {"imovel_id": f"r00{i}"}, i) for i in range(1, 4)]
    llm = ScriptedChat(messages=iter([*loop[:2], _final("Resposta forçada: r001 e r002")]))

    async def _go() -> Any:
        async with MCPToolkit() as tk:
            system = build_systems(tk, cfg, ["langgraph_unico"])["langgraph_unico"]
            system.llm = llm
            return await system.answer("?")

    res = run(_go())
    assert res.erro_execucao is None
    assert res.resposta == "Resposta forçada: r001 e r002"
    assert res.detalhes["limites_de_chamadas_atingidos"] == 1
    assert res.chamadas_modelo == 3
    assert res.ferramentas_usadas == ["detalhes_imovel", "detalhes_imovel"]


# ── Runner (src.run_agents) and telemetry helpers ──────────────────────────────


def test_run_benchmark_writes_contract_and_resumes(tmp_path: Any, monkeypatch: Any) -> None:
    """End-to-end runner with a fake LLM: JSONL contract, trajectories and --resume."""
    from src import run_agents
    from src.schema import load_executions_jsonl

    script = [
        _final("ok"),  # warm-up call
        _tool_call("detalhes_imovel", {"imovel_id": "r001"}),
        _final("O r001 custa R$ 86.098,58."),
        _tool_call("detalhes_imovel", {"imovel_id": "r9999"}),
        _final("O imóvel r9999 não consta no estoque."),
    ]
    llm = ScriptedChat(messages=iter(script))
    monkeypatch.setattr(AgentConfig, "build_llm", lambda self: llm)
    out, traj = tmp_path / "ex.jsonl", tmp_path / "tr.jsonl"

    total = run(run_agents.run_benchmark(["langgraph_unico"], [7, 18], out, traj, AgentConfig()))
    assert total == 2
    recs = load_executions_jsonl(out)
    assert [(r.sistema, r.pergunta_id) for r in recs] == [
        ("langgraph_unico", 7),
        ("langgraph_unico", 18),
    ]
    assert recs[1].erros_ferramentas == 1  # r9999 -> {"erro": ...}
    assert all(r.erro_execucao is None and r.num_agentes == 1 for r in recs)
    trajs = [json.loads(line) for line in traj.read_text(encoding="utf-8").splitlines()]
    assert trajs[1]["detalhes"]["chamadas_ferramentas"] == [
        {
            "agente": "agente_unico",
            "ferramenta": "detalhes_imovel",
            "argumentos": {"imovel_id": "r9999"},
        }
    ]

    # Resume: nothing left to run (the exhausted fake LLM would fail otherwise, except warm-up)
    llm2 = ScriptedChat(messages=iter([_final("ok")]))
    monkeypatch.setattr(AgentConfig, "build_llm", lambda self: llm2)
    again = run(
        run_agents.run_benchmark(
            ["langgraph_unico"], [7, 18], out, traj, AgentConfig(), resume=True
        )
    )
    assert again == 0
    assert len(load_executions_jsonl(out)) == 2


def test_done_pairs_and_parser_defaults(tmp_path: Any) -> None:
    from src import run_agents

    path = tmp_path / "ex.jsonl"
    assert run_agents._done_pairs(path) == set()
    path.write_text(
        '{"sistema": "langgraph_unico", "pergunta_id": 3}\n\n'
        '{"sistema": "langgraph_supervisor", "pergunta_id": "20"}\n',
        encoding="utf-8",
    )
    assert run_agents._done_pairs(path) == {
        ("langgraph_unico", 3),
        ("langgraph_supervisor", 20),
    }
    args = run_agents.build_parser().parse_args([])
    assert args.sistemas == ["langgraph_unico", "langgraph_verificador", "langgraph_supervisor"]
    assert args.perguntas == list(range(1, 21))
    assert len(args.sistemas) * len(args.perguntas) == 60


def test_collect_environment_without_ollama() -> None:
    from src import run_agents

    cfg = AgentConfig()
    cfg.base_url = "http://127.0.0.1:9"  # nothing listens here
    env = run_agents.collect_environment(cfg)
    assert env["cpu_nucleos_logicos"] >= 1 and env["ram_total_gb"] > 0
    assert env["agente"]["modelo"] == cfg.model
    assert env["agente"]["num_predict"] == cfg.num_predict
    assert env["bibliotecas"]["langgraph"] and env["bibliotecas"]["fastmcp"]
    assert "ollama_erro" in env


def test_config_reads_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv("AGENT_MODEL", "qwen3:8b")
    monkeypatch.setenv("AGENT_REASONING", "sim")
    monkeypatch.setenv("AGENT_MAX_CHAMADAS_AGENTE", "5")
    cfg = AgentConfig()
    assert (cfg.model, cfg.reasoning, cfg.max_chamadas_agente) == ("qwen3:8b", True, 5)
    assert cfg.recursion_limit > 4 * cfg.max_chamadas_agente  # graceful cap fires first
    llm = cfg.build_llm()
    assert llm.model == "qwen3:8b" and llm.num_predict == cfg.num_predict


def test_strip_nulls_simplifies_optional_schema() -> None:
    from src.agents.mcp_tools import _strip_nulls

    schema = {
        "properties": {
            "cidade": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "multi": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        }
    }
    out = _strip_nulls(schema)
    assert out["properties"]["cidade"] == {"type": "string", "default": None}
    assert "anyOf" in out["properties"]["multi"]


def test_message_helpers() -> None:
    from langchain_core.messages import ToolMessage

    from src.agents.base import final_ai_text, message_text, tool_evidence

    assert message_text(AIMessage(content="<think>hmm</think> Resposta")) == "Resposta"
    assert message_text(AIMessage(content=[{"type": "text", "text": "a"}, "b"])) == "ab"
    msgs = [
        _tool_call("detalhes_imovel", {"imovel_id": "r001"}, 1),
        ToolMessage(content="x" * 5000, tool_call_id="call_1", name="detalhes_imovel"),
        _final("fim"),
        AIMessage(content=""),
    ]
    assert final_ai_text(msgs) == "fim"
    ev = tool_evidence(msgs, max_chars=100)
    assert ev[0]["ferramenta"] == "detalhes_imovel"
    assert ev[0]["argumentos"] == {"imovel_id": "r001"}
    assert ev[0]["resultado"].endswith("…[truncado]") and len(ev[0]["resultado"]) < 120


def test_tracker_and_resource_monitor() -> None:
    import time
    from uuid import uuid4

    from langchain_core.outputs import ChatGeneration, LLMResult

    from src.agents.telemetry import ExecutionTracker, ResourceMonitor

    tracker = ExecutionTracker()
    assert tracker.tokens == (None, None)  # no usage reported -> None, not 0
    tracker.on_chat_model_start({}, [[]], run_id=uuid4(), metadata={"agente": "gerador"})
    tracker.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=_final("x"))]]), run_id=uuid4()
    )
    assert tracker.tokens == (20, 7)
    assert tracker.chamadas_por_agente == {"gerador": 1}

    with ResourceMonitor(intervalo_s=0.05) as mon:
        sum(i * i for i in range(300_000))
        time.sleep(0.2)
    snap = mon.snapshot()
    assert snap.ram_mb_pico is not None and snap.ram_mb_pico > 10
    assert snap.cpu_pct_medio is not None and 0 <= snap.cpu_pct_medio <= 100
    assert snap.cpu_pct_pico is not None and snap.cpu_pct_pico >= snap.cpu_pct_medio
