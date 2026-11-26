"""End-to-end tests for schemas, JSONL I/O, metrics pipeline, resume, and reporting."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.judge import JudgeConfig, LLMStepResponse, LLMToolRequest, MCPJudge
from src.metrics import evaluate_executions
from src.report import compute_cohens_kappa, generate_report
from src.schema import ExecutionRecord, append_execution_jsonl, load_executions_jsonl
from tests.test_judge_calibration import build_calibration_cases


class _PipelineMockBackend:
    """Mock judge backend returning controllable verdicts for pipeline tests."""

    def __init__(self, expected_by_prompt: dict[str, tuple[bool, str, str]]) -> None:
        self._expected = expected_by_prompt
        self.calls = 0

    async def generate_step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMStepResponse:
        self.calls += 1
        user_prompt = str(messages[0]["content"])
        if len(messages) == 1:
            return LLMStepResponse(
                content="",
                tool_calls=[
                    LLMToolRequest(
                        name="detalhes_imovel",
                        arguments={"imovel_id": "r001"},
                        call_id="pipe_1",
                    )
                ],
            )
        for key, (ok, veredito, just) in self._expected.items():
            if key in user_prompt:
                return LLMStepResponse(
                    content=json.dumps(
                        {
                            "correta": ok,
                            "veredito_detalhado": veredito,
                            "justificativa": just,
                        },
                        ensure_ascii=False,
                    ),
                    tool_calls=[],
                )
        return LLMStepResponse(
            content=json.dumps(
                {
                    "correta": True,
                    "veredito_detalhado": "correta",
                    "justificativa": "Verificado via MCP.",
                }
            ),
            tool_calls=[],
        )


def test_execution_record_and_jsonl_roundtrip(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "nested" / "executions.jsonl"
    rec1 = ExecutionRecord(
        sistema="  langgraph_unico  ",
        pergunta_id=1,
        resposta="Recomendo o imóvel r471.",
        ferramentas_usadas=["buscar_imoveis"],
        chamadas_modelo=2,
        latencia_segundos=1.25,
        tokens_entrada=300,
        tokens_saida=90,
        erros_ferramentas=1,
        num_agentes=1,
        cpu_pct_medio=45.5,
        cpu_pct_pico=78.0,
        ram_mb_pico=1024.0,
        gpu_pct_medio=68.0,
        gpu_mem_mb_pico=4096.0,
        modelo_agente="qwen3:8b",
    )
    rec2 = ExecutionRecord(
        sistema="langgraph_supervisor",
        pergunta_id=18,
        resposta="O imóvel r9999 não existe.",
        ferramentas_usadas=["detalhes_imovel"],
        chamadas_modelo=2,
        latencia_segundos=0.95,
    )
    append_execution_jsonl(jsonl_path, rec1)
    append_execution_jsonl(jsonl_path, rec2)

    loaded = load_executions_jsonl(jsonl_path)
    assert len(loaded) == 2
    assert loaded[0].sistema == "langgraph_unico"
    assert loaded[0].erros_ferramentas == 1
    assert loaded[0].num_agentes == 1
    assert loaded[0].cpu_pct_medio == 45.5
    assert loaded[0].cpu_pct_pico == 78.0
    assert loaded[0].ram_mb_pico == 1024.0
    assert loaded[0].gpu_pct_medio == 68.0
    assert loaded[0].gpu_mem_mb_pico == 4096.0
    assert loaded[1].pergunta_id == 18
    assert loaded[1].cpu_pct_medio is None
    assert loaded[1].gpu_pct_medio is None

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"sistema": "ok"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="linha 1"):
        load_executions_jsonl(bad_path)


def test_metrics_pipeline_resume_and_report(tmp_path: Path) -> None:
    cases = build_calibration_cases()
    c_pos_q1 = cases[0]
    c_pos_q12 = cases[1]
    c_neg_q8 = cases[4]

    # Create 6-row synthetic executions across 3 systems:
    # - 1 crash/erro_execucao (should not invoke LLM judge)
    # - 1 auto/judge divergence (Q1 with valid ID & tool, but judge flags parcial/incorreta)
    div_rec = ExecutionRecord(
        sistema="langgraph_verificador",
        pergunta_id=1,
        resposta=c_pos_q1["execution"].resposta + " Preço errado inventado R$ 999.",
        ferramentas_usadas=["buscar_imoveis"],
        chamadas_modelo=3,
        latencia_segundos=2.1,
        tokens_entrada=500,
        tokens_saida=120,
        modelo_agente="qwen3:8b",
    )
    records = [
        ExecutionRecord(
            sistema="langgraph_unico",
            pergunta_id=1,
            resposta=c_pos_q1["execution"].resposta,
            ferramentas_usadas=["buscar_imoveis"],
            chamadas_modelo=2,
            latencia_segundos=1.4,
            tokens_entrada=250,
            tokens_saida=80,
            erros_ferramentas=0,
            num_agentes=1,
            cpu_pct_medio=40.0,
            cpu_pct_pico=65.0,
            ram_mb_pico=1000.0,
            gpu_pct_medio=60.0,
            gpu_mem_mb_pico=3000.0,
            modelo_agente="qwen3:8b",
        ),
        ExecutionRecord(
            sistema="langgraph_unico",
            pergunta_id=12,
            resposta=c_pos_q12["execution"].resposta,
            ferramentas_usadas=["avaliar_investimento", "detalhes_imovel"],
            chamadas_modelo=2,
            latencia_segundos=1.2,
            tokens_entrada=280,
            tokens_saida=95,
            erros_ferramentas=0,
            num_agentes=1,
            cpu_pct_medio=60.0,
            cpu_pct_pico=85.0,
            ram_mb_pico=1500.0,
            gpu_pct_medio=80.0,
            gpu_mem_mb_pico=5000.0,
            modelo_agente="qwen3:8b",
        ),
        div_rec,
        ExecutionRecord(
            sistema="langgraph_verificador",
            pergunta_id=8,
            resposta=c_neg_q8["execution"].resposta,
            ferramentas_usadas=["comparar_imoveis"],
            chamadas_modelo=3,
            latencia_segundos=1.8,
            tokens_entrada=420,
            tokens_saida=110,
            modelo_agente="qwen3:8b",
        ),
        ExecutionRecord(
            sistema="langgraph_supervisor",
            pergunta_id=12,
            resposta="",
            ferramentas_usadas=[],
            chamadas_modelo=1,
            latencia_segundos=5.0,
            erro_execucao="TimeoutError após 5s",
            modelo_agente="qwen3:8b",
        ),
        ExecutionRecord(
            sistema="langgraph_supervisor",
            pergunta_id=18,
            resposta="O imóvel r9999 fica em Ponta Negra por R$ 300.000.",
            ferramentas_usadas=["detalhes_imovel"],
            chamadas_modelo=2,
            latencia_segundos=1.1,
            tokens_entrada=220,
            tokens_saida=60,
            modelo_agente="qwen3:8b",
        ),
    ]

    jsonl_file = tmp_path / "executions.jsonl"
    for r in records:
        append_execution_jsonl(jsonl_file, r)

    out_csv = tmp_path / "benchmark_60_execucoes.csv"
    expected_map = {
        "Preço errado inventado R$ 999": (False, "parcial", "Preço divergente no anúncio."),
        c_neg_q8["execution"].resposta: (False, "incorreta", "Ranking invertido."),
        "O imóvel r9999 fica em Ponta Negra": (False, "alucinacao", "Inventou r9999."),
    }
    backend = _PipelineMockBackend(expected_map)
    cfg = JudgeConfig(provider="gemini", model="gemini-2.5-flash", min_delay_seconds=0.0)

    loaded_records = load_executions_jsonl(jsonl_file)
    rows = asyncio.run(
        evaluate_executions(
            records=loaded_records,
            output_csv=out_csv,
            judge=MCPJudge(config=cfg, backend=backend),
        )
    )
    assert len(rows) == 6
    # Error row skipped judge LLM call: 5 non-error rows * 2 steps each = 10 calls
    assert backend.calls == 10

    df_csv = pd.read_csv(out_csv)
    assert len(df_csv) == 6
    assert "consultas_juiz" in df_csv.columns
    for raw_json in df_csv["consultas_juiz"]:
        parsed = json.loads(str(raw_json))
        assert isinstance(parsed, list)

    # Verify divergence row was detected (auto=True, judge=False)
    div_row = df_csv.loc[
        (df_csv["sistema"] == "langgraph_verificador") & (df_csv["pergunta_id"] == 1)
    ].iloc[0]
    assert bool(div_row["acerto_automatico"]) is True
    assert bool(div_row["acerto_juiz"]) is False
    assert bool(div_row["concordancia_auto_juiz"]) is False

    # Verify --resume makes 0 additional LLM calls and keeps 6 rows
    calls_before_resume = backend.calls
    rows_resumed = asyncio.run(
        evaluate_executions(
            records=loaded_records,
            output_csv=out_csv,
            judge=MCPJudge(config=cfg, backend=backend),
            resume=True,
        )
    )
    assert len(rows_resumed) == 6
    assert backend.calls == calls_before_resume

    # Generate report and audit CSV
    artifacts = generate_report(out_csv, tmp_path)
    summary_text = artifacts["summary_md"].read_text(encoding="utf-8")
    assert "Tabela 1 — Resumo Geral por Sistema/Arquitetura" in summary_text
    assert "Taxa Acerto (%)" in summary_text
    assert "Taxa Alucinação (%)" in summary_text
    assert "Chamadas LLM (Total / Média)" in summary_text
    assert "Chamadas Tools (Total / Média)" in summary_text
    assert "Chamadas Total (Total / Média)" in summary_text
    assert "Tabela 2 — Matriz de Acerto do Juiz por Categoria × Sistema" in summary_text
    assert "Tabela 3 — Distribuição de Vereditos Detalhados por Sistema" in summary_text
    assert "Tabela 4 — Estatísticas Computacionais por Sistema (CPU, RAM e GPU)" in summary_text
    assert "| langgraph_unico | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 100.0% | 0.0% |" in summary_text
    assert "| 4 / 2.00 | 3 / 1.50 | 7 / 3.50 |" in summary_text
    assert (
        "| langgraph_unico | 2/2 | 50.0% | 85.0% | 1250.0 MB / 1500.0 MB "
        "| 70.0% / 80.0% | 4000.0 MB / 5000.0 MB |"
    ) in summary_text
    assert "| langgraph_supervisor | 0/2 | N/A | N/A | N/A | N/A | N/A |" in summary_text

    df_div = pd.read_csv(artifacts["divergencias_csv"])
    assert len(df_div) == 4  # 4 rows with veredito_detalhado != 'correta' or divergence
    assert set(df_div["veredito_detalhado"]) == {"parcial", "incorreta", "alucinacao"}


def test_report_legacy_csv_and_skip_judge_fallback(tmp_path: Path) -> None:
    legacy_csv = tmp_path / "legacy.csv"
    df_legacy = pd.DataFrame(
        [
            {
                "sistema": "langgraph_unico",
                "pergunta_id": 1,
                "categoria": "busca_com_filtro",
                "ferramenta_esperada": "buscar_imoveis",
                "ferramentas_usadas": '["buscar_imoveis"]',
                "ferramenta_correta": True,
                "num_tool_calls": 1,
                "chamadas_modelo": 2,
                "latencia_segundos": 1.0,
                "tokens_entrada": 100,
                "tokens_saida": 50,
                "acerto_automatico": True,
                "falhas_automaticas": "",
                "acerto_juiz": None,
                "veredito_detalhado": None,
                "resposta_agente": "Imóvel r001.",
            },
            {
                "sistema": "langgraph_unico",
                "pergunta_id": 2,
                "categoria": "busca_com_filtro",
                "ferramenta_esperada": "buscar_imoveis",
                "ferramentas_usadas": '["buscar_imoveis", "detalhes_imovel"]',
                "ferramenta_correta": True,
                "num_tool_calls": 2,
                "chamadas_modelo": 3,
                "latencia_segundos": 2.0,
                "tokens_entrada": 200,
                "tokens_saida": 70,
                "acerto_automatico": False,
                "falhas_automaticas": "faltou_valor",
                "acerto_juiz": None,
                "veredito_detalhado": None,
                "resposta_agente": "Sem valor.",
            },
        ]
    )
    df_legacy.to_csv(legacy_csv, index=False)

    artifacts = generate_report(legacy_csv, tmp_path / "out")
    summary_text = artifacts["summary_md"].read_text(encoding="utf-8")
    assert "| langgraph_unico | 2 | 1/2 (50.0%) | N/A | 50.0% | N/A |" in summary_text
    assert "| 5 / 2.50 | 3 / 1.50 | 8 / 4.00 |" in summary_text
    assert "| langgraph_unico | 0/2 | N/A | N/A | N/A | N/A | N/A |" in summary_text


def test_compute_cohens_kappa() -> None:
    assert compute_cohens_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    neg_kappa = compute_cohens_kappa([True, False], [False, True])
    assert neg_kappa is not None and neg_kappa < 0.0
    assert compute_cohens_kappa([], []) is None
