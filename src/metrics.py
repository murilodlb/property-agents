#!/usr/bin/env python3
"""Evaluation pipeline combining automatic regex/trajectory grading and MCP Frontier Judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.judge import JudgeConfig, MCPJudge
from src.questions import PERGUNTAS, checagem_automatica
from src.schema import BenchmarkRow, ExecutionRecord, JudgeVerdict, load_executions_jsonl

BENCHMARK_COLUMNS: list[str] = list(BenchmarkRow.model_fields.keys())


def _load_existing_rows(csv_path: Path) -> dict[tuple[str, int], BenchmarkRow]:
    """Load previously evaluated rows from ``csv_path`` keyed by ``(sistema, pergunta_id)``."""
    df_existing = pd.read_csv(csv_path)
    existing: dict[tuple[str, int], BenchmarkRow] = {}
    for _, raw in df_existing.iterrows():
        cleaned: dict[str, Any] = {}
        for col in BENCHMARK_COLUMNS:
            val = raw.get(col)
            if val is None or bool(pd.isna(val)):
                if col in (
                    "revisao_manual",
                    "nota_manual",
                    "falhas_automaticas",
                    "resposta_agente",
                ):
                    cleaned[col] = ""
                elif col == "consultas_juiz":
                    cleaned[col] = "[]"
                else:
                    cleaned[col] = None
            else:
                cleaned[col] = val
        row = BenchmarkRow.model_validate(cleaned)
        existing[(row.sistema, row.pergunta_id)] = row
    return existing


def _save_rows_csv(rows: Sequence[BenchmarkRow], output_path: Path) -> None:
    """Persist ``rows`` deterministically sorted by ``(sistema, pergunta_id)``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r.sistema, r.pergunta_id))
    data = [r.model_dump(mode="python") for r in ordered]
    df_out = pd.DataFrame(data, columns=BENCHMARK_COLUMNS)
    tmp_path = Path(f"{output_path}.tmp")
    df_out.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)


def _check_expected_tool(pergunta_dict: dict[str, Any], ferramentas_usadas: list[str]) -> bool:
    """Return True if at least one called tool matches ``pergunta_dict['ferramenta']``."""
    pattern = str(pergunta_dict.get("ferramenta") or "").strip()
    if not pattern:
        return True
    regex = re.compile(pattern, re.IGNORECASE)
    return any(regex.search(tool) for tool in ferramentas_usadas)


async def evaluate_executions(
    records: Sequence[ExecutionRecord],
    output_csv: str | Path | None = None,
    judge: MCPJudge | None = None,
    skip_judge: bool = False,
    include_reference: bool = True,
    resume: bool = False,
) -> list[BenchmarkRow]:
    """Evaluate all agent ``records`` with automatic checks and optional MCP Frontier Judge."""
    questions_by_id: dict[int, dict[str, Any]] = {int(q["id"]): q for q in PERGUNTAS}
    out_path = Path(output_csv) if output_csv is not None else None

    rows_by_key: dict[tuple[str, int], BenchmarkRow] = {}
    if resume and out_path is not None and out_path.is_file():
        rows_by_key = _load_existing_rows(out_path)

    pending: list[ExecutionRecord] = []
    for rec in records:
        key = (rec.sistema, rec.pergunta_id)
        if resume and key in rows_by_key:
            continue
        pending.append(rec)

    active_judge = judge if (not skip_judge) else None
    if active_judge is None and not skip_judge and pending:
        active_judge = MCPJudge()

    async def _evaluate_all(connected_judge: MCPJudge | None) -> None:
        for rec in pending:
            pergunta = questions_by_id.get(rec.pergunta_id)
            if pergunta is None:
                raise ValueError(
                    f"pergunta_id inválido ({rec.pergunta_id}) para sistema {rec.sistema!r}."
                )

            ferramenta_correta = _check_expected_tool(pergunta, rec.ferramentas_usadas)
            falhas = checagem_automatica(
                pergunta,
                {"resposta": rec.resposta, "ferramentas_usadas": rec.ferramentas_usadas},
            )
            has_exec_error = bool(rec.erro_execucao and rec.erro_execucao.strip()) or not bool(
                rec.resposta.strip()
            )
            if has_exec_error:
                err_note = f"erro_execucao: {rec.erro_execucao or 'resposta vazia'}"
                if err_note not in falhas:
                    falhas.insert(0, err_note)
                acerto_automatico = False
            else:
                acerto_automatico = len(falhas) == 0

            verdict: JudgeVerdict | None = None
            if not skip_judge:
                if has_exec_error:
                    verdict = JudgeVerdict(
                        correta=False,
                        veredito_detalhado="incorreta",
                        justificativa=(
                            f"Falha na execução do agente: {rec.erro_execucao or 'resposta vazia'}."
                        ),
                        consultas_realizadas=[],
                        provedor_juiz=(
                            connected_judge.config.provider if connected_judge else "pipeline"
                        ),
                        modelo_juiz=(
                            connected_judge.config.model if connected_judge else "pipeline"
                        ),
                    )
                else:
                    assert connected_judge is not None
                    verdict = await connected_judge.evaluate(
                        pergunta, rec, include_reference=include_reference
                    )

            acerto_juiz = verdict.correta if verdict is not None else None
            concordancia = (
                (acerto_automatico == acerto_juiz) if acerto_juiz is not None else None
            )

            row = BenchmarkRow(
                sistema=rec.sistema,
                pergunta_id=rec.pergunta_id,
                categoria=str(pergunta["categoria"]),
                ferramenta_esperada=str(pergunta["ferramenta"]),
                modelo_agente=rec.modelo_agente,
                ferramentas_usadas=json.dumps(rec.ferramentas_usadas, ensure_ascii=False),
                ferramenta_correta=ferramenta_correta,
                num_tool_calls=len(rec.ferramentas_usadas),
                chamadas_modelo=rec.chamadas_modelo,
                latencia_segundos=round(rec.latencia_segundos, 3),
                tokens_entrada=rec.tokens_entrada,
                tokens_saida=rec.tokens_saida,
                erro_execucao=rec.erro_execucao,
                erros_ferramentas=rec.erros_ferramentas,
                num_agentes=rec.num_agentes,
                cpu_pct_medio=rec.cpu_pct_medio,
                cpu_pct_pico=rec.cpu_pct_pico,
                ram_mb_pico=rec.ram_mb_pico,
                gpu_pct_medio=rec.gpu_pct_medio,
                gpu_mem_mb_pico=rec.gpu_mem_mb_pico,
                acerto_automatico=acerto_automatico,
                falhas_automaticas="; ".join(falhas),
                acerto_juiz=acerto_juiz,
                veredito_detalhado=verdict.veredito_detalhado if verdict else None,
                justificativa_juiz=verdict.justificativa if verdict else None,
                consultas_juiz=(
                    json.dumps(
                        [c.model_dump(mode="json") for c in verdict.consultas_realizadas],
                        ensure_ascii=False,
                    )
                    if verdict
                    else "[]"
                ),
                num_consultas_juiz=len(verdict.consultas_realizadas) if verdict else 0,
                provedor_juiz=verdict.provedor_juiz if verdict else None,
                modelo_juiz=verdict.modelo_juiz if verdict else None,
                concordancia_auto_juiz=concordancia,
                resposta_agente=rec.resposta,
            )
            rows_by_key[(row.sistema, row.pergunta_id)] = row
            if out_path is not None:
                _save_rows_csv(list(rows_by_key.values()), out_path)

    if active_judge is not None and any(
        not (bool(r.erro_execucao and r.erro_execucao.strip()) or not r.resposta.strip())
        for r in pending
    ):
        async with active_judge as connected:
            await _evaluate_all(connected)
    else:
        await _evaluate_all(active_judge)

    ordered_rows = sorted(rows_by_key.values(), key=lambda r: (r.sistema, r.pergunta_id))
    if out_path is not None:
        _save_rows_csv(ordered_rows, out_path)
    return ordered_rows


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``python -m src.metrics``."""
    parser = argparse.ArgumentParser(
        description="Pipeline de avaliação do benchmark property-agents (60 execuções)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/executions.jsonl",
        help="Caminho do arquivo JSONL de execuções dos agentes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark_60_execucoes.csv",
        help="Caminho do CSV consolidado de métricas.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai_compatible", "ollama"],
        default=None,
        help="Provedor do Juiz Frontier (padrão: env JUDGE_PROVIDER ou gemini).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo do Juiz Frontier (não usar modelos qwen*).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Intervalo mínimo em segundos entre chamadas ao juiz.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reaproveita pares (sistema, pergunta_id) já salvos em --output.",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Executa apenas a checagem automática e telemetria offline (sem juiz LLM).",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Omite o gabarito do prompt do juiz para auditoria puramente via MCP.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for evaluating ``executions.jsonl``."""
    args = build_parser().parse_args(argv)
    records = load_executions_jsonl(args.input)

    judge: MCPJudge | None = None
    if not args.skip_judge:
        cfg = JudgeConfig.from_env(
            provider=args.provider,
            model=args.model,
            min_delay_seconds=args.delay,
        )
        judge = MCPJudge(config=cfg)

    rows = asyncio.run(
        evaluate_executions(
            records=records,
            output_csv=args.output,
            judge=judge,
            skip_judge=args.skip_judge,
            include_reference=not args.no_reference,
            resume=args.resume,
        )
    )
    print(f"Avaliação concluída: {len(rows)} execução(ões) salvas em {args.output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
