#!/usr/bin/env python3
"""Benchmark analysis reporter and manual audit divergence exporter (PT-BR)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

CATEGORY_ORDER: list[str] = [
    "busca_com_filtro",
    "consulta_especifica",
    "comparacao",
    "calculo_financeiro",
    "pegadinha",
    "fora_do_escopo",
]

VERDICT_ORDER: list[str] = ["correta", "parcial", "incorreta", "alucinacao"]

AUDIT_COLUMNS: list[str] = [
    "sistema",
    "pergunta_id",
    "categoria",
    "acerto_automatico",
    "falhas_automaticas",
    "acerto_juiz",
    "veredito_detalhado",
    "justificativa_juiz",
    "consultas_juiz",
    "resposta_agente",
    "revisao_manual",
    "nota_manual",
]


def _to_bool(value: Any) -> bool | None:
    """Coerce CSV boolean/string cells into ``bool | None``."""
    if value is None or bool(pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "sim"):
        return True
    if text in ("false", "0", "nao", "não"):
        return False
    return None


def compute_cohens_kappa(
    auto_series: Sequence[bool],
    judge_series: Sequence[bool],
) -> float | None:
    """Compute Cohen's kappa between two binary rater sequences without external deps."""
    n = len(auto_series)
    if n == 0 or n != len(judge_series):
        return None

    tp = sum(1 for a, j in zip(auto_series, judge_series, strict=True) if a and j)
    tn = sum(1 for a, j in zip(auto_series, judge_series, strict=True) if not a and not j)
    fp = sum(1 for a, j in zip(auto_series, judge_series, strict=True) if a and not j)
    fn = sum(1 for a, j in zip(auto_series, judge_series, strict=True) if not a and j)

    p_o = (tp + tn) / n
    p_auto_yes = (tp + fp) / n
    p_judge_yes = (tp + fn) / n
    p_e = (p_auto_yes * p_judge_yes) + ((1.0 - p_auto_yes) * (1.0 - p_judge_yes))

    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    return round((p_o - p_e) / denom, 4)


def _format_pct(num: int, total: int) -> str:
    if total == 0:
        return "0/0 (0.0%)"
    return f"{num}/{total} ({(num / total) * 100.0:.1f}%)"


def _build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_report(
    input_csv: str | Path,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Generate PT-BR ``benchmark_summary.md`` and ``divergencias_auditoria_manual.csv``."""
    in_path = Path(input_csv)
    if not in_path.is_file():
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {in_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df["_auto_bool"] = [_to_bool(v) for v in df["acerto_automatico"]]
    df["_juiz_bool"] = [_to_bool(v) for v in df["acerto_juiz"]]
    df["_tool_bool"] = [_to_bool(v) for v in df["ferramenta_correta"]]

    systems = sorted(df["sistema"].dropna().astype(str).unique().tolist())

    # Tabela 1 — Resumo Geral por Sistema/Arquitetura
    t1_headers = [
        "Sistema",
        "Execuções",
        "Acerto Auto",
        "Acerto Juiz",
        "Ferramenta Correta (%)",
        "Taxa Alucinação (%)",
        "Concordância Auto/Juiz (%)",
        "Kappa Cohen",
        "Chamadas Modelo (Total / Média)",
        "Latência (Total s / Mediana s)",
        "Tokens Totais",
    ]
    t1_rows: list[list[str]] = []
    for sys_name in systems:
        sub = df.loc[df["sistema"] == sys_name]
        n = int(len(sub))
        auto_ok = sum(1 for v in sub["_auto_bool"] if v is True)
        tool_ok = sum(1 for v in sub["_tool_bool"] if v is True)

        judge_pairs = [
            (bool(a), bool(j))
            for a, j in zip(sub["_auto_bool"], sub["_juiz_bool"], strict=True)
            if a is not None and j is not None
        ]
        if judge_pairs:
            judge_ok = sum(1 for _, j in judge_pairs if j)
            judge_str = _format_pct(judge_ok, len(judge_pairs))
            agree = sum(1 for a, j in judge_pairs if a == j)
            agree_str = f"{(agree / len(judge_pairs)) * 100.0:.1f}%"
            kappa = compute_cohens_kappa(
                [a for a, _ in judge_pairs], [j for _, j in judge_pairs]
            )
            kappa_str = f"{kappa:.4f}" if kappa is not None else "N/A"
            verdicts = (
                sub["veredito_detalhado"].astype("string").fillna("").str.strip().str.lower()
            )
            halluc_cnt = int((verdicts == "alucinacao").sum())
            halluc_str = f"{(halluc_cnt / len(judge_pairs)) * 100.0:.1f}%"
        else:
            judge_str = "N/A"
            agree_str = "N/A"
            kappa_str = "N/A"
            halluc_str = "N/A"

        calls_total = int(sub["chamadas_modelo"].fillna(0).sum())
        calls_avg = calls_total / n if n else 0.0
        lat_total = float(sub["latencia_segundos"].fillna(0.0).sum())
        lat_med = float(sub["latencia_segundos"].fillna(0.0).median()) if n else 0.0
        tok_in = int(sub["tokens_entrada"].fillna(0).sum())
        tok_out = int(sub["tokens_saida"].fillna(0).sum())
        tok_total = tok_in + tok_out

        t1_rows.append(
            [
                sys_name,
                str(n),
                _format_pct(auto_ok, n),
                judge_str,
                f"{(tool_ok / n) * 100.0:.1f}%" if n else "0.0%",
                halluc_str,
                agree_str,
                kappa_str,
                f"{calls_total} / {calls_avg:.2f}",
                f"{lat_total:.1f}s / {lat_med:.2f}s",
                str(tok_total),
            ]
        )

    # Tabela 2 — Matriz de Acerto do Juiz por Categoria × Sistema
    has_any_judge = any(v is not None for v in df["_juiz_bool"])
    metric_col = "_juiz_bool" if has_any_judge else "_auto_bool"
    t2_title = (
        "Tabela 2 — Matriz de Acerto do Juiz por Categoria × Sistema"
        if has_any_judge
        else "Tabela 2 — Matriz de Acerto Automático por Categoria × Sistema (sem juiz)"
    )
    present_cats = [c for c in CATEGORY_ORDER if c in set(df["categoria"].astype(str))]
    for extra in sorted(set(df["categoria"].astype(str)) - set(present_cats)):
        present_cats.append(extra)

    t2_headers = ["Categoria", *systems]
    t2_rows: list[list[str]] = []
    for cat in present_cats:
        row_cells = [cat]
        for sys_name in systems:
            sub = df.loc[(df["categoria"] == cat) & (df["sistema"] == sys_name)]
            vals = [v for v in sub[metric_col] if v is not None]
            ok_cnt = sum(1 for v in vals if v)
            row_cells.append(_format_pct(ok_cnt, len(vals)))
        t2_rows.append(row_cells)

    # Tabela 3 — Distribuição de Vereditos Detalhados por Sistema
    t3_headers = ["Sistema", *VERDICT_ORDER, "Sem Juiz"]
    t3_rows: list[list[str]] = []
    for sys_name in systems:
        sub = df.loc[df["sistema"] == sys_name]
        verdicts = (
            sub["veredito_detalhado"].astype("string").fillna("").str.strip().str.lower()
        )
        counts = [str(int((verdicts == v).sum())) for v in VERDICT_ORDER]
        no_judge = str(int((verdicts == "").sum()))
        t3_rows.append([sys_name, *counts, no_judge])

    summary_md = "\n\n".join(
        [
            "# Relatório Consolidado do Benchmark — Property Agents (RN)",
            f"Arquivo avaliado: `{in_path}` ({len(df)} execuções)",
            "## Tabela 1 — Resumo Geral por Sistema/Arquitetura",
            _build_markdown_table(t1_headers, t1_rows),
            f"## {t2_title}",
            _build_markdown_table(t2_headers, t2_rows),
            "## Tabela 3 — Distribuição de Vereditos Detalhados por Sistema",
            _build_markdown_table(t3_headers, t3_rows),
        ]
    )

    summary_path = out_path / "benchmark_summary.md"
    summary_path.write_text(summary_md + "\n", encoding="utf-8")

    # Export divergences & non-"correta" rows for manual audit
    verdicts_all = df["veredito_detalhado"].astype("string").fillna("").str.strip().str.lower()
    div_mask = [
        (a != j if j is not None else a is False) or (v != "" and v != "correta")
        for a, j, v in zip(df["_auto_bool"], df["_juiz_bool"], verdicts_all, strict=True)
    ]
    df_div = df.loc[div_mask].copy()
    for col in AUDIT_COLUMNS:
        if col not in df_div.columns:
            df_div[col] = ""
    df_div = df_div[AUDIT_COLUMNS]
    div_path = out_path / "divergencias_auditoria_manual.csv"
    df_div.to_csv(div_path, index=False)

    print(summary_md)
    print(f"\nArquivos gerados: {summary_path} e {div_path} ({len(df_div)} divergência(s)).")
    return {"summary_md": summary_path, "divergencias_csv": div_path}


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for ``python -m src.report``."""
    parser = argparse.ArgumentParser(
        description="Gera relatório consolidado em PT-BR e CSV de auditoria manual."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/benchmark_60_execucoes.csv",
        help="CSV gerado por src.metrics.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Diretório de saída para benchmark_summary.md e divergencias_auditoria_manual.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for generating benchmark summary and audit divergence files."""
    args = build_parser().parse_args(argv)
    generate_report(args.input, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
