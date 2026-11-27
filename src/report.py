#!/usr/bin/env python3
"""Benchmark analysis reporter and manual audit divergence exporter (PT-BR)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
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

RESOURCE_COLUMNS: tuple[str, ...] = (
    "cpu_pct_medio",
    "cpu_pct_pico",
    "ram_mb_pico",
    "gpu_pct_medio",
    "gpu_mem_mb_pico",
)


MANUAL_TRUE = {"correta", "correto", "sim", "s", "true", "1", "ok", "v", "acerto"}
MANUAL_FALSE = {
    "incorreta",
    "incorreto",
    "parcial",
    "alucinacao",
    "alucinação",
    "nao",
    "não",
    "n",
    "false",
    "0",
    "x",
    "erro",
}


def parse_manual_label(value: Any) -> bool | None:
    """Map a free-text ``revisao_manual`` cell to correct/incorrect (``None`` if blank)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text in MANUAL_TRUE:
        return True
    if text in MANUAL_FALSE:
        return False
    raise ValueError(
        f"revisao_manual inválida: {value!r}. Use 'correta' ou 'incorreta' "
        "(também aceitos: sim/nao, true/false, 1/0, parcial, alucinacao)."
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in 0..100); ``None`` for empty input."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _acerto_series(df: pd.DataFrame) -> pd.Series:
    """Per-row correctness used across the report: judge verdict, else automatic check."""
    if any(v is not None for v in df["_juiz_bool"]):
        return df["_juiz_bool"]
    return df["_auto_bool"]


def _load_manual_reviews(path: Path) -> dict[tuple[str, int], tuple[str, str]]:
    """Read already-filled ``revisao_manual``/``nota_manual`` cells of a previous audit CSV."""
    if not path.is_file():
        return {}
    prev = pd.read_csv(path, dtype={"revisao_manual": "string", "nota_manual": "string"})
    reviews: dict[tuple[str, int], tuple[str, str]] = {}
    for _, row in prev.iterrows():
        rev = row.get("revisao_manual")
        nota = row.get("nota_manual")
        rev_s = "" if rev is None or pd.isna(rev) else str(rev).strip()
        nota_s = "" if nota is None or pd.isna(nota) else str(nota).strip()
        if rev_s or nota_s:
            reviews[(str(row["sistema"]), int(row["pergunta_id"]))] = (rev_s, nota_s)
    return reviews


def plot_accuracy_vs_cost(summary: pd.DataFrame, output_png: Path) -> Path:
    """Scatter of accuracy (%) vs mean tokens and vs mean latency, one point per system."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#2a6fdb", "#e0782b", "#2f9e5b", "#b23a8f", "#6b6b6b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    panels = [
        ("tokens_medios", "Tokens médios por execução (entrada + saída)"),
        ("latencia_media_s", "Latência média por execução (s)"),
    ]
    for ax, (col, xlabel) in zip(axes, panels, strict=True):
        for i, row in enumerate(summary.itertuples(index=False)):
            color = palette[i % len(palette)]
            x = float(getattr(row, col))
            y = float(row.taxa_acerto_pct)
            ax.scatter([x], [y], s=110, color=color, zorder=3, label=row.sistema)
            ax.annotate(
                row.sistema,
                (x, y),
                textcoords="offset points",
                xytext=(7, 6),
                fontsize=9,
                color=color,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Taxa de acerto (%)")
        ax.set_ylim(0, 105)
        ax.set_xlim(left=0, right=max(float(summary[col].max()) * 1.35, 1.0))
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Acerto × custo por arquitetura (mais acima e à esquerda = melhor)")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


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


def _extract_floats(
    sub: pd.DataFrame,
    col: str,
    *,
    fill_na: float | None = None,
) -> list[float]:
    if col not in sub.columns:
        return []
    out: list[float] = []
    for raw in sub[col].tolist():
        if raw is None or bool(pd.isna(raw)):
            if fill_na is not None:
                out.append(fill_na)
            continue
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            if fill_na is not None:
                out.append(fill_na)
    return out


def _sum_int_column(sub: pd.DataFrame, col: str) -> int:
    return int(sum(_extract_floats(sub, col)))


def _format_mean(sub: pd.DataFrame, col: str, unit: str, decimals: int = 1) -> str:
    vals = _extract_floats(sub, col)
    if not vals:
        return "N/A"
    return f"{sum(vals) / len(vals):.{decimals}f}{unit}"


def _format_max(sub: pd.DataFrame, col: str, unit: str, decimals: int = 1) -> str:
    vals = _extract_floats(sub, col)
    if not vals:
        return "N/A"
    return f"{max(vals):.{decimals}f}{unit}"


def _format_mean_max(sub: pd.DataFrame, col: str, unit: str, decimals: int = 1) -> str:
    vals = _extract_floats(sub, col)
    if not vals:
        return "N/A"
    mean_val = sum(vals) / len(vals)
    max_val = max(vals)
    return f"{mean_val:.{decimals}f}{unit} / {max_val:.{decimals}f}{unit}"


def _count_resource_samples(sub: pd.DataFrame) -> int:
    present = [col for col in RESOURCE_COLUMNS if col in sub.columns]
    if not present or len(sub) == 0:
        return 0
    has_sample = [False] * len(sub)
    for col in present:
        for idx, raw in enumerate(sub[col].tolist()):
            if has_sample[idx] or raw is None or bool(pd.isna(raw)):
                continue
            try:
                float(raw)
                has_sample[idx] = True
            except (TypeError, ValueError):
                continue
    return sum(has_sample)


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
        "Taxa Acerto (%)",
        "Taxa Alucinação (%)",
        "Ferramenta Correta (%)",
        "Concordância Auto/Juiz (%)",
        "Kappa Cohen",
        "Chamadas LLM (Total / Média)",
        "Chamadas Tools (Total / Média)",
        "Chamadas Total (Total / Média)",
        "Latência Média / Mediana / P95 (s)",
        "Tokens Entrada / Saída (Total)",
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
            taxa_acerto_str = f"{(judge_ok / len(judge_pairs)) * 100.0:.1f}%"
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
            taxa_acerto_str = f"{(auto_ok / n) * 100.0:.1f}%" if n else "0.0%"
            agree_str = "N/A"
            kappa_str = "N/A"
            halluc_str = "N/A"

        llm_calls_total = _sum_int_column(sub, "chamadas_modelo")
        llm_calls_avg = llm_calls_total / n if n else 0.0
        tool_calls_total = _sum_int_column(sub, "num_tool_calls")
        tool_calls_avg = tool_calls_total / n if n else 0.0
        total_calls = llm_calls_total + tool_calls_total
        total_calls_avg = total_calls / n if n else 0.0

        lat_vals = _extract_floats(sub, "latencia_segundos", fill_na=0.0)
        lat_mean = float(mean(lat_vals)) if lat_vals else 0.0
        lat_med = float(median(lat_vals)) if lat_vals else 0.0
        lat_p95 = percentile(lat_vals, 95) or 0.0
        tok_in = _sum_int_column(sub, "tokens_entrada")
        tok_out = _sum_int_column(sub, "tokens_saida")
        tok_total = tok_in + tok_out

        t1_rows.append(
            [
                sys_name,
                str(n),
                _format_pct(auto_ok, n),
                judge_str,
                taxa_acerto_str,
                halluc_str,
                f"{(tool_ok / n) * 100.0:.1f}%" if n else "0.0%",
                agree_str,
                kappa_str,
                f"{llm_calls_total} / {llm_calls_avg:.2f}",
                f"{tool_calls_total} / {tool_calls_avg:.2f}",
                f"{total_calls} / {total_calls_avg:.2f}",
                f"{lat_mean:.2f} / {lat_med:.2f} / {lat_p95:.2f}",
                f"{tok_in} / {tok_out}",
                str(tok_total),
            ]
        )

    t1_note = (
        "_Nota: `Taxa Acerto (%)` usa o veredito do Juiz MCP quando disponível "
        "(ou `Acerto Auto` no modo `--skip-judge`); "
        "`Chamadas Total = Chamadas LLM + Chamadas Tools`; P95 com interpolação linear._"
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

    # Tabela 4 — Estatísticas Computacionais por Sistema (CPU, RAM e GPU)
    t4_headers = [
        "Sistema",
        "Amostras Telemetria",
        "CPU Médio (%)",
        "CPU Pico (%)",
        "RAM Pico (MB) (Média / Máx)",
        "GPU Média (%) (Média / Máx)",
        "VRAM GPU Pico (MB) (Média / Máx)",
    ]
    t4_rows: list[list[str]] = []
    for sys_name in systems:
        sub = df.loc[df["sistema"] == sys_name]
        n = int(len(sub))
        samples = _count_resource_samples(sub)
        t4_rows.append(
            [
                sys_name,
                f"{samples}/{n}",
                _format_mean(sub, "cpu_pct_medio", "%", decimals=1),
                _format_max(sub, "cpu_pct_pico", "%", decimals=1),
                _format_mean_max(sub, "ram_mb_pico", " MB", decimals=1),
                _format_mean_max(sub, "gpu_pct_medio", "%", decimals=1),
                _format_mean_max(sub, "gpu_mem_mb_pico", " MB", decimals=1),
            ]
        )

    # Tabela 5 — Acerto por Pergunta × Sistema (análise de erros por pergunta)
    df["_acerto"] = _acerto_series(df)
    perguntas = sorted({int(p) for p in df["pergunta_id"]})
    t5_headers = ["Pergunta", "Categoria", *systems, "Acertos"]
    t5_rows: list[list[str]] = []
    erros_rows: list[dict[str, Any]] = []
    for pid in perguntas:
        sub_q = df.loc[df["pergunta_id"] == pid]
        cat = str(sub_q["categoria"].iloc[0])
        cells: list[str] = []
        ok_q = 0
        n_q = 0
        for sys_name in systems:
            row_sq = sub_q.loc[sub_q["sistema"] == sys_name]
            if row_sq.empty:
                cells.append("—")
                continue
            r = row_sq.iloc[0]
            acerto = r["_acerto"]
            ver = r.get("veredito_detalhado")
            ver_s = "" if ver is None or pd.isna(ver) else str(ver)
            if acerto is None:
                cells.append("?")
                continue
            n_q += 1
            if acerto:
                ok_q += 1
                cells.append("✓")
            else:
                cells.append(f"✗ ({ver_s})" if ver_s else "✗")
                erros_rows.append(
                    {
                        "pergunta_id": pid,
                        "categoria": cat,
                        "sistema": sys_name,
                        "veredito_detalhado": ver_s,
                        "ferramenta_correta": r.get("ferramenta_correta"),
                        "ferramentas_usadas": r.get("ferramentas_usadas"),
                        "erro_execucao": r.get("erro_execucao"),
                        "falhas_automaticas": r.get("falhas_automaticas"),
                        "justificativa_juiz": r.get("justificativa_juiz"),
                    }
                )
        t5_rows.append([f"Q{pid:02d}", cat, *cells, f"{ok_q}/{n_q}"])
    erros_path = out_path / "erros_por_pergunta.csv"
    pd.DataFrame(
        erros_rows,
        columns=[
            "pergunta_id",
            "categoria",
            "sistema",
            "veredito_detalhado",
            "ferramenta_correta",
            "ferramentas_usadas",
            "erro_execucao",
            "falhas_automaticas",
            "justificativa_juiz",
        ],
    ).to_csv(erros_path, index=False)

    # Gráfico — Acerto × Custo
    chart_rows = []
    for sys_name in systems:
        sub = df.loc[df["sistema"] == sys_name]
        vals = [v for v in sub["_acerto"] if v is not None]
        n = int(len(sub))
        tok = _sum_int_column(sub, "tokens_entrada") + _sum_int_column(sub, "tokens_saida")
        lat = _extract_floats(sub, "latencia_segundos", fill_na=0.0)
        chart_rows.append(
            {
                "sistema": sys_name,
                "taxa_acerto_pct": (sum(1 for v in vals if v) / len(vals) * 100.0)
                if vals
                else 0.0,
                "tokens_medios": tok / n if n else 0.0,
                "latencia_media_s": float(mean(lat)) if lat else 0.0,
            }
        )
    chart_df = pd.DataFrame(chart_rows)
    chart_path = plot_accuracy_vs_cost(chart_df, out_path / "acerto_vs_custo.png")
    chart_df.round(3).to_csv(out_path / "acerto_vs_custo.csv", index=False)

    # Revisão manual: preserve cells already filled in a previous audit CSV
    div_path = out_path / "divergencias_auditoria_manual.csv"
    manual_reviews = _load_manual_reviews(div_path)
    df["_manual_bool"] = [
        parse_manual_label(manual_reviews.get((str(s_), int(p_)), ("", ""))[0])
        for s_, p_ in zip(df["sistema"], df["pergunta_id"], strict=True)
    ]

    # Tabela 6 — Verificação Manual dos Vereditos (concordância Manual × Juiz)
    t6_headers = [
        "Sistema",
        "Revisadas",
        "Concordância Manual/Juiz (%)",
        "Kappa Manual/Juiz",
        "Taxa Acerto Pós-Auditoria (%)",
    ]
    t6_rows: list[list[str]] = []
    for sys_name in [*systems, "TOTAL"]:
        sub = df if sys_name == "TOTAL" else df.loc[df["sistema"] == sys_name]
        pairs = [
            (bool(m), bool(j))
            for m, j in zip(sub["_manual_bool"], sub["_juiz_bool"], strict=True)
            if m is not None and j is not None
        ]
        final = [
            m if m is not None else a
            for m, a in zip(sub["_manual_bool"], sub["_acerto"], strict=True)
        ]
        final_vals = [v for v in final if v is not None]
        if pairs:
            agree = sum(1 for m, j in pairs if m == j) / len(pairs) * 100.0
            kappa_m = compute_cohens_kappa([m for m, _ in pairs], [j for _, j in pairs])
            agree_s = f"{agree:.1f}%"
            kappa_s = f"{kappa_m:.4f}" if kappa_m is not None else "N/A"
        else:
            agree_s, kappa_s = "N/A", "N/A"
        final_s = (
            f"{sum(1 for v in final_vals if v) / len(final_vals) * 100.0:.1f}%"
            if final_vals
            else "N/A"
        )
        reviewed = sum(1 for m in sub["_manual_bool"] if m is not None)
        t6_rows.append([sys_name, f"{reviewed}/{len(sub)}", agree_s, kappa_s, final_s])
    t6_note = (
        "_Preencha `revisao_manual` em `divergencias_auditoria_manual.csv` com `correta` ou "
        "`incorreta` e rode o relatório novamente; as revisões já preenchidas são "
        "preservadas. Linhas não revisadas mantêm o veredito do juiz._"
    )

    summary_md = "\n\n".join(
        [
            "# Relatório Consolidado do Benchmark — Property Agents (RN)",
            f"Arquivo avaliado: `{in_path}` ({len(df)} execuções)",
            "## Tabela 1 — Resumo Geral por Sistema/Arquitetura",
            _build_markdown_table(t1_headers, t1_rows),
            t1_note,
            f"## {t2_title}",
            _build_markdown_table(t2_headers, t2_rows),
            "## Tabela 3 — Distribuição de Vereditos Detalhados por Sistema",
            _build_markdown_table(t3_headers, t3_rows),
            "## Tabela 4 — Estatísticas Computacionais por Sistema (CPU, RAM e GPU)",
            _build_markdown_table(t4_headers, t4_rows),
            "## Tabela 5 — Acerto por Pergunta × Sistema",
            _build_markdown_table(t5_headers, t5_rows),
            "_✓ = correta; ✗ (veredito) = incorreta; detalhes em `erros_por_pergunta.csv`._",
            "## Tabela 6 — Verificação Manual dos Vereditos (Manual × Juiz)",
            _build_markdown_table(t6_headers, t6_rows),
            t6_note,
            "## Gráfico — Acerto × Custo",
            f"![Acerto × custo]({chart_path.name})",
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
    df_div["revisao_manual"] = df_div["revisao_manual"].astype("object")
    df_div["nota_manual"] = df_div["nota_manual"].astype("object")
    for idx, row in df_div.iterrows():
        prev = manual_reviews.get((str(row["sistema"]), int(row["pergunta_id"])))
        if prev:
            df_div.at[idx, "revisao_manual"], df_div.at[idx, "nota_manual"] = prev
    df_div.to_csv(div_path, index=False)

    print(summary_md)
    print(
        f"\nArquivos gerados: {summary_path}, {div_path} ({len(df_div)} divergência(s)), "
        f"{erros_path} e {chart_path}."
    )
    return {
        "summary_md": summary_path,
        "divergencias_csv": div_path,
        "erros_csv": erros_path,
        "grafico_png": chart_path,
    }


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
