"""Tests for the analysis artifacts of ``src.report`` required by the assignment.

Covers: latency mean/median/P95, input/output tokens, per-question error table,
accuracy × cost chart, and manual verdict review (agreement with the judge, kappa,
post-audit accuracy and preservation of filled reviews across report re-runs).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.report import generate_report, parse_manual_label, percentile
from src.schema import BenchmarkRow

SYSTEMS = ("langgraph_unico", "langgraph_verificador")


def _row(
    sistema: str,
    pid: int,
    *,
    juiz: bool | None,
    auto: bool = True,
    veredito: str | None = None,
    lat: float = 1.0,
    tok_in: int = 100,
    tok_out: int = 10,
    categoria: str = "busca_com_filtro",
) -> BenchmarkRow:
    return BenchmarkRow(
        sistema=sistema,
        pergunta_id=pid,
        categoria=categoria,
        ferramenta_esperada="buscar_imoveis",
        ferramentas_usadas='["buscar_imoveis"]',
        ferramenta_correta=True,
        num_tool_calls=1,
        chamadas_modelo=2,
        latencia_segundos=lat,
        tokens_entrada=tok_in,
        tokens_saida=tok_out,
        acerto_automatico=auto,
        falhas_automaticas="" if auto else "faltou r001",
        acerto_juiz=juiz,
        veredito_detalhado=veredito
        if veredito is not None
        else (None if juiz is None else ("correta" if juiz else "incorreta")),
        justificativa_juiz=None if juiz is None else "ok",
        resposta_agente=f"resposta {sistema} {pid}",
    )


def _write_csv(rows: list[BenchmarkRow], path: Path) -> Path:
    pd.DataFrame([r.model_dump(mode="python") for r in rows]).to_csv(path, index=False)
    return path


@pytest.fixture()
def bench_csv(tmp_path: Path) -> Path:
    rows: list[BenchmarkRow] = []
    # unico: latencies 1..20 s, all correct except Q3 (incorreta) and Q18 (alucinacao)
    for pid in range(1, 21):
        juiz = pid not in (3, 18)
        veredito = "alucinacao" if pid == 18 else None
        rows.append(_row("langgraph_unico", pid, juiz=juiz, veredito=veredito, lat=float(pid)))
    # verificador: 3x the tokens, 10 s each, wrong on Q3 only
    for pid in range(1, 21):
        rows.append(
            _row("langgraph_verificador", pid, juiz=pid != 3, lat=10.0, tok_in=300, tok_out=30)
        )
    return _write_csv(rows, tmp_path / "bench.csv")


def test_percentile_matches_numpy_linear() -> None:
    vals = [float(v) for v in range(1, 21)]
    assert percentile(vals, 95) == pytest.approx(19.05)
    assert percentile(vals, 50) == pytest.approx(10.5)
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("correta", True),
        (" Sim ", True),
        ("1", True),
        ("incorreta", False),
        ("parcial", False),
        ("Alucinação", False),
        ("não", False),
        ("", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_parse_manual_label(raw: object, expected: bool | None) -> None:
    assert parse_manual_label(raw) is expected


def test_parse_manual_label_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="revisao_manual inválida"):
        parse_manual_label("talvez")


def test_latency_mean_median_p95_and_tokens(bench_csv: Path, tmp_path: Path) -> None:
    art = generate_report(bench_csv, tmp_path / "out")
    md = art["summary_md"].read_text(encoding="utf-8")
    assert "Latência Média / Mediana / P95 (s)" in md
    assert "Tokens Entrada / Saída (Total)" in md
    # unico: mean 10.5, median 10.5, p95 19.05; tokens 2000 in / 200 out
    assert "| 10.50 / 10.50 / 19.05 | 2000 / 200 | 2200 |" in md
    # verificador: constant 10 s; 6000 in / 600 out
    assert "| 10.00 / 10.00 / 10.00 | 6000 / 600 | 6600 |" in md


def test_per_question_table_and_errors_csv(bench_csv: Path, tmp_path: Path) -> None:
    art = generate_report(bench_csv, tmp_path / "out")
    md = art["summary_md"].read_text(encoding="utf-8")
    assert "## Tabela 5 — Acerto por Pergunta × Sistema" in md
    assert "| Q03 | busca_com_filtro | ✗ (incorreta) | ✗ (incorreta) | 0/2 |" in md
    assert "| Q18 | busca_com_filtro | ✗ (alucinacao) | ✓ | 1/2 |" in md
    assert "| Q01 | busca_com_filtro | ✓ | ✓ | 2/2 |" in md

    erros = pd.read_csv(art["erros_csv"])
    assert len(erros) == 3
    assert set(zip(erros["sistema"], erros["pergunta_id"], strict=True)) == {
        ("langgraph_unico", 3),
        ("langgraph_unico", 18),
        ("langgraph_verificador", 3),
    }
    assert {"veredito_detalhado", "justificativa_juiz", "ferramentas_usadas"} <= set(erros.columns)


def test_accuracy_vs_cost_chart(bench_csv: Path, tmp_path: Path) -> None:
    art = generate_report(bench_csv, tmp_path / "out")
    png = art["grafico_png"]
    assert png.is_file() and png.stat().st_size > 5_000
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    pts = pd.read_csv(png.with_suffix(".csv")).set_index("sistema")
    assert pts.loc["langgraph_unico", "taxa_acerto_pct"] == pytest.approx(90.0)
    assert pts.loc["langgraph_verificador", "taxa_acerto_pct"] == pytest.approx(95.0)
    assert pts.loc["langgraph_unico", "tokens_medios"] == pytest.approx(110.0)
    assert pts.loc["langgraph_verificador", "tokens_medios"] == pytest.approx(330.0)
    assert pts.loc["langgraph_unico", "latencia_media_s"] == pytest.approx(10.5)
    assert "![Acerto × custo](acerto_vs_custo.png)" in art["summary_md"].read_text("utf-8")


def test_chart_falls_back_to_automatic_check_without_judge(tmp_path: Path) -> None:
    rows = [
        _row("langgraph_unico", 1, juiz=None, auto=True),
        _row("langgraph_unico", 2, juiz=None, auto=False),
    ]
    art = generate_report(_write_csv(rows, tmp_path / "b.csv"), tmp_path / "out")
    pts = pd.read_csv(art["grafico_png"].with_suffix(".csv"))
    assert pts["taxa_acerto_pct"].tolist() == [50.0]
    md = art["summary_md"].read_text(encoding="utf-8")
    assert "| Q02 | busca_com_filtro | ✗ |" in md
    assert "| langgraph_unico | 0/2 | N/A | N/A | 50.0% |" in md


def test_manual_review_agreement_and_preservation(bench_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    art = generate_report(bench_csv, out)
    md = art["summary_md"].read_text(encoding="utf-8")
    assert "## Tabela 6 — Verificação Manual dos Vereditos (Manual × Juiz)" in md
    assert "| TOTAL | 0/40 | N/A | N/A | 92.5% |" in md

    div = pd.read_csv(
        art["divergencias_csv"], dtype={"revisao_manual": "string", "nota_manual": "string"}
    )
    assert len(div) == 3  # the three judge-incorrect rows
    reviews = {
        ("langgraph_unico", 3): ("incorreta", "concordo com o juiz"),
        ("langgraph_unico", 18): ("correta", "juiz errou: resposta cita r9999 inexistente"),
        ("langgraph_verificador", 3): ("incorreta", ""),
    }
    for idx, row in div.iterrows():
        rev, nota = reviews[(row["sistema"], int(row["pergunta_id"]))]
        div.at[idx, "revisao_manual"] = rev
        div.at[idx, "nota_manual"] = nota
    div.to_csv(art["divergencias_csv"], index=False)

    art2 = generate_report(bench_csv, out)  # re-run must keep the manual cells
    md2 = art2["summary_md"].read_text(encoding="utf-8")
    # unico: 2 reviewed, 1 agrees with judge -> 50%; Q18 flipped to correct -> 95%
    assert "| langgraph_unico | 2/20 | 50.0% |" in md2
    assert "| 95.0% |" in md2
    # verificador: 1 reviewed, agrees -> 100%; accuracy unchanged at 95%
    assert "| langgraph_verificador | 1/20 | 100.0% |" in md2
    # total: 3 reviewed, 2 agree; post-audit accuracy 38/40 = 95%
    assert "| TOTAL | 3/40 | 66.7% |" in md2
    assert md2.rstrip().count("95.0%") >= 3

    div2 = pd.read_csv(art2["divergencias_csv"], dtype={"nota_manual": "string"})
    kept = {
        (r.sistema, int(r.pergunta_id)): (r.revisao_manual, r.nota_manual)
        for r in div2.itertuples()
    }
    assert kept[("langgraph_unico", 18)] == (
        "correta",
        "juiz errou: resposta cita r9999 inexistente",
    )
    assert kept[("langgraph_unico", 3)][0] == "incorreta"


def test_manual_review_invalid_label_fails_loudly(bench_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    art = generate_report(bench_csv, out)
    div = pd.read_csv(art["divergencias_csv"], dtype={"revisao_manual": "string"})
    div.loc[0, "revisao_manual"] = "acho que sim"
    div.to_csv(art["divergencias_csv"], index=False)
    with pytest.raises(ValueError, match="revisao_manual inválida"):
        generate_report(bench_csv, out)
