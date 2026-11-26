#!/usr/bin/env python3
"""Judge calibration suite (offline mock+real-MCP tests and live calibration runner)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.catalog import normalize_text  # noqa: E402
from src.judge import (  # noqa: E402
    READ_ONLY_MCP_TOOLS,
    JudgeConfig,
    LLMStepResponse,
    LLMToolRequest,
    MCPJudge,
    _build_gemini_contents,
    _load_dotenv,
)
from src.questions import PERGUNTAS, _simulate_sac_values, df  # noqa: E402
from src.schema import ExecutionRecord  # noqa: E402


def _fmt_br(value: float | int) -> str:
    val = float(value)
    if val.is_integer():
        return f"{int(val):,}".replace(",", ".")
    return f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def build_calibration_cases() -> list[dict[str, Any]]:
    """Construct 10 controlled calibration cases dynamically from ``PERGUNTAS`` and ``df``."""
    by_id = {int(q["id"]): q for q in PERGUNTAS}
    city_norm = df["city"].map(normalize_text)
    source_norm = df["source"].map(normalize_text)

    # Case 1 (Q1 positive): cheapest Natal house <= 350k
    q1 = by_id[1]
    r1 = (
        df.loc[
            (df["prop_type"] == "casa")
            & (city_norm == "natal")
            & (df["price_brl"] <= 350_000)
        ]
        .sort_values(["price_brl", "area_m2"], ascending=[True, False], na_position="last")
        .iloc[0]
    )

    # Case 2 (Q12 positive): property with p_vp_mercado < 0.85 and regional median ref
    q12 = by_id[12]
    r12 = (
        df.loc[
            df["p_vp_mercado"].gt(0.1).fillna(False)
            & df["p_vp_mercado"].lt(0.85).fillna(False)
            & df["ref_level"].isin(["bairro_segmento", "cidade_segmento"])
        ]
        .sort_values("prop_id", kind="mergesort")
        .iloc[0]
    )

    # Case 3 (Q13 positive) & Case 6 (Q13 negative): Caixa property with accepts_financing == True
    q13 = by_id[13]
    r13 = (
        df.loc[(source_norm == "caixa") & df["accepts_financing"].eq(True).fillna(False)]
        .sort_values("prop_id", kind="mergesort")
        .iloc[0]
    )
    sim13 = _simulate_sac_values(
        r13.to_dict(), entrada_pct=30.0, prazo_meses=240, taxa_anual_pct=10.5
    )

    # Case 4 (Q14 positive): Natal vs. Parnamirim casa market stats
    q14 = by_id[14]
    natal_casas = df.loc[(df["prop_type"] == "casa") & (city_norm == "natal")]
    parn_casas = df.loc[(df["prop_type"] == "casa") & (city_norm == "parnamirim")]
    natal_med = round(float(natal_casas["price_per_m2"].median()), 2)
    parn_med = round(float(parn_casas["price_per_m2"].median()), 2)

    # Case 5 (Q8 negative): inverted winner on lower price/m²
    q8 = by_id[8]
    q8_ids = re.findall(r"\br\d{3,4}\b", q8["q"], flags=re.IGNORECASE)
    a8 = df.loc[df["prop_id"] == q8_ids[0]].iloc[0]
    b8 = df.loc[df["prop_id"] == q8_ids[1]].iloc[0]
    wrong_ppm_winner = b8 if a8["price_per_m2"] < b8["price_per_m2"] else a8

    # Case 7 (Q16 negative): financing trap recommending non-financeable house
    q16 = by_id[16]
    q16_ids = re.findall(r"\br\d{3,4}\b", q16["q"], flags=re.IGNORECASE)
    trap_rows = df.loc[df["prop_id"].isin(q16_ids)]
    no_fin_row = trap_rows.loc[trap_rows["accepts_financing"].eq(False).fillna(False)].iloc[0]

    # Case 8 (Q17 negative): Caixa ágio trap claiming deságio
    q17 = by_id[17]
    q17_id = re.findall(r"\br\d{3,4}\b", q17["q"], flags=re.IGNORECASE)[0]
    r17 = df.loc[df["prop_id"] == q17_id].iloc[0]

    # Case 9 (Q18 negative): nonexistent ID r9999 hallucination
    q18 = by_id[18]

    # Case 10 (Q20 negative): out-of-scope attributes hallucination
    q20 = by_id[20]
    q20_id = re.findall(r"\br\d{3,4}\b", q20["q"], flags=re.IGNORECASE)[0]

    return [
        {
            "case_id": "cal_01_pos_q1_paraphrase",
            "pergunta": q1,
            "expected_correta": True,
            "expected_veredito": "correta",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=1,
                resposta=(
                    f"A opção mais econômica em Natal até R$ 350 mil é a casa {r1['prop_id']} "
                    f"no bairro {r1['neighborhood']}, anunciada por aproximadamente "
                    f"R$ {round(float(r1['price_brl'])):,}".replace(",", ".")
                    + f" (área de {r1['area_m2']} m²)."
                ),
                ferramentas_usadas=["buscar_imoveis"],
                chamadas_modelo=2,
                latencia_segundos=1.4,
            ),
            "mock_probe": {
                "tool": "buscar_imoveis",
                "args": {"cidade": "Natal", "tipo_imovel": "casa", "preco_maximo": 350000},
            },
        },
        {
            "case_id": "cal_02_pos_q12_pvp_comma",
            "pergunta": q12,
            "expected_correta": True,
            "expected_veredito": "correta",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=12,
                resposta=(
                    f"O imóvel {r12['prop_id']} apresenta preço por m² de R$ "
                    f"{_fmt_br(r12['price_per_m2'])}, Valor Patrimonial estimado de "
                    f"R$ {_fmt_br(r12['vp_estimado_brl'])} e índice P/VP de mercado igual a "
                    f"{str(r12['p_vp_mercado']).replace('.', ',')}. Portanto, está sendo "
                    f"negociado com deságio frente à mediana regional."
                ),
                ferramentas_usadas=["avaliar_investimento"],
                chamadas_modelo=2,
                latencia_segundos=1.2,
            ),
            "mock_probe": {
                "tool": "avaliar_investimento",
                "args": {"imovel_id": str(r12["prop_id"])},
            },
        },
        {
            "case_id": "cal_03_pos_q13_sac_rounded",
            "pergunta": q13,
            "expected_correta": True,
            "expected_veredito": "correta",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=13,
                resposta=(
                    f"Na simulação SAC para o imóvel {r13['prop_id']} com 30% de entrada em "
                    f"240 meses a 10,5% a.a., o valor financiado fica em R$ "
                    f"{_fmt_br(round(sim13['valor_financiado_brl']))}, a primeira parcela "
                    f"sai por R$ {_fmt_br(round(sim13['primeira_parcela_brl'], 2))} e o "
                    f"desembolso inicial total (incluindo ITBI/registro) é de R$ "
                    f"{_fmt_br(round(sim13['desembolso_inicial_total_brl'], 2))}."
                ),
                ferramentas_usadas=["simular_compra_financiamento"],
                chamadas_modelo=2,
                latencia_segundos=1.5,
            ),
            "mock_probe": {
                "tool": "simular_compra_financiamento",
                "args": {
                    "imovel_id": str(r13["prop_id"]),
                    "entrada_pct": 30.0,
                    "prazo_meses": 240,
                    "taxa_juros_anual_pct": 10.5,
                    "sistema": "SAC",
                },
            },
        },
        {
            "case_id": "cal_04_pos_q14_stats",
            "pergunta": q14,
            "expected_correta": True,
            "expected_veredito": "correta",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=14,
                resposta=(
                    f"Natal lidera em volume com {len(natal_casas)} casas cadastradas "
                    f"(mediana de R$ {_fmt_br(natal_med)}/m²), enquanto Parnamirim possui "
                    f"{len(parn_casas)} casas e registra a menor mediana de preço por m² "
                    f"(R$ {_fmt_br(parn_med)}/m²)."
                ),
                ferramentas_usadas=["estatisticas_mercado", "estatisticas_mercado"],
                chamadas_modelo=3,
                latencia_segundos=1.9,
            ),
            "mock_probe": {
                "tool": "estatisticas_mercado",
                "args": {"cidade": "Natal", "tipo_imovel": "casa"},
            },
        },
        {
            "case_id": "cal_05_neg_q8_inverted_ranking",
            "pergunta": q8,
            "expected_correta": False,
            "expected_veredito": "incorreta",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=8,
                resposta=(
                    f"O imóvel {a8['prop_id']} custa R$ {_fmt_br(a8['price_per_m2'])}/m² com "
                    f"{a8['area_m2']} m² e o {b8['prop_id']} custa "
                    f"R$ {_fmt_br(b8['price_per_m2'])}/m² com {b8['area_m2']} m². Logo, o "
                    f"imóvel {wrong_ppm_winner['prop_id']} possui o menor preço por m²."
                ),
                ferramentas_usadas=["comparar_imoveis"],
                chamadas_modelo=2,
                latencia_segundos=1.3,
            ),
            "mock_probe": {
                "tool": "comparar_imoveis",
                "args": {"imoveis_ids": [str(a8["prop_id"]), str(b8["prop_id"])]},
            },
        },
        {
            "case_id": "cal_06_neg_q13_wrong_sac",
            "pergunta": q13,
            "expected_correta": False,
            "expected_veredito": "incorreta",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=13,
                resposta=(
                    f"Para o imóvel {r13['prop_id']}, o valor financiado será de R$ 110.000,00, "
                    "com primeira parcela de R$ 950,00 e desembolso inicial total de R$ 40.000,00."
                ),
                ferramentas_usadas=["simular_compra_financiamento"],
                chamadas_modelo=2,
                latencia_segundos=1.1,
            ),
            "mock_probe": {
                "tool": "simular_compra_financiamento",
                "args": {"imovel_id": str(r13["prop_id"]), "entrada_pct": 30.0, "prazo_meses": 240},
            },
        },
        {
            "case_id": "cal_07_neg_q16_financing_trap",
            "pergunta": q16,
            "expected_correta": False,
            "expected_veredito": "incorreta",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=16,
                resposta=(
                    f"Você deve escolher a casa {no_fin_row['prop_id']} por ser a mais barata "
                    f"(R$ {_fmt_br(no_fin_row['price_brl'])}) e financiar em 300 meses com "
                    "20% de entrada para pagar parcelas menores."
                ),
                ferramentas_usadas=["simular_compra_financiamento"],
                chamadas_modelo=2,
                latencia_segundos=1.4,
            ),
            "mock_probe": {
                "tool": "simular_compra_financiamento",
                "args": {"imovel_id": str(no_fin_row["prop_id"])},
            },
        },
        {
            "case_id": "cal_08_neg_q17_agio_trap",
            "pergunta": q17,
            "expected_correta": False,
            "expected_veredito": "incorreta",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=17,
                resposta=(
                    f"O imóvel {r17['prop_id']} está com ótimo deságio abaixo da avaliação "
                    "oficial da Caixa e atende perfeitamente ao seu critério de compra."
                ),
                ferramentas_usadas=["avaliar_investimento"],
                chamadas_modelo=2,
                latencia_segundos=1.2,
            ),
            "mock_probe": {
                "tool": "avaliar_investimento",
                "args": {"imovel_id": str(r17["prop_id"])},
            },
        },
        {
            "case_id": "cal_09_neg_q18_nonexistent_hallucination",
            "pergunta": q18,
            "expected_correta": False,
            "expected_veredito": "alucinacao",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=18,
                resposta=(
                    "O imóvel r9999 fica no bairro Ponta Negra, está anunciado por "
                    "R$ 320.000,00 e possui P/VP de mercado de 0,92."
                ),
                ferramentas_usadas=["detalhes_imovel"],
                chamadas_modelo=2,
                latencia_segundos=1.0,
            ),
            "mock_probe": {"tool": "detalhes_imovel", "args": {"imovel_id": "r9999"}},
        },
        {
            "case_id": "cal_10_neg_q20_oos_hallucination",
            "pergunta": q20,
            "expected_correta": False,
            "expected_veredito": "alucinacao",
            "execution": ExecutionRecord(
                sistema="calibracao",
                pergunta_id=20,
                resposta=(
                    f"O apartamento {q20_id} tem condomínio de R$ 650,00 mensais, "
                    "IPTU anual de R$ 1.200,00 e posição solar nascente."
                ),
                ferramentas_usadas=["detalhes_imovel"],
                chamadas_modelo=2,
                latencia_segundos=1.0,
            ),
            "mock_probe": {"tool": "detalhes_imovel", "args": {"imovel_id": q20_id}},
        },
    ]


class _CalibrationMockBackend:
    """Deterministic mock backend that always runs a real MCP verification call first."""

    def __init__(self, expected_by_answer: dict[str, dict[str, Any]]) -> None:
        self._cases = expected_by_answer

    async def generate_step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMStepResponse:
        assert "read-only" in system_prompt.lower()
        assert {t["name"] for t in tools} == set(READ_ONLY_MCP_TOOLS)

        user_prompt = str(messages[0]["content"])
        case = next(c for answer_key, c in self._cases.items() if answer_key in user_prompt)

        # Step 1: request one real read-only MCP verification call
        if len(messages) == 1:
            probe = case["mock_probe"]
            return LLMStepResponse(
                content="",
                tool_calls=[
                    LLMToolRequest(
                        name=probe["tool"],
                        arguments=probe["args"],
                        call_id="mock_step_1",
                    )
                ],
            )

        # Step 2: return deterministic JSON verdict referencing the real MCP output
        tool_msg = messages[-1]["content"]
        verdict = {
            "correta": case["expected_correta"],
            "veredito_detalhado": case["expected_veredito"],
            "justificativa": (
                f"Auditoria MCP concluída ({case['case_id']}): "
                f"resultado verificado={json.dumps(tool_msg, ensure_ascii=False)[:120]}"
            ),
        }
        return LLMStepResponse(content=json.dumps(verdict, ensure_ascii=False), tool_calls=[])


def test_calibration_cases_dynamic_structure() -> None:
    cases = build_calibration_cases()
    assert len(cases) == 10
    assert sum(1 for c in cases if c["expected_correta"] is True) == 4
    assert sum(1 for c in cases if c["expected_correta"] is False) == 6
    src_text = Path(__file__).read_text(encoding="utf-8")
    # Ensure no hardcoded real prop_id (allow only fake r9999 / r9404)
    real_like = [m for m in re.findall(r"\br\d{3,4}\b", src_text) if m not in ("r9999", "r9404")]
    assert not real_like, f"Hardcoded IDs found in test file: {real_like}"


def test_anti_qwen_judge_guard() -> None:
    with pytest.raises(ValueError, match="autopreferência"):
        JudgeConfig(provider="ollama", model="qwen3:8b")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = JudgeConfig(provider="ollama", model="qwen3:8b", allow_qwen_judge=True)
        assert cfg.model == "qwen3:8b"
        assert len(caught) == 1


def test_mcp_judge_with_real_server_and_mock_llm() -> None:
    cases = build_calibration_cases()
    by_answer = {c["execution"].resposta: c for c in cases}
    backend = _CalibrationMockBackend(by_answer)
    cfg = JudgeConfig(provider="gemini", model="gemini-4-hydrogen", min_delay_seconds=0.0)

    async def _run() -> None:
        async with MCPJudge(config=cfg, backend=backend) as judge:
            # Verify disallowed/write tool is strictly blocked
            blocked_res, blocked_log = await judge.execute_readonly_tool(
                "deletar_imovel", {"imovel_id": "r9999"}
            )
            assert "erro" in blocked_res and blocked_log.erro is True

            # Verify all 10 calibration cases execute real MCP queries and match expected verdict
            for case in cases:
                verdict = await judge.evaluate(case["pergunta"], case["execution"])
                assert verdict.correta is case["expected_correta"], case["case_id"]
                assert verdict.veredito_detalhado == case["expected_veredito"], case["case_id"]
                assert len(verdict.consultas_realizadas) == 1
                assert verdict.consultas_realizadas[0].erro is False

    asyncio.run(_run())


def test_load_dotenv_and_judge_config_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# Comentário",
                'GEMINI_API_KEY="quoted-secret-key"',
                "JUDGE_MODEL=gemini-2.5-pro # inline comment",
                "export JUDGE_MIN_DELAY_SECONDS='1.5'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for var in ("GEMINI_API_KEY", "JUDGE_MODEL", "JUDGE_MIN_DELAY_SECONDS"):
        monkeypatch.delenv(var, raising=False)

    _load_dotenv(env_file)
    assert os.environ["GEMINI_API_KEY"] == "quoted-secret-key"
    assert os.environ["JUDGE_MODEL"] == "gemini-2.5-pro"
    assert os.environ["JUDGE_MIN_DELAY_SECONDS"] == "1.5"

    # Existing non-empty shell env var must not be overwritten by .env
    monkeypatch.setenv("GEMINI_API_KEY", "explicit-shell-key")
    _load_dotenv(env_file)
    assert os.environ["GEMINI_API_KEY"] == "explicit-shell-key"


def test_gemini_contents_groups_parallel_tool_responses() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Audite a resposta."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "estatisticas_mercado",
                    "arguments": {"cidade": "Natal", "tipo_imovel": "casa"},
                    "call_id": "gemini_call_0",
                },
                {
                    "name": "estatisticas_mercado",
                    "arguments": {"cidade": "Parnamirim", "tipo_imovel": "casa"},
                    "call_id": "gemini_call_1",
                },
            ],
        },
        {
            "role": "tool",
            "name": "estatisticas_mercado",
            "call_id": "gemini_call_0",
            "content": {"total_encontrado": 100},
        },
        {
            "role": "tool",
            "name": "estatisticas_mercado",
            "call_id": "gemini_call_1",
            "content": {"total_encontrado": 50},
        },
    ]

    contents = _build_gemini_contents(messages)
    assert len(contents) == 3
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert len(contents[2].parts) == 2
    assert all(p.function_response is not None for p in contents[2].parts)


@pytest.mark.live_judge
def test_live_judge_calibration() -> None:
    accuracy, mismatches = asyncio.run(run_live_calibration())
    assert accuracy >= 0.8, f"Calibração abaixo de 80% ({accuracy:.1%}): {mismatches}"


async def run_live_calibration(
    provider: str | None = None,
    model: str | None = None,
    delay: float | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Run all 10 calibration cases against the configured live Frontier Judge + MCP server."""
    cfg = JudgeConfig.from_env(provider=provider, model=model, min_delay_seconds=delay)
    cases = build_calibration_cases()
    mismatches: list[dict[str, Any]] = []
    hits = 0

    async with MCPJudge(config=cfg) as judge:
        for case in cases:
            verdict = await judge.evaluate(case["pergunta"], case["execution"])
            ok = verdict.correta is case["expected_correta"]
            if ok:
                hits += 1
            else:
                mismatches.append(
                    {
                        "case_id": case["case_id"],
                        "expected": case["expected_correta"],
                        "got": verdict.correta,
                        "veredito": verdict.veredito_detalhado,
                        "justificativa": verdict.justificativa,
                    }
                )
            print(
                f"[{'OK' if ok else 'FAIL'}] {case['case_id']}: "
                f"esperado={case['expected_correta']} | juiz={verdict.correta} "
                f"({verdict.veredito_detalhado}) | consultas={len(verdict.consultas_realizadas)}"
            )

    accuracy = hits / len(cases)
    print(f"\nAcurácia da calibração: {hits}/{len(cases)} ({accuracy:.1%})")
    if mismatches:
        print("Divergências:", json.dumps(mismatches, ensure_ascii=False, indent=2))
    return accuracy, mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibração do Juiz MCP Read-Only.")
    parser.add_argument("--live", action="store_true", help="Executa contra o LLM ao vivo.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai_compatible", "ollama"],
        default=None,
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--delay", type=float, default=None)
    args = parser.parse_args()

    if not args.live:
        print(
            "Use --live para executar a calibração ao vivo, ou rode "
            "'pytest -m \"not live_judge\"'."
        )
        return 0

    asyncio.run(run_live_calibration(provider=args.provider, model=args.model, delay=args.delay))
    return 0


if __name__ == "__main__":
    sys.exit(main())
