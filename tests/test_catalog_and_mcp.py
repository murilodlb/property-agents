"""Unit tests for dataset invariants, hierarchical valuation, 6 MCP tools, and benchmark."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import mcp_server.server as srv
from src.catalog import load_catalog
from src.questions import PERGUNTAS, SYSTEM_PROMPT_PTBR, checagem_automatica


def test_real_catalog_cleaning_invariants() -> None:
    df = load_catalog()
    assert len(df) >= 790
    assert "condo_fee_brl" not in df.columns
    assert "iptu_brl" not in df.columns
    assert bool(df["area_m2"].gt(0).all())
    assert bool(df["price_per_m2"].notna().all())

    terrenos = df.loc[df["prop_type"] == "terreno"]
    assert (terrenos["bedrooms"] == 0).all()
    assert (terrenos["bathrooms"] == 0).all()
    assert (terrenos["garage_spots"] == 0).all()

    caixa = df.loc[df["source"] == "caixa"]
    agencies = df.loc[df["source"] != "caixa"]
    assert (caixa["market_segment"] == "caixa").all()
    assert (agencies["market_segment"] == "imobiliaria").all()
    assert caixa["accepts_financing"].notna().all()
    assert agencies["accepts_financing"].isna().all()
    assert agencies["appraisal_value_brl"].isna().all()
    assert agencies["discount_pct"].isna().all()
    assert set(df["ref_level"].unique()).issubset(
        {"bairro_segmento", "cidade_segmento", "cidade_tipo", "estado_tipo"}
    )


def test_hierarchical_median_tiers_synthetic(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    # Tier 1: 5 Caixa houses in Natal/Ponta Negra -> price/m2 = [1000, 1100, 1200, 1300, 1400]
    for i, ppm in enumerate([1000, 1100, 1200, 1300, 1400], start=1):
        rows.append(
            {
                "prop_id": f"r{i:03d}",
                "source": "caixa",
                "external_code": str(i),
                "title": f"Caixa T1 {i}",
                "prop_type": "casa",
                "city": "Natal",
                "neighborhood": "Ponta Negra",
                "price_brl": ppm * 100.0,
                "appraisal_value_brl": ppm * 120.0,
                "discount_pct": 16.67,
                "area_m2": 100.0,
                "bedrooms": 3,
                "bathrooms": 2,
                "garage_spots": 1,
                "sale_modality": "Venda Online",
                "accepts_financing": True,
                "url": f"https://example.com/{i}",
            }
        )
    # Tier 2 fallback: 5 Caixa houses in Mossoró across small neighborhoods -> median = 900
    for i, ppm in enumerate([700, 800, 900, 1000, 2000], start=6):
        rows.append(
            {
                "prop_id": f"r{i:03d}",
                "source": "caixa",
                "external_code": str(i),
                "title": f"Caixa T2 {i}",
                "prop_type": "casa",
                "city": "Mossoró",
                "neighborhood": f"Bairro {i}",
                "price_brl": ppm * 100.0,
                "appraisal_value_brl": ppm * 110.0,
                "discount_pct": 9.09,
                "area_m2": 100.0,
                "bedrooms": 2,
                "bathrooms": 1,
                "garage_spots": 1,
                "sale_modality": "Venda Online",
                "accepts_financing": False,
                "url": f"https://example.com/{i}",
            }
        )
    # Tier 3 fallback: 3 Caixa + 2 Agency apartments in Extremoz -> combined count == 5
    for i, (src, ppm) in enumerate(
        [
            ("caixa", 1500),
            ("caixa", 1600),
            ("caixa", 1700),
            ("agencia.com", 2500),
            ("agencia.com", 2700),
        ],
        start=11,
    ):
        rows.append(
            {
                "prop_id": f"r{i:03d}",
                "source": src,
                "external_code": str(i),
                "title": f"Apto T3 {i}",
                "prop_type": "apartamento",
                "city": "Extremoz",
                "neighborhood": f"Nb {i}",
                "price_brl": ppm * 50.0,
                "appraisal_value_brl": (ppm * 60.0) if src == "caixa" else None,
                "discount_pct": 16.67 if src == "caixa" else None,
                "area_m2": 50.0,
                "bedrooms": 2,
                "bathrooms": 1,
                "garage_spots": 1,
                "sale_modality": "Venda Padrão",
                "accepts_financing": True if src == "caixa" else None,
                "url": f"https://example.com/{i}",
            }
        )
    # Tier 4 fallback: single terrain -> statewide median (price/m2 = 500)
    rows.append(
        {
            "prop_id": "r016",
            "source": "agencia.com",
            "external_code": "16",
            "title": "Terreno T4",
            "prop_type": "terreno",
            "city": "Acari",
            "neighborhood": "",
            "price_brl": 100_000.0,
            "appraisal_value_brl": None,
            "discount_pct": None,
            "area_m2": 200.0,
            "bedrooms": None,
            "bathrooms": None,
            "garage_spots": None,
            "sale_modality": "Venda Padrão",
            "accepts_financing": None,
            "url": "https://example.com/16",
        }
    )

    csv_file = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    df = load_catalog(csv_file)

    assert df.loc[df["prop_id"] == "r001", "ref_level"].iloc[0] == "bairro_segmento"
    assert df.loc[df["prop_id"] == "r001", "ref_m2_brl"].iloc[0] == 1200.0
    assert df.loc[df["prop_id"] == "r006", "ref_level"].iloc[0] == "cidade_segmento"
    assert df.loc[df["prop_id"] == "r006", "ref_m2_brl"].iloc[0] == 900.0
    assert df.loc[df["prop_id"] == "r011", "ref_level"].iloc[0] == "cidade_tipo"
    assert df.loc[df["prop_id"] == "r011", "ref_m2_brl"].iloc[0] == 1700.0
    assert df.loc[df["prop_id"] == "r016", "ref_level"].iloc[0] == "estado_tipo"
    assert df.loc[df["prop_id"] == "r016", "ref_m2_brl"].iloc[0] == 500.0
    assert df.loc[df["prop_id"] == "r016", "financing_status"].iloc[0] == "nao_informado"
    assert int(df.loc[df["prop_id"] == "r016", "bathrooms"].iloc[0]) == 0


def test_mcp_tools_registration_and_json_safety() -> None:
    async def _list() -> tuple[list[str], list[str], list[str]]:
        tools = [t.name for t in await srv.mcp.list_tools()]
        resources = [str(r.uri) for r in await srv.mcp.list_resources()]
        prompts = [p.name for p in await srv.mcp.list_prompts()]
        return tools, resources, prompts

    tools, resources, prompts = asyncio.run(_list())
    assert set(tools) == {
        "buscar_imoveis",
        "detalhes_imovel",
        "comparar_imoveis",
        "avaliar_investimento",
        "simular_compra_financiamento",
        "estatisticas_mercado",
    }
    assert "estoque://esquema" in resources
    assert "recomendar_imovel" in prompts

    for payload in [
        srv.buscar_imoveis(cidade="Mossoro", segmento_mercado="caixa"),
        srv.detalhes_imovel("r001"),
        srv.comparar_imoveis(["r001", "r002", "r9404"]),
        srv.avaliar_investimento("r001"),
        srv.estatisticas_mercado(cidade="Natal", tipo_imovel="casa"),
    ]:
        dumped = json.dumps(payload, allow_nan=False, ensure_ascii=False)
        assert "NaN" not in dumped


def test_simular_compra_financiamento_all_cases() -> None:
    df = srv.df
    fin_true_id = str(df.loc[df["accepts_financing"].eq(True).fillna(False), "prop_id"].iloc[0])
    fin_false_id = str(df.loc[df["accepts_financing"].eq(False).fillna(False), "prop_id"].iloc[0])
    fin_na_id = str(df.loc[df["accepts_financing"].isna(), "prop_id"].iloc[0])
    auction_id = str(
        df.loc[
            (df["source"] == "caixa")
            & df["sale_modality"].astype(str).str.contains("Leil", case=False, na=False),
            "prop_id",
        ].iloc[0]
    )

    sac = srv.simular_compra_financiamento(
        fin_true_id, entrada_pct=30.0, prazo_meses=240, taxa_juros_anual_pct=10.5, sistema="SAC"
    )
    assert sac["financiamento_permitido"] is True
    assert sac["status_financiamento"] == "aceita"
    assert sac["primeira_parcela_brl"] > sac["ultima_parcela_brl"]
    assert sac["aviso"] is None
    json.dumps(sac, allow_nan=False)

    price = srv.simular_compra_financiamento(
        fin_true_id, entrada_pct=20.0, prazo_meses=360, taxa_juros_anual_pct=10.5, sistema="PRICE"
    )
    assert price["financiamento_permitido"] is True
    assert price["primeira_parcela_brl"] == price["ultima_parcela_brl"]
    json.dumps(price, allow_nan=False)

    blocked = srv.simular_compra_financiamento(fin_false_id)
    assert blocked["financiamento_permitido"] is False
    assert blocked["status_financiamento"] == "nao_aceita"
    assert blocked["valor_financiado_brl"] is None
    assert blocked["custo_total_a_vista_brl"] > blocked["preco_brl"]
    assert "NÃO aceita financiamento" in blocked["aviso"]
    json.dumps(blocked, allow_nan=False)

    agency = srv.simular_compra_financiamento(fin_na_id)
    assert agency["financiamento_permitido"] is True
    assert agency["status_financiamento"] == "nao_informado"
    assert "imobiliária não informa" in agency["aviso"]
    json.dumps(agency, allow_nan=False)

    auction = srv.simular_compra_financiamento(auction_id)
    assert auction["comissao_leiloeiro_brl"] == round(auction["preco_brl"] * 0.05, 2)

    assert "erro" in srv.simular_compra_financiamento("r9999")
    assert "erro" in srv.simular_compra_financiamento(fin_true_id, entrada_pct=100.0)
    assert "erro" in srv.simular_compra_financiamento(fin_true_id, prazo_meses=0)
    assert "erro" in srv.simular_compra_financiamento(fin_true_id, taxa_juros_anual_pct=40.0)


def test_estatisticas_mercado_and_subprocess_isolation() -> None:
    stats = srv.estatisticas_mercado(cidade="Natal", tipo_imovel="casa")
    assert stats["total_imoveis"] > 5
    assert stats["por_tipo"] == {"casa": stats["total_imoveis"]}
    assert stats["preco_por_m2_brl"]["mediana"] > 0
    json.dumps(stats, allow_nan=False)

    empty = srv.estatisticas_mercado(cidade="CidadeInexistente")
    assert empty["total_imoveis"] == 0
    assert "mensagem" in empty

    # Verify Colab-style subprocess launch from outside repo root
    server_path = Path(srv.__file__).resolve()
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib.util; s=importlib.util.spec_from_file_location('s','{server_path}');"
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print(len(m.df))",
        ],
        cwd="/tmp",
        env={"PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    assert int(proc.stdout.strip()) == len(srv.df)


def test_questions_suite_and_checagem() -> None:
    assert len(PERGUNTAS) == 20
    assert "imovel_id" in SYSTEM_PROMPT_PTBR
    cat_counts = Counter(q["categoria"] for q in PERGUNTAS)
    assert dict(cat_counts) == {
        "busca_com_filtro": 4,
        "consulta_especifica": 3,
        "comparacao": 4,
        "calculo_financeiro": 4,
        "pegadinha": 2,
        "fora_do_escopo": 3,
    }
    for q in PERGUNTAS:
        primary_tool = q["ferramenta"].split("|")[0]
        errs = checagem_automatica(
            q, {"resposta": q["gabarito"], "ferramentas_usadas": [primary_tool]}
        )
        assert not errs, f"Q{q['id']} self-check failed: {errs}"
