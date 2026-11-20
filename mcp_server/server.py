#!/usr/bin/env python3
"""FastMCP server ('imoveis-rn') exposing 6 PT-BR real-estate search & valuation tools."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastmcp import FastMCP

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.catalog import (  # noqa: E402
    clean_value,
    limpar_nulos,
    load_catalog,
    normalize_text,
    valuation_status,
)

_limpar_nulos = limpar_nulos
_normalize_text = normalize_text

mcp = FastMCP("imoveis-rn")

SORT_SPECS: dict[str, tuple[list[str], list[bool]]] = {
    "mais_barato": (["price_brl", "area_m2"], [True, False]),
    "maior_desconto": (["discount_pct", "price_brl"], [False, True]),
    "maior_area": (["area_m2", "price_brl"], [False, True]),
    "menor_preco_m2": (["price_per_m2", "price_brl"], [True, True]),
    "menor_p_vp": (["p_vp_mercado", "price_brl"], [True, True]),
}

COMPARE_METRICS: list[tuple[str, str, str]] = [
    ("menor_preco", "price_brl", "min"),
    ("maior_area", "area_m2", "max"),
    ("menor_preco_m2", "price_per_m2", "min"),
    ("maior_desconto", "discount_pct", "max"),
    ("menor_p_vp_mercado", "p_vp_mercado", "min"),
    ("menor_p_vp_caixa", "p_vp_caixa", "min"),
    ("mais_quartos", "bedrooms", "max"),
    ("menor_preco_por_quarto", "price_per_bedroom", "min"),
]

df: pd.DataFrame = load_catalog()

_CITY_NORM = pd.Series([normalize_text(v) for v in df["city"]], index=df.index)
_NEIGHBORHOOD_NORM = pd.Series([normalize_text(v) for v in df["neighborhood"]], index=df.index)
_SOURCE_NORM = pd.Series([normalize_text(v) for v in df["source"]], index=df.index)
_TYPE_NORM = pd.Series([normalize_text(v) for v in df["prop_type"]], index=df.index)
_SEGMENT_NORM = pd.Series([normalize_text(v) for v in df["market_segment"]], index=df.index)
_ID_TO_POS: dict[str, int] = {
    str(pid).strip().lower(): pos for pos, pid in enumerate(df["prop_id"])
}


def _filter_mask(
    preco_maximo: float | None = None,
    tipo_imovel: str | None = None,
    cidade: str | None = None,
    bairro: str | None = None,
    fonte: str | None = None,
    segmento_mercado: str | None = None,
    quartos_minimo: int | None = None,
    aceita_financiamento: bool | None = None,
) -> np.ndarray:
    """Build a boolean mask for filtering ``df`` across all search and aggregate tools."""
    mask = np.ones(len(df), dtype=bool)
    if preco_maximo is not None:
        mask &= df["price_brl"].le(preco_maximo).fillna(False).to_numpy(dtype=bool)
    if tipo_imovel is not None:
        mask &= _TYPE_NORM.eq(normalize_text(tipo_imovel)).to_numpy(dtype=bool)
    if cidade is not None:
        mask &= _CITY_NORM.str.contains(normalize_text(cidade), regex=False).to_numpy(dtype=bool)
    if bairro is not None:
        mask &= (
            _NEIGHBORHOOD_NORM.str.contains(normalize_text(bairro), regex=False)
            .to_numpy(dtype=bool)
        )
    if fonte is not None:
        mask &= _SOURCE_NORM.str.contains(normalize_text(fonte), regex=False).to_numpy(dtype=bool)
    if segmento_mercado is not None:
        mask &= _SEGMENT_NORM.eq(normalize_text(segmento_mercado)).to_numpy(dtype=bool)
    if quartos_minimo is not None:
        mask &= df["bedrooms"].ge(quartos_minimo).fillna(False).to_numpy(dtype=bool)
    if aceita_financiamento is not None:
        mask &= (
            df["accepts_financing"]
            .eq(bool(aceita_financiamento))
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    return mask


def _serialize_search_row(row: pd.Series) -> dict[str, Any]:
    """Serialize one row into the PT-BR summary schema returned by ``buscar_imoveis``."""
    return limpar_nulos(
        {
            "imovel_id": row["prop_id"],
            "fonte": row["source"],
            "segmento_mercado": row["market_segment"],
            "titulo": row["title"],
            "tipo_imovel": row["prop_type"],
            "cidade": row["city"],
            "bairro": row["neighborhood"],
            "preco_brl": row["price_brl"],
            "desconto_pct": row["discount_pct"],
            "area_m2": row["area_m2"],
            "quartos": row["bedrooms"],
            "aceita_financiamento": row["accepts_financing"],
            "status_financiamento": row["financing_status"],
            "p_vp_mercado": row["p_vp_mercado"],
        }
    )


def _serialize_detail_row(row: pd.Series) -> dict[str, Any]:
    """Serialize one row into the full PT-BR detail schema."""
    return limpar_nulos(
        {
            "imovel_id": row["prop_id"],
            "fonte": row["source"],
            "segmento_mercado": row["market_segment"],
            "codigo_externo": row["external_code"],
            "titulo": row["title"],
            "tipo_imovel": row["prop_type"],
            "cidade": row["city"],
            "bairro": row["neighborhood"],
            "preco_brl": row["price_brl"],
            "valor_avaliacao_brl": row["appraisal_value_brl"],
            "desconto_pct": row["discount_pct"],
            "area_m2": row["area_m2"],
            "quartos": row["bedrooms"],
            "banheiros": row["bathrooms"],
            "vagas_garagem": row["garage_spots"],
            "modalidade_venda": row["sale_modality"],
            "aceita_financiamento": row["accepts_financing"],
            "status_financiamento": row["financing_status"],
            "preco_por_m2_brl": row["price_per_m2"],
            "preco_m2_mediano_referencia_brl": row["ref_m2_brl"],
            "nivel_referencia_m2": row["ref_level"],
            "valor_patrimonial_estimado_brl": row["vp_estimado_brl"],
            "p_vp_mercado": row["p_vp_mercado"],
            "p_vp_caixa": row["p_vp_caixa"],
            "economia_desagio_brl": row["economia_brl"],
            "preco_por_quarto_brl": row["price_per_bedroom"],
            "url": row["url"],
        }
    )


def _serialize_investment_row(row: pd.Series) -> dict[str, Any]:
    """Serialize one row into the PT-BR valuation and financing breakdown schema."""
    return limpar_nulos(
        {
            "imovel_id": row["prop_id"],
            "fonte": row["source"],
            "segmento_mercado": row["market_segment"],
            "titulo": row["title"],
            "tipo_imovel": row["prop_type"],
            "cidade": row["city"],
            "bairro": row["neighborhood"],
            "modalidade_venda": row["sale_modality"],
            "aceita_financiamento": row["accepts_financing"],
            "status_financiamento": row["financing_status"],
            "preco_brl": row["price_brl"],
            "valor_avaliacao_brl": row["appraisal_value_brl"],
            "desconto_pct": row["discount_pct"],
            "economia_desagio_brl": row["economia_brl"],
            "p_vp_caixa": row["p_vp_caixa"],
            "status_avaliacao_caixa": valuation_status(row["p_vp_caixa"]),
            "area_m2": row["area_m2"],
            "preco_por_m2_brl": row["price_per_m2"],
            "preco_m2_mediano_referencia_brl": row["ref_m2_brl"],
            "nivel_referencia_m2": row["ref_level"],
            "valor_patrimonial_estimado_brl": row["vp_estimado_brl"],
            "p_vp_mercado": row["p_vp_mercado"],
            "status_avaliacao_mercado": valuation_status(row["p_vp_mercado"]),
            "quartos": row["bedrooms"],
            "preco_por_quarto_brl": row["price_per_bedroom"],
            "entrada_20_pct_brl": row["entrada_20_pct_brl"],
            "itbi_registro_est_brl": row["itbi_registro_est_brl"],
        }
    )


def _select_winners(sub: pd.DataFrame, col: str, mode: str) -> list[str]:
    """Return all ``prop_id``s achieving the min/max on ``col``, ignoring NA values."""
    valid = sub[col].dropna()
    if valid.empty:
        return []
    target = valid.max() if mode == "max" else valid.min()
    mask = sub[col].eq(target).fillna(False).to_numpy(dtype=bool)
    return [str(pid) for pid in sub.loc[mask, "prop_id"].tolist()]


@mcp.tool
def buscar_imoveis(
    preco_maximo: float | None = None,
    tipo_imovel: Literal["casa", "apartamento", "terreno"] | None = None,
    cidade: str | None = None,
    bairro: str | None = None,
    fonte: str | None = None,
    segmento_mercado: Literal["caixa", "imobiliaria"] | None = None,
    quartos_minimo: int | None = None,
    aceita_financiamento: bool | None = None,
    ordenar_por: Literal[
        "mais_barato", "maior_desconto", "maior_area", "menor_preco_m2", "menor_p_vp"
    ] = "mais_barato",
) -> dict[str, Any]:
    """Busca imóveis reais no Rio Grande do Norte aplicando filtros e ordenação.

    Parâmetros de filtro (todos opcionais):
    - ``preco_maximo``: preço máximo em reais (BRL).
    - ``tipo_imovel``: ``"casa"``, ``"apartamento"`` ou ``"terreno"``.
    - ``cidade``: nome ou trecho da cidade (sem acentos/maiúsculas, ex.: "Mossoro").
    - ``bairro``: nome ou trecho do bairro (sem acentos/maiúsculas).
    - ``fonte``: origem/imobiliária do anúncio (ex.: "caixa", "pereseperes").
    - ``segmento_mercado``: ``"caixa"`` ou ``"imobiliaria"``.
    - ``quartos_minimo``: quantidade mínima de quartos.
    - ``aceita_financiamento``: ``True`` para imóveis financiáveis, ``False`` para não.
    - ``ordenar_por``: ``"mais_barato"``, ``"maior_desconto"``, ``"maior_area"``,
      ``"menor_preco_m2"`` ou ``"menor_p_vp"``.

    Retorna ``total_encontrados`` e os 5 primeiros imóveis em ``imoveis``.
    Cite sempre o ``imovel_id`` (ex.: r001) de todo imóvel mencionado ao usuário.
    """
    mask = _filter_mask(
        preco_maximo=preco_maximo,
        tipo_imovel=tipo_imovel,
        cidade=cidade,
        bairro=bairro,
        fonte=fonte,
        segmento_mercado=segmento_mercado,
        quartos_minimo=quartos_minimo,
        aceita_financiamento=aceita_financiamento,
    )
    sort_cols, ascending = SORT_SPECS.get(ordenar_por, SORT_SPECS["mais_barato"])
    filtered = df.loc[mask].sort_values(
        by=sort_cols, ascending=ascending, na_position="last", kind="mergesort"
    )
    return {
        "total_encontrados": int(len(filtered)),
        "imoveis": [_serialize_search_row(row) for _, row in filtered.head(5).iterrows()],
    }


@mcp.tool
def detalhes_imovel(imovel_id: str) -> dict[str, Any]:
    """Retorna todos os detalhes cadastrais e indicadores de valuation de um único imóvel.

    Inclui cidade, bairro, fonte, segmento de mercado, modalidade de venda, status de
    financiamento, área em m², quartos, banheiros, vagas de garagem, avaliação oficial,
    percentual de desconto, referência mediana de R$/m² (e seu nível hierárquico),
    P/VP de mercado, P/VP Caixa, economia/deságio e preço por quarto.
    Cite sempre o ``imovel_id`` na resposta.
    """
    clean_id = imovel_id.strip().lower()
    pos = _ID_TO_POS.get(clean_id)
    if pos is None:
        return {"erro": f"Nenhum imóvel encontrado com o código {imovel_id}."}
    return _serialize_detail_row(df.iloc[pos])


@mcp.tool
def comparar_imoveis(imoveis_ids: list[str]) -> dict[str, Any]:
    """Compara de 2 a 4 imóveis lado a lado e já calcula o vencedor em cada critério.

    Retorna os dados completos dos imóveis encontrados, o dicionário
    ``melhor_em_cada_criterio`` (com listas de ``imovel_id`` vencedores em cada métrica,
    usando P/VP mediano hierárquico e ignorando valores ausentes) e ``nao_encontrados``
    para códigos inexistentes.
    """
    positions: list[int] = []
    seen_pos: set[int] = set()
    found_lower: set[str] = set()
    for raw_id in imoveis_ids:
        clean_id = raw_id.strip().lower()
        pos = _ID_TO_POS.get(clean_id)
        if pos is not None:
            found_lower.add(clean_id)
            if pos not in seen_pos:
                seen_pos.add(pos)
                positions.append(pos)

    if not positions:
        return {"erro": f"Nenhum dos códigos informados existe no estoque: {imoveis_ids}."}

    sub = df.iloc[positions]
    best = {
        label: _select_winners(sub, col, mode)
        for label, col, mode in COMPARE_METRICS
    }
    missing = [raw_id for raw_id in imoveis_ids if raw_id.strip().lower() not in found_lower]
    return {
        "imoveis": [_serialize_detail_row(row) for _, row in sub.iterrows()],
        "melhor_em_cada_criterio": best,
        "nao_encontrados": missing,
    }


@mcp.tool
def avaliar_investimento(imovel_id: str) -> dict[str, Any]:
    """Calcula a análise financeira completa e de financiamento de um único imóvel.

    Retorna preço por m², mediana regional de referência (e seu nível hierárquico),
    Valor Patrimonial estimado, índices P/VP Caixa e P/VP de mercado com classificação
    (``desagio``, ``agio`` ou ``par``), economia absoluta em reais, preço por quarto,
    estimativa de entrada (20%) e custos estimados de ITBI + registro (4,5%).
    """
    clean_id = imovel_id.strip().lower()
    pos = _ID_TO_POS.get(clean_id)
    if pos is None:
        return {"erro": f"Nenhum imóvel encontrado com o código {imovel_id}."}
    return _serialize_investment_row(df.iloc[pos])


@mcp.tool
def simular_compra_financiamento(
    imovel_id: str,
    entrada_pct: float = 20.0,
    prazo_meses: int = 360,
    taxa_juros_anual_pct: float = 10.5,
    sistema: Literal["SAC", "PRICE"] = "SAC",
) -> dict[str, Any]:
    """Simula compra financiada (SAC ou PRICE) ou à vista e custos cartoriais de um imóvel.

    Parâmetros:
    - ``imovel_id``: código do imóvel (ex.: ``"r047"``).
    - ``entrada_pct``: percentual de entrada entre 0 (inclusive) e 100 (exclusive).
    - ``prazo_meses``: prazo do financiamento entre 1 e 420 meses.
    - ``taxa_juros_anual_pct``: taxa de juros nominal anual entre 0 e 35%.
    - ``sistema``: sistema de amortização ``"SAC"`` ou ``"PRICE"``.

    Verifica automaticamente se o imóvel aceita financiamento (``aceita_financiamento``) e
    inclui estimativa de ITBI/registro (4,5%) e comissão de leiloeiro (5% quando aplicável).
    """
    if not (0.0 <= entrada_pct < 100.0):
        return {"erro": "entrada_pct deve estar entre 0 (inclusive) e 100 (exclusive)."}
    if not (1 <= prazo_meses <= 420):
        return {"erro": "prazo_meses deve estar entre 1 e 420 meses."}
    if not (0.0 <= taxa_juros_anual_pct <= 35.0):
        return {"erro": "taxa_juros_anual_pct deve estar entre 0 e 35%."}
    sistema_upper = str(sistema).strip().upper()
    if sistema_upper not in ("SAC", "PRICE"):
        return {"erro": "sistema deve ser 'SAC' ou 'PRICE'."}

    clean_id = imovel_id.strip().lower()
    pos = _ID_TO_POS.get(clean_id)
    if pos is None:
        return {"erro": f"Nenhum imóvel encontrado com o código {imovel_id}."}

    row = df.iloc[pos]
    preco = float(row["price_brl"])
    itbi_registro_est_brl = round(preco * 0.045, 2)

    source_norm = normalize_text(row["source"])
    modality_norm = normalize_text(row["sale_modality"])
    is_auction = source_norm == "caixa" and (
        "leilao" in modality_norm or "licitacao" in modality_norm
    )
    comissao_leiloeiro_brl = round(preco * 0.05, 2) if is_auction else 0.0
    custo_total_a_vista_brl = round(
        preco + itbi_registro_est_brl + comissao_leiloeiro_brl, 2
    )

    accepts_fin = clean_value(row["accepts_financing"])
    status_fin = str(row["financing_status"])

    base_payload: dict[str, Any] = {
        "imovel_id": row["prop_id"],
        "fonte": row["source"],
        "segmento_mercado": row["market_segment"],
        "titulo": row["title"],
        "cidade": row["city"],
        "modalidade_venda": row["sale_modality"],
        "preco_brl": round(preco, 2),
        "itbi_registro_est_brl": itbi_registro_est_brl,
        "comissao_leiloeiro_brl": comissao_leiloeiro_brl,
        "custo_total_a_vista_brl": custo_total_a_vista_brl,
        "aceita_financiamento": accepts_fin,
        "status_financiamento": status_fin,
    }

    if accepts_fin is False:
        return limpar_nulos(
            {
                **base_payload,
                "financiamento_permitido": False,
                "sistema": sistema_upper,
                "entrada_pct": float(entrada_pct),
                "prazo_meses": int(prazo_meses),
                "taxa_juros_anual_pct": float(taxa_juros_anual_pct),
                "valor_entrada_brl": None,
                "valor_financiado_brl": None,
                "desembolso_inicial_total_brl": None,
                "primeira_parcela_brl": None,
                "ultima_parcela_brl": None,
                "total_juros_brl": None,
                "total_pago_financiamento_brl": None,
                "aviso": (
                    "Este imóvel da Caixa NÃO aceita financiamento bancário; "
                    "a compra deve ser feita integralmente com recursos próprios."
                ),
            }
        )

    valor_entrada_brl = round(preco * (entrada_pct / 100.0), 2)
    valor_financiado_brl = round(preco - valor_entrada_brl, 2)
    desembolso_inicial_total_brl = round(
        valor_entrada_brl + itbi_registro_est_brl + comissao_leiloeiro_brl, 2
    )
    taxa_mensal = (taxa_juros_anual_pct / 100.0) / 12.0

    if sistema_upper == "SAC":
        amortizacao = valor_financiado_brl / prazo_meses
        primeira_parcela_brl = round(amortizacao + valor_financiado_brl * taxa_mensal, 2)
        ultima_parcela_brl = round(amortizacao + amortizacao * taxa_mensal, 2)
        total_juros_brl = round(
            valor_financiado_brl * taxa_mensal * (prazo_meses + 1) / 2.0, 2
        )
        total_pago_financiamento_brl = round(valor_financiado_brl + total_juros_brl, 2)
    else:
        if taxa_mensal > 0.0:
            factor = (1.0 + taxa_mensal) ** prazo_meses
            pmt = valor_financiado_brl * (taxa_mensal * factor) / (factor - 1.0)
        else:
            pmt = valor_financiado_brl / prazo_meses
        primeira_parcela_brl = round(pmt, 2)
        ultima_parcela_brl = round(pmt, 2)
        total_pago_financiamento_brl = round(pmt * prazo_meses, 2)
        total_juros_brl = round(total_pago_financiamento_brl - valor_financiado_brl, 2)

    aviso = (
        None
        if accepts_fin is True
        else (
            "Simulação estimada: o anúncio da imobiliária não informa elegibilidade a "
            "financiamento bancário; confirme diretamente com o anunciante."
        )
    )

    return limpar_nulos(
        {
            **base_payload,
            "financiamento_permitido": True,
            "sistema": sistema_upper,
            "entrada_pct": float(entrada_pct),
            "prazo_meses": int(prazo_meses),
            "taxa_juros_anual_pct": float(taxa_juros_anual_pct),
            "valor_entrada_brl": valor_entrada_brl,
            "valor_financiado_brl": valor_financiado_brl,
            "desembolso_inicial_total_brl": desembolso_inicial_total_brl,
            "primeira_parcela_brl": primeira_parcela_brl,
            "ultima_parcela_brl": ultima_parcela_brl,
            "total_juros_brl": total_juros_brl,
            "total_pago_financiamento_brl": total_pago_financiamento_brl,
            "aviso": aviso,
        }
    )


@mcp.tool
def estatisticas_mercado(
    cidade: str | None = None,
    bairro: str | None = None,
    tipo_imovel: Literal["casa", "apartamento", "terreno"] | None = None,
    fonte: str | None = None,
    segmento_mercado: Literal["caixa", "imobiliaria"] | None = None,
    aceita_financiamento: bool | None = None,
) -> dict[str, Any]:
    """Calcula estatísticas agregadas de mercado sobre TODO o estoque filtrado.

    Ao contrário de ``buscar_imoveis`` (limitado aos 5 primeiros resultados), esta ferramenta
    analisa todos os imóveis que atendem aos filtros informados e retorna contagens por
    tipo e segmento, estatísticas de preço, preço por m² (incluindo mediana), área em m²,
    descontos da Caixa e distribuição de elegibilidade a financiamento.
    """
    filtros_aplicados = limpar_nulos(
        {
            "cidade": cidade,
            "bairro": bairro,
            "tipo_imovel": tipo_imovel,
            "fonte": fonte,
            "segmento_mercado": segmento_mercado,
            "aceita_financiamento": aceita_financiamento,
        }
    )
    mask = _filter_mask(
        tipo_imovel=tipo_imovel,
        cidade=cidade,
        bairro=bairro,
        fonte=fonte,
        segmento_mercado=segmento_mercado,
        aceita_financiamento=aceita_financiamento,
    )
    sub = df.loc[mask]
    if sub.empty:
        return {
            "filtros_aplicados": filtros_aplicados,
            "total_imoveis": 0,
            "mensagem": "Nenhum imóvel encontrado para os filtros informados.",
        }

    discounts = sub["discount_pct"].dropna()
    discount_stats = (
        {
            "quantidade_com_desconto": int(len(discounts)),
            "mediana": round(float(discounts.median()), 2),
            "media": round(float(discounts.mean()), 2),
            "maximo": round(float(discounts.max()), 2),
        }
        if not discounts.empty
        else {
            "quantidade_com_desconto": 0,
            "mediana": None,
            "media": None,
            "maximo": None,
        }
    )

    fin_series = sub["accepts_financing"]
    por_tipo = {str(k): int(v) for k, v in sub["prop_type"].value_counts().items()}
    por_segmento = {str(k): int(v) for k, v in sub["market_segment"].value_counts().items()}

    return limpar_nulos(
        {
            "filtros_aplicados": filtros_aplicados,
            "total_imoveis": int(len(sub)),
            "por_tipo": por_tipo,
            "por_segmento": por_segmento,
            "preco_brl": {
                "minimo": round(float(sub["price_brl"].min()), 2),
                "mediana": round(float(sub["price_brl"].median()), 2),
                "media": round(float(sub["price_brl"].mean()), 2),
                "maximo": round(float(sub["price_brl"].max()), 2),
            },
            "preco_por_m2_brl": {
                "minimo": round(float(sub["price_per_m2"].min()), 2),
                "mediana": round(float(sub["price_per_m2"].median()), 2),
                "media": round(float(sub["price_per_m2"].mean()), 2),
                "maximo": round(float(sub["price_per_m2"].max()), 2),
            },
            "area_m2": {
                "minimo": round(float(sub["area_m2"].min()), 2),
                "mediana": round(float(sub["area_m2"].median()), 2),
                "maximo": round(float(sub["area_m2"].max()), 2),
            },
            "desconto_caixa_pct": discount_stats,
            "financiamento": {
                "aceita_sim": int(fin_series.eq(True).fillna(False).sum()),
                "aceita_nao": int(fin_series.eq(False).fillna(False).sum()),
                "nao_informado": int(fin_series.isna().sum()),
            },
        }
    )


@mcp.resource("estoque://esquema")
def catalog_schema() -> str:
    """Retorna a documentação em português do esquema de campos, ferramentas e valuation."""
    return (
        f"# Esquema do Estoque Real de Imóveis (RN) — {len(df)} Registros\n\n"
        "## Ferramentas Disponíveis (6)\n"
        "- `buscar_imoveis`: busca e filtra o estoque (retorna total e top 5).\n"
        "- `detalhes_imovel`: ficha completa de um único `imovel_id`.\n"
        "- `comparar_imoveis`: compara de 2 a 4 imóveis com vencedores por critério.\n"
        "- `avaliar_investimento`: análise de P/VP (Caixa e mercado mediano), entrada e ITBI.\n"
        "- `simular_compra_financiamento`: simula SAC/PRICE ou compra à vista com custos.\n"
        "- `estatisticas_mercado`: agrega min/mediana/média/máx sobre todo o estoque filtrado.\n\n"
        "## Campos Disponíveis (100% em português nas respostas)\n"
        "- `imovel_id` (str): Código identificador único no catálogo (ex.: 'r001').\n"
        "- `fonte` (str) e `segmento_mercado` ('caixa' | 'imobiliaria').\n"
        "- `codigo_externo`, `titulo`, `tipo_imovel` ('casa' | 'apartamento' | 'terreno').\n"
        "- `cidade` e `bairro`: Localização no Rio Grande do Norte.\n"
        "- `preco_brl` (float): Preço anunciado em reais.\n"
        "- `valor_avaliacao_brl` e `desconto_pct` (Caixa; null em imobiliárias).\n"
        "- `area_m2` (float): Área em m² (100% preenchida).\n"
        "- `quartos`, `banheiros`, `vagas_garagem` (int | null; 0 em terrenos).\n"
        "- `modalidade_venda`, `aceita_financiamento` (bool | null) e "
        "`status_financiamento` ('aceita' | 'nao_aceita' | 'nao_informado').\n"
        "- `url`: Link do anúncio original.\n\n"
        "## Atributos NÃO Registrados no Catálogo\n"
        "O catálogo NÃO registra valor de **condomínio**, valor de **IPTU**, "
        "**posição solar** (nascente/poente), presença de **poço artesiano** nem "
        "política de **permuta/troca por veículo**. Nunca invente esses dados.\n\n"
        "## Metodologia de Valuation (Mediana Hierárquica em 4 Níveis, MIN_GROUP_SIZE=5)\n"
        "- `preco_por_m2_brl` = round(preco_brl / area_m2, 2)\n"
        "- `preco_m2_mediano_referencia_brl` (`nivel_referencia_m2`): mediana de R$/m² "
        "do nível mais específico com ao menos 5 imóveis:\n"
        "  1. `bairro_segmento`: (segmento_mercado, cidade, bairro, tipo_imovel)\n"
        "  2. `cidade_segmento`: (segmento_mercado, cidade, tipo_imovel)\n"
        "  3. `cidade_tipo`: (cidade, tipo_imovel)\n"
        "  4. `estado_tipo`: mediana estadual por tipo_imovel\n"
        "- `valor_patrimonial_estimado_brl` = round(area_m2 * preco_m2_mediano_referencia_brl, 2)\n"
        "- `p_vp_mercado` = round(preco_brl / valor_patrimonial_estimado_brl, 2) "
        "(< 1.0 = deságio, > 1.0 = ágio, 1.0 = par)\n"
        "- `p_vp_caixa` = round(preco_brl / valor_avaliacao_brl, 2)\n"
        "- `economia_desagio_brl` = round(valor_avaliacao_brl - preco_brl, 2)\n"
        "- `entrada_20_pct_brl` = round(preco_brl * 0.20, 2) e "
        "`itbi_registro_est_brl` = round(preco_brl * 0.045, 2)\n"
    )


@mcp.prompt
def recomendar_imovel(
    orcamento: int,
    tipo_imovel: str,
    cidade: str | None = None,
) -> str:
    """Gera um prompt em português do Brasil para recomendação de imóveis no estoque."""
    city_clause = f" na cidade de '{cidade}'" if cidade else " no Rio Grande do Norte"
    return (
        f"Você é um consultor imobiliário especialista no Rio Grande do Norte. "
        f"Responda sempre em português do Brasil (PT-BR). Um cliente busca um imóvel "
        f"do tipo '{tipo_imovel}'{city_clause} com orçamento máximo de R$ {orcamento:,.2f}. "
        f"Use `buscar_imoveis` e `estatisticas_mercado` para situar as melhores opções "
        f"reais no estoque, utilize `comparar_imoveis`, `avaliar_investimento` ou "
        f"`simular_compra_financiamento` para comparar preços, área, desconto oficial, "
        f"elegibilidade a financiamento e índice P/VP mediano, cite obrigatoriamente o "
        f"`imovel_id` de cada imóvel mencionado e nunca invente atributos ausentes."
    )


if __name__ == "__main__":
    mcp.run()
