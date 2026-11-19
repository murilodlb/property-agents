#!/usr/bin/env python3
"""FastMCP server ('imoveis-rn') exposing real-estate search and valuation tools in PT-BR."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastmcp import FastMCP

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


def _resolve_csv_path() -> Path:
    """Resolve ``real_properties.csv`` from working directory or repository root."""
    candidates = [
        Path("data/real_properties.csv"),
        Path("real_properties.csv"),
        Path(__file__).resolve().parent.parent / "data" / "real_properties.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not locate data/real_properties.csv or real_properties.csv"
    )


def _normalize_text(value: Any) -> str:
    """Return lowercase ASCII-folded text without accents or surrounding whitespace."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if pd.isna(value):
        return ""
    folded = unicodedata.normalize("NFD", str(value).strip().lower())
    return "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")


def _clean_value(value: Any) -> Any:
    """Convert numpy/pandas scalars and NA/NaN values to native JSON-safe Python types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return _limpar_nulos(value)
    if isinstance(value, (list, tuple)):
        return [_clean_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        num_val = float(value)
        return None if (math.isnan(num_val) or math.isinf(num_val)) else num_val
    if bool(pd.isna(value)):
        return None
    return value


def _limpar_nulos(record: dict[str, Any]) -> dict[str, Any]:
    """Convert any NaN/NA/numpy scalars in ``record`` into native Python types or None."""
    return {str(key): _clean_value(val) for key, val in record.items()}


def load_catalog() -> pd.DataFrame:
    """Load ``real_properties.csv`` and compute deterministic real-estate valuation metrics."""
    catalog = pd.read_csv(_resolve_csv_path())
    for int_col in ("bedrooms", "bathrooms", "garage_spots"):
        catalog[int_col] = catalog[int_col].astype("Int64")
    catalog["accepts_financing"] = catalog["accepts_financing"].astype("boolean")

    area_vals = catalog["area_m2"].to_numpy(dtype=float, na_value=np.nan)
    price_vals = catalog["price_brl"].to_numpy(dtype=float, na_value=np.nan)
    area_pos = catalog["area_m2"].gt(0).fillna(False).to_numpy(dtype=bool)
    catalog["price_per_m2"] = np.round(
        np.divide(price_vals, area_vals, out=np.full_like(price_vals, np.nan), where=area_pos),
        2,
    )
    group_ref_m2 = catalog.groupby(["city", "prop_type"])["price_per_m2"].transform("mean")
    type_ref_m2 = catalog.groupby("prop_type")["price_per_m2"].transform("mean")
    ref_m2_vals = np.where(
        pd.notna(group_ref_m2),
        np.asarray(group_ref_m2, dtype=float),
        np.asarray(type_ref_m2, dtype=float),
    )
    catalog["ref_m2_brl"] = np.round(ref_m2_vals, 2)
    catalog["vp_estimado_brl"] = (catalog["area_m2"] * catalog["ref_m2_brl"]).round(2)

    vp_vals = catalog["vp_estimado_brl"].to_numpy(dtype=float, na_value=np.nan)
    vp_pos = catalog["vp_estimado_brl"].gt(0).fillna(False).to_numpy(dtype=bool)
    catalog["p_vp_mercado"] = np.round(
        np.divide(price_vals, vp_vals, out=np.full_like(price_vals, np.nan), where=vp_pos),
        2,
    )

    appr_vals = catalog["appraisal_value_brl"].to_numpy(dtype=float, na_value=np.nan)
    appr_pos = catalog["appraisal_value_brl"].gt(0).fillna(False).to_numpy(dtype=bool)
    catalog["p_vp_caixa"] = np.round(
        np.divide(price_vals, appr_vals, out=np.full_like(price_vals, np.nan), where=appr_pos),
        2,
    )
    catalog["economia_brl"] = (
        catalog["appraisal_value_brl"] - catalog["price_brl"]
    ).round(2)

    bed_pos = catalog["bedrooms"].gt(0).fillna(False).to_numpy(dtype=bool)
    bedrooms_float = catalog["bedrooms"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    catalog["price_per_bedroom"] = np.round(
        np.divide(price_vals, bedrooms_float, out=np.full_like(price_vals, np.nan), where=bed_pos),
        2,
    )
    catalog["entrada_20_pct_brl"] = (catalog["price_brl"] * 0.20).round(2)
    catalog["itbi_registro_est_brl"] = (catalog["price_brl"] * 0.045).round(2)
    return catalog


df: pd.DataFrame = load_catalog()

_CITY_NORM = pd.Series([_normalize_text(v) for v in df["city"]], index=df.index)
_NEIGHBORHOOD_NORM = pd.Series([_normalize_text(v) for v in df["neighborhood"]], index=df.index)
_SOURCE_NORM = pd.Series([_normalize_text(v) for v in df["source"]], index=df.index)
_TYPE_NORM = pd.Series([_normalize_text(v) for v in df["prop_type"]], index=df.index)
_ID_LOWER = pd.Series([str(v).strip().lower() for v in df["prop_id"]], index=df.index)
_ID_TO_POS: dict[str, int] = {
    str(pid).strip().lower(): pos for pos, pid in enumerate(df["prop_id"])
}


def _valuation_status(p_vp: Any) -> str | None:
    """Classify P/VP ratio as 'desagio', 'agio', 'par', or None."""
    cleaned = _clean_value(p_vp)
    if cleaned is None:
        return None
    if cleaned < 1.0:
        return "desagio"
    if cleaned > 1.0:
        return "agio"
    return "par"


def _serialize_search_row(row: pd.Series) -> dict[str, Any]:
    """Serialize one row into the PT-BR summary schema returned by ``buscar_imoveis``."""
    return _limpar_nulos(
        {
            "imovel_id": row["prop_id"],
            "fonte": row["source"],
            "titulo": row["title"],
            "tipo_imovel": row["prop_type"],
            "cidade": row["city"],
            "bairro": row["neighborhood"],
            "preco_brl": row["price_brl"],
            "desconto_pct": row["discount_pct"],
            "area_m2": row["area_m2"],
            "quartos": row["bedrooms"],
            "aceita_financiamento": row["accepts_financing"],
            "p_vp_mercado": row["p_vp_mercado"],
        }
    )


def _serialize_detail_row(row: pd.Series) -> dict[str, Any]:
    """Serialize one row into the full PT-BR detail schema."""
    return _limpar_nulos(
        {
            "imovel_id": row["prop_id"],
            "fonte": row["source"],
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
            "condominio_brl": row["condo_fee_brl"],
            "iptu_brl": row["iptu_brl"],
            "modalidade_venda": row["sale_modality"],
            "aceita_financiamento": row["accepts_financing"],
            "preco_por_m2_brl": row["price_per_m2"],
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
    return _limpar_nulos(
        {
            "imovel_id": row["prop_id"],
            "fonte": row["source"],
            "titulo": row["title"],
            "tipo_imovel": row["prop_type"],
            "cidade": row["city"],
            "bairro": row["neighborhood"],
            "modalidade_venda": row["sale_modality"],
            "aceita_financiamento": row["accepts_financing"],
            "preco_brl": row["price_brl"],
            "valor_avaliacao_brl": row["appraisal_value_brl"],
            "desconto_pct": row["discount_pct"],
            "economia_desagio_brl": row["economia_brl"],
            "p_vp_caixa": row["p_vp_caixa"],
            "status_avaliacao_caixa": _valuation_status(row["p_vp_caixa"]),
            "area_m2": row["area_m2"],
            "preco_por_m2_brl": row["price_per_m2"],
            "preco_m2_medio_regiao_brl": row["ref_m2_brl"],
            "valor_patrimonial_estimado_brl": row["vp_estimado_brl"],
            "p_vp_mercado": row["p_vp_mercado"],
            "status_avaliacao_mercado": _valuation_status(row["p_vp_mercado"]),
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
    - ``cidade``: nome ou trecho da cidade (busca sem acentos e sem maiúsculas, ex.: "Mossoro").
    - ``bairro``: nome ou trecho do bairro (busca sem acentos e sem maiúsculas).
    - ``fonte``: origem/imobiliária do anúncio (ex.: "caixa", "pereseperes").
    - ``quartos_minimo``: quantidade mínima de quartos.
    - ``aceita_financiamento``: ``True`` para imóveis financiáveis, ``False`` para não financiáveis.
    - ``ordenar_por``: ``"mais_barato"``, ``"maior_desconto"``, ``"maior_area"``,
      ``"menor_preco_m2"`` ou ``"menor_p_vp"``.

    Retorna o total encontrado (``total_encontrados``) e os 5 primeiros imóveis em ``imoveis``.
    Cite sempre o ``imovel_id`` (ex.: r001) de todo imóvel mencionado ao usuário.
    """
    mask = np.ones(len(df), dtype=bool)
    if preco_maximo is not None:
        mask &= df["price_brl"].le(preco_maximo).fillna(False).to_numpy(dtype=bool)
    if tipo_imovel is not None:
        target_type = _normalize_text(tipo_imovel)
        mask &= _TYPE_NORM.eq(target_type).to_numpy(dtype=bool)
    if cidade is not None:
        target_city = _normalize_text(cidade)
        mask &= _CITY_NORM.str.contains(target_city, regex=False).to_numpy(dtype=bool)
    if bairro is not None:
        target_nb = _normalize_text(bairro)
        mask &= _NEIGHBORHOOD_NORM.str.contains(target_nb, regex=False).to_numpy(dtype=bool)
    if fonte is not None:
        target_source = _normalize_text(fonte)
        mask &= _SOURCE_NORM.str.contains(target_source, regex=False).to_numpy(dtype=bool)
    if quartos_minimo is not None:
        mask &= df["bedrooms"].ge(quartos_minimo).fillna(False).to_numpy(dtype=bool)
    if aceita_financiamento is not None:
        mask &= (
            df["accepts_financing"]
            .eq(bool(aceita_financiamento))
            .fillna(False)
            .to_numpy(dtype=bool)
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

    Inclui cidade, bairro, fonte, modalidade de venda, elegibilidade a financiamento,
    área em m², quartos, banheiros, vagas de garagem, avaliação oficial, percentual de
    desconto, P/VP de mercado, P/VP Caixa, economia/deságio e preço por quarto.
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
    já tratando empates e valores ausentes) e ``nao_encontrados`` para códigos inexistentes.
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

    Retorna preço por m², preço médio regional de referência, Valor Patrimonial estimado,
    índices P/VP Caixa e P/VP de mercado com classificação (``desagio``, ``agio`` ou
    ``par``), economia absoluta em reais, preço por quarto, estimativa de entrada (20%)
    e custos estimados de ITBI + registro (4,5%).
    """
    clean_id = imovel_id.strip().lower()
    pos = _ID_TO_POS.get(clean_id)
    if pos is None:
        return {"erro": f"Nenhum imóvel encontrado com o código {imovel_id}."}
    return _serialize_investment_row(df.iloc[pos])


@mcp.resource("estoque://esquema")
def catalog_schema() -> str:
    """Retorna a documentação em português do esquema de campos e fórmulas de valuation."""
    return (
        f"# Esquema do Estoque Real de Imóveis (RN) — {len(df)} Registros\n\n"
        "## Campos Disponíveis (em português nas respostas das ferramentas)\n"
        "- `imovel_id` (str): Código identificador único no catálogo (ex.: 'r001').\n"
        "- `fonte` (str): Origem do anúncio ('caixa' ou domínio da imobiliária local).\n"
        "- `codigo_externo` (str): Código original na Caixa ou imobiliária.\n"
        "- `titulo` (str): Título ou endereço resumido do anúncio.\n"
        "- `tipo_imovel` (str): 'casa', 'apartamento' ou 'terreno'.\n"
        "- `cidade` (str) e `bairro` (str): Localização no Rio Grande do Norte.\n"
        "- `preco_brl` (float): Preço anunciado em reais (BRL).\n"
        "- `valor_avaliacao_brl` (float | null): Avaliação oficial (disponível na Caixa).\n"
        "- `desconto_pct` (float | null): Percentual de desconto oficial (Caixa).\n"
        "- `area_m2` (float | null): Área em m².\n"
        "- `quartos`, `banheiros`, `vagas_garagem` (int | null): Cômodos e vagas.\n"
        "- `condominio_brl`, `iptu_brl` (float | null): Taxas quando informadas.\n"
        "- `modalidade_venda` (str): Modalidade (ex.: 'Venda Online', 'Venda Padrão').\n"
        "- `aceita_financiamento` (bool | null): Elegibilidade a financiamento.\n"
        "- `url` (str): Link do anúncio original.\n\n"
        "## Atributos NÃO Registrados no Catálogo\n"
        "O catálogo NÃO possui informações sobre posição solar (nascente/poente), "
        "poço artesiano, política de permuta ou aceitação de veículo usado como parte "
        "do pagamento. Nunca invente esses dados.\n\n"
        "## Fórmulas Determinísticas de Valuation\n"
        "- `preco_por_m2_brl` = round(preco_brl / area_m2, 2) (se area_m2 > 0)\n"
        "- `preco_m2_medio_regiao_brl` = média de preço/m² por (cidade, tipo_imovel) "
        "(com fallback para média estadual de tipo_imovel)\n"
        "- `valor_patrimonial_estimado_brl` = round(area_m2 * preco_m2_medio_regiao_brl, 2)\n"
        "- `p_vp_mercado` = round(preco_brl / valor_patrimonial_estimado_brl, 2) "
        "(< 1.0 = deságio, > 1.0 = ágio, 1.0 = par)\n"
        "- `p_vp_caixa` = round(preco_brl / valor_avaliacao_brl, 2) "
        "(se valor_avaliacao_brl > 0)\n"
        "- `economia_desagio_brl` = round(valor_avaliacao_brl - preco_brl, 2) "
        "(positivo = deságio/economia, negativo = ágio)\n"
        "- `preco_por_quarto_brl` = round(preco_brl / quartos, 2) (se quartos > 0)\n"
        "- `entrada_20_pct_brl` = round(preco_brl * 0.20, 2)\n"
        "- `itbi_registro_est_brl` = round(preco_brl * 0.045, 2)\n"
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
        f"Use `buscar_imoveis` para encontrar as melhores opções reais no estoque, "
        f"utilize `comparar_imoveis` ou `avaliar_investimento` para comparar preços, "
        f"área, desconto oficial, aceitação de financiamento e índice P/VP, cite "
        f"obrigatoriamente o `imovel_id` de cada imóvel mencionado e nunca invente "
        f"atributos ausentes."
    )


if __name__ == "__main__":
    mcp.run()
