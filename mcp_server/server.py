#!/usr/bin/env python3
"""FastMCP server ('imoveis-rn') exposing real-estate search and valuation tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("imoveis-rn")

BASE_COLUMNS: list[str] = [
    "prop_id",
    "builder",
    "title",
    "prop_type",
    "year_delivery",
    "dist_beach_km",
    "price_brl",
    "documentation",
    "water_source",
    "yield_rate_pct",
    "monthly_fee_brl",
    "area_m2",
    "sun_position",
    "city",
]

SEARCH_COLUMNS: list[str] = [
    "prop_id",
    "builder",
    "title",
    "prop_type",
    "city",
    "year_delivery",
    "dist_beach_km",
    "price_brl",
    "documentation",
    "yield_rate_pct",
    "p_vp",
]

DETAILS_COLUMNS: list[str] = [
    *BASE_COLUMNS,
    "price_per_m2",
    "vp_estimado_brl",
    "p_vp",
    "noi_annual_brl",
    "net_cap_rate_pct",
    "payback_years",
]

COMPARE_COLUMNS: list[str] = [
    "prop_id",
    "builder",
    "title",
    "prop_type",
    "city",
    "year_delivery",
    "dist_beach_km",
    "price_brl",
    "documentation",
    "area_m2",
    "monthly_fee_brl",
    "yield_rate_pct",
    "price_per_m2",
    "p_vp",
    "noi_annual_brl",
    "net_cap_rate_pct",
    "payback_years",
]

SORT_SPECS: dict[str, tuple[list[str], list[bool]]] = {
    "newest": (["year_delivery", "dist_beach_km"], [False, True]),
    "yield": (["yield_rate_pct", "year_delivery", "dist_beach_km"], [False, False, True]),
    "cheapest": (["price_brl", "year_delivery"], [True, False]),
    "lowest_p_vp": (["p_vp", "price_brl"], [True, True]),
}

COMPARE_METRICS: list[tuple[str, str, str]] = [
    ("largest_area", "area_m2", "max"),
    ("cheapest_price", "price_brl", "min"),
    ("lowest_monthly_fee", "monthly_fee_brl", "min"),
    ("highest_gross_yield", "yield_rate_pct", "max"),
    ("highest_net_cap_rate", "net_cap_rate_pct", "max"),
    ("Highest_NOI", "noi_annual_brl", "max"),
    ("lowest_p_vp", "p_vp", "min"),
    ("shortest_payback", "payback_years", "min"),
    ("closest_to_beach", "dist_beach_km", "min"),
    ("newest", "year_delivery", "max"),
]


def _resolve_csv_path() -> Path:
    """Resolve ``properties.csv`` from the working directory or repository root."""
    candidates = [
        Path("data/properties.csv"),
        Path("properties.csv"),
        Path(__file__).resolve().parent.parent / "data" / "properties.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not locate data/properties.csv or properties.csv"
    )


def load_catalog() -> pd.DataFrame:
    """Load ``properties.csv`` and precompute the 9 deterministic valuation columns."""
    catalog = pd.read_csv(_resolve_csv_path())
    catalog["price_per_m2"] = (catalog["price_brl"] / catalog["area_m2"]).round(2)
    group_ref_m2 = catalog.groupby(["city", "prop_type"])["price_per_m2"].transform("mean")
    catalog["vp_estimado_brl"] = (catalog["area_m2"] * group_ref_m2).round(2)
    catalog["p_vp"] = (catalog["price_brl"] / catalog["vp_estimado_brl"]).round(2)
    catalog["gross_annual_brl"] = (
        catalog["price_brl"] * (catalog["yield_rate_pct"] / 100.0)
    ).round(2)
    catalog["annual_fee_brl"] = catalog["monthly_fee_brl"] * 12
    catalog["noi_annual_brl"] = (
        catalog["gross_annual_brl"] - catalog["annual_fee_brl"]
    ).round(2)
    catalog["net_cap_rate_pct"] = (
        (catalog["noi_annual_brl"] / catalog["price_brl"]) * 100.0
    ).round(2)
    catalog["payback_years"] = (
        catalog["price_brl"] / catalog["noi_annual_brl"]
    ).round(2)
    return catalog


df: pd.DataFrame = load_catalog()


@mcp.tool
def search_properties(
    max_price: float | None = None,
    prop_type: Literal["casa", "apartamento", "terreno"] | None = None,
    city: str | None = None,
    documentation: Literal["escriturado", "posse"] | None = None,
    sort_by: Literal["newest", "yield", "cheapest", "lowest_p_vp"] = "newest",
) -> dict[str, Any]:
    """Filter real-estate listings and return the top 5 results sorted by ``sort_by``."""
    r = df
    if max_price is not None:
        r = r[r["price_brl"] <= max_price]
    if prop_type is not None:
        r = r[r["prop_type"].str.lower() == prop_type.strip().lower()]
    if city is not None:
        r = r[r["city"].str.lower() == city.strip().lower()]
    if documentation is not None:
        r = r[r["documentation"].str.lower() == documentation.strip().lower()]

    sort_cols, ascending = SORT_SPECS.get(sort_by, SORT_SPECS["newest"])
    r = r.sort_values(by=sort_cols, ascending=ascending)
    return {
        "total_found": len(r),
        "properties": r[SEARCH_COLUMNS].head(5).to_dict("records"),
    }


@mcp.tool
def property_details(prop_id: str) -> dict[str, Any]:
    """Return full catalog and valuation details for a single ``prop_id``."""
    clean_id = prop_id.strip().lower()
    matches = df[df["prop_id"].str.lower() == clean_id]
    if matches.empty:
        return {"error": f"no property with id {prop_id}"}
    return matches[DETAILS_COLUMNS].iloc[0].to_dict()


def _select_winner(sub: pd.DataFrame, col: str, mode: str) -> str | list[str]:
    """Return the winning ``prop_id`` (or list of tied ``prop_id``s) for a metric."""
    target = sub[col].max() if mode == "max" else sub[col].min()
    winners = sub.loc[sub[col] == target, "prop_id"].tolist()
    return winners[0] if len(winners) == 1 else winners


@mcp.tool
def compare_properties(prop_ids: list[str]) -> dict[str, Any]:
    """Compare 2 to 4 properties side by side and compute category winners."""
    r = df[df["prop_id"].isin(prop_ids)]
    if r.empty:
        return {"error": f"none of these ids exist: {prop_ids}"}

    best = {
        label: _select_winner(r, col, mode)
        for label, col, mode in COMPARE_METRICS
    }
    found_ids = set(r.prop_id)
    missing = [i for i in prop_ids if i not in found_ids]
    return {
        "properties": r[COMPARE_COLUMNS].to_dict("records"),
        "best": best,
        "missing": missing,
    }


@mcp.tool
def evaluate_investment(prop_id: str) -> dict[str, Any]:
    """Compute a complete financial and valuation breakdown for ``prop_id``."""
    clean_id = prop_id.strip().lower()
    matches = df[df["prop_id"].str.lower() == clean_id]
    if matches.empty:
        return {"error": f"no property with id {prop_id}"}

    row = matches.iloc[0]
    p_vp = float(row["p_vp"])
    if p_vp < 1.0:
        valuation_status = "desagio"
    elif p_vp > 1.0:
        valuation_status = "agio"
    else:
        valuation_status = "justo"

    return {
        "prop_id": str(row["prop_id"]),
        "title": str(row["title"]),
        "city": str(row["city"]),
        "prop_type": str(row["prop_type"]),
        "documentation": str(row["documentation"]),
        "financing_eligible": bool(row["documentation"] == "escriturado"),
        "price_brl": int(row["price_brl"]),
        "area_m2": int(row["area_m2"]),
        "price_per_m2": float(row["price_per_m2"]),
        "vp_estimado_brl": float(row["vp_estimado_brl"]),
        "p_vp": p_vp,
        "valuation_status": valuation_status,
        "yield_rate_pct": float(row["yield_rate_pct"]),
        "gross_annual_brl": float(row["gross_annual_brl"]),
        "monthly_fee_brl": int(row["monthly_fee_brl"]),
        "annual_fee_brl": int(row["annual_fee_brl"]),
        "noi_annual_brl": float(row["noi_annual_brl"]),
        "net_cap_rate_pct": float(row["net_cap_rate_pct"]),
        "payback_years": float(row["payback_years"]),
    }


@mcp.resource("estoque://schema")
def estoque_schema() -> str:
    """Return the property catalog column definitions and valuation metric formulas."""
    return (
        "# Schema do Estoque de Imóveis (RN) — 360 Registros\n\n"
        "## Colunas Base (14)\n"
        "- prop_id (str): Identificador único ('p001' a 'p360')\n"
        "- builder (str): Construtora ('Moura Dubeux', 'MRV', 'Alliance', 'Capuche', "
        "'Ecomax', 'Direcional', 'Cyrela', 'Particular')\n"
        "- title (str): Nome do empreendimento ou loteamento\n"
        "- prop_type (str): Tipo do imóvel ('casa', 'apartamento', 'terreno')\n"
        "- year_delivery (int): Ano de entrega (2015 a 2026)\n"
        "- dist_beach_km (float): Distância até a praia em km (0.4 a 38.0)\n"
        "- price_brl (int): Preço em R$ (múltiplos de 500 ou 1000)\n"
        "- documentation (str): Situação documental ('escriturado' = elegível a "
        "financiamento bancário; 'posse' = não financiável)\n"
        "- water_source (str): Fonte de água ('caern', 'poco_artesiano', 'ambos')\n"
        "- yield_rate_pct (float): Retorno bruto anual estimado em % (4.5 a 11.5)\n"
        "- monthly_fee_brl (int): Taxa mensal fixa de condomínio + IPTU em R$ (60 a 1600)\n"
        "- area_m2 (int): Área em m²\n"
        "- sun_position (str): Posição solar ('nascente', 'poente', 'norte', 'sul')\n"
        "- city (str): Município no RN ('Natal', 'Parnamirim', 'São José de Mipibu', "
        "'Nísia Floresta', 'Macaíba', 'Extremoz', 'Mossoró')\n\n"
        "## Métricas de Valuation Pré-calculadas (9)\n"
        "- price_per_m2 = round(price_brl / area_m2, 2)\n"
        "- group_ref_m2 = média de price_per_m2 agrupada por (city, prop_type)\n"
        "- vp_estimado_brl = round(area_m2 * group_ref_m2, 2)\n"
        "- p_vp = round(price_brl / vp_estimado_brl, 2) "
        "(< 1.0 = deságio, > 1.0 = ágio, 1.0 = justo)\n"
        "- gross_annual_brl = round(price_brl * (yield_rate_pct / 100.0), 2)\n"
        "- annual_fee_brl = monthly_fee_brl * 12\n"
        "- noi_annual_brl = round(gross_annual_brl - annual_fee_brl, 2)\n"
        "- net_cap_rate_pct = round((noi_annual_brl / price_brl) * 100.0, 2)\n"
        "- payback_years = round(price_brl / noi_annual_brl, 2)\n"
    )


@mcp.prompt
def recomendar_imovel(orcamento: int, tipo: str) -> str:
    """Generate a Portuguese recommendation prompt for a budget and property type."""
    return (
        f"Você é um consultor imobiliário especialista no Rio Grande do Norte. "
        f"Um cliente busca um imóvel do tipo '{tipo}' com orçamento máximo de "
        f"R$ {orcamento:,.2f}. Utilize a ferramenta `search_properties` para "
        f"encontrar as melhores opções dentro desse limite, compare os imóveis "
        f"mais promissores com `compare_properties` ou `evaluate_investment`, "
        f"verifique a documentação ('escriturado' vs 'posse') e recomende a "
        f"opção mais vantajosa justificando com dados de preço, distância da "
        f"praia, yield líquido (Cap Rate) e P/VP."
    )


if __name__ == "__main__":
    mcp.run()
