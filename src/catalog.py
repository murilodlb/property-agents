#!/usr/bin/env python3
"""Shared catalog loader, text normalization, JSON null-sanitizer, and valuation math."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIN_GROUP_SIZE = 5


def resolve_csv_path(csv_path: str | Path | None = None) -> Path:
    """Resolve ``real_properties.csv`` from an explicit path, CWD, or repo root."""
    if csv_path is not None:
        explicit = Path(csv_path)
        if explicit.is_file():
            return explicit
        raise FileNotFoundError(f"Could not locate CSV at {explicit}")

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


def normalize_text(value: Any) -> str:
    """Return lowercase ASCII-folded text without accents or surrounding whitespace."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if bool(pd.isna(value)):
        return ""
    folded = unicodedata.normalize("NFD", str(value).strip().lower())
    return "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")


def clean_value(value: Any) -> Any:
    """Convert numpy/pandas scalars and NA/NaN values to native JSON-safe Python types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return limpar_nulos(value)
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
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


def limpar_nulos(record: dict[str, Any]) -> dict[str, Any]:
    """Convert any NaN/NA/numpy scalars in ``record`` into native Python types or None."""
    return {str(key): clean_value(val) for key, val in record.items()}


def valuation_status(p_vp: Any) -> str | None:
    """Classify a P/VP ratio as ``'desagio'``, ``'agio'``, ``'par'``, or ``None``."""
    cleaned = clean_value(p_vp)
    if cleaned is None:
        return None
    if cleaned < 1.0:
        return "desagio"
    if cleaned > 1.0:
        return "agio"
    return "par"


def financing_status_label(value: Any) -> str:
    """Map nullable ``accepts_financing`` into ``aceita`` / ``nao_aceita`` / ``nao_informado``."""
    cleaned = clean_value(value)
    if cleaned is True:
        return "aceita"
    if cleaned is False:
        return "nao_aceita"
    return "nao_informado"


def _compute_hierarchical_ref_m2(
    catalog: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 4-tier hierarchical median reference R$/m² and tier labels."""
    nb_valid = (
        catalog["neighborhood"].fillna("").astype(str).str.strip().ne("").to_numpy(dtype=bool)
    )

    t1_keys = ["market_segment", "city", "neighborhood", "prop_type"]
    t1_cnt = np.asarray(
        catalog.groupby(t1_keys, dropna=False)["price_per_m2"].transform("count"),
        dtype=int,
    )
    t1_med = np.asarray(
        catalog.groupby(t1_keys, dropna=False)["price_per_m2"].transform("median"),
        dtype=float,
    )

    t2_keys = ["market_segment", "city", "prop_type"]
    t2_cnt = np.asarray(
        catalog.groupby(t2_keys, dropna=False)["price_per_m2"].transform("count"),
        dtype=int,
    )
    t2_med = np.asarray(
        catalog.groupby(t2_keys, dropna=False)["price_per_m2"].transform("median"),
        dtype=float,
    )

    t3_keys = ["city", "prop_type"]
    t3_cnt = np.asarray(
        catalog.groupby(t3_keys, dropna=False)["price_per_m2"].transform("count"),
        dtype=int,
    )
    t3_med = np.asarray(
        catalog.groupby(t3_keys, dropna=False)["price_per_m2"].transform("median"),
        dtype=float,
    )

    t4_med = np.asarray(
        catalog.groupby("prop_type", dropna=False)["price_per_m2"].transform("median"),
        dtype=float,
    )

    use_t1 = nb_valid & (t1_cnt >= MIN_GROUP_SIZE) & ~np.isnan(t1_med)
    use_t2 = (~use_t1) & (t2_cnt >= MIN_GROUP_SIZE) & ~np.isnan(t2_med)
    use_t3 = (~use_t1) & (~use_t2) & (t3_cnt >= MIN_GROUP_SIZE) & ~np.isnan(t3_med)

    ref_m2 = np.where(
        use_t1,
        t1_med,
        np.where(use_t2, t2_med, np.where(use_t3, t3_med, t4_med)),
    )
    ref_level = np.where(
        use_t1,
        "bairro_segmento",
        np.where(
            use_t2,
            "cidade_segmento",
            np.where(use_t3, "cidade_tipo", "estado_tipo"),
        ),
    )
    return np.round(ref_m2, 2), ref_level


def load_catalog(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load ``real_properties.csv`` and compute hierarchical median valuation columns."""
    catalog = pd.read_csv(resolve_csv_path(csv_path))
    dead_cols = [c for c in ("condo_fee_brl", "iptu_brl") if c in catalog.columns]
    if dead_cols:
        catalog = catalog.drop(columns=dead_cols)

    is_terreno = catalog["prop_type"] == "terreno"
    for int_col in ("bedrooms", "bathrooms", "garage_spots"):
        catalog.loc[is_terreno, int_col] = 0
        catalog[int_col] = catalog[int_col].astype("Int64")
    catalog["accepts_financing"] = catalog["accepts_financing"].astype("boolean")

    source_lower = catalog["source"].astype(str).str.strip().str.lower()
    catalog["market_segment"] = np.where(source_lower == "caixa", "caixa", "imobiliaria")

    area_vals = catalog["area_m2"].to_numpy(dtype=float, na_value=np.nan)
    price_vals = catalog["price_brl"].to_numpy(dtype=float, na_value=np.nan)
    area_pos = catalog["area_m2"].gt(0).fillna(False).to_numpy(dtype=bool)
    catalog["price_per_m2"] = np.round(
        np.divide(price_vals, area_vals, out=np.full_like(price_vals, np.nan), where=area_pos),
        2,
    )

    ref_m2, ref_level = _compute_hierarchical_ref_m2(catalog)
    catalog["ref_m2_brl"] = ref_m2
    catalog["ref_level"] = ref_level
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
    catalog["financing_status"] = [
        financing_status_label(val) for val in catalog["accepts_financing"]
    ]
    return catalog
