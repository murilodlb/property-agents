#!/usr/bin/env python3
"""Deterministic synthetic dataset generator for the property-agents benchmark.

Generates 360 real-estate listings across Rio Grande do Norte, Brazil, writes
them to ``data/properties.csv``, and runs schema + ground-truth verification.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

np: Any
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional in pure-stdlib environments
    np = None

pd: Any
try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional in pure-stdlib environments
    pd = None


SEED = 42
TOTAL_ROWS = 360

COLUMNS = [
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

VALID_BUILDERS = [
    "Moura Dubeux",
    "MRV",
    "Alliance",
    "Capuche",
    "Ecomax",
    "Direcional",
    "Cyrela",
    "Particular",
]
TERRENO_BUILDERS = {"Ecomax", "Capuche", "Direcional", "Particular"}
VALID_PROP_TYPES = ["casa", "apartamento", "terreno"]
VALID_DOCUMENTATION = ["escriturado", "posse"]
VALID_WATER_SOURCES = ["caern", "poco_artesiano", "ambos"]
VALID_SUN_POSITIONS = ["nascente", "poente", "norte", "sul"]
VALID_CITIES = [
    "Natal",
    "Parnamirim",
    "São José de Mipibu",
    "Nísia Floresta",
    "Macaíba",
    "Extremoz",
    "Mossoró",
]

CITY_BEACH_RANGES: dict[str, tuple[float, float]] = {
    "Natal": (0.4, 16.0),
    "Parnamirim": (0.4, 16.0),
    "Nísia Floresta": (0.4, 16.0),
    "Extremoz": (0.4, 16.0),
    "São José de Mipibu": (18.0, 32.0),
    "Macaíba": (18.0, 32.0),
    "Mossoró": (32.0, 38.0),
}

CITY_PRICE_FACTORS: dict[str, float] = {
    "Natal": 1.06,
    "Parnamirim": 1.00,
    "Nísia Floresta": 0.97,
    "Extremoz": 0.95,
    "Mossoró": 0.92,
    "Macaíba": 0.90,
    "São José de Mipibu": 0.89,
}

PRICE_BOUNDS: dict[str, tuple[int, int]] = {
    "terreno": (65_000, 320_000),
    "apartamento": (140_000, 750_000),
    "casa": (180_000, 950_000),
}

AREA_BOUNDS: dict[str, tuple[int, int]] = {
    "apartamento": (48, 185),
    "casa": (90, 450),
    "terreno": (200, 1200),
}

DOC_ESCRITURADO_PROB: dict[str, float] = {
    "apartamento": 0.90,
    "casa": 0.75,
    "terreno": 0.58,
}

WATER_SOURCE_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "apartamento": [("caern", 0.65), ("ambos", 0.30), ("poco_artesiano", 0.05)],
    "casa": [("caern", 0.40), ("ambos", 0.35), ("poco_artesiano", 0.25)],
    "terreno": [("poco_artesiano", 0.40), ("caern", 0.30), ("ambos", 0.30)],
}

SUN_POSITION_WEIGHTS: list[tuple[str, float]] = [
    ("nascente", 0.40),
    ("poente", 0.25),
    ("norte", 0.20),
    ("sul", 0.15),
]

YIELD_TYPE_BONUS: dict[str, float] = {
    "terreno": 0.9,
    "casa": 0.4,
    "apartamento": 0.2,
}

ONE_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]$")
POSITIVE_INT_RE = re.compile(r"^[1-9][0-9]*$")
INT_COLUMNS = ("year_delivery", "price_brl", "monthly_fee_brl", "area_m2")

# 24 distinct developments/allotments summing to 360 rows:
# 110 apartamento (40 + 70), 150 casa, 100 terreno.
# Row ranges align with anchor prop_ids so each anchor sits naturally within
# its matching development block, and city weights respect RN geography.
DEVELOPMENTS: list[dict[str, Any]] = [
    # --- Apartamentos part 1: p001..p040 (40 rows) ---
    {
        "title": "Solar Potengi",
        "builder": "Moura Dubeux",
        "prop_type": "apartamento",
        "count": 15,
        "cities": [("Natal", 1.0)],
        "area_range": (72, 145),
        "price_per_m2": 4200,
        "fee_per_m2": 8.4,
    },
    {
        "title": "Mirante de Ponta Negra",
        "builder": "Cyrela",
        "prop_type": "apartamento",
        "count": 13,
        "cities": [("Natal", 1.0)],
        "area_range": (95, 185),
        "price_per_m2": 3600,
        "fee_per_m2": 8.0,
    },
    {
        "title": "Residencial Via Costeira",
        "builder": "Moura Dubeux",
        "prop_type": "apartamento",
        "count": 12,
        "cities": [("Natal", 1.0)],
        "area_range": (105, 185),
        "price_per_m2": 3550,
        "fee_per_m2": 7.8,
    },
    # --- Casas: p041..p190 (150 rows) ---
    {
        "title": "Recanto de Pium",
        "builder": "Ecomax",
        "prop_type": "casa",
        "count": 14,
        "cities": [("Parnamirim", 0.55), ("Nísia Floresta", 0.45)],
        "area_range": (110, 260),
        "price_per_m2": 2250,
        "fee_per_m2": 2.6,
    },
    {
        "title": "Reserva Nísia",
        "builder": "MRV",
        "prop_type": "casa",
        "count": 15,
        "cities": [
            ("Nísia Floresta", 0.45),
            ("Macaíba", 0.25),
            ("Parnamirim", 0.15),
            ("São José de Mipibu", 0.15),
        ],
        "area_range": (95, 195),
        "price_per_m2": 2350,
        "fee_per_m2": 2.3,
    },
    {
        "title": "Bosque das Dunas",
        "builder": "Ecomax",
        "prop_type": "casa",
        "count": 15,
        "cities": [("Natal", 0.45), ("Parnamirim", 0.35), ("Extremoz", 0.20)],
        "area_range": (120, 280),
        "price_per_m2": 2050,
        "fee_per_m2": 2.4,
    },
    {
        "title": "Jardim das Mangueiras",
        "builder": "Particular",
        "prop_type": "casa",
        "count": 15,
        "cities": [("Natal", 0.30), ("Parnamirim", 0.25), ("Macaíba", 0.25), ("Mossoró", 0.20)],
        "area_range": (105, 310),
        "price_per_m2": 2150,
        "fee_per_m2": 2.1,
    },
    {
        "title": "Vila Atlântica",
        "builder": "Alliance",
        "prop_type": "casa",
        "count": 15,
        "cities": [
            ("Nísia Floresta", 0.35),
            ("Parnamirim", 0.25),
            ("Natal", 0.20),
            ("Extremoz", 0.20),
        ],
        "area_range": (120, 290),
        "price_per_m2": 2250,
        "fee_per_m2": 2.8,
    },
    {
        "title": "Condomínio Morada do Sol",
        "builder": "Capuche",
        "prop_type": "casa",
        "count": 15,
        "cities": [("Parnamirim", 0.25), ("Mossoró", 0.35), ("Natal", 0.15), ("Macaíba", 0.25)],
        "area_range": (130, 320),
        "price_per_m2": 2150,
        "fee_per_m2": 2.5,
    },
    {
        "title": "Vilas de Extremoz",
        "builder": "Direcional",
        "prop_type": "casa",
        "count": 15,
        "cities": [("Extremoz", 1.0)],
        "area_range": (95, 210),
        "price_per_m2": 2250,
        "fee_per_m2": 2.2,
    },
    {
        "title": "Alphaville Potiguar",
        "builder": "Cyrela",
        "prop_type": "casa",
        "count": 15,
        "cities": [("Natal", 0.40), ("Parnamirim", 0.35), ("Mossoró", 0.25)],
        "area_range": (200, 445),
        "price_per_m2": 2020,
        "fee_per_m2": 2.6,
    },
    {
        "title": "Portal de Parnamirim",
        "builder": "MRV",
        "prop_type": "casa",
        "count": 15,
        "cities": [("Parnamirim", 1.0)],
        "area_range": (95, 190),
        "price_per_m2": 2300,
        "fee_per_m2": 2.2,
    },
    {
        "title": "Residencial Lagoa Azul",
        "builder": "Particular",
        "prop_type": "casa",
        "count": 16,
        "cities": [
            ("Nísia Floresta", 0.25),
            ("São José de Mipibu", 0.30),
            ("Extremoz", 0.20),
            ("Mossoró", 0.15),
            ("Macaíba", 0.10),
        ],
        "area_range": (115, 360),
        "price_per_m2": 2050,
        "fee_per_m2": 2.0,
    },
    # --- Terrenos: p191..p290 (100 rows) ---
    {
        "title": "Quintas de Mipibu",
        "builder": "Capuche",
        "prop_type": "terreno",
        "count": 12,
        "cities": [("São José de Mipibu", 0.80), ("Nísia Floresta", 0.20)],
        "area_range": (280, 950),
        "price_per_m2": 335,
        "fee_per_m2": 0.28,
    },
    {
        "title": "Loteamento Brisa do Sul",
        "builder": "Direcional",
        "prop_type": "terreno",
        "count": 15,
        "cities": [("Extremoz", 0.45), ("Parnamirim", 0.30), ("Nísia Floresta", 0.25)],
        "area_range": (240, 820),
        "price_per_m2": 300,
        "fee_per_m2": 0.27,
    },
    {
        "title": "Loteamento Mar do Norte",
        "builder": "Ecomax",
        "prop_type": "terreno",
        "count": 15,
        "cities": [("Extremoz", 0.40), ("Natal", 0.35), ("Nísia Floresta", 0.25)],
        "area_range": (230, 780),
        "price_per_m2": 360,
        "fee_per_m2": 0.30,
    },
    {
        "title": "Terras de São José",
        "builder": "Capuche",
        "prop_type": "terreno",
        "count": 15,
        "cities": [("São José de Mipibu", 0.65), ("Macaíba", 0.20), ("Nísia Floresta", 0.15)],
        "area_range": (260, 950),
        "price_per_m2": 315,
        "fee_per_m2": 0.27,
    },
    {
        "title": "Loteamento Sol Nascente",
        "builder": "Particular",
        "prop_type": "terreno",
        "count": 14,
        "cities": [("Parnamirim", 0.30), ("Natal", 0.25), ("Mossoró", 0.25), ("Macaíba", 0.20)],
        "area_range": (220, 750),
        "price_per_m2": 330,
        "fee_per_m2": 0.30,
    },
    {
        "title": "Chácaras do Potengi",
        "builder": "Particular",
        "prop_type": "terreno",
        "count": 14,
        "cities": [("Macaíba", 0.40), ("São José de Mipibu", 0.35), ("Extremoz", 0.25)],
        "area_range": (450, 1200),
        "price_per_m2": 250,
        "fee_per_m2": 0.20,
    },
    {
        "title": "Loteamento Costa Branca",
        "builder": "Ecomax",
        "prop_type": "terreno",
        "count": 15,
        "cities": [("Mossoró", 1.0)],
        "area_range": (235, 850),
        "price_per_m2": 330,
        "fee_per_m2": 0.29,
    },
    # --- Apartamentos part 2: p291..p360 (70 rows) ---
    {
        "title": "Torre do Tirol",
        "builder": "Cyrela",
        "prop_type": "apartamento",
        "count": 18,
        "cities": [("Natal", 1.0)],
        "area_range": (85, 175),
        "price_per_m2": 3650,
        "fee_per_m2": 7.8,
    },
    {
        "title": "Edifício Capim Macio",
        "builder": "Alliance",
        "prop_type": "apartamento",
        "count": 18,
        "cities": [("Natal", 1.0)],
        "area_range": (65, 135),
        "price_per_m2": 3850,
        "fee_per_m2": 7.8,
    },
    {
        "title": "Parque dos Cajueiros",
        "builder": "MRV",
        "prop_type": "apartamento",
        "count": 17,
        "cities": [("Parnamirim", 0.45), ("Mossoró", 0.30), ("Macaíba", 0.15), ("Natal", 0.10)],
        "area_range": (48, 88),
        "price_per_m2": 3200,
        "fee_per_m2": 6.8,
    },
    {
        "title": "Residencial Brisas Potiguares",
        "builder": "Direcional",
        "prop_type": "apartamento",
        "count": 17,
        "cities": [("Parnamirim", 0.45), ("Extremoz", 0.30), ("Mossoró", 0.15), ("Macaíba", 0.10)],
        "area_range": (55, 110),
        "price_per_m2": 3400,
        "fee_per_m2": 7.2,
    },
]

ANCHOR_ROWS: dict[str, dict[str, Any]] = {
    # Anchor Group 1 — Top Houses up to R$ 350,000
    "p082": {
        "prop_id": "p082",
        "builder": "Ecomax",
        "title": "Bosque das Dunas",
        "prop_type": "casa",
        "year_delivery": 2024,
        "dist_beach_km": 6.5,
        "price_brl": 340000,
        "documentation": "escriturado",
        "water_source": "ambos",
        "yield_rate_pct": 7.4,
        "monthly_fee_brl": 480,
        "area_m2": 180,
        "sun_position": "poente",
        "city": "Parnamirim",
    },
    "p081": {
        "prop_id": "p081",
        "builder": "Ecomax",
        "title": "Bosque das Dunas",
        "prop_type": "casa",
        "year_delivery": 2024,
        "dist_beach_km": 8.2,
        "price_brl": 315000,
        "documentation": "escriturado",
        "water_source": "caern",
        "yield_rate_pct": 8.9,
        "monthly_fee_brl": 390,
        "area_m2": 180,
        "sun_position": "nascente",
        "city": "Natal",
    },
    "p109": {
        "prop_id": "p109",
        "builder": "Alliance",
        "title": "Vila Atlântica",
        "prop_type": "casa",
        "year_delivery": 2024,
        "dist_beach_km": 11.0,
        "price_brl": 345000,
        "documentation": "escriturado",
        "water_source": "ambos",
        "yield_rate_pct": 9.6,
        "monthly_fee_brl": 520,
        "area_m2": 155,
        "sun_position": "nascente",
        "city": "Nísia Floresta",
    },
    # Anchor Group 2 — Top Terrains up to R$ 150,000
    "p199": {
        "prop_id": "p199",
        "builder": "Capuche",
        "title": "Quintas de Mipibu",
        "prop_type": "terreno",
        "year_delivery": 2026,
        "dist_beach_km": 21.5,
        "price_brl": 145000,
        "documentation": "escriturado",
        "water_source": "poco_artesiano",
        "yield_rate_pct": 10.8,
        "monthly_fee_brl": 110,
        "area_m2": 450,
        "sun_position": "nascente",
        "city": "São José de Mipibu",
    },
    "p204": {
        "prop_id": "p204",
        "builder": "Direcional",
        "title": "Loteamento Brisa do Sul",
        "prop_type": "terreno",
        "year_delivery": 2025,
        "dist_beach_km": 14.0,
        "price_brl": 138000,
        "documentation": "posse",
        "water_source": "caern",
        "yield_rate_pct": 8.2,
        "monthly_fee_brl": 90,
        "area_m2": 600,
        "sun_position": "sul",
        "city": "Extremoz",
    },
    # Anchor Group 3 — Specific Comparison & Detail Targets
    "p003": {
        "prop_id": "p003",
        "builder": "Moura Dubeux",
        "title": "Solar Potengi",
        "prop_type": "apartamento",
        "year_delivery": 2022,
        "dist_beach_km": 2.1,
        "price_brl": 420000,
        "documentation": "escriturado",
        "water_source": "caern",
        "yield_rate_pct": 6.8,
        "monthly_fee_brl": 890,
        "area_m2": 92,
        "sun_position": "nascente",
        "city": "Natal",
    },
    "p061": {
        "prop_id": "p061",
        "builder": "MRV",
        "title": "Reserva Nísia",
        "prop_type": "casa",
        "year_delivery": 2023,
        "dist_beach_km": 9.4,
        "price_brl": 348000,
        "documentation": "posse",
        "water_source": "poco_artesiano",
        "yield_rate_pct": 7.1,
        "monthly_fee_brl": 310,
        "area_m2": 135,
        "sun_position": "poente",
        "city": "Macaíba",
    },
}

ANCHOR_SORT_KEYS: frozenset[tuple[int, float]] = frozenset(
    (row["year_delivery"], row["dist_beach_km"]) for row in ANCHOR_ROWS.values()
)


def _weighted_choice(items_weights: list[tuple[str, float]]) -> str:
    """Select one string option according to its relative weight."""
    items = [item for item, _ in items_weights]
    weights = [weight for _, weight in items_weights]
    return random.choices(items, weights=weights, k=1)[0]


def _clamp(val: float, low: float, high: float) -> float:
    """Clamp ``val`` to the inclusive ``[low, high]`` interval."""
    return max(low, min(high, val))


def _compute_price(
    prop_type: str,
    area_m2: int,
    price_per_m2: float,
    year_delivery: int,
    dist_beach_km: float,
    documentation: str,
    city: str,
    noise: float,
) -> int:
    """Compute a property price rounded to R$ 500 and clamped to type bounds."""
    year_factor = 1.0 + (year_delivery - 2020) * 0.022
    beach_factor = max(0.88, 1.07 - dist_beach_km * 0.0055)
    doc_factor = 1.0 if documentation == "escriturado" else 0.88
    city_factor = CITY_PRICE_FACTORS[city]

    raw = area_m2 * price_per_m2 * year_factor * beach_factor * doc_factor * city_factor * noise
    rounded = int(round(raw / 500.0) * 500)
    low, high = PRICE_BOUNDS[prop_type]
    return max(low, min(high, rounded))


def _sample_unique_dist(
    year_delivery: int,
    city: str,
    used_keys: set[tuple[int, float]],
) -> float:
    """Sample a city-realistic beach distance that avoids sort-key ties."""
    low, high = CITY_BEACH_RANGES[city]
    base = round(random.uniform(low, high), 1)
    if (year_delivery, base) not in used_keys:
        used_keys.add((year_delivery, base))
        return base
    candidates = [
        round(low + step * 0.1, 1)
        for step in range(int(round((high - low) * 10)) + 1)
    ]
    candidates.sort(key=lambda c: abs(c - base))
    for cand in candidates:
        if (year_delivery, cand) not in used_keys:
            used_keys.add((year_delivery, cand))
            return cand
    used_keys.add((year_delivery, base))
    return base


def _sample_doc_and_water(prop_type: str) -> tuple[str, str]:
    """Sample documentation status and water source conditioned on property type."""
    is_escriturado = random.random() < DOC_ESCRITURADO_PROB[prop_type]
    documentation = "escriturado" if is_escriturado else "posse"
    water_source = _weighted_choice(WATER_SOURCE_WEIGHTS[prop_type])
    return documentation, water_source


def _enforce_anchor_headroom(
    prop_type: str,
    area_m2: int,
    price_per_m2: float,
    year_delivery: int,
    dist_beach_km: float,
    price_brl: int,
    documentation: str,
    city: str,
    noise: float,
    used_sort_keys: set[tuple[int, float]],
) -> tuple[int, float, int]:
    """Resample year, distance, and price if a non-anchor row infringes anchor headroom."""
    max_year: int | None = None
    if prop_type == "casa" and price_brl <= 350_000:
        if year_delivery > 2024 or (year_delivery == 2024 and dist_beach_km <= 11.5):
            max_year = 2023
    elif prop_type == "terreno" and price_brl <= 150_000 and year_delivery >= 2025:
        max_year = 2024

    if max_year is None:
        return year_delivery, dist_beach_km, price_brl

    old_key = (year_delivery, dist_beach_km)
    if old_key not in ANCHOR_SORT_KEYS:
        used_sort_keys.discard(old_key)
    year_delivery = random.randint(2015, max_year)
    dist_beach_km = _sample_unique_dist(year_delivery, city, used_sort_keys)
    price_brl = _compute_price(
        prop_type=prop_type,
        area_m2=area_m2,
        price_per_m2=price_per_m2,
        year_delivery=year_delivery,
        dist_beach_km=dist_beach_km,
        documentation=documentation,
        city=city,
        noise=noise,
    )
    return year_delivery, dist_beach_km, price_brl


def _compute_yield_rate(prop_type: str, dist_beach_km: float, year_delivery: int) -> float:
    """Compute annual yield percentage rounded to 1 decimal place in [4.5, 11.5]."""
    type_bonus = YIELD_TYPE_BONUS[prop_type]
    beach_bonus = max(0.0, (12.0 - dist_beach_km) * 0.08)
    year_bonus = (year_delivery - 2015) * 0.12
    yield_raw = 6.0 + type_bonus + beach_bonus + year_bonus + random.uniform(-2.2, 3.2)
    return round(_clamp(yield_raw, 4.5, 11.5), 1)


def _compute_monthly_fee(area_m2: int, fee_per_m2: float) -> int:
    """Compute monthly fee as a multiple of 10 BRL clamped to [60, 1600]."""
    fee_raw = area_m2 * fee_per_m2 * random.uniform(0.84, 1.18)
    return int(_clamp(round(fee_raw / 10.0) * 10, 60, 1600))


def _generate_listing(
    prop_id: str,
    dev: dict[str, Any],
    used_sort_keys: set[tuple[int, float]],
) -> dict[str, Any]:
    """Generate a single synthetic non-anchor listing within a development block."""
    prop_type = dev["prop_type"]
    area_min, area_max = dev["area_range"]

    city = _weighted_choice(dev["cities"])
    area_m2 = random.randint(area_min, area_max)
    year_delivery = random.randint(2015, 2026)
    documentation, water_source = _sample_doc_and_water(prop_type)
    sun_position = _weighted_choice(SUN_POSITION_WEIGHTS)
    dist_beach_km = _sample_unique_dist(year_delivery, city, used_sort_keys)
    noise = random.uniform(0.92, 1.08)

    price_brl = _compute_price(
        prop_type=prop_type,
        area_m2=area_m2,
        price_per_m2=dev["price_per_m2"],
        year_delivery=year_delivery,
        dist_beach_km=dist_beach_km,
        documentation=documentation,
        city=city,
        noise=noise,
    )
    year_delivery, dist_beach_km, price_brl = _enforce_anchor_headroom(
        prop_type=prop_type,
        area_m2=area_m2,
        price_per_m2=dev["price_per_m2"],
        year_delivery=year_delivery,
        dist_beach_km=dist_beach_km,
        price_brl=price_brl,
        documentation=documentation,
        city=city,
        noise=noise,
        used_sort_keys=used_sort_keys,
    )
    yield_rate_pct = _compute_yield_rate(prop_type, dist_beach_km, year_delivery)
    monthly_fee_brl = _compute_monthly_fee(area_m2, dev["fee_per_m2"])

    return {
        "prop_id": prop_id,
        "builder": dev["builder"],
        "title": dev["title"],
        "prop_type": prop_type,
        "year_delivery": year_delivery,
        "dist_beach_km": dist_beach_km,
        "price_brl": price_brl,
        "documentation": documentation,
        "water_source": water_source,
        "yield_rate_pct": yield_rate_pct,
        "monthly_fee_brl": monthly_fee_brl,
        "area_m2": area_m2,
        "sun_position": sun_position,
        "city": city,
    }


def generate_rows() -> list[dict[str, Any]]:
    """Generate all 360 deterministic property listings with planted anchor rows."""
    random.seed(SEED)
    if np is not None:
        np.random.seed(SEED)

    # Pre-register anchor (year_delivery, dist_beach_km) keys so no generated
    # row ties any anchor—or any other row—on the default sort key.
    used_sort_keys: set[tuple[int, float]] = set(ANCHOR_SORT_KEYS)

    rows: list[dict[str, Any]] = []
    idx = 1
    for dev in DEVELOPMENTS:
        for _ in range(dev["count"]):
            prop_id = f"p{idx:03d}"
            idx += 1

            if prop_id in ANCHOR_ROWS:
                anchor = dict(ANCHOR_ROWS[prop_id])
                expected_meta = (dev["title"], dev["builder"], dev["prop_type"])
                actual_meta = (anchor["title"], anchor["builder"], anchor["prop_type"])
                assert actual_meta == expected_meta, (
                    f"Anchor {prop_id} placed in mismatched development block {dev['title']}"
                )
                rows.append(anchor)
            else:
                rows.append(_generate_listing(prop_id, dev, used_sort_keys))

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write property rows to ``output_path`` using exact column formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["dist_beach_km"] = f"{row['dist_beach_km']:.1f}"
            formatted["yield_rate_pct"] = f"{row['yield_rate_pct']:.1f}"
            writer.writerow(formatted)


def _validate_row_constraints(row: dict[str, Any]) -> None:
    """Validate domain enums, numeric bounds, and city-to-beach realism for one row."""
    assert row["builder"] in VALID_BUILDERS, f"Bad builder: {row}"
    assert row["prop_type"] in VALID_PROP_TYPES, f"Bad prop_type: {row}"
    if row["prop_type"] == "terreno":
        assert row["builder"] in TERRENO_BUILDERS, f"Invalid terreno builder: {row}"
    assert 2015 <= row["year_delivery"] <= 2026, f"Bad year_delivery: {row}"
    assert 0.4 <= row["dist_beach_km"] <= 38.0, f"Bad dist_beach_km: {row}"
    assert row["city"] in VALID_CITIES, f"Bad city: {row}"
    # Only anchor p061 (Macaíba at 9.4 km) is intentionally outside its city's range.
    if row["prop_id"] != "p061":
        c_low, c_high = CITY_BEACH_RANGES[row["city"]]
        assert c_low <= row["dist_beach_km"] <= c_high, f"Unrealistic city dist: {row}"

    p_low, p_high = PRICE_BOUNDS[row["prop_type"]]
    assert p_low <= row["price_brl"] <= p_high, f"Bad price_brl: {row}"
    assert row["price_brl"] % 500 == 0, f"Price not multiple of 500: {row}"
    assert row["documentation"] in VALID_DOCUMENTATION, f"Bad documentation: {row}"
    assert row["water_source"] in VALID_WATER_SOURCES, f"Bad water_source: {row}"
    assert 4.5 <= row["yield_rate_pct"] <= 11.5, f"Bad yield_rate_pct: {row}"
    assert 60 <= row["monthly_fee_brl"] <= 1600, f"Bad monthly_fee_brl: {row}"
    assert row["monthly_fee_brl"] % 10 == 0, f"Fee not multiple of 10: {row}"
    a_low, a_high = AREA_BOUNDS[row["prop_type"]]
    assert a_low <= row["area_m2"] <= a_high, f"Bad area_m2: {row}"
    assert row["sun_position"] in VALID_SUN_POSITIONS, f"Bad sun_position: {row}"


def _parse_and_validate_row(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    """Parse one CSV row dict into typed values and validate per-row constraints."""
    assert set(raw.keys()) == set(COLUMNS), (
        f"Unexpected or missing columns at row {idx}: {list(raw.keys())}"
    )
    for col in COLUMNS:
        val = raw[col]
        assert isinstance(val, str) and val == val.strip() and val != "", (
            f"Malformed/unstripped cell {col} at row {idx}: {val!r}"
        )
    for int_col in INT_COLUMNS:
        assert POSITIVE_INT_RE.fullmatch(raw[int_col]), (
            f"{int_col} not a canonical integer at row {idx}: {raw[int_col]!r}"
        )
    assert ONE_DECIMAL_RE.fullmatch(raw["dist_beach_km"]), (
        f"dist_beach_km not 1 decimal at row {idx}: {raw['dist_beach_km']!r}"
    )
    assert ONE_DECIMAL_RE.fullmatch(raw["yield_rate_pct"]), (
        f"yield_rate_pct not 1 decimal at row {idx}: {raw['yield_rate_pct']!r}"
    )

    row: dict[str, Any] = {
        "prop_id": raw["prop_id"],
        "builder": raw["builder"],
        "title": raw["title"],
        "prop_type": raw["prop_type"],
        "year_delivery": int(raw["year_delivery"]),
        "dist_beach_km": float(raw["dist_beach_km"]),
        "price_brl": int(raw["price_brl"]),
        "documentation": raw["documentation"],
        "water_source": raw["water_source"],
        "yield_rate_pct": float(raw["yield_rate_pct"]),
        "monthly_fee_brl": int(raw["monthly_fee_brl"]),
        "area_m2": int(raw["area_m2"]),
        "sun_position": raw["sun_position"],
        "city": raw["city"],
    }

    assert row["prop_id"] == f"p{idx:03d}", f"Bad prop_id at {idx}: {row['prop_id']}"
    _validate_row_constraints(row)
    return row


def _validate_dataset_invariants(rows: list[dict[str, Any]]) -> None:
    """Validate dataset-level counts, title coherence, unique sort keys, and anchors."""
    assert len(rows) == TOTAL_ROWS, f"Expected {TOTAL_ROWS} rows, got {len(rows)}"
    distinct_titles = {r["title"] for r in rows}
    assert len(distinct_titles) == 24, (
        f"Expected 24 distinct titles, got {len(distinct_titles)}"
    )

    pt_counts = Counter(r["prop_type"] for r in rows)
    assert 140 <= pt_counts["casa"] <= 160, f"Unexpected casa count: {pt_counts['casa']}"
    assert 100 <= pt_counts["apartamento"] <= 120, (
        f"Unexpected apartamento count: {pt_counts['apartamento']}"
    )
    assert 90 <= pt_counts["terreno"] <= 110, (
        f"Unexpected terreno count: {pt_counts['terreno']}"
    )

    doc_counts = Counter(r["documentation"] for r in rows)
    escriturado_ratio = doc_counts["escriturado"] / len(rows)
    assert 0.70 <= escriturado_ratio <= 0.80, (
        f"Unexpected escriturado ratio: {escriturado_ratio:.3f}"
    )

    assert {r["builder"] for r in rows} == set(VALID_BUILDERS), "Missing expected builders"
    assert {r["city"] for r in rows} == set(VALID_CITIES), "Missing expected cities"
    assert {r["water_source"] for r in rows} == set(VALID_WATER_SOURCES), (
        "Missing expected water sources"
    )
    assert {r["sun_position"] for r in rows} == set(VALID_SUN_POSITIONS), (
        "Missing expected sun positions"
    )

    title_signatures: dict[str, tuple[str, str]] = {}
    for r in rows:
        sig = (r["prop_type"], r["builder"])
        if r["title"] in title_signatures:
            assert title_signatures[r["title"]] == sig, (
                f"Inconsistent (prop_type, builder) for title {r['title']}: "
                f"{title_signatures[r['title']]} vs {sig}"
            )
        else:
            title_signatures[r["title"]] = sig

    sort_keys = [(r["year_delivery"], r["dist_beach_km"]) for r in rows]
    assert len(sort_keys) == len(set(sort_keys)), (
        "Duplicate (year_delivery, dist_beach_km) sort keys found"
    )

    by_id = {r["prop_id"]: r for r in rows}
    for pid, expected in ANCHOR_ROWS.items():
        assert pid in by_id, f"Missing anchor {pid}"
        assert by_id[pid] == expected, (
            f"Anchor {pid} mismatch: {by_id[pid]} != {expected}"
        )

    # Ensure no boundary clamping pile-ups (>2 listings at exact price bound)
    for pt, (p_low, p_high) in PRICE_BOUNDS.items():
        pt_prices = [r["price_brl"] for r in rows if r["prop_type"] == pt]
        assert pt_prices.count(p_low) <= 2, (
            f"Lower price pile-up for {pt}: {pt_prices.count(p_low)}"
        )
        assert pt_prices.count(p_high) <= 2, (
            f"Upper price pile-up for {pt}: {pt_prices.count(p_high)}"
        )


def _load_and_validate_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load ``csv_path`` and validate row schemas and dataset-wide invariants."""
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == COLUMNS, f"Column mismatch: {reader.fieldnames}"
        parsed = [_parse_and_validate_row(raw, idx) for idx, raw in enumerate(reader, start=1)]

    _validate_dataset_invariants(parsed)
    return parsed


def _print_ranked_rows(heading: str, rows: list[dict[str, Any]], limit: int) -> None:
    """Print a formatted top-N summary table for a filtered listing subset."""
    print(f"\n{heading}:")
    for rank, r in enumerate(rows[:limit], start=1):
        print(
            f"  #{rank}: {r['prop_id']} | {r['title']} | builder={r['builder']} | "
            f"year={r['year_delivery']} | dist={r['dist_beach_km']:.1f}km | "
            f"price=R${r['price_brl']} | area={r['area_m2']}m2 | {r['city']}"
        )


def _verify_anchor_group_casas(rows: list[dict[str, Any]]) -> None:
    """Verify and print Anchor Group 1 (top houses up to R$ 350,000)."""
    casas_350k = sorted(
        [r for r in rows if r["prop_type"] == "casa" and r["price_brl"] <= 350_000],
        key=lambda r: (-r["year_delivery"], r["dist_beach_km"]),
    )
    assert len(casas_350k) >= 3, f"Expected >= 3 casas <= 350k, got {len(casas_350k)}"
    assert all(r["year_delivery"] <= 2024 for r in casas_350k), (
        "Found casa <= 350k newer than 2024"
    )
    assert [r["prop_id"] for r in casas_350k[:3]] == ["p082", "p081", "p109"], (
        f"Unexpected top 3 casas <= 350k: {[r['prop_id'] for r in casas_350k[:5]]}"
    )
    assert all(r["dist_beach_km"] > 11.0 for r in casas_350k[3:] if r["year_delivery"] == 2024), (
        "Non-anchor 2024 casa <= 350k closer than p109"
    )
    _print_ranked_rows(
        "Top 5 houses (prop_type == 'casa' & price_brl <= 350000)",
        casas_350k,
        limit=5,
    )


def _verify_anchor_group_terrenos(rows: list[dict[str, Any]]) -> None:
    """Verify and print Anchor Group 2 (top terrains up to R$ 150,000)."""
    terrenos_150k = sorted(
        [r for r in rows if r["prop_type"] == "terreno" and r["price_brl"] <= 150_000],
        key=lambda r: (-r["year_delivery"], r["dist_beach_km"]),
    )
    assert len(terrenos_150k) >= 2, (
        f"Expected >= 2 terrenos <= 150k, got {len(terrenos_150k)}"
    )
    assert terrenos_150k[0]["prop_id"] == "p199", (
        f"Unexpected #1 terreno <= 150k: {terrenos_150k[0]['prop_id']}"
    )
    assert terrenos_150k[1]["prop_id"] == "p204", (
        f"Unexpected #2 terreno <= 150k: {terrenos_150k[1]['prop_id']}"
    )
    assert all(r["year_delivery"] <= 2024 for r in terrenos_150k[2:]), (
        "Found non-anchor terreno <= 150k with year_delivery >= 2025"
    )
    _print_ranked_rows(
        "Top 3 terrains (prop_type == 'terreno' & price_brl <= 150000)",
        terrenos_150k,
        limit=3,
    )


def _verify_pandas_crosscheck(csv_path: Path) -> None:
    """Cross-check CSV schema and anchor ordering with pandas when installed."""
    if pd is None:
        return
    df = pd.read_csv(csv_path)
    assert len(df) == TOTAL_ROWS
    assert list(df.columns) == COLUMNS
    top_casas = df[
        (df["prop_type"] == "casa") & (df["price_brl"] <= 350_000)
    ].sort_values(by=["year_delivery", "dist_beach_km"], ascending=[False, True])
    assert top_casas["prop_id"].head(3).tolist() == ["p082", "p081", "p109"]
    top_terrenos = df[
        (df["prop_type"] == "terreno") & (df["price_brl"] <= 150_000)
    ].sort_values(by=["year_delivery", "dist_beach_km"], ascending=[False, True])
    assert top_terrenos["prop_id"].head(2).tolist() == ["p199", "p204"]


def verify_and_report(csv_path: Path) -> None:
    """Validate ``csv_path`` and print the benchmark verification report."""
    rows = _load_and_validate_csv(csv_path)

    print("=== Verification Report ===")
    print(f"File: {csv_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Columns ({len(COLUMNS)}): {COLUMNS}")
    print(f"Distinct titles: {len({r['title'] for r in rows})}")
    print(f"prop_type counts: {dict(Counter(r['prop_type'] for r in rows))}")
    print(f"documentation counts: {dict(Counter(r['documentation'] for r in rows))}")
    print(f"builder counts: {dict(Counter(r['builder'] for r in rows))}")
    print(f"city counts: {dict(Counter(r['city'] for r in rows))}")

    _verify_anchor_group_casas(rows)
    _verify_anchor_group_terrenos(rows)
    _verify_pandas_crosscheck(csv_path)

    print("\nAll schema and anchor ground-truth checks PASSED.")


def main() -> None:
    """Generate ``data/properties.csv`` and run full verification reporting."""
    output_path = Path(__file__).resolve().parent / "properties.csv"
    rows = generate_rows()
    write_csv(rows, output_path)
    verify_and_report(output_path)


if __name__ == "__main__":
    main()
