"""Shared catalog loader, benchmark suite, and automatic grading helpers."""

from src.catalog import (
    MIN_GROUP_SIZE,
    clean_value,
    financing_status_label,
    limpar_nulos,
    load_catalog,
    normalize_text,
    valuation_status,
)
from src.questions import (
    PERGUNTAS,
    SYSTEM_PROMPT_PTBR,
    brl_re,
    checagem_automatica,
    df,
    num,
)

__all__ = [
    "MIN_GROUP_SIZE",
    "PERGUNTAS",
    "SYSTEM_PROMPT_PTBR",
    "brl_re",
    "checagem_automatica",
    "clean_value",
    "df",
    "financing_status_label",
    "limpar_nulos",
    "load_catalog",
    "normalize_text",
    "num",
    "valuation_status",
]
