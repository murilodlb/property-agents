"""Shared catalog loader, schemas, benchmark suite, evaluation pipeline, and MCP judge."""

from src.catalog import (
    MIN_GROUP_SIZE,
    clean_value,
    financing_status_label,
    limpar_nulos,
    load_catalog,
    normalize_text,
    valuation_status,
)
from src.judge import READ_ONLY_MCP_TOOLS, JudgeConfig, MCPJudge
from src.questions import (
    PERGUNTAS,
    SYSTEM_PROMPT_PTBR,
    brl_re,
    checagem_automatica,
    df,
    num,
)
from src.schema import (
    BenchmarkRow,
    ExecutionRecord,
    JudgeToolCall,
    JudgeVerdict,
    VereditoDetalhado,
    append_execution_jsonl,
    load_executions_jsonl,
)

__all__ = [
    "BenchmarkRow",
    "ExecutionRecord",
    "JudgeConfig",
    "JudgeToolCall",
    "JudgeVerdict",
    "MCPJudge",
    "MIN_GROUP_SIZE",
    "PERGUNTAS",
    "READ_ONLY_MCP_TOOLS",
    "SYSTEM_PROMPT_PTBR",
    "VereditoDetalhado",
    "append_execution_jsonl",
    "brl_re",
    "checagem_automatica",
    "clean_value",
    "df",
    "financing_status_label",
    "limpar_nulos",
    "load_catalog",
    "load_executions_jsonl",
    "normalize_text",
    "num",
    "valuation_status",
]
