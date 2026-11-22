"""Pydantic v2 contracts for agent execution JSONL records, judge verdicts, and CSV rows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

VereditoDetalhado = Literal["correta", "parcial", "incorreta", "alucinacao"]


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


class ExecutionRecord(BaseModel):
    """Contract for a single line in ``results/executions.jsonl`` produced by an agent run."""

    sistema: str
    pergunta_id: int = Field(ge=1, le=20)
    resposta: str
    ferramentas_usadas: list[str] = Field(default_factory=list)
    chamadas_modelo: int = Field(ge=0)
    latencia_segundos: float = Field(ge=0.0)
    tokens_entrada: int | None = Field(default=None, ge=0)
    tokens_saida: int | None = Field(default=None, ge=0)
    modelo_agente: str | None = None
    erro_execucao: str | None = None
    timestamp_utc: str = Field(default_factory=_utc_now_iso)

    @field_validator("sistema")
    @classmethod
    def _validate_sistema(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("O campo 'sistema' não pode ser vazio.")
        return cleaned


class JudgeToolCall(BaseModel):
    """Audit record of a single read-only MCP tool invocation made by the judge."""

    ferramenta: str
    argumentos: dict[str, Any] = Field(default_factory=dict)
    resumo_resultado: str
    erro: bool = False


class JudgeVerdict(BaseModel):
    """Structured evaluation verdict returned by the read-only MCP Frontier Judge."""

    correta: bool
    veredito_detalhado: VereditoDetalhado
    justificativa: str
    consultas_realizadas: list[JudgeToolCall] = Field(default_factory=list)
    provedor_juiz: str
    modelo_juiz: str


class BenchmarkRow(BaseModel):
    """Flat row representation for ``results/benchmark_60_execucoes.csv``."""

    # Identification
    sistema: str
    pergunta_id: int = Field(ge=1, le=20)
    categoria: str
    ferramenta_esperada: str
    modelo_agente: str | None = None

    # Agent trajectory & cost
    ferramentas_usadas: str
    ferramenta_correta: bool
    num_tool_calls: int = Field(ge=0)
    chamadas_modelo: int = Field(ge=0)
    latencia_segundos: float = Field(ge=0.0)
    tokens_entrada: int | None = None
    tokens_saida: int | None = None
    erro_execucao: str | None = None

    # Automatic grading
    acerto_automatico: bool
    falhas_automaticas: str

    # Frontier Judge (MCP Read-Only)
    acerto_juiz: bool | None = None
    veredito_detalhado: str | None = None
    justificativa_juiz: str | None = None
    consultas_juiz: str = "[]"
    num_consultas_juiz: int = Field(default=0, ge=0)
    provedor_juiz: str | None = None
    modelo_juiz: str | None = None

    # Manual audit
    concordancia_auto_juiz: bool | None = None
    revisao_manual: str = ""
    nota_manual: str = ""

    # Traceability
    resposta_agente: str


def append_execution_jsonl(path: str | Path, record: ExecutionRecord) -> None:
    """Append one validated ``ExecutionRecord`` as a compact UTF-8 JSONL line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    validated = (
        record
        if isinstance(record, ExecutionRecord)
        else ExecutionRecord.model_validate(record)
    )
    payload = json.dumps(validated.model_dump(mode="json"), ensure_ascii=False)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(payload + "\n")


def load_executions_jsonl(path: str | Path) -> list[ExecutionRecord]:
    """Read and validate all non-empty lines from a JSONL file into ``ExecutionRecord`` items."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Arquivo de execuções não encontrado: {target}")

    records: list[ExecutionRecord] = []
    with target.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido na linha {line_num} de {target}: {exc}"
                ) from exc
            try:
                records.append(ExecutionRecord.model_validate(data))
            except ValidationError as exc:
                raise ValueError(
                    f"Registro inválido na linha {line_num} de {target}: {exc}"
                ) from exc
    return records
