#!/usr/bin/env python3
"""Run the 20 benchmark questions on each agent system and write ``executions.jsonl``.

Usage::

    uv run python -m src.run_agents                       # 3 systems x 20 questions = 60 runs
    uv run python -m src.run_agents --sistemas langgraph_unico --perguntas 1 2 3
    uv run python -m src.run_agents --resume              # skip (sistema, pergunta) already done

Outputs (``results/``):

- ``executions.jsonl``: one ``ExecutionRecord`` per run (input of ``src.metrics``);
- ``trajetorias.jsonl``: per-run internals (verifier verdicts, supervisor delegations,
  LLM calls per agent) for the error analysis;
- ``ambiente.json``: hardware, model, quantization and library versions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil
import requests

from src.agents import SYSTEM_CLASSES, AgentConfig, MCPToolkit, build_systems
from src.questions import PERGUNTAS
from src.schema import append_execution_jsonl

LIBS = [
    "langchain",
    "langchain-core",
    "langgraph",
    "langchain-ollama",
    "ollama",
    "fastmcp",
    "mcp",
    "pydantic",
    "pandas",
    "psutil",
]


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _gpu_info() -> list[str]:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        return [
            str(pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i)))
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5).stdout
        return [ln.split(": ", 1)[-1] for ln in out.splitlines() if "VGA" in ln or "3D" in ln]
    except Exception:
        return []


def collect_environment(config: AgentConfig) -> dict[str, Any]:
    """Hardware, model/quantization (Ollama API) and library versions."""
    env: dict[str, Any] = {
        "sistema_operacional": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": _cpu_model(),
        "cpu_nucleos_logicos": psutil.cpu_count(logical=True),
        "cpu_nucleos_fisicos": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "gpu": _gpu_info(),
        "bibliotecas": {},
        "agente": {
            "modelo": config.model,
            "temperatura": config.temperature,
            "seed": config.seed,
            "num_ctx": config.num_ctx,
            "num_predict": config.num_predict,
            "reasoning": config.reasoning,
            "recursion_limit": config.recursion_limit,
            "max_revisoes": config.max_revisoes,
            "max_delegacoes": config.max_delegacoes,
        },
    }
    for lib in LIBS:
        try:
            env["bibliotecas"][lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            env["bibliotecas"][lib] = None
    try:
        env["ollama_versao"] = requests.get(f"{config.base_url}/api/version", timeout=5).json()
        show = requests.post(
            f"{config.base_url}/api/show", json={"model": config.model}, timeout=10
        ).json()
        details = show.get("details", {})
        env["modelo_detalhes"] = {
            "familia": details.get("family"),
            "parametros": details.get("parameter_size"),
            "quantizacao": details.get("quantization_level"),
            "formato": details.get("format"),
            "capacidades": show.get("capabilities"),
        }
        ps = requests.get(f"{config.base_url}/api/ps", timeout=5).json()
        env["ollama_modelos_carregados"] = [
            {"nome": m.get("name"), "tamanho": m.get("size"), "tamanho_vram": m.get("size_vram")}
            for m in ps.get("models", [])
        ]
    except Exception as exc:
        env["ollama_erro"] = str(exc)
    return env


def _done_pairs(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    done: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            done.add((data["sistema"], int(data["pergunta_id"])))
    return done


async def run_benchmark(
    sistemas: Sequence[str],
    pergunta_ids: Sequence[int],
    output: Path,
    trajetorias: Path,
    config: AgentConfig,
    resume: bool = False,
) -> int:
    perguntas = {int(q["id"]): q for q in PERGUNTAS}
    done = _done_pairs(output) if resume else set()
    total = 0
    async with MCPToolkit(config.server_script) as toolkit:
        systems = build_systems(toolkit, config, list(sistemas))
        # Warm-up: load the model into memory so the first run's latency is not inflated.
        await systems[sistemas[0]].llm.ainvoke("ok")
        for nome, system in systems.items():
            for pid in pergunta_ids:
                if (nome, pid) in done:
                    print(f"[skip] {nome} q{pid:02d}")
                    continue
                result = await system.answer(str(perguntas[pid]["q"]))
                append_execution_jsonl(output, result.to_record(pid))
                with trajetorias.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "sistema": nome,
                                "pergunta_id": pid,
                                "ferramentas_usadas": result.ferramentas_usadas,
                                "detalhes": result.detalhes,
                            },
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
                total += 1
                status = f"ERRO: {result.erro_execucao}" if result.erro_execucao else "ok"
                print(
                    f"[{nome}] q{pid:02d} {result.latencia_segundos:6.1f}s "
                    f"llm={result.chamadas_modelo} tools={result.ferramentas_usadas} {status}",
                    flush=True,
                )
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa os 3 sistemas de agentes (60 runs).")
    parser.add_argument(
        "--sistemas", nargs="+", choices=list(SYSTEM_CLASSES), default=list(SYSTEM_CLASSES)
    )
    parser.add_argument("--perguntas", nargs="+", type=int, default=[q["id"] for q in PERGUNTAS])
    parser.add_argument("--output", type=Path, default=Path("results/executions.jsonl"))
    parser.add_argument("--trajetorias", type=Path, default=Path("results/trajetorias.jsonl"))
    parser.add_argument("--ambiente", type=Path, default=Path("results/ambiente.json"))
    parser.add_argument("--model", type=str, default=None, help="Modelo Ollama (AGENT_MODEL).")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AgentConfig()
    if args.model:
        config.model = args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trajetorias.parent.mkdir(parents=True, exist_ok=True)
    args.ambiente.parent.mkdir(parents=True, exist_ok=True)
    total = asyncio.run(
        run_benchmark(
            args.sistemas, args.perguntas, args.output, args.trajetorias, config, args.resume
        )
    )
    args.ambiente.write_text(
        json.dumps(collect_environment(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{total} execução(ões) gravadas em {args.output}; ambiente em {args.ambiente}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
