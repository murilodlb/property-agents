"""Per-execution telemetry: LLM calls, tokens, tool trajectory and CPU/RAM/GPU sampling."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import psutil
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

_OLLAMA_NAMES = ("ollama",)


class ExecutionTracker(BaseCallbackHandler):
    """LangChain callback that aggregates usage across *all* agents of a system.

    It is passed in the ``config`` of the top-level graph invocation and propagated to
    every nested agent/LLM call, so tokens and calls are summed over all agents.
    """

    run_inline = True  # keep tool ordering deterministic in async graphs

    def __init__(self) -> None:
        self.chamadas_modelo = 0
        self.tokens_entrada = 0
        self.tokens_saida = 0
        self._tokens_disponiveis = False
        self.ferramentas_usadas: list[str] = []
        self.chamadas_por_agente: dict[str, int] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.chamadas_modelo += 1
        agente = str((metadata or {}).get("agente") or "desconhecido")
        self.chamadas_por_agente[agente] = self.chamadas_por_agente.get(agente, 0) + 1

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    self._tokens_disponiveis = True
                    self.tokens_entrada += int(usage.get("input_tokens", 0) or 0)
                    self.tokens_saida += int(usage.get("output_tokens", 0) or 0)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "desconhecida"
        self.ferramentas_usadas.append(str(name))

    @property
    def tokens(self) -> tuple[int | None, int | None]:
        if not self._tokens_disponiveis:
            return None, None
        return self.tokens_entrada, self.tokens_saida


def _find_ollama_processes() -> list[psutil.Process]:
    procs: list[psutil.Process] = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            argv0 = ((proc.info.get("cmdline") or [""])[0] or "").lower()
            if any(n in name or n in argv0 for n in _OLLAMA_NAMES):
                procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs


class _NvidiaSampler:
    """Optional NVIDIA GPU sampler (``pynvml``); silently disabled when unavailable."""

    def __init__(self) -> None:
        self.ok = False
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
            ]
            self.ok = bool(self._handles)
        except Exception:
            self.ok = False

    def sample(self) -> tuple[float, float] | None:
        if not self.ok:
            return None
        util = 0.0
        mem = 0.0
        for h in self._handles:
            util = max(util, float(self._nvml.nvmlDeviceGetUtilizationRates(h).gpu))
            mem += float(self._nvml.nvmlDeviceGetMemoryInfo(h).used) / 2**20
        return util, mem


@dataclass(slots=True)
class ResourceSnapshot:
    cpu_pct_medio: float | None = None
    cpu_pct_pico: float | None = None
    ram_mb_pico: float | None = None
    gpu_pct_medio: float | None = None
    gpu_mem_mb_pico: float | None = None


@dataclass
class ResourceMonitor:
    """Background sampler of CPU/RAM (this process tree + Ollama) and GPU (if NVIDIA).

    ``cpu_pct`` is normalized to the whole machine (0-100% = all logical cores busy), and
    ``ram_mb`` is the summed RSS of the Python process, its children (MCP server) and the
    Ollama server/runner processes — i.e. the memory footprint of the agent stack.
    """

    intervalo_s: float = 0.5
    _cpu: list[float] = field(default_factory=list)
    _ram: list[float] = field(default_factory=list)
    _gpu: list[float] = field(default_factory=list)
    _gpu_mem: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _procs: dict[int, psutil.Process] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._ncpu = psutil.cpu_count(logical=True) or 1
        self._gpu_sampler = _NvidiaSampler()

    def _refresh_procs(self) -> None:
        me = psutil.Process()
        candidates = [me, *me.children(recursive=True), *_find_ollama_processes()]
        for p in candidates:
            if p.pid not in self._procs:
                try:
                    p.cpu_percent(None)  # prime the counter
                    self._procs[p.pid] = p
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

    def _sample_once(self) -> None:
        self._refresh_procs()
        cpu = 0.0
        ram = 0.0
        for pid, p in list(self._procs.items()):
            try:
                cpu += p.cpu_percent(None)
                ram += p.memory_info().rss / 2**20
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._procs.pop(pid, None)
        self._cpu.append(cpu / self._ncpu)
        self._ram.append(ram)
        gpu = self._gpu_sampler.sample()
        if gpu is not None:
            self._gpu.append(gpu[0])
            self._gpu_mem.append(gpu[1])

    def _loop(self) -> None:
        while not self._stop.wait(self.intervalo_s):
            self._sample_once()

    def __enter__(self) -> ResourceMonitor:
        self._refresh_procs()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample_once()

    def snapshot(self) -> ResourceSnapshot:
        def avg(xs: list[float]) -> float | None:
            return round(sum(xs) / len(xs), 2) if xs else None

        def peak(xs: list[float]) -> float | None:
            return round(max(xs), 2) if xs else None

        return ResourceSnapshot(
            cpu_pct_medio=avg(self._cpu),
            cpu_pct_pico=peak(self._cpu),
            ram_mb_pico=peak(self._ram),
            gpu_pct_medio=avg(self._gpu),
            gpu_mem_mb_pico=peak(self._gpu_mem),
        )


class Stopwatch:
    def __enter__(self) -> Stopwatch:
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start
