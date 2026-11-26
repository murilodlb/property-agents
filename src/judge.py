"""Read-only MCP Frontier Judge with multi-provider support and rate-limit resilience."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import fastmcp
import requests
from fastmcp.client.transports import PythonStdioTransport

from src.schema import ExecutionRecord, JudgeToolCall, JudgeVerdict, VereditoDetalhado

READ_ONLY_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "buscar_imoveis",
        "detalhes_imovel",
        "comparar_imoveis",
        "avaliar_investimento",
        "simular_compra_financiamento",
        "estatisticas_mercado",
    }
)

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-4-hydrogen",
    "openai_compatible": "llama-3.3-70b-versatile",
    "ollama": "gemma3:12b",
}

DEFAULT_DELAYS: dict[str, float] = {
    "gemini": 4.0,
    "openai_compatible": 0.5,
    "ollama": 0.0,
}

JUDGE_SYSTEM_PROMPT_PTBR = (
    "Você é um auditor independente e rigoroso de um benchmark imobiliário no Rio Grande "
    "do Norte (RN). Sua função é verificar se a resposta do agente avaliado está correta "
    "consultando diretamente as ferramentas read-only do servidor MCP.\n\n"
    "Regras obrigatórias de auditoria:\n"
    "1. Sempre que o agente citar um imovel_id, preço, desconto, área, P/VP, simulação "
    "SAC/PRICE, ranking ou estatística de mercado, execute pelo menos 1 consulta nas "
    "ferramentas MCP disponíveis para comprovar os fatos diretamente no estoque real.\n"
    "2. Para perguntas sobre códigos inexistentes (ex.: r9999) ou atributos ausentes no "
    "catálogo (condomínio, IPTU, posição solar, poço artesiano, permuta), verifique no "
    "MCP e aprove somente se o agente recusar sem inventar dados.\n"
    "3. Aprove como 'correta' (correta=true) pequenas diferenças de arredondamento "
    "ou paráfrases fiéis aos dados.\n"
    "4. Reprove como 'incorreta' ou 'alucinacao' (correta=false) quando houver ranking "
    "invertido, erro de cálculo SAC/PRICE/P-VP, recomendação de imóvel não financiável "
    "(aceita_financiamento=False) em cenários de financiamento, confusão entre ágio e "
    "deságio, ou invenção de imóveis/atributos inexistentes. Use 'parcial' (correta=false) "
    "quando acertar parte relevante mas omitir ou errar requisito importante.\n"
    "5. Responda ao final EXCLUSIVAMENTE com um objeto JSON válido (sem texto fora do JSON) "
    "no formato:\n"
    '{"correta": true, "veredito_detalhado": "correta", '
    '"justificativa": "Justificativa concisa em PT-BR citando os números/fatos '
    'verificados no MCP."}\n'
    'Valores válidos para "veredito_detalhado": "correta", "parcial", "incorreta", "alucinacao".'
)


@dataclass
class LLMToolRequest:
    """Normalized tool call requested by any LLM provider backend."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class LLMStepResponse:
    """Normalized response from one LLM step (either tool calls or final text)."""

    content: str
    tool_calls: list[LLMToolRequest]
    raw_parts: list[Any] | None = None


class JudgeLLMBackend(Protocol):
    """Provider-agnostic async interface for one step of tool-calling generation."""

    async def generate_step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMStepResponse: ...


def _load_dotenv(env_path: Path | None = None) -> None:
    """Load key=value pairs from ``.env`` files into ``os.environ`` without overwriting."""
    if env_path is not None:
        candidates = [env_path]
    else:
        cwd_env = (Path.cwd() / ".env").resolve()
        repo_env = (Path(__file__).resolve().parent.parent / ".env").resolve()
        candidates = [cwd_env] if cwd_env == repo_env else [cwd_env, repo_env]

    for path in candidates:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, raw_val = line.split("=", 1)
            key = key.strip()
            raw_val = raw_val.strip()
            if not key:
                continue
            if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in ("'", '"'):
                val = raw_val[1:-1]
            else:
                val = re.split(r"\s+#", raw_val, maxsplit=1)[0].strip()
            if not os.environ.get(key):
                os.environ[key] = val


@dataclass
class JudgeConfig:
    """Configuration for the cross-family read-only MCP Frontier Judge."""

    provider: Literal["gemini", "openai_compatible", "ollama"] = "gemini"
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    ollama_url: str = "http://localhost:11434"
    min_delay_seconds: float | None = None
    max_steps: int = 6
    max_retries: int = 4
    allow_qwen_judge: bool = False
    server_script: str | Path | None = None
    server_env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            self.model = DEFAULT_MODELS[self.provider]
        else:
            self.model = self.model.strip()

        if self.min_delay_seconds is None:
            self.min_delay_seconds = DEFAULT_DELAYS[self.provider]

        if self.model.lower().startswith("qwen"):
            if not self.allow_qwen_judge:
                raise ValueError(
                    f"Modelo '{self.model}' bloqueado como juiz: para evitar viés de "
                    f"autopreferência sobre agentes Qwen, utilize outra família de modelo "
                    f"(ex.: Gemini, Llama ou Gemma) ou defina allow_qwen_judge=True."
                )
            warnings.warn(
                f"Usando modelo Qwen ('{self.model}') como juiz por override explícito; "
                f"há risco de viés de autopreferência.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_env(
        cls,
        provider: str | None = None,
        model: str | None = None,
        min_delay_seconds: float | None = None,
        allow_qwen_judge: bool = False,
        server_script: str | Path | None = None,
        server_env: dict[str, str] | None = None,
    ) -> JudgeConfig:
        """Build a ``JudgeConfig`` from explicit overrides and environment variables."""
        _load_dotenv()
        raw_prov = (provider or os.getenv("JUDGE_PROVIDER") or "gemini").strip().lower()
        if raw_prov not in ("gemini", "openai_compatible", "ollama"):
            raise ValueError(
                f"Provedor de juiz inválido: {raw_prov!r}. "
                f"Use 'gemini', 'openai_compatible' ou 'ollama'."
            )
        typed_prov: Literal["gemini", "openai_compatible", "ollama"] = raw_prov  # type: ignore[assignment]

        env_model = (model or os.getenv("JUDGE_MODEL") or DEFAULT_MODELS[typed_prov]).strip()
        env_delay: float | None = min_delay_seconds
        if env_delay is None:
            raw_delay = os.getenv("JUDGE_MIN_DELAY_SECONDS")
            env_delay = float(raw_delay) if raw_delay else DEFAULT_DELAYS[typed_prov]

        api_key = (
            os.getenv("GEMINI_API_KEY")
            if typed_prov == "gemini"
            else os.getenv("JUDGE_API_KEY")
        )
        if not api_key and typed_prov == "openai_compatible":
            api_key = os.getenv("OPENAI_API_KEY")

        return cls(
            provider=typed_prov,
            model=env_model,
            api_key=api_key,
            base_url=os.getenv("JUDGE_BASE_URL"),
            ollama_url=os.getenv("OLLAMA_URL") or "http://localhost:11434",
            min_delay_seconds=env_delay,
            allow_qwen_judge=allow_qwen_judge,
            server_script=server_script,
            server_env=server_env,
        )


def _strip_additional_properties(schema: Any) -> Any:
    """Recursively clean JSON schema nodes for strict provider tool-declaration parsers."""
    if isinstance(schema, dict):
        return {
            k: _strip_additional_properties(v)
            for k, v in schema.items()
            if k != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_strip_additional_properties(item) for item in schema]
    return schema


def _build_gemini_contents(messages: list[dict[str, Any]]) -> list[Any]:
    """Convert normalized chat history into ``google.genai`` ``Content`` turns."""
    from google.genai import types

    contents: list[types.Content] = []
    for msg in messages:
        role = msg["role"]
        if role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=str(msg["content"]))],
                )
            )
        elif role == "assistant":
            raw_parts = msg.get("raw_parts")
            if raw_parts:
                contents.append(types.Content(role="model", parts=list(raw_parts)))
            else:
                parts: list[types.Part] = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=str(msg["content"])))
                for tc in msg.get("tool_calls", []):
                    parts.append(
                        types.Part.from_function_call(
                            name=tc["name"], args=tc.get("arguments", {})
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            safe_content = json.loads(
                json.dumps(msg["content"], ensure_ascii=False, default=str)
            )
            fn_resp = types.Part.from_function_response(
                name=msg["name"],
                response={"result": safe_content},
            )
            if (
                contents
                and contents[-1].role == "user"
                and contents[-1].parts
                and all(
                    getattr(p, "function_response", None) is not None
                    for p in contents[-1].parts
                )
            ):
                contents[-1].parts.append(fn_resp)
            else:
                contents.append(types.Content(role="user", parts=[fn_resp]))
    return contents


class GeminiJudgeBackend:
    """Google GenAI (Gemini) tool-calling backend using ``google-genai``."""

    def __init__(self, config: JudgeConfig) -> None:
        from google import genai

        api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            _load_dotenv()
            api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Chave GEMINI_API_KEY não configurada para o provedor 'gemini'."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = config.model

    async def generate_step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMStepResponse:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters_json_schema=_strip_additional_properties(
                    tool.get("input_schema", {"type": "object", "properties": {}})
                ),
            )
            for tool in tools
        ]
        genai_tools = [types.Tool(function_declarations=declarations)] if declarations else None

        contents = _build_gemini_contents(messages)

        gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            tools=genai_tools,  # type: ignore[arg-type]
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,  # type: ignore[arg-type]
            config=gen_config,
        )

        raw_parts: list[Any] = []
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "content", None) is not None:
            raw_parts = list(getattr(candidates[0].content, "parts", None) or [])

        text_chunks: list[str] = []
        for part in raw_parts:
            part_text = getattr(part, "text", None)
            if part_text and not getattr(part, "thought", False):
                text_chunks.append(str(part_text))

        tool_calls: list[LLMToolRequest] = []
        if response.function_calls:
            for idx, fc in enumerate(response.function_calls):
                if fc.name:
                    tool_calls.append(
                        LLMToolRequest(
                            name=str(fc.name),
                            arguments=dict(fc.args or {}),
                            call_id=f"gemini_call_{idx}",
                        )
                    )

        content_text = "".join(text_chunks)
        if not content_text and not tool_calls:
            content_text = response.text or ""
        return LLMStepResponse(
            content=content_text,
            tool_calls=tool_calls,
            raw_parts=raw_parts or None,
        )


class OpenAICompatibleJudgeBackend:
    """OpenAI-compatible tool-calling backend (Groq, OpenRouter, or OpenAI)."""

    def __init__(self, config: JudgeConfig) -> None:
        from openai import AsyncOpenAI

        api_key = config.api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            _load_dotenv()
            api_key = (
                config.api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
            )
        if not api_key:
            raise ValueError(
                "Chave JUDGE_API_KEY (ou OPENAI_API_KEY) não configurada para "
                "o provedor 'openai_compatible'."
            )
        self._client = AsyncOpenAI(api_key=api_key, base_url=config.base_url)
        self._model = config.model

    async def generate_step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMStepResponse:
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"]
            if role == "user":
                oai_messages.append({"role": "user", "content": str(msg["content"])})
            elif role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content") or None,
                }
                if msg.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": tc.get("call_id") or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(
                                    tc.get("arguments", {}), ensure_ascii=False
                                ),
                            },
                        }
                        for i, tc in enumerate(msg["tool_calls"])
                    ]
                oai_messages.append(entry)
            elif role == "tool":
                oai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("call_id") or "call_0",
                        "content": json.dumps(msg["content"], ensure_ascii=False),
                    }
                )

        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "temperature": 0.0,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls: list[LLMToolRequest] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                func = getattr(tc, "function", None)
                if func is None:
                    continue
                raw_args = func.arguments or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {}
                tool_calls.append(
                    LLMToolRequest(
                        name=str(func.name),
                        arguments=parsed_args if isinstance(parsed_args, dict) else {},
                        call_id=str(tc.id),
                    )
                )
        return LLMStepResponse(content=message.content or "", tool_calls=tool_calls)


class OllamaJudgeBackend:
    """Local Ollama ``/api/chat`` tool-calling backend for non-Qwen models."""

    def __init__(self, config: JudgeConfig) -> None:
        self._base_url = config.ollama_url.rstrip("/")
        self._model = config.model

    async def generate_step(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMStepResponse:
        ollama_msgs: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"]
            if role == "user":
                ollama_msgs.append({"role": "user", "content": str(msg["content"])})
            elif role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content", ""),
                }
                if msg.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "function": {
                                "name": tc["name"],
                                "arguments": tc.get("arguments", {}),
                            }
                        }
                        for tc in msg["tool_calls"]
                    ]
                ollama_msgs.append(entry)
            elif role == "tool":
                ollama_msgs.append(
                    {
                        "role": "tool",
                        "content": json.dumps(msg["content"], ensure_ascii=False),
                    }
                )

        ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": ollama_msgs,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        def _post() -> dict[str, Any]:
            resp = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            return dict(resp.json())

        data = await asyncio.to_thread(_post)
        message = data.get("message", {})
        tool_calls: list[LLMToolRequest] = []
        for idx, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            if fn.get("name"):
                tool_calls.append(
                    LLMToolRequest(
                        name=str(fn["name"]),
                        arguments=raw_args if isinstance(raw_args, dict) else {},
                        call_id=f"ollama_{idx}",
                    )
                )
        return LLMStepResponse(content=str(message.get("content") or ""), tool_calls=tool_calls)


def _create_backend(config: JudgeConfig) -> JudgeLLMBackend:
    """Instantiate the configured LLM backend."""
    if config.provider == "gemini":
        return GeminiJudgeBackend(config)
    if config.provider == "openai_compatible":
        return OpenAICompatibleJudgeBackend(config)
    if config.provider == "ollama":
        return OllamaJudgeBackend(config)
    raise ValueError(f"Provedor não suportado: {config.provider!r}")


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for transient rate-limit or server-overload exceptions."""
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = ("429", "503", "502", "rate", "quota", "resourceexhausted", "timeout", "unavailable")
    return any(m in text for m in markers)


def _extract_json_block(text: str) -> dict[str, Any]:
    """Parse JSON from a raw model response, tolerating markdown fences and surrounding text."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Resposta JSON do juiz não é um objeto.")
    return parsed


def _parse_verdict_json(
    raw_text: str,
    consultas: list[JudgeToolCall],
    config: JudgeConfig,
) -> JudgeVerdict:
    """Normalize LLM text output into a validated ``JudgeVerdict``."""
    try:
        payload = _extract_json_block(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        return JudgeVerdict(
            correta=False,
            veredito_detalhado="incorreta",
            justificativa=(
                f"Falha ao interpretar veredito JSON do juiz ({exc}). "
                f"Saída bruta: {raw_text[:200]}"
            ),
            consultas_realizadas=consultas,
            provedor_juiz=config.provider,
            modelo_juiz=config.model,
        )

    raw_verdict = str(payload.get("veredito_detalhado", "")).strip().lower()
    raw_correta = bool(payload.get("correta", False))

    valid_verdicts: tuple[VereditoDetalhado, ...] = (
        "correta",
        "parcial",
        "incorreta",
        "alucinacao",
    )
    veredito: VereditoDetalhado
    if raw_verdict in valid_verdicts:
        veredito = raw_verdict  # type: ignore[assignment]
    else:
        veredito = "correta" if raw_correta else "incorreta"

    correta = veredito == "correta"
    justificativa = str(payload.get("justificativa") or "").strip()
    if not justificativa:
        justificativa = "Sem justificativa detalhada retornada pelo juiz."

    return JudgeVerdict(
        correta=correta,
        veredito_detalhado=veredito,
        justificativa=justificativa,
        consultas_realizadas=consultas,
        provedor_juiz=config.provider,
        modelo_juiz=config.model,
    )


def _summarize_tool_result(data: Any, limit: int = 240) -> str:
    """Create a compact single-line summary of an MCP tool output."""
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class MCPJudge:
    """Agentic Frontier Judge connected to the read-only ``imoveis-rn`` MCP server."""

    def __init__(
        self,
        config: JudgeConfig | None = None,
        backend: JudgeLLMBackend | None = None,
    ) -> None:
        self.config = config or JudgeConfig.from_env()
        self._backend: JudgeLLMBackend | None = backend
        self._client: fastmcp.Client[Any] | None = None
        self._tools: list[dict[str, Any]] = []
        self._last_llm_call_ts: float = 0.0

    def _get_backend(self) -> JudgeLLMBackend:
        if self._backend is None:
            self._backend = _create_backend(self.config)
        return self._backend

    def _default_server_script(self) -> Path:
        if self.config.server_script is not None:
            return Path(self.config.server_script)
        return Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"

    async def __aenter__(self) -> MCPJudge:
        transport = PythonStdioTransport(
            script_path=self._default_server_script(),
            env=self.config.server_env,
        )
        self._client = fastmcp.Client(transport)
        await self._client.__aenter__()
        raw_tools = await self._client.list_tools()
        self._tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": (
                    getattr(t, "input_schema", None)
                    or getattr(t, "inputSchema", None)
                    or {"type": "object", "properties": {}}
                ),
            }
            for t in raw_tools
            if t.name in READ_ONLY_MCP_TOOLS
        ]
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None
            self._tools = []

    async def execute_readonly_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], JudgeToolCall]:
        """Execute a single MCP tool call only if ``name`` is in ``READ_ONLY_MCP_TOOLS``."""
        if name not in READ_ONLY_MCP_TOOLS:
            err_msg = f"Ferramenta '{name}' bloqueada pela política read-only do juiz."
            return (
                {"erro": err_msg},
                JudgeToolCall(
                    ferramenta=name,
                    argumentos=arguments,
                    resumo_resultado=err_msg,
                    erro=True,
                ),
            )
        if self._client is None:
            raise RuntimeError("MCPJudge precisa estar conectado (use 'async with MCPJudge(...)').")

        try:
            result = await self._client.call_tool(name, arguments)
            payload = result.data if result.data is not None else result.structured_content
            if payload is None and result.content:
                payload = {"texto": str(result.content[0])}
            if not isinstance(payload, dict):
                payload = {"resultado": payload}
            is_err = bool(getattr(result, "is_error", False))
            call_log = JudgeToolCall(
                ferramenta=name,
                argumentos=arguments,
                resumo_resultado=_summarize_tool_result(payload),
                erro=is_err,
            )
            return payload, call_log
        except Exception as exc:
            err_payload = {"erro": f"{type(exc).__name__}: {exc}"}
            return (
                err_payload,
                JudgeToolCall(
                    ferramenta=name,
                    argumentos=arguments,
                    resumo_resultado=_summarize_tool_result(err_payload),
                    erro=True,
                ),
            )

    async def _call_backend_with_backoff(
        self,
        messages: list[dict[str, Any]],
    ) -> LLMStepResponse:
        """Invoke the LLM backend while honoring ``min_delay_seconds`` and retrying rate limits."""
        delay = float(self.config.min_delay_seconds or 0.0)
        backend = self._get_backend()

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            if delay > 0.0 and self._last_llm_call_ts > 0.0:
                elapsed = time.monotonic() - self._last_llm_call_ts
                if elapsed < delay:
                    await asyncio.sleep(delay - elapsed)
            self._last_llm_call_ts = time.monotonic()
            try:
                return await backend.generate_step(
                    system_prompt=JUDGE_SYSTEM_PROMPT_PTBR,
                    messages=messages,
                    tools=self._tools,
                )
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_error(exc) or attempt + 1 >= self.config.max_retries:
                    raise
                backoff = min(60.0, 2.0 * (2**attempt))
                await asyncio.sleep(backoff)

        assert last_exc is not None
        raise last_exc

    def _build_evaluation_prompt(
        self,
        pergunta_dict: dict[str, Any],
        execution: ExecutionRecord,
        include_reference: bool,
    ) -> str:
        """Format the PT-BR evaluation case prompt for the judge."""
        parts = [
            f"Pergunta ID: {pergunta_dict['id']} (Categoria: {pergunta_dict['categoria']})",
            f"Pergunta do usuário:\n{pergunta_dict['q']}",
            f"Ferramenta(s) esperada(s): {pergunta_dict.get('ferramenta', 'N/A')}",
            f"Ferramentas chamadas pelo agente ({execution.sistema}): "
            f"{execution.ferramentas_usadas}",
            f"Resposta final do agente:\n{execution.resposta}",
        ]
        if include_reference and pergunta_dict.get("gabarito"):
            parts.append(
                "Gabarito de referência (use como guia auxiliar, mas confirme os fatos "
                f"no MCP read-only):\n{pergunta_dict['gabarito']}"
            )
        parts.append(
            "Use as ferramentas MCP disponíveis para auditar os fatos mencionados pelo agente "
            "e retorne seu veredito final exclusivamente em JSON."
        )
        return "\n\n".join(parts)

    async def evaluate(
        self,
        pergunta_dict: dict[str, Any],
        execution: ExecutionRecord,
        include_reference: bool = True,
    ) -> JudgeVerdict:
        """Audit one ``ExecutionRecord`` against the read-only MCP server."""
        if self._client is None:
            async with self:
                return await self.evaluate(
                    pergunta_dict, execution, include_reference=include_reference
                )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._build_evaluation_prompt(
                    pergunta_dict, execution, include_reference=include_reference
                ),
            }
        ]
        consultas: list[JudgeToolCall] = []

        for _step in range(self.config.max_steps):
            step_resp = await self._call_backend_with_backoff(messages)
            if not step_resp.tool_calls:
                return _parse_verdict_json(step_resp.content, consultas, self.config)

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": step_resp.content,
                "tool_calls": [
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "call_id": tc.call_id,
                    }
                    for tc in step_resp.tool_calls
                ],
            }
            if step_resp.raw_parts is not None:
                assistant_msg["raw_parts"] = step_resp.raw_parts
            messages.append(assistant_msg)
            for tc in step_resp.tool_calls:
                tool_result, call_log = await self.execute_readonly_tool(tc.name, tc.arguments)
                consultas.append(call_log)
                messages.append(
                    {
                        "role": "tool",
                        "name": tc.name,
                        "call_id": tc.call_id,
                        "content": tool_result,
                    }
                )

        return JudgeVerdict(
            correta=False,
            veredito_detalhado="incorreta",
            justificativa=(
                f"O juiz atingiu o limite máximo de {self.config.max_steps} passos sem "
                "emitir veredito JSON final."
            ),
            consultas_realizadas=consultas,
            provedor_juiz=self.config.provider,
            modelo_juiz=self.config.model,
        )
