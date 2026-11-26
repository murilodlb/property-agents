"""Thin MCP -> LangChain adapter built on ``fastmcp.Client`` (stdio).

``langchain-mcp-adapters`` does not support the MCP SDK 2.x used by FastMCP 4, so this
module converts the server tools into ``StructuredTool`` objects directly. A single stdio
session is kept open for the whole benchmark run (the catalog is loaded only once).

Error contract: tool failures (invalid arguments, unknown ids, validation errors) are
never raised to the graph. They are returned to the LLM as a clear ``ERRO`` message so the
ReAct loop can recover (requirement 2.2 of the assignment).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fastmcp
from fastmcp.client.transports import PythonStdioTransport
from langchain_core.tools import BaseTool, StructuredTool

from src.agents.config import DEFAULT_SERVER_SCRIPT

ERROR_PREFIX = "ERRO"


def _strip_nulls(schema: Any) -> Any:
    """Simplify ``anyOf: [{type: X}, {type: null}]`` into ``{type: X}`` for small models."""
    if isinstance(schema, dict):
        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            non_null = [s for s in any_of if not (isinstance(s, dict) and s.get("type") == "null")]
            if len(non_null) == 1 and len(non_null) < len(any_of):
                merged = {k: v for k, v in schema.items() if k != "anyOf"}
                merged.update(non_null[0])
                schema = merged
        return {k: _strip_nulls(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_strip_nulls(s) for s in schema]
    return schema


def _result_to_text(result: Any) -> tuple[str, bool]:
    """Serialize a FastMCP ``CallToolResult`` into text and flag logical errors."""
    texts = [getattr(c, "text", "") for c in (result.content or []) if getattr(c, "text", None)]
    if result.is_error:
        return "\n".join(texts) or "erro desconhecido", True
    data = result.data if result.data is not None else result.structured_content
    if data is not None:
        is_logical_error = isinstance(data, dict) and "erro" in data
        return json.dumps(data, ensure_ascii=False, default=str), is_logical_error
    return "\n".join(texts), False


class MCPToolkit:
    """Async context manager exposing the MCP server as LangChain tools, resource and prompt."""

    def __init__(self, server_script: str | Path = DEFAULT_SERVER_SCRIPT) -> None:
        self.server_script = Path(server_script)
        self._client: fastmcp.Client[Any] | None = None
        self.tools: list[BaseTool] = []
        self.tool_errors: int = 0
        self.repeated_calls: int = 0
        self._call_cache: dict[str, str] = {}

    def reset_cache(self) -> None:
        """Forget previous results (called at the start of every ``answer``)."""
        self._call_cache = {}

    async def __aenter__(self) -> MCPToolkit:
        transport = PythonStdioTransport(script_path=self.server_script)
        self._client = fastmcp.Client(transport)
        await self._client.__aenter__()
        raw_tools = await self._client.list_tools()
        self.tools = [self._make_tool(t) for t in raw_tools]
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None
        self.tools = []

    @property
    def client(self) -> fastmcp.Client[Any]:
        if self._client is None:
            raise RuntimeError("MCPToolkit não conectado; use 'async with MCPToolkit()'.")
        return self._client

    def _make_tool(self, mcp_tool: Any) -> BaseTool:
        name: str = mcp_tool.name
        schema = getattr(mcp_tool, "input_schema", None) or {"type": "object", "properties": {}}
        schema = _strip_nulls(dict(schema))
        schema.pop("additionalProperties", None)
        schema.setdefault("title", name)
        schema.setdefault("description", mcp_tool.description or name)

        async def _call(**kwargs: Any) -> str:
            # Drop explicit nulls produced by small models for optional parameters.
            args = {k: v for k, v in kwargs.items() if v is not None}
            key = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            if key in self._call_cache:
                # Loop guard: small models tend to repeat the same failing call forever.
                self.repeated_calls += 1
                return (
                    "AVISO: esta ferramenta já foi chamada com exatamente estes argumentos "
                    "nesta tarefa. Não repita a chamada; use o resultado abaixo e escreva a "
                    f"resposta.\nResultado anterior: {self._call_cache[key]}"
                )
            text = await self._invoke(name, args)
            self._call_cache[key] = text
            return text

        return StructuredTool(
            name=name,
            description=mcp_tool.description or name,
            args_schema=schema,
            coroutine=_call,
        )

    async def _invoke(self, name: str, args: dict[str, Any]) -> str:
        """Call the MCP tool, converting any failure into a recoverable ERRO message."""
        try:
            result = await self.client.call_tool(name, args, raise_on_error=False)
        except Exception as exc:  # transport/protocol failure: still recoverable by LLM
            self.tool_errors += 1
            return f"{ERROR_PREFIX} ao chamar '{name}': {exc}. Revise os argumentos."
        text, is_error = _result_to_text(result)
        if is_error:
            self.tool_errors += 1
            return (
                f"{ERROR_PREFIX} retornado pela ferramenta '{name}' com argumentos "
                f"{json.dumps(args, ensure_ascii=False)}: {text}\n"
                "Corrija os argumentos e tente novamente, ou informe ao usuário que o "
                "dado não existe no estoque."
            )
        return text

    def select(self, names: Iterable[str]) -> list[BaseTool]:
        """Return the subset of tools with the given names (keeps the requested order)."""
        by_name = {t.name: t for t in self.tools}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"Ferramentas MCP inexistentes: {missing}")
        return [by_name[n] for n in names]

    async def read_resource(self, uri: str = "estoque://esquema") -> str:
        """Read a text MCP resource (used to ground system prompts on the catalog schema)."""
        contents = await self.client.read_resource(uri)
        return "\n".join(getattr(c, "text", "") or "" for c in contents)

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        """Render an MCP prompt template into plain text."""
        result = await self.client.get_prompt(name, arguments)
        parts: list[str] = []
        for msg in result.messages:
            content = msg.content
            parts.append(getattr(content, "text", None) or str(content))
        return "\n".join(parts)
