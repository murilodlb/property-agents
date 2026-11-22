# Property Agents — Benchmark Imobiliário RN com MCP e Multiagentes

Projeto de avaliação comparativa de arquiteturas de agentes (**LangGraph**) atuando sobre um estoque imobiliário real do Rio Grande do Norte (RN) via **Model Context Protocol (MCP)**.

## 1. Visão Geral e Estrutura do Repositório

```text
property-agents/
├── data/
│   ├── Lista_imoveis_RN.csv          # Catálogo oficial da Caixa Econômica Federal (RN)
│   ├── agencies_snapshot.csv         # Snapshot offline dos anúncios das imobiliárias locais
│   ├── scrape_real_properties.py     # Pipeline reproduzível de coleta, limpeza e deduplicação
│   └── real_properties.csv           # Base consolidada e limpa (801 imóveis reais únicos)
├── mcp_server/
│   └── server.py                     # Servidor FastMCP ('imoveis-rn') com 6 ferramentas PT-BR
├── src/
│   ├── catalog.py                    # Fonte única de verdade: carga, limpeza e P/VP mediano
│   ├── questions.py                  # Suíte de 20 perguntas PT-BR + checagem_automatica
│   ├── schema.py                     # Contratos Pydantic v2 (executions.jsonl, juiz e CSV)
│   ├── judge.py                      # Juiz Frontier com acesso read-only ao servidor MCP
│   ├── metrics.py                    # Pipeline das 60 execuções (automático + Juiz MCP)
│   └── report.py                     # Relatórios consolidados e CSV de auditoria manual
├── tests/
│   ├── test_catalog_and_mcp.py       # Testes unitários do catálogo e das 6 ferramentas MCP
│   ├── test_judge_calibration.py     # Suíte de calibração (10 casos controlados + MCP real)
│   └── test_metrics_pipeline.py      # Testes ponta a ponta de JSONL, métricas, resume e report
├── .env.example                      # Modelo de variáveis de ambiente para o Juiz Frontier
├── pyproject.toml                    # Dependências (uv), pytest, ruff e pyright
└── README.md
```

---

## 2. Catálogo Real do RN (`data/real_properties.csv`)

- **Total de registros:** **801 imóveis reais únicos** (`616` da Caixa + `185` de imobiliárias locais de Natal e Região Metropolitana).
- **Cobertura:** `100%` dos imóveis possuem `area_m2 > 0` e `price_per_m2` válido. Terrenos possuem `bedrooms=0`, `bathrooms=0` e `garage_spots=0`; colunas mortas (`condo_fee_brl`, `iptu_brl`) foram removidas e `accepts_financing` preserva `True`, `False` (Caixa) ou `<NA>` (imobiliárias).
- **Metodologia de Valuation (Mediana Hierárquica em 4 Níveis, `MIN_GROUP_SIZE = 5`):**
  1. `bairro_segmento`: `(segmento_mercado, cidade, bairro, tipo_imovel)`
  2. `cidade_segmento`: `(segmento_mercado, cidade, tipo_imovel)`
  3. `cidade_tipo`: `(cidade, tipo_imovel)`
  4. `estado_tipo`: mediana estadual por `tipo_imovel`

Para reconstruir `data/real_properties.csv` de forma determinística e offline:

```bash
uv run python data/scrape_real_properties.py \
  --mode combined \
  --caixa-file data/Lista_imoveis_RN.csv \
  --agencies-csv data/agencies_snapshot.csv \
  --output data/real_properties.csv
```

---

## 3. Ferramentas Read-Only do Servidor MCP (`mcp_server/server.py`)

O servidor `FastMCP("imoveis-rn")` expõe **6 ferramentas 100% em português (PT-BR)**, 1 recurso (`estoque://esquema`) e 1 prompt (`recomendar_imovel`):

| Ferramenta | Finalidade |
|---|---|
| `buscar_imoveis` | Filtra por preço máximo, tipo, cidade, bairro, fonte, segmento (`caixa` / `imobiliaria`), quartos mínimos e financiamento; ordena e retorna o total + top 5. |
| `detalhes_imovel` | Retorna a ficha cadastral completa e os indicadores de valuation de um `imovel_id`. |
| `comparar_imoveis` | Compara de 2 a 4 imóveis lado a lado e já calcula os vencedores em 8 critérios objetivos. |
| `avaliar_investimento` | Calcula P/VP Caixa, P/VP de mercado (mediana hierárquica), economia/deságio, entrada de 20% e ITBI/registro estimado (4,5%). |
| `simular_compra_financiamento` | Simula financiamento **SAC** ou **PRICE** (entrada, parcelas, juros, desembolso inicial, ITBI/registro e comissão de leilão 5%) e bloqueia imóveis com `aceita_financiamento=False`. |
| `estatisticas_mercado` | Agrega estatísticas (`min`, `mediana`, `média`, `máximo` e contagens) sobre **todo** o estoque filtrado (sem corte de 5 linhas). |

Inspeção via CLI:

```bash
uv run fastmcp list mcp_server/server.py
uv run fastmcp call mcp_server/server.py buscar_imoveis '{"cidade":"Natal","tipo_imovel":"casa","preco_maximo":350000}'
```

---

## 4. Contrato de Integração para os Agentes (`executions.jsonl`)

Todos os notebooks/agentes da equipe (**LangGraph / CrewAI**) devem gravar suas **20 execuções por arquitetura** (ex.: `langgraph_unico`, `langgraph_verificador`, `langgraph_supervisor` $\rightarrow$ `60` linhas no total) em `results/executions.jsonl`, seguindo exatamente o schema `ExecutionRecord` em `src/schema.py`:

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `sistema` | `str` | Sim | Identificador da arquitetura (ex.: `"langgraph_unico"`, `"langgraph_verificador"`, `"langgraph_supervisor"`). |
| `pergunta_id` | `int` (`1..20`) | Sim | ID da pergunta em `PERGUNTAS` (`src/questions.py`). |
| `resposta` | `str` | Sim | Resposta final em PT-BR retornada pelo agente (deve citar `imovel_id`, ex.: `r001`). |
| `ferramentas_usadas` | `list[str]` | Sim (default `[]`) | Lista ordenada dos nomes das ferramentas MCP chamadas durante a execução. |
| `chamadas_modelo` | `int` (`>= 0`) | Sim | Número total de invocações ao LLM realizadas pelo grafo/equipe. |
| `latencia_segundos` | `float` (`>= 0.0`) | Sim | Tempo total de execução fim a fim em segundos. |
| `tokens_entrada` | `int \| None` | Opcional | Total de tokens de entrada consumidos (`None` se indisponível). |
| `tokens_saida` | `int \| None` | Opcional | Total de tokens de saída gerados (`None` se indisponível). |
| `modelo_agente` | `str \| None` | Opcional | Nome do modelo usado pelo agente (ex.: `"qwen3:8b"`). |
| `erro_execucao` | `str \| None` | Opcional | Mensagem de erro caso a execução tenha falhado/dado timeout (`None` quando ok). |
| `timestamp_utc` | `str` | Automático | Timestamp ISO-8601 UTC preenchido automaticamente pelo `ExecutionRecord`. |

### Exemplo de linha válida em `results/executions.jsonl`

```json
{"sistema":"langgraph_unico","pergunta_id":1,"resposta":"A casa mais barata em Natal até R$ 350 mil é a r471 (R$ 85.466,19).","ferramentas_usadas":["buscar_imoveis"],"chamadas_modelo":2,"latencia_segundos":1.84,"tokens_entrada":420,"tokens_saida":95,"modelo_agente":"qwen3:8b","erro_execucao":null,"timestamp_utc":"2026-11-22T17:00:00+00:00"}
```

### Trecho pronto para uso nos notebooks (LangGraph)

```python
import time
from src.questions import PERGUNTAS, SYSTEM_PROMPT_PTBR
from src.schema import ExecutionRecord, append_execution_jsonl

OUTPUT_JSONL = "results/executions.jsonl"

for item in PERGUNTAS:
    t0 = time.perf_counter()
    try:
        # Chame aqui seu agente passando SYSTEM_PROMPT_PTBR + item["q"]
        resposta, tools_chamadas, n_chamadas, tok_in, tok_out = executar_meu_agente(
            system_prompt=SYSTEM_PROMPT_PTBR,
            pergunta=item["q"],
        )
        erro = None
    except Exception as exc:
        resposta, tools_chamadas, n_chamadas, tok_in, tok_out = "", [], 0, None, None
        erro = f"{type(exc).__name__}: {exc}"

    registro = ExecutionRecord(
        sistema="langgraph_unico",
        pergunta_id=item["id"],
        resposta=resposta,
        ferramentas_usadas=tools_chamadas,
        chamadas_modelo=n_chamadas,
        latencia_segundos=round(time.perf_counter() - t0, 3),
        tokens_entrada=tok_in,
        tokens_saida=tok_out,
        modelo_agente="qwen3:8b",
        erro_execucao=erro,
    )
    append_execution_jsonl(OUTPUT_JSONL, registro)
```

---

## 5. Como Executar a Avaliação, Calibração e Relatórios

### 1. Instalar dependências com `uv`

```bash
uv sync
cp .env.example .env
```

### 2. Teste rápido offline (sem gastar cota de API)

Valida o `results/executions.jsonl`, calcula `checagem_automatica`, contagem de chamadas e gera `results/benchmark_60_execucoes.csv`:

```bash
uv run python -m src.metrics --input results/executions.jsonl --output results/benchmark_60_execucoes.csv --skip-judge
```

### 3. Avaliação completa com Juiz Frontier (Gemini + MCP Read-Only)

Por padrão, o juiz usa **Gemini (`gemini-2.5-flash`)** com intervalo mínimo de `4.0s` entre chamadas e suporte a `--resume` (salva cada linha imediatamente no CSV para retomar caso a conexão caia):

```bash
export GEMINI_API_KEY="sua_chave_aqui"
uv run python -m src.metrics --input results/executions.jsonl --output results/benchmark_60_execucoes.csv --resume
```

Fallbacks para outro modelo **não-Qwen** (evitando viés de autopreferência):

```bash
# Groq (OpenAI-compatible)
export JUDGE_PROVIDER=openai_compatible
export JUDGE_BASE_URL=https://api.groq.com/openai/v1
export JUDGE_API_KEY="sua_chave_groq"
uv run python -m src.metrics --provider openai_compatible --model llama-3.3-70b-versatile --delay 2.0 --resume

# Ollama local (Gemma 3 ou Llama 3.1)
uv run python -m src.metrics --provider ollama --model gemma3:12b --delay 0.0 --resume
```

> **Política anti-viés:** modelos `qwen*` são bloqueados como juiz por padrão (`allow_qwen_judge=False`).

### 4. Gerar relatório consolidado e planilha de auditoria manual

```bash
uv run python -m src.report --input results/benchmark_60_execucoes.csv --out-dir results/
```

Gera:
- `results/benchmark_summary.md` (Tabelas de resumo por sistema, matriz Categoria × Sistema e distribuição de vereditos + Kappa de Cohen)
- `results/divergencias_auditoria_manual.csv` (casos em que automático e juiz divergem ou veredito != `correta`, com colunas `revisao_manual` e `nota_manual`)

### 5. Calibração do Juiz e Suíte de Testes

```bash
# Testes automatizados offline (incluindo servidor MCP real com juiz mockado)
uv run pytest -m "not live_judge" -v

# Calibração ao vivo nos 10 casos controlados (4 positivos + 6 pegadinhas/alucinações)
GEMINI_API_KEY="sua_chave_aqui" uv run python tests/test_judge_calibration.py --live --provider gemini
```
