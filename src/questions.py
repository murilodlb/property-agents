#!/usr/bin/env python3
"""20-question benchmark suite, dynamic ground truths, and automatic trajectory/regex grader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

CategoriaType = Literal[
    "busca_com_filtro",
    "consulta_especifica",
    "comparacao",
    "calculo_financeiro",
    "pegadinha",
    "fora_do_escopo",
]

PROP_ID_RE = re.compile(r"\bp\d{3}\b", re.IGNORECASE)


def _resolve_csv_path() -> Path:
    """Resolve ``properties.csv`` from the working directory or repository root."""
    candidates = [
        Path("data/properties.csv"),
        Path("properties.csv"),
        Path(__file__).resolve().parent.parent / "data" / "properties.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not locate data/properties.csv or properties.csv")


def _load_catalog() -> pd.DataFrame:
    """Load ``properties.csv`` and compute all 9 deterministic valuation columns."""
    catalog = pd.read_csv(_resolve_csv_path())
    catalog["price_per_m2"] = (catalog["price_brl"] / catalog["area_m2"]).round(2)
    group_ref_m2 = catalog.groupby(["city", "prop_type"])["price_per_m2"].transform("mean")
    catalog["vp_estimado_brl"] = (catalog["area_m2"] * group_ref_m2).round(2)
    catalog["p_vp"] = (catalog["price_brl"] / catalog["vp_estimado_brl"]).round(2)
    catalog["gross_annual_brl"] = (
        catalog["price_brl"] * (catalog["yield_rate_pct"] / 100.0)
    ).round(2)
    catalog["annual_fee_brl"] = catalog["monthly_fee_brl"] * 12
    catalog["noi_annual_brl"] = (
        catalog["gross_annual_brl"] - catalog["annual_fee_brl"]
    ).round(2)
    catalog["net_cap_rate_pct"] = (
        (catalog["noi_annual_brl"] / catalog["price_brl"]) * 100.0
    ).round(2)
    catalog["payback_years"] = (
        catalog["price_brl"] / catalog["noi_annual_brl"]
    ).round(2)
    return catalog


df: pd.DataFrame = _load_catalog()
_BY_ID: dict[str, dict[str, Any]] = {
    str(row["prop_id"]): row.to_dict() for _, row in df.iterrows()
}


def num(x: float | int | str) -> str:
    """Convert a decimal number into a regex accepting '.' or ',' as decimal separator."""
    s = f"{x:g}" if isinstance(x, float) else str(x)
    return s.replace(".", "[,.]")


def _row(prop_id: str) -> dict[str, Any]:
    """Fetch a single property dict by ``prop_id``."""
    return _BY_ID[prop_id]


def _build_perguntas() -> list[dict[str, Any]]:
    """Construct the 20 benchmark questions with dynamic ground-truth strings from ``df``."""
    # Pre-fetch dynamic rows used across gabaritos
    p082 = _row("p082")
    p081 = _row("p081")
    p109 = _row("p109")
    p199 = _row("p199")
    p204 = _row("p204")
    p322 = _row("p322")
    p321 = _row("p321")
    p004 = _row("p004")
    p166 = _row("p166")
    p092 = _row("p092")
    p003 = _row("p003")
    p061 = _row("p061")
    p297 = _row("p297")
    p318 = _row("p318")
    p206 = _row("p206")

    return [
        {
            "id": 1,
            "categoria": "busca_com_filtro",
            "q": "Quero uma casa para a família até R$ 350 mil. O que você recomenda?",
            "ferramenta": "search_properties",
            "deve": ["p082"],
            "filtros": {"prop_type": "casa", "max_price": 350000},
            "gabarito": (
                f"Top 3 casas até R$ 350 mil (ordenadas por entrega mais recente e menor "
                f"distância da praia): 1º {p082['prop_id']} ({p082['title']}, {p082['city']}, "
                f"ano {p082['year_delivery']}, {p082['dist_beach_km']} km da praia, "
                f"R$ {p082['price_brl']}, taxa R$ {p082['monthly_fee_brl']}/mês, yield "
                f"{p082['yield_rate_pct']}%, P/VP {p082['p_vp']}); 2º {p081['prop_id']} "
                f"({p081['title']}, {p081['city']}, ano {p081['year_delivery']}, "
                f"{p081['dist_beach_km']} km, R$ {p081['price_brl']}, taxa "
                f"R$ {p081['monthly_fee_brl']}/mês, yield {p081['yield_rate_pct']}%, P/VP "
                f"{p081['p_vp']}); 3º {p109['prop_id']} ({p109['title']}, {p109['city']}, "
                f"ano {p109['year_delivery']}, {p109['dist_beach_km']} km, "
                f"R$ {p109['price_brl']})."
            ),
        },
        {
            "id": 2,
            "categoria": "busca_com_filtro",
            "q": (
                "Procuro um terreno até R$ 150 mil para investir. "
                "Qual é o mais recente do estoque?"
            ),
            "ferramenta": "search_properties",
            "deve": ["p199", "2026"],
            "filtros": {"prop_type": "terreno", "max_price": 150000},
            "gabarito": (
                f"O terreno mais recente até R$ 150 mil é o {p199['prop_id']} "
                f"({p199['title']}, construtora {p199['builder']}, {p199['city']}), com "
                f"entrega em {p199['year_delivery']}, a {p199['dist_beach_km']} km da praia, "
                f"preço de R$ {p199['price_brl']}, área de {p199['area_m2']} m², documentação "
                f"{p199['documentation']}, yield bruto de {p199['yield_rate_pct']}% e P/VP "
                f"{p199['p_vp']}. O segundo mais recente é o {p204['prop_id']} "
                f"(entrega em {p204['year_delivery']}, R$ {p204['price_brl']}, posse)."
            ),
        },
        {
            "id": 3,
            "categoria": "busca_com_filtro",
            "q": (
                "Quais apartamentos escriturados existem em Natal até R$ 300 mil? "
                "Cite o mais novo."
            ),
            "ferramenta": "search_properties",
            "deve": ["p322"],
            "filtros": {
                "prop_type": "apartamento",
                "city": "Natal",
                "max_price": 300000,
                "documentation": "escriturado",
            },
            "gabarito": (
                f"Existem 3 apartamentos escriturados em Natal até R$ 300 mil: o mais novo é o "
                f"{p322['prop_id']} ({p322['title']}, {p322['builder']}, ano "
                f"{p322['year_delivery']}, {p322['dist_beach_km']} km da praia, "
                f"R$ {p322['price_brl']}, {p322['area_m2']} m², yield {p322['yield_rate_pct']}%, "
                f"P/VP {p322['p_vp']}), seguido por {p321['prop_id']} ({p321['title']}, ano "
                f"{p321['year_delivery']}, R$ {p321['price_brl']}) e {p004['prop_id']} "
                f"({p004['title']}, ano {p004['year_delivery']}, R$ {p004['price_brl']})."
            ),
        },
        {
            "id": 4,
            "categoria": "busca_com_filtro",
            "q": "Qual é o imóvel com maior retorno bruto (yield_rate_pct) em Parnamirim?",
            "ferramenta": "search_properties",
            "deve": ["p166|p092", "11([,.]0)?"],
            "filtros": {"city": "Parnamirim"},
            "gabarito": (
                f"Em Parnamirim, há empate no maior yield bruto ({p166['yield_rate_pct']}%) "
                f"entre {p166['prop_id']} ({p166['title']}, {p166['builder']}, casa, ano "
                f"{p166['year_delivery']}, {p166['dist_beach_km']} km da praia, "
                f"R$ {p166['price_brl']}) e {p092['prop_id']} ({p092['title']}, "
                f"{p092['builder']}, casa, ano {p092['year_delivery']}, "
                f"{p092['dist_beach_km']} km da praia, R$ {p092['price_brl']}). Pelo critério "
                f"de desempate de ano de entrega, {p166['prop_id']} aparece em 1º lugar."
            ),
        },
        {
            "id": 5,
            "categoria": "consulta_especifica",
            "q": (
                "Em que cidade fica o imóvel p109, qual é a posição solar dele e "
                "qual a fonte de água?"
            ),
            "ferramenta": "property_details",
            "deve": ["n[íi]sia floresta", "nascente", "ambos"],
            "gabarito": (
                f"O imóvel {p109['prop_id']} ({p109['title']}, construtora {p109['builder']}, "
                f"tipo {p109['prop_type']}) fica na cidade de {p109['city']}, possui posição "
                f"solar '{p109['sun_position']}' e fonte de água '{p109['water_source']}' "
                f"(CAERN + poço artesiano)."
            ),
        },
        {
            "id": 6,
            "categoria": "consulta_especifica",
            "q": (
                "Qual é a área em m², a taxa mensal (condomínio/IPTU) e o yield bruto "
                "do apartamento p003?"
            ),
            "ferramenta": "property_details",
            "deve": ["92", "890", num(p003["yield_rate_pct"])],
            "gabarito": (
                f"O apartamento {p003['prop_id']} ({p003['title']}, {p003['builder']}, "
                f"{p003['city']}) possui área de {p003['area_m2']} m², taxa mensal fixa "
                f"(condomínio + IPTU) de R$ {p003['monthly_fee_brl']} e yield bruto anual de "
                f"{p003['yield_rate_pct']}% (preço R$ {p003['price_brl']})."
            ),
        },
        {
            "id": 7,
            "categoria": "consulta_especifica",
            "q": (
                "Qual é a construtora, a situação da documentação e a distância para a "
                "praia do imóvel p061?"
            ),
            "ferramenta": "property_details",
            "deve": ["mrv", "posse", num(p061["dist_beach_km"])],
            "gabarito": (
                f"O imóvel {p061['prop_id']} ({p061['title']}, casa em {p061['city']}) foi "
                f"construído pela construtora {p061['builder']}, possui documentação em "
                f"regime de '{p061['documentation']}' (não elegível a financiamento bancário) "
                f"e fica a {p061['dist_beach_km']} km da praia."
            ),
        },
        {
            "id": 8,
            "categoria": "comparacao",
            "q": (
                "Entre o p081 e o p082, qual tem a menor taxa mensal de manutenção e "
                "qual entrega o maior yield bruto?"
            ),
            "ferramenta": "compare_properties|property_details",
            "deve": ["p081", "390", num(p081["yield_rate_pct"])],
            "gabarito": (
                f"O imóvel {p081['prop_id']} vence em ambos os critérios: tem a menor taxa "
                f"mensal de manutenção (R$ {p081['monthly_fee_brl']}/mês contra "
                f"R$ {p082['monthly_fee_brl']}/mês do {p082['prop_id']}) e entrega o maior "
                f"yield bruto ({p081['yield_rate_pct']}% ao ano contra "
                f"{p082['yield_rate_pct']}% do {p082['prop_id']})."
            ),
        },
        {
            "id": 9,
            "categoria": "comparacao",
            "q": (
                "Compare o p109 e o p061: qual tem a maior área construída e qual fica "
                "mais próximo da praia?"
            ),
            "ferramenta": "compare_properties|property_details",
            "deve": ["p109", "155", "p061", num(p061["dist_beach_km"])],
            "gabarito": (
                f"Comparando {p109['prop_id']} e {p061['prop_id']}: o {p109['prop_id']} "
                f"({p109['title']}, {p109['city']}) tem a maior área construída "
                f"({p109['area_m2']} m² contra {p061['area_m2']} m² do {p061['prop_id']}), "
                f"enquanto o {p061['prop_id']} ({p061['title']}, {p061['city']}) fica mais "
                f"próximo da praia ({p061['dist_beach_km']} km contra "
                f"{p109['dist_beach_km']} km do {p109['prop_id']})."
            ),
        },
        {
            "id": 10,
            "categoria": "comparacao",
            "q": (
                "Quero uma casa até R$ 350 mil. Compare as duas mais novas do estoque "
                "e me diga qual você escolheria e por quê."
            ),
            "ferramenta": "compare_properties",
            "deve": ["p082", "p081"],
            "filtros": {"prop_type": "casa", "max_price": 350000},
            "gabarito": (
                f"As duas casas mais novas até R$ 350 mil são {p082['prop_id']} e "
                f"{p081['prop_id']} (ambas {p082['title']}/{p082['builder']}, entrega "
                f"{p082['year_delivery']}, {p082['area_m2']} m², escrituradas). "
                f"{p082['prop_id']} ({p082['city']}) é mais próxima da praia "
                f"({p082['dist_beach_km']} km vs {p081['dist_beach_km']} km), porém "
                f"{p081['prop_id']} ({p081['city']}) é mais barata (R$ {p081['price_brl']} "
                f"vs R$ {p082['price_brl']}), tem menor condomínio/IPTU "
                f"(R$ {p081['monthly_fee_brl']} vs R$ {p082['monthly_fee_brl']}), posição "
                f"nascente (vs poente), maior yield bruto ({p081['yield_rate_pct']}% vs "
                f"{p082['yield_rate_pct']}%), maior Cap Rate líquido "
                f"({p081['net_cap_rate_pct']}% vs {p082['net_cap_rate_pct']}%) e menor P/VP "
                f"({p081['p_vp']} vs {p082['p_vp']})."
            ),
        },
        {
            "id": 11,
            "categoria": "comparacao",
            "q": (
                "Busque os dois apartamentos mais novos em Natal até R$ 450 mil e "
                "compare-os: qual fica mais perto da praia e qual é o mais barato?"
            ),
            "ferramenta": "compare_properties|search_properties",
            "deve": ["p297", "p318", num(p297["dist_beach_km"])],
            "filtros": {"prop_type": "apartamento", "city": "Natal", "max_price": 450000},
            "gabarito": (
                f"Os dois apartamentos mais novos em Natal até R$ 450 mil são "
                f"{p297['prop_id']} ({p297['title']}, {p297['builder']}, entrega "
                f"{p297['year_delivery']}) e {p318['prop_id']} ({p318['title']}, "
                f"{p318['builder']}, entrega {p318['year_delivery']}). O {p297['prop_id']} "
                f"fica mais perto da praia ({p297['dist_beach_km']} km contra "
                f"{p318['dist_beach_km']} km do {p318['prop_id']}), enquanto o "
                f"{p318['prop_id']} é o mais barato (R$ {p318['price_brl']} contra "
                f"R$ {p297['price_brl']} do {p297['prop_id']})."
            ),
        },
        {
            "id": 12,
            "categoria": "calculo_financeiro",
            "q": (
                "Calcule o preço por m², o Valor Patrimonial estimado e o índice P/VP "
                "da casa p081. Ela está sendo negociada com ágio ou deságio?"
            ),
            "ferramenta": "evaluate_investment|property_details",
            "deve": ["1750", num(p081["p_vp"]), "des[áa]gio|abaixo"],
            "gabarito": (
                f"Para a casa {p081['prop_id']} (preço R$ {p081['price_brl']}, área "
                f"{p081['area_m2']} m² em {p081['city']}): preço por m² = "
                f"R$ {p081['price_per_m2']:.2f}/m², Valor Patrimonial estimado (VP) = "
                f"R$ {p081['vp_estimado_brl']:.2f} e índice P/VP = {p081['p_vp']:.2f}. "
                f"Como P/VP < 1.0 ({p081['p_vp']:.2f}), ela está sendo negociada com "
                f"deságio (abaixo do valor patrimonial médio de casas em Natal)."
            ),
        },
        {
            "id": 13,
            "categoria": "calculo_financeiro",
            "q": (
                "Qual é a receita líquida anual (NOI), o Cap Rate líquido e o tempo de "
                "payback em anos do apartamento p003?"
            ),
            "ferramenta": "evaluate_investment|property_details",
            "deve": [
                "17[._]?880",
                num(p003["net_cap_rate_pct"]),
                f"{num(p003['payback_years'])}|{num(round(p003['payback_years'], 1))}",
            ],
            "gabarito": (
                f"Para o apartamento {p003['prop_id']} (preço R$ {p003['price_brl']}, yield "
                f"bruto {p003['yield_rate_pct']}%, receita bruta anual "
                f"R$ {p003['gross_annual_brl']:.2f}, custo anual R$ {p003['annual_fee_brl']}): "
                f"receita líquida anual (NOI) = R$ {p003['noi_annual_brl']:.2f}, Cap Rate "
                f"líquido = {p003['net_cap_rate_pct']:.2f}% ao ano e payback líquido = "
                f"{p003['payback_years']:.2f} anos."
            ),
        },
        {
            "id": 14,
            "categoria": "calculo_financeiro",
            "q": (
                "Compare os indicadores de valuation dos terrenos p199 e p204: qual "
                "possui o menor P/VP e qual tem o menor payback líquido?"
            ),
            "ferramenta": "compare_properties|evaluate_investment|property_details",
            "deve": [
                "p204",
                num(p204["p_vp"]),
                "p199",
                f"{num(p199['payback_years'])}|{num(round(p199['payback_years'], 1))}",
            ],
            "gabarito": (
                f"Entre {p199['prop_id']} e {p204['prop_id']}: o terreno {p204['prop_id']} "
                f"possui o menor P/VP ({p204['p_vp']:.2f}, com deságio, contra "
                f"{p199['p_vp']:.2f} do {p199['prop_id']}), enquanto o terreno "
                f"{p199['prop_id']} tem o menor payback líquido ({p199['payback_years']:.2f} "
                f"anos, NOI R$ {p199['noi_annual_brl']:.2f} e Cap Rate líquido "
                f"{p199['net_cap_rate_pct']:.2f}%, contra payback de "
                f"{p204['payback_years']:.2f} anos e Cap Rate {p204['net_cap_rate_pct']:.2f}% "
                f"do {p204['prop_id']})."
            ),
        },
        {
            "id": 15,
            "categoria": "calculo_financeiro",
            "q": (
                "Em Parnamirim, os imóveis p166 e p092 empatam no maior yield bruto "
                "(11%). Comparando os dois, qual entrega a maior renda líquida anual "
                "(NOI) / Cap Rate líquido, e qual possui o menor P/VP?"
            ),
            "ferramenta": "compare_properties|evaluate_investment|property_details",
            "deve": [
                "p166",
                f"36[._]?865|{num(p166['net_cap_rate_pct'])}",
                "p092",
                num(p092["p_vp"]),
            ],
            "gabarito": (
                f"Comparando {p166['prop_id']} e {p092['prop_id']} (ambos em Parnamirim com "
                f"yield bruto de 11.0%): o {p166['prop_id']} entrega a maior renda líquida "
                f"anual (NOI = R$ {p166['noi_annual_brl']:.2f} vs "
                f"R$ {p092['noi_annual_brl']:.2f}) e o maior Cap Rate líquido "
                f"({p166['net_cap_rate_pct']:.2f}% vs {p092['net_cap_rate_pct']:.2f}%), "
                f"enquanto o {p092['prop_id']} possui o menor P/VP ({p092['p_vp']:.2f} vs "
                f"{p166['p_vp']:.2f} do {p166['prop_id']})."
            ),
        },
        {
            "id": 16,
            "categoria": "pegadinha",
            "q": (
                "Preciso comprar um terreno usando financiamento bancário e estou em "
                "dúvida entre o p204 (R$ 138 mil) e o p199 (R$ 145 mil). Qual dos dois "
                "eu devo escolher para financiar pagando menos?"
            ),
            "ferramenta": "compare_properties|evaluate_investment|property_details",
            "deve": ["p199", "posse", "escriturado"],
            "gabarito": (
                f"Pegadinha de elegibilidade a financiamento bancário: embora o "
                f"{p204['prop_id']} seja mais barato (R$ {p204['price_brl']}), sua "
                f"documentação é '{p204['documentation']}' (financing_eligible=False) e não "
                f"pode ser financiado por banco. Portanto, o {p199['prop_id']} "
                f"(R$ {p199['price_brl']}, documentação '{p199['documentation']}', "
                f"financing_eligible=True) é a única escolha viável para financiamento."
            ),
        },
        {
            "id": 17,
            "categoria": "pegadinha",
            "q": (
                "Procuro um terreno em Extremoz até R$ 150 mil que seja obrigatoriamente "
                "escriturado. Qual é o mais recente?"
            ),
            "ferramenta": "search_properties",
            "deve": ["p206", "2022", "110[._]?000|110 mil"],
            "filtros": {
                "prop_type": "terreno",
                "city": "Extremoz",
                "max_price": 150000,
                "documentation": "escriturado",
            },
            "gabarito": (
                f"O terreno escriturado mais recente em Extremoz até R$ 150 mil é o "
                f"{p206['prop_id']} ({p206['title']}, {p206['builder']}), entregue em "
                f"{p206['year_delivery']}, a {p206['dist_beach_km']} km da praia, por "
                f"R$ {p206['price_brl']} ({p206['area_m2']} m²). Os terrenos mais novos em "
                f"Extremoz nessa faixa de preço (p204 de 2025, p219 e p221 de 2024, e p205 "
                f"de 2023) possuem documentação de 'posse' e devem ser descartados."
            ),
        },
        {
            "id": 18,
            "categoria": "fora_do_escopo",
            "q": "Qual é a posição solar, a fonte de água e o P/VP do imóvel p999?",
            "ferramenta": "property_details|evaluate_investment",
            "deve": [r"n[ãa]o (existe|encontr|há|ha|consta|dispon)|nenhum|inexist"],
            "proibido": ["nascente", "poente", "caern", "poco_artesiano"],
            "gabarito": (
                "O imóvel p999 não existe no catálogo (que contém apenas p001 a p360; "
                "a ferramenta retorna {'error': 'no property with id p999'}). O assistente "
                "deve informar que o código p999 é inexistente sem inventar posição solar, "
                "fonte de água ou P/VP."
            ),
        },
        {
            "id": 19,
            "categoria": "fora_do_escopo",
            "q": (
                "Compare lado a lado os imóveis p081, p082 e p404, indicando qual é "
                "o mais barato."
            ),
            "ferramenta": "compare_properties",
            "deve": ["p081", "p082", "p404"],
            "gabarito": (
                f"O código p404 não existe no estoque e é retornado em missing=['p404'], "
                f"devendo ser reportado como inexistente/não encontrado. Entre os dois "
                f"imóveis válidos ({p081['prop_id']} por R$ {p081['price_brl']} e "
                f"{p082['prop_id']} por R$ {p082['price_brl']}), o {p081['prop_id']} vence "
                f"como o mais barato por R$ {p081['price_brl']}."
            ),
        },
        {
            "id": 20,
            "categoria": "fora_do_escopo",
            "q": (
                "Quantos quartos e vagas de garagem tem a casa p081, e o proprietário "
                "aceita um carro usado como parte do pagamento?"
            ),
            "ferramenta": "property_details",
            "deve": ["p081", r"n[ãa]o (consta|inform|possui|registr|dispon|tem|há|ha)"],
            "gabarito": (
                f"O catálogo não registra número de quartos, vagas de garagem ou política "
                f"de permuta/aceitação de veículo usado como parte do pagamento. Para a casa "
                f"{p081['prop_id']} ({p081['title']}, {p081['city']}), os dados disponíveis "
                f"são área de {p081['area_m2']} m², preço de R$ {p081['price_brl']}, "
                f"documentação '{p081['documentation']}', fonte de água "
                f"'{p081['water_source']}' e posição solar '{p081['sun_position']}'."
            ),
        },
    ]


PERGUNTAS: list[dict[str, Any]] = _build_perguntas()


def _extract_response_text(resultado_execucao: dict[str, Any]) -> str:
    """Extract the assistant's final response string from ``resultado_execucao``."""
    for key in ("resposta", "answer", "final_answer", "output", "text"):
        val = resultado_execucao.get(key)
        if isinstance(val, str):
            return val
    return ""


def _extract_called_tools(resultado_execucao: dict[str, Any]) -> list[str]:
    """Extract the list of called tool names from ``resultado_execucao``."""
    for key in ("ferramentas", "ferramentas_usadas", "tools", "tool_calls"):
        raw = resultado_execucao.get(key)
        if isinstance(raw, (list, tuple, set)):
            names: list[str] = []
            for item in raw:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("tool") or item.get("ferramenta")
                    if isinstance(name, str):
                        names.append(name)
            return names
        if isinstance(raw, str) and raw:
            return [raw]
    return []


def _check_cited_ids_against_filters(
    pergunta_dict: dict[str, Any],
    resposta: str,
) -> list[str]:
    """Verify that any ``p\\d{3}`` cited in ``resposta`` satisfies ``pergunta_dict['filtros']``."""
    filtros = pergunta_dict.get("filtros")
    if not filtros:
        return []

    errors: list[str] = []
    prompt_ids = {m.lower() for m in PROP_ID_RE.findall(pergunta_dict.get("q", ""))}
    cited_ids = []
    for match in PROP_ID_RE.findall(resposta):
        pid = match.lower()
        if pid not in cited_ids:
            cited_ids.append(pid)

    for pid in cited_ids:
        if pid in prompt_ids:
            continue
        if pid not in _BY_ID:
            errors.append(f"violacao_filtro: {pid} inexistente no estoque")
            continue

        row = _BY_ID[pid]
        if "prop_type" in filtros and row["prop_type"] != filtros["prop_type"]:
            errors.append(
                f"violacao_filtro: {pid} tem prop_type={row['prop_type']!r} "
                f"(esperado {filtros['prop_type']!r})"
            )
        if "max_price" in filtros and row["price_brl"] > filtros["max_price"]:
            errors.append(
                f"violacao_filtro: {pid} tem price_brl={row['price_brl']} "
                f"> max_price={filtros['max_price']}"
            )
        if "city" in filtros and row["city"].lower() != str(filtros["city"]).lower():
            errors.append(
                f"violacao_filtro: {pid} fica em {row['city']!r} "
                f"(esperado {filtros['city']!r})"
            )
        if "documentation" in filtros and row["documentation"] != filtros["documentation"]:
            # Allow citing a non-matching ID in a 'pegadinha' response only when the
            # response explicitly flags 'posse' to reject it in favor of the 'escriturado' ID.
            is_rejected_posse = (
                pergunta_dict.get("categoria") == "pegadinha"
                and bool(re.search(r"\bposse\b", resposta, re.IGNORECASE))
            )
            if not is_rejected_posse:
                errors.append(
                    f"violacao_filtro: {pid} tem documentation={row['documentation']!r} "
                    f"(esperado {filtros['documentation']!r})"
                )

    return errors


def checagem_automatica(
    pergunta_dict: dict[str, Any],
    resultado_execucao: dict[str, Any],
) -> list[str]:
    """Grade an execution trace and final answer against deterministic question rules.

    Returns a list of failure descriptions; an empty list ``[]`` indicates all
    automatic checks passed.
    """
    falhas: list[str] = []
    resposta = _extract_response_text(resultado_execucao)
    ferramentas_usadas = _extract_called_tools(resultado_execucao)

    expected_tool_pattern = pergunta_dict.get("ferramenta", "")
    if expected_tool_pattern:
        tool_re = re.compile(expected_tool_pattern, re.IGNORECASE)
        if not any(tool_re.search(t) for t in ferramentas_usadas):
            falhas.append(
                f"ferramenta_ausente: esperado /{expected_tool_pattern}/, "
                f"usadas={ferramentas_usadas}"
            )

    for padrao in pergunta_dict.get("deve", []):
        if not re.search(padrao, resposta, re.IGNORECASE):
            falhas.append(f"ausente_deve: /{padrao}/")

    for padrao in pergunta_dict.get("proibido", []):
        if re.search(padrao, resposta, re.IGNORECASE):
            falhas.append(f"presente_proibido: /{padrao}/")

    falhas.extend(_check_cited_ids_against_filters(pergunta_dict, resposta))
    return falhas


def main() -> None:
    """Print a summary table of all 20 benchmark questions and run self-checks."""
    assert len(PERGUNTAS) == 20, f"Expected 20 questions, got {len(PERGUNTAS)}"
    header = f"{'ID':>2} | {'Categoria':<20} | {'Ferramenta':<49} | {'Gabarito (110 chars)'}"
    print(header)
    print("-" * len(header))
    for item in PERGUNTAS:
        qid = item["id"]
        cat = item["categoria"]
        tool = item["ferramenta"]
        gab_preview = item["gabarito"][:110]
        print(f"{qid:>2} | {cat:<20} | {tool:<49} | {gab_preview}")

        # Self-verify that each question's own gabarito satisfies its 'deve' / 'proibido' rules
        # (using the primary expected tool).
        primary_tool = tool.split("|")[0]
        synthetic_trace = {
            "resposta": item["gabarito"],
            "ferramentas_usadas": [primary_tool],
        }
        # Only run automatic check on gabarito when proibido words aren't mentioned
        # meta-linguistically in Q18's explanation.
        if qid != 18:
            errs = checagem_automatica(item, synthetic_trace)
            assert not errs, f"Self-check failed on Q{qid}: {errs}"


if __name__ == "__main__":
    main()
