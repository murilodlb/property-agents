#!/usr/bin/env python3
"""20-question PT-BR benchmark suite, dynamic ground truths, and automatic grader."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Literal

import pandas as pd

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.catalog import load_catalog, normalize_text  # noqa: E402

SYSTEM_PROMPT_PTBR = (
    "Você é um assistente especialista em imóveis no Rio Grande do Norte (Natal e região). "
    "Responda sempre em português do Brasil (PT-BR), de forma clara e objetiva. "
    "Consulte sempre as ferramentas do estoque para buscar imóveis reais; nunca invente "
    "imóveis, preços, bairros ou atributos ausentes. Cite obrigatoriamente o imovel_id "
    "(ex: r001) de todo imóvel mencionado."
)

CategoriaType = Literal[
    "busca_com_filtro",
    "consulta_especifica",
    "comparacao",
    "calculo_financeiro",
    "pegadinha",
    "fora_do_escopo",
]

PROP_ID_RE = re.compile(r"\br\d{3,4}\b", re.IGNORECASE)

_ACCENT_CHAR_CLASS: dict[str, str] = {
    "a": "[aáàâã]",
    "e": "[eéê]",
    "i": "[ií]",
    "o": "[oóôõ]",
    "u": "[uú]",
    "c": "[cç]",
}

df: pd.DataFrame = load_catalog()
_BY_ID: dict[str, dict[str, Any]] = {
    str(row["prop_id"]).lower(): row.to_dict() for _, row in df.iterrows()
}


def num(x: float | int | str) -> str:
    """Convert a numeric value into a regex accepting '.' or ',' decimal separators."""
    if isinstance(x, float):
        text = f"{x:.2f}".rstrip("0").rstrip(".")
    else:
        text = str(x).strip()
    return text.replace(".", "[,.]")


def _group_thousands_regex(int_val: int) -> str:
    """Build a regex matching an integer with optional '.', ',', or '_' thousands separators."""
    raw = str(int_val)
    if len(raw) <= 3:
        return raw
    groups: list[str] = []
    while len(raw) > 3:
        groups.append(raw[-3:])
        raw = raw[:-3]
    groups.append(raw)
    return r"[.,_]?".join(reversed(groups))


def brl_re(value: float | int) -> str:
    """Build a regex matching a currency/area figure in PT-BR, raw, or rounded formats."""
    abs_val = abs(float(value))
    int_part = int(abs_val)
    cents_val = round((abs_val - int_part) * 100)
    if cents_val == 100:
        int_part += 1
        cents_val = 0

    int_pat = _group_thousands_regex(int_part)
    variants: list[str] = []
    if cents_val == 0:
        variants.append(rf"{int_pat}(?:[,.]00?)?")
        if int_part >= 1000 and int_part % 1000 == 0:
            variants.append(rf"{int_part // 1000}\s*mil")
    else:
        cents_str = f"{cents_val:02d}"
        cents_pat = f"{cents_str[0]}0?" if cents_str.endswith("0") else cents_str
        variants.append(rf"{int_pat}[,.]{cents_pat}")
        rounded_int = round(abs_val)
        variants.append(_group_thousands_regex(rounded_int))
        if rounded_int != int_part:
            variants.append(int_pat)

    return r"\b(?:" + "|".join(dict.fromkeys(variants)) + r")\b"


def _fmt_brl(value: float | int) -> str:
    """Format a number in Brazilian Portuguese style (e.g. '65.270,92')."""
    num_val = float(value)
    if num_val.is_integer():
        return f"{int(num_val):,}".replace(",", ".")
    return f"{num_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _accent_flexible_re(text: str) -> str:
    """Build a regex matching ``text`` regardless of Portuguese diacritics."""
    folded = normalize_text(text)
    parts: list[str] = []
    for ch in folded:
        if ch in _ACCENT_CHAR_CLASS:
            parts.append(_ACCENT_CHAR_CLASS[ch])
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _pick_distinct_pair(
    sub: pd.DataFrame,
    col_a: str,
    mode_a: Literal["min", "max"],
    col_b: str,
    mode_b: Literal["min", "max"],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Pick two rows where row A strictly wins ``col_a`` and row B strictly wins ``col_b``."""
    ordered = sub.sort_values("prop_id", kind="mergesort")
    for _, row_a in ordered.iterrows():
        val_a1 = row_a[col_a]
        val_a2 = row_a[col_b]
        if bool(pd.isna(val_a1)) or bool(pd.isna(val_a2)):
            continue
        cond_a = ordered[col_a] > val_a1 if mode_a == "min" else ordered[col_a] < val_a1
        cond_b = ordered[col_b] < val_a2 if mode_b == "min" else ordered[col_b] > val_a2
        cands = ordered.loc[cond_a & cond_b]
        if not cands.empty:
            return row_a.to_dict(), cands.iloc[0].to_dict()
    return None


def _simulate_sac_values(
    row: dict[str, Any],
    entrada_pct: float = 30.0,
    prazo_meses: int = 240,
    taxa_anual_pct: float = 10.5,
) -> dict[str, float]:
    """Compute exact SAC simulation numbers matching ``simular_compra_financiamento``."""
    preco = float(row["price_brl"])
    itbi = round(preco * 0.045, 2)
    source_norm = normalize_text(row["source"])
    modality_norm = normalize_text(row["sale_modality"])
    is_auction = source_norm == "caixa" and (
        "leilao" in modality_norm or "licitacao" in modality_norm
    )
    comissao = round(preco * 0.05, 2) if is_auction else 0.0
    valor_entrada = round(preco * (entrada_pct / 100.0), 2)
    valor_financiado = round(preco - valor_entrada, 2)
    desembolso_inicial = round(valor_entrada + itbi + comissao, 2)
    taxa_mensal = (taxa_anual_pct / 100.0) / 12.0
    amortizacao = valor_financiado / prazo_meses
    primeira_parcela = round(amortizacao + valor_financiado * taxa_mensal, 2)
    return {
        "valor_entrada_brl": valor_entrada,
        "valor_financiado_brl": valor_financiado,
        "itbi_registro_est_brl": itbi,
        "comissao_leiloeiro_brl": comissao,
        "desembolso_inicial_total_brl": desembolso_inicial,
        "primeira_parcela_brl": primeira_parcela,
    }


def _build_perguntas() -> list[dict[str, Any]]:
    """Construct the 20 PT-BR benchmark questions dynamically from ``df``."""
    city_norm = pd.Series([normalize_text(v) for v in df["city"]], index=df.index)
    nb_norm = pd.Series([normalize_text(v) for v in df["neighborhood"]], index=df.index)
    source_norm = pd.Series([normalize_text(v) for v in df["source"]], index=df.index)

    # Q1: Cheapest house in Natal up to R$ 350k
    q1_sub = df.loc[
        (df["prop_type"] == "casa")
        & (city_norm == "natal")
        & (df["price_brl"] <= 350_000)
    ].sort_values(["price_brl", "area_m2"], ascending=[True, False], na_position="last")
    r_q1 = q1_sub.iloc[0].to_dict()
    r_q1_2 = q1_sub.iloc[1].to_dict()

    # Q2: Largest terrain in RN up to R$ 200k
    q2_sub = df.loc[
        (df["prop_type"] == "terreno") & (df["price_brl"] <= 200_000)
    ].sort_values(["area_m2", "price_brl"], ascending=[False, True], na_position="last")
    r_q2 = q2_sub.iloc[0].to_dict()

    # Q3: Cheapest apartment in Natal up to R$ 350k with >= 2 bedrooms
    q3_sub = df.loc[
        (df["prop_type"] == "apartamento")
        & (city_norm == "natal")
        & (df["price_brl"] <= 350_000)
        & (df["bedrooms"].ge(2).fillna(False))
    ].sort_values(["price_brl", "area_m2"], ascending=[True, False], na_position="last")
    r_q3 = q3_sub.iloc[0].to_dict()

    # Q4: Property with highest discount in Parnamirim
    q4_sub = df.loc[city_norm == "parnamirim"].sort_values(
        ["discount_pct", "price_brl"], ascending=[False, True], na_position="last"
    )
    r_q4 = q4_sub.iloc[0].to_dict()

    # Q5: Natal Caixa listing accepting financing (fallback: any Natal Caixa listing)
    q5_fin = df.loc[
        (source_norm == "caixa")
        & (city_norm == "natal")
        & (df["accepts_financing"].eq(True).fillna(False))
    ].sort_values("prop_id", kind="mergesort")
    if q5_fin.empty:
        q5_fin = df.loc[(source_norm == "caixa") & (city_norm == "natal")].sort_values(
            "prop_id", kind="mergesort"
        )
    r_q5 = q5_fin.iloc[0].to_dict()

    # Q6: Agency apartment in Ponta Negra with area, bedrooms, bathrooms, and garage
    q6_sub = df.loc[
        (source_norm != "caixa")
        & (df["prop_type"] == "apartamento")
        & (nb_norm == "ponta negra")
        & df["area_m2"].notna()
        & df["bedrooms"].notna()
        & df["bathrooms"].notna()
        & df["garage_spots"].notna()
    ].sort_values("prop_id", kind="mergesort")
    r_q6 = q6_sub.iloc[0].to_dict()

    # Q7: Caixa house with appraisal, discount, and accepts_financing
    q7_sub = df.loc[
        (source_norm == "caixa")
        & (df["prop_type"] == "casa")
        & df["appraisal_value_brl"].notna()
        & df["discount_pct"].notna()
        & df["accepts_financing"].notna()
    ].sort_values("prop_id", kind="mergesort")
    r_q7 = q7_sub.iloc[0].to_dict()
    q7_fin_regex = r"\bsim\b|aceita" if bool(r_q7["accepts_financing"]) else r"n[ãa]o"
    q7_fin_label = "Sim" if bool(r_q7["accepts_financing"]) else "Não"

    # Q8: 2 properties where A has lower price_per_m2 and B has larger area_m2
    q8_sub = df.loc[df["area_m2"].gt(0).fillna(False) & df["price_per_m2"].notna()]
    pair8 = _pick_distinct_pair(q8_sub, "price_per_m2", "min", "area_m2", "max")
    assert pair8 is not None, "Não foi possível encontrar par distinto para Q8"
    a_q8, b_q8 = pair8

    # Q9: 2 Caixa properties where A is cheaper and B has higher discount_pct
    q9_sub = df.loc[(source_norm == "caixa") & df["discount_pct"].gt(0).fillna(False)]
    pair9 = _pick_distinct_pair(q9_sub, "price_brl", "min", "discount_pct", "max")
    assert pair9 is not None, "Não foi possível encontrar par distinto para Q9"
    a_q9, b_q9 = pair9

    # Q10: 2 cheapest apartments in Ponta Negra (Natal) up to R$ 450k
    q10_sub = df.loc[
        (df["prop_type"] == "apartamento")
        & (city_norm == "natal")
        & (nb_norm.str.contains("ponta negra", regex=False))
        & (df["price_brl"] <= 450_000)
    ].sort_values(["price_brl", "area_m2"], ascending=[True, False], na_position="last")
    a_q10, b_q10 = q10_sub.iloc[0].to_dict(), q10_sub.iloc[1].to_dict()

    # Q11: 2 houses with highest discount in Parnamirim
    q11_sub = df.loc[
        (df["prop_type"] == "casa") & (city_norm == "parnamirim")
    ].sort_values(["discount_pct", "price_brl"], ascending=[False, True], na_position="last")
    a_q11, b_q11 = q11_sub.iloc[0].to_dict(), q11_sub.iloc[1].to_dict()
    q11_area_win = a_q11 if a_q11["area_m2"] >= b_q11["area_m2"] else b_q11
    q11_ppm_win = a_q11 if a_q11["price_per_m2"] <= b_q11["price_per_m2"] else b_q11

    # Q12: Property with p_vp_mercado < 0.85 and ref_level in ("bairro_segmento", "cidade_segmento")
    q12_sub = df.loc[
        df["p_vp_mercado"].gt(0.1).fillna(False)
        & df["p_vp_mercado"].lt(0.85).fillna(False)
        & df["ref_level"].isin(["bairro_segmento", "cidade_segmento"])
    ].sort_values("prop_id", kind="mergesort")
    r_q12 = q12_sub.iloc[0].to_dict()

    # Q13: Caixa property with accepts_financing == True for SAC simulation (30% down, 240m, 10.5%)
    q13_sub = df.loc[
        (source_norm == "caixa") & df["accepts_financing"].eq(True).fillna(False)
    ].sort_values("prop_id", kind="mergesort")
    r_q13 = q13_sub.iloc[0].to_dict()
    sim13 = _simulate_sac_values(r_q13, entrada_pct=30.0, prazo_meses=240, taxa_anual_pct=10.5)

    # Q14: Market stats comparison for casas in Natal vs. Parnamirim
    natal_casas = df.loc[(df["prop_type"] == "casa") & (city_norm == "natal")]
    parn_casas = df.loc[(df["prop_type"] == "casa") & (city_norm == "parnamirim")]
    natal_cnt = int(len(natal_casas))
    parn_cnt = int(len(parn_casas))
    natal_med_ppm = round(float(natal_casas["price_per_m2"].median()), 2)
    parn_med_ppm = round(float(parn_casas["price_per_m2"].median()), 2)
    more_casas_city = "Natal" if natal_cnt >= parn_cnt else "Parnamirim"
    cheaper_m2_city = "Parnamirim" if parn_med_ppm <= natal_med_ppm else "Natal"
    cheaper_m2_val = min(natal_med_ppm, parn_med_ppm)

    # Q15: 2 terrains with distinct winners on price_per_m2 vs p_vp_mercado
    q15_sub = df.loc[
        (df["prop_type"] == "terreno")
        & df["area_m2"].gt(0).fillna(False)
        & df["p_vp_mercado"].gt(0.1).fillna(False)
    ].sort_values("prop_id", kind="mergesort")
    pair15 = _pick_distinct_pair(q15_sub, "price_per_m2", "min", "p_vp_mercado", "min")
    if pair15 is not None:
        a_q15, b_q15 = pair15
    else:
        a_q15, b_q15 = q15_sub.iloc[0].to_dict(), q15_sub.iloc[1].to_dict()
    q15_ppm_win = a_q15 if a_q15["price_per_m2"] <= b_q15["price_per_m2"] else b_q15
    q15_pvp_win = a_q15 if a_q15["p_vp_mercado"] <= b_q15["p_vp_mercado"] else b_q15

    # Q16: Financing simulation trap (Ceará-Mirim or Mossoró)
    q16_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    for city_key in ("ceara-mirim", "mossoro"):
        sub_city = df.loc[
            (source_norm == "caixa") & (df["prop_type"] == "casa") & (city_norm == city_key)
        ]
        no_fin_df = sub_city.loc[sub_city["accepts_financing"].eq(False).fillna(False)].sort_values(
            "price_brl", kind="mergesort"
        )
        yes_fin_df = sub_city.loc[sub_city["accepts_financing"].eq(True).fillna(False)].sort_values(
            "price_brl", kind="mergesort"
        )
        if not no_fin_df.empty and not yes_fin_df.empty:
            cand_no = no_fin_df.iloc[0].to_dict()
            cand_yes = yes_fin_df.iloc[0].to_dict()
            if cand_no["price_brl"] < cand_yes["price_brl"]:
                q16_pair = (cand_no, cand_yes)
                break
    assert q16_pair is not None, "Não foi possível montar pegadinha de financiamento (Q16)"
    no_fin_16, yes_fin_16 = q16_pair
    sim16_yes = _simulate_sac_values(
        yes_fin_16, entrada_pct=20.0, prazo_meses=300, taxa_anual_pct=10.5
    )

    # Q17: Caixa ágio trap (price_brl > appraisal_value_brl), preferring Parnamirim
    q17_all = df.loc[
        (source_norm == "caixa")
        & df["appraisal_value_brl"].gt(0).fillna(False)
        & (df["price_brl"] > df["appraisal_value_brl"])
        & df["p_vp_caixa"].gt(1.0).fillna(False)
        & df["neighborhood"].astype(str).str.strip().ne("")
    ].sort_values("prop_id", kind="mergesort")
    q17_parn = q17_all.loc[city_norm.loc[q17_all.index] == "parnamirim"]
    r_q17 = (q17_parn if not q17_parn.empty else q17_all).iloc[0].to_dict()

    # Q19 & Q20 dynamic valid targets
    a_q19, b_q19 = a_q10, b_q10
    cheapest_19 = a_q19 if a_q19["price_brl"] <= b_q19["price_brl"] else b_q19
    r_q20 = (
        df.loc[df["prop_type"] == "apartamento"]
        .sort_values("prop_id", kind="mergesort")
        .iloc[0]
        .to_dict()
    )

    return [
        {
            "id": 1,
            "categoria": "busca_com_filtro",
            "q": (
                "Quero comprar uma casa em Natal de até R$ 350 mil. Quais são as opções "
                "mais baratas disponíveis no estoque?"
            ),
            "ferramenta": "buscar_imoveis",
            "deve": [r_q1["prop_id"]],
            "filtros": {"prop_type": "casa", "city": "Natal", "max_price": 350_000},
            "gabarito": (
                f"A casa mais barata em Natal até R$ 350 mil é a {r_q1['prop_id']} "
                f"({r_q1['title']}, bairro {r_q1['neighborhood']}, fonte {r_q1['source']}), "
                f"por R$ {_fmt_brl(r_q1['price_brl'])}, {r_q1['area_m2']} m² e P/VP de mercado "
                f"{r_q1['p_vp_mercado']}; em 2º lugar aparece a {r_q1_2['prop_id']} "
                f"({r_q1_2['neighborhood']}, R$ {_fmt_brl(r_q1_2['price_brl'])})."
            ),
        },
        {
            "id": 2,
            "categoria": "busca_com_filtro",
            "q": (
                "Procuro um terreno no Rio Grande do Norte de até R$ 200 mil. "
                "Qual tem a maior área em m²?"
            ),
            "ferramenta": "buscar_imoveis",
            "deve": [r_q2["prop_id"], brl_re(r_q2["area_m2"])],
            "filtros": {"prop_type": "terreno", "max_price": 200_000},
            "gabarito": (
                f"O terreno de maior área até R$ 200 mil no RN é o {r_q2['prop_id']} "
                f"({r_q2['title']}, {r_q2['city']}, fonte {r_q2['source']}), com "
                f"{_fmt_brl(r_q2['area_m2'])} m² por R$ {_fmt_brl(r_q2['price_brl'])}."
            ),
        },
        {
            "id": 3,
            "categoria": "busca_com_filtro",
            "q": (
                "Quais apartamentos em Natal de até R$ 350 mil possuem pelo menos "
                "2 quartos? Cite qual é o mais barato."
            ),
            "ferramenta": "buscar_imoveis",
            "deve": [r_q3["prop_id"]],
            "filtros": {
                "prop_type": "apartamento",
                "city": "Natal",
                "max_price": 350_000,
                "min_bedrooms": 2,
            },
            "gabarito": (
                f"Entre os apartamentos em Natal até R$ 350 mil com pelo menos 2 quartos "
                f"({len(q3_sub)} opções), o mais barato é o {r_q3['prop_id']} "
                f"({r_q3['title']}, bairro {r_q3['neighborhood']}, {int(r_q3['bedrooms'])} "
                f"quartos, {r_q3['area_m2']} m²), por R$ {_fmt_brl(r_q3['price_brl'])}."
            ),
        },
        {
            "id": 4,
            "categoria": "busca_com_filtro",
            "q": "Qual é o imóvel com o maior desconto percentual disponível em Parnamirim?",
            "ferramenta": "buscar_imoveis",
            "deve": [r_q4["prop_id"], num(r_q4["discount_pct"])],
            "filtros": {"city": "Parnamirim"},
            "gabarito": (
                f"Em Parnamirim, o imóvel com maior desconto percentual é o {r_q4['prop_id']} "
                f"({r_q4['title']}, {r_q4['prop_type']}, bairro {r_q4['neighborhood']}), "
                f"com {r_q4['discount_pct']}% de desconto sobre a avaliação oficial "
                f"(de R$ {_fmt_brl(r_q4['appraisal_value_brl'])} por "
                f"R$ {_fmt_brl(r_q4['price_brl'])})."
            ),
        },
        {
            "id": 5,
            "categoria": "consulta_especifica",
            "q": (
                f"Em qual cidade e bairro fica o imóvel {r_q5['prop_id']}, qual é a origem "
                f"(fonte) do anúncio e qual a modalidade de venda?"
            ),
            "ferramenta": "detalhes_imovel",
            "deve": [
                _accent_flexible_re(r_q5["city"]),
                _accent_flexible_re(r_q5["neighborhood"]),
                re.escape(r_q5["source"]),
                _accent_flexible_re(r_q5["sale_modality"]),
            ],
            "gabarito": (
                f"O imóvel {r_q5['prop_id']} ({r_q5['title']}) fica em {r_q5['city']}, "
                f"no bairro {r_q5['neighborhood']}; a origem do anúncio é '{r_q5['source']}' "
                f"e a modalidade de venda é '{r_q5['sale_modality']}'."
            ),
        },
        {
            "id": 6,
            "categoria": "consulta_especifica",
            "q": (
                f"Qual é a área em m², a quantidade de quartos, banheiros e vagas de "
                f"garagem do apartamento {r_q6['prop_id']}?"
            ),
            "ferramenta": "detalhes_imovel",
            "deve": [
                num(r_q6["area_m2"]),
                str(int(r_q6["bedrooms"])),
                str(int(r_q6["bathrooms"])),
                str(int(r_q6["garage_spots"])),
            ],
            "gabarito": (
                f"O apartamento {r_q6['prop_id']} ({r_q6['title']}, {r_q6['source']}) possui "
                f"área de {r_q6['area_m2']} m², {int(r_q6['bedrooms'])} quartos, "
                f"{int(r_q6['bathrooms'])} banheiros e {int(r_q6['garage_spots'])} vaga(s) "
                f"de garagem (preço R$ {_fmt_brl(r_q6['price_brl'])})."
            ),
        },
        {
            "id": 7,
            "categoria": "consulta_especifica",
            "q": (
                f"Qual é o preço de venda, o valor de avaliação oficial, o percentual de "
                f"desconto e se o imóvel {r_q7['prop_id']} aceita financiamento?"
            ),
            "ferramenta": "detalhes_imovel",
            "deve": [
                brl_re(r_q7["price_brl"]),
                brl_re(r_q7["appraisal_value_brl"]),
                num(r_q7["discount_pct"]),
                q7_fin_regex,
            ],
            "gabarito": (
                f"O imóvel {r_q7['prop_id']} ({r_q7['title']}, {r_q7['city']}) tem preço de "
                f"venda de R$ {_fmt_brl(r_q7['price_brl'])}, valor de avaliação oficial de "
                f"R$ {_fmt_brl(r_q7['appraisal_value_brl'])}, desconto de "
                f"{r_q7['discount_pct']}% e aceita financiamento: {q7_fin_label}."
            ),
        },
        {
            "id": 8,
            "categoria": "comparacao",
            "q": (
                f"Entre os imóveis {a_q8['prop_id']} e {b_q8['prop_id']}, qual possui o "
                f"menor preço por m² e qual tem a maior área em m²?"
            ),
            "ferramenta": "comparar_imoveis|detalhes_imovel",
            "deve": [
                a_q8["prop_id"],
                b_q8["prop_id"],
                brl_re(a_q8["price_per_m2"]),
                num(b_q8["area_m2"]),
            ],
            "gabarito": (
                f"Comparando {a_q8['prop_id']} e {b_q8['prop_id']}: o {a_q8['prop_id']} "
                f"tem o menor preço por m² (R$ {_fmt_brl(a_q8['price_per_m2'])}/m² contra "
                f"R$ {_fmt_brl(b_q8['price_per_m2'])}/m²), enquanto o {b_q8['prop_id']} tem "
                f"a maior área ({b_q8['area_m2']} m² contra {a_q8['area_m2']} m²)."
            ),
        },
        {
            "id": 9,
            "categoria": "comparacao",
            "q": (
                f"Compare os imóveis {a_q9['prop_id']} e {b_q9['prop_id']}: qual é mais "
                f"barato no preço total e qual oferece o maior desconto percentual sobre a "
                f"avaliação oficial?"
            ),
            "ferramenta": "comparar_imoveis|detalhes_imovel",
            "deve": [
                a_q9["prop_id"],
                b_q9["prop_id"],
                brl_re(a_q9["price_brl"]),
                num(b_q9["discount_pct"]),
            ],
            "gabarito": (
                f"Entre {a_q9['prop_id']} e {b_q9['prop_id']}: o {a_q9['prop_id']} é mais "
                f"barato no preço total (R$ {_fmt_brl(a_q9['price_brl'])} contra "
                f"R$ {_fmt_brl(b_q9['price_brl'])}), já o {b_q9['prop_id']} oferece o maior "
                f"desconto percentual ({b_q9['discount_pct']}% contra "
                f"{a_q9['discount_pct']}%)."
            ),
        },
        {
            "id": 10,
            "categoria": "comparacao",
            "q": (
                "Busque os dois apartamentos mais baratos em Ponta Negra (Natal) de até "
                "R$ 450 mil, compare-os lado a lado e me diga qual você escolheria e por quê."
            ),
            "ferramenta": "comparar_imoveis",
            "deve": [a_q10["prop_id"], b_q10["prop_id"]],
            "filtros": {"prop_type": "apartamento", "city": "Natal", "max_price": 450_000},
            "gabarito": (
                f"Os dois apartamentos mais baratos em Ponta Negra (Natal) até R$ 450 mil "
                f"são {a_q10['prop_id']} (R$ {_fmt_brl(a_q10['price_brl'])}, "
                f"{a_q10['area_m2']} m², P/VP de mercado {a_q10['p_vp_mercado']}, fonte "
                f"{a_q10['source']}) e {b_q10['prop_id']} (R$ {_fmt_brl(b_q10['price_brl'])}, "
                f"{b_q10['area_m2']} m², P/VP de mercado {b_q10['p_vp_mercado']}, fonte "
                f"{b_q10['source']}). O {a_q10['prop_id']} destaca-se pelo menor preço total, "
                f"menor preço/m² e menor P/VP de mercado."
            ),
        },
        {
            "id": 11,
            "categoria": "comparacao",
            "q": (
                "Busque as duas casas com maior desconto em Parnamirim e compare-as: "
                "qual tem a maior área em m² e qual tem o menor preço por m²?"
            ),
            "ferramenta": "comparar_imoveis|buscar_imoveis",
            "deve": [
                a_q11["prop_id"],
                b_q11["prop_id"],
                num(q11_area_win["area_m2"]),
                brl_re(q11_ppm_win["price_per_m2"]),
            ],
            "filtros": {"prop_type": "casa", "city": "Parnamirim"},
            "gabarito": (
                f"As duas casas com maior desconto em Parnamirim são {a_q11['prop_id']} "
                f"({a_q11['discount_pct']}% de desconto, {a_q11['area_m2']} m², "
                f"R$ {_fmt_brl(a_q11['price_per_m2'])}/m²) e {b_q11['prop_id']} "
                f"({b_q11['discount_pct']}% de desconto, {b_q11['area_m2']} m², "
                f"R$ {_fmt_brl(b_q11['price_per_m2'])}/m²). A maior área é da "
                f"{q11_area_win['prop_id']} ({q11_area_win['area_m2']} m²) e o menor preço "
                f"por m² é da {q11_ppm_win['prop_id']} "
                f"(R$ {_fmt_brl(q11_ppm_win['price_per_m2'])}/m²)."
            ),
        },
        {
            "id": 12,
            "categoria": "calculo_financeiro",
            "q": (
                f"Calcule o preço por m², o Valor Patrimonial estimado de mercado e o índice "
                f"P/VP de mercado do imóvel {r_q12['prop_id']}. Ele está sendo negociado "
                f"com ágio ou deságio em relação à mediana regional?"
            ),
            "ferramenta": "avaliar_investimento|detalhes_imovel",
            "deve": [
                brl_re(r_q12["price_per_m2"]),
                brl_re(r_q12["vp_estimado_brl"]),
                num(r_q12["p_vp_mercado"]),
                r"des[áa]gio|abaixo",
            ],
            "gabarito": (
                f"Para o imóvel {r_q12['prop_id']} ({r_q12['city']}, {r_q12['prop_type']}, "
                f"referência mediano nível '{r_q12['ref_level']}' = "
                f"R$ {_fmt_brl(r_q12['ref_m2_brl'])}/m²): preço por m² = "
                f"R$ {_fmt_brl(r_q12['price_per_m2'])}/m², Valor Patrimonial estimado de "
                f"mercado = R$ {_fmt_brl(r_q12['vp_estimado_brl'])} e índice P/VP de mercado "
                f"= {r_q12['p_vp_mercado']}. Como P/VP < 1,0, está sendo negociado com "
                f"deságio em relação à mediana regional."
            ),
        },
        {
            "id": 13,
            "categoria": "calculo_financeiro",
            "q": (
                f"Simule o financiamento do imóvel {r_q13['prop_id']} pelo sistema SAC com "
                f"entrada de 30%, prazo de 240 meses e juros de 10,5% ao ano. Qual será o "
                f"valor financiado, a primeira parcela e o desembolso inicial total "
                f"(incluindo ITBI/registro)?"
            ),
            "ferramenta": "simular_compra_financiamento",
            "deve": [
                r_q13["prop_id"],
                brl_re(sim13["valor_financiado_brl"]),
                brl_re(sim13["primeira_parcela_brl"]),
                brl_re(sim13["desembolso_inicial_total_brl"]),
            ],
            "gabarito": (
                f"Simulação SAC para {r_q13['prop_id']} ({r_q13['city']}, preço "
                f"R$ {_fmt_brl(r_q13['price_brl'])}): entrada de 30% = "
                f"R$ {_fmt_brl(sim13['valor_entrada_brl'])}, valor financiado = "
                f"R$ {_fmt_brl(sim13['valor_financiado_brl'])}, primeira parcela SAC = "
                f"R$ {_fmt_brl(sim13['primeira_parcela_brl'])}, ITBI/registro estimado = "
                f"R$ {_fmt_brl(sim13['itbi_registro_est_brl'])} e desembolso inicial total "
                f"= R$ {_fmt_brl(sim13['desembolso_inicial_total_brl'])}."
            ),
        },
        {
            "id": 14,
            "categoria": "calculo_financeiro",
            "q": (
                "Compare as estatísticas de mercado de casas em Natal e em Parnamirim: "
                "qual cidade tem mais casas cadastradas e qual apresenta a menor mediana de "
                "preço por m²?"
            ),
            "ferramenta": "estatisticas_mercado",
            "deve": [
                "Natal",
                str(natal_cnt),
                "Parnamirim",
                brl_re(cheaper_m2_val),
            ],
            "gabarito": (
                f"Nas estatísticas de mercado de casas: Natal possui {natal_cnt} casas "
                f"cadastradas (mediana de R$ {_fmt_brl(natal_med_ppm)}/m²) contra "
                f"{parn_cnt} casas em Parnamirim (mediana de "
                f"R$ {_fmt_brl(parn_med_ppm)}/m²). Portanto, {more_casas_city} tem mais "
                f"casas cadastradas e {cheaper_m2_city} apresenta a menor mediana de preço "
                f"por m² (R$ {_fmt_brl(cheaper_m2_val)}/m²)."
            ),
        },
        {
            "id": 15,
            "categoria": "calculo_financeiro",
            "q": (
                f"Comparando os terrenos {a_q15['prop_id']} e {b_q15['prop_id']}, qual "
                f"oferece o menor preço por m² e qual possui o menor índice P/VP de mercado?"
            ),
            "ferramenta": "comparar_imoveis|avaliar_investimento|detalhes_imovel",
            "deve": [
                q15_ppm_win["prop_id"],
                brl_re(q15_ppm_win["price_per_m2"]),
                q15_pvp_win["prop_id"],
                num(q15_pvp_win["p_vp_mercado"]),
            ],
            "gabarito": (
                f"Comparando os terrenos {a_q15['prop_id']} ({a_q15['city']}, "
                f"R$ {_fmt_brl(a_q15['price_per_m2'])}/m², P/VP {a_q15['p_vp_mercado']}) e "
                f"{b_q15['prop_id']} ({b_q15['city']}, "
                f"R$ {_fmt_brl(b_q15['price_per_m2'])}/m², P/VP {b_q15['p_vp_mercado']}): "
                f"o menor preço por m² é do {q15_ppm_win['prop_id']} "
                f"(R$ {_fmt_brl(q15_ppm_win['price_per_m2'])}/m²) e o menor P/VP de mercado "
                f"é do {q15_pvp_win['prop_id']} ({q15_pvp_win['p_vp_mercado']})."
            ),
        },
        {
            "id": 16,
            "categoria": "pegadinha",
            "q": (
                f"Simule o financiamento (20% de entrada em 300 meses) para a casa mais "
                f"barata de {yes_fin_16['city']} ({no_fin_16['prop_id']}, "
                f"R$ {_fmt_brl(no_fin_16['price_brl'])}) e compare com "
                f"{yes_fin_16['prop_id']} (R$ {_fmt_brl(yes_fin_16['price_brl'])}). "
                f"Posso financiar a mais barata ou qual devo escolher?"
            ),
            "ferramenta": "simular_compra_financiamento|comparar_imoveis|detalhes_imovel",
            "deve": [
                yes_fin_16["prop_id"],
                (
                    r"n[ãa]o (?:aceita|financi|é financi|permite)"
                    r"|sem financiamento|recursos pr[óo]prios|imposs[íi]vel"
                ),
            ],
            "gabarito": (
                f"Pegadinha de financiamento: a casa {no_fin_16['prop_id']} "
                f"(R$ {_fmt_brl(no_fin_16['price_brl'])}) NÃO aceita financiamento bancário "
                f"(financiamento_permitido=False), exigindo compra à vista com recursos "
                f"próprios. Assim, para financiar com 20% de entrada em 300 meses, você "
                f"deve escolher a {yes_fin_16['prop_id']} "
                f"(R$ {_fmt_brl(yes_fin_16['price_brl'])}, primeira parcela SAC aproximada "
                f"de R$ {_fmt_brl(sim16_yes['primeira_parcela_brl'])})."
            ),
        },
        {
            "id": 17,
            "categoria": "pegadinha",
            "q": (
                f"Quero comprar um imóvel da Caixa em {r_q17['city']} pagando abaixo da "
                f"avaliação oficial do banco. O imóvel {r_q17['prop_id']} (no bairro "
                f"{r_q17['neighborhood']}) está com deságio ou ágio em relação à avaliação? "
                f"Ele atende ao meu critério?"
            ),
            "ferramenta": "avaliar_investimento|detalhes_imovel",
            "deve": [
                r"[áa]gio|acima",
                r"n[ãa]o atende|n[ãa]o cumpr|acima da avalia",
                num(r_q17["p_vp_caixa"]),
            ],
            "gabarito": (
                f"Pegadinha de ágio na Caixa: o imóvel {r_q17['prop_id']} ({r_q17['city']}, "
                f"bairro {r_q17['neighborhood']}) está anunciado por "
                f"R$ {_fmt_brl(r_q17['price_brl'])}, acima da avaliação oficial de "
                f"R$ {_fmt_brl(r_q17['appraisal_value_brl'])} (P/VP Caixa = "
                f"{r_q17['p_vp_caixa']}, ágio de R$ {_fmt_brl(abs(r_q17['economia_brl']))}). "
                f"Portanto, está com ágio e não atende ao critério de pagar abaixo da "
                f"avaliação."
            ),
        },
        {
            "id": 18,
            "categoria": "fora_do_escopo",
            "q": "Qual é o bairro, o preço e o índice P/VP do imóvel r9999?",
            "ferramenta": "detalhes_imovel|avaliar_investimento",
            "deve": [r"n[ãa]o (existe|encontr|há|ha|consta|dispon)|nenhum|inexist"],
            "proibido": ["ponta negra", "tirol", "planalto", "nova esperan[çc]a"],
            "gabarito": (
                "O imóvel r9999 não existe no catálogo (a ferramenta retorna erro "
                "informando que nenhum imóvel foi encontrado com esse código). O assistente "
                "deve avisar que r9999 é inexistente sem inventar bairro, preço ou P/VP."
            ),
        },
        {
            "id": 19,
            "categoria": "fora_do_escopo",
            "q": (
                f"Compare lado a lado os imóveis {a_q19['prop_id']}, {b_q19['prop_id']} e "
                f"r9404, indicando qual é o mais barato."
            ),
            "ferramenta": "comparar_imoveis",
            "deve": [
                a_q19["prop_id"],
                b_q19["prop_id"],
                "r9404",
                brl_re(cheapest_19["price_brl"]),
            ],
            "gabarito": (
                f"O código r9404 não existe no estoque (retornado em nao_encontrados) e "
                f"deve ser reportado como inexistente. Entre os dois imóveis válidos "
                f"({a_q19['prop_id']} por R$ {_fmt_brl(a_q19['price_brl'])} e "
                f"{b_q19['prop_id']} por R$ {_fmt_brl(b_q19['price_brl'])}), o mais barato "
                f"é o {cheapest_19['prop_id']} por R$ {_fmt_brl(cheapest_19['price_brl'])}."
            ),
        },
        {
            "id": 20,
            "categoria": "fora_do_escopo",
            "q": (
                f"Qual é o valor mensal do condomínio, o valor do IPTU anual e a posição "
                f"solar (nascente/poente) do apartamento {r_q20['prop_id']}?"
            ),
            "ferramenta": "detalhes_imovel",
            "deve": [
                r_q20["prop_id"],
                r"n[ãa]o (consta|inform|possui|registr|dispon|tem|há|ha)",
            ],
            "gabarito": (
                f"O catálogo não registra valor de condomínio, valor de IPTU anual nem "
                f"posição solar (nascente/poente). Para o apartamento {r_q20['prop_id']} "
                f"({r_q20['city']}, {r_q20['neighborhood']}), constam apenas os dados "
                f"cadastrais e de valuation disponíveis no estoque."
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
    """Verify that any ``r\\d{3,4}`` cited in ``resposta`` satisfies ``filtros``."""
    filtros = pergunta_dict.get("filtros")
    if not filtros:
        return []

    errors: list[str] = []
    prompt_ids = {m.lower() for m in PROP_ID_RE.findall(pergunta_dict.get("q", ""))}
    cited_ids: list[str] = []
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
        if "city" in filtros:
            expected_city = normalize_text(filtros["city"])
            actual_city = normalize_text(row["city"])
            if expected_city not in actual_city and actual_city != expected_city:
                errors.append(
                    f"violacao_filtro: {pid} fica em {row['city']!r} "
                    f"(esperado {filtros['city']!r})"
                )
        if "min_bedrooms" in filtros:
            beds = row.get("bedrooms")
            if beds is None or bool(pd.isna(beds)) or int(beds) < int(filtros["min_bedrooms"]):
                errors.append(
                    f"violacao_filtro: {pid} tem bedrooms={beds!r} "
                    f"(< {filtros['min_bedrooms']})"
                )
        if "accepts_financing" in filtros:
            fin = row.get("accepts_financing")
            expected_fin = bool(filtros["accepts_financing"])
            if fin is None or bool(pd.isna(fin)) or bool(fin) != expected_fin:
                errors.append(
                    f"violacao_filtro: {pid} tem accepts_financing={fin!r} "
                    f"(esperado {expected_fin!r})"
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
    """Print a summary table of all 20 PT-BR benchmark questions and run self-checks."""
    assert len(PERGUNTAS) == 20, f"Expected 20 questions, got {len(PERGUNTAS)}"
    header = f"{'ID':>2} | {'Categoria':<20} | {'Ferramenta':<56} | {'Gabarito (110 chars)'}"
    print(header)
    print("-" * len(header))
    for item in PERGUNTAS:
        qid = item["id"]
        cat = item["categoria"]
        tool = item["ferramenta"]
        gab_preview = item["gabarito"][:110]
        print(f"{qid:>2} | {cat:<20} | {tool:<56} | {gab_preview}")

        primary_tool = tool.split("|")[0]
        synthetic_trace = {
            "resposta": item["gabarito"],
            "ferramentas_usadas": [primary_tool],
        }
        errs = checagem_automatica(item, synthetic_trace)
        assert not errs, f"Self-check failed on Q{qid}: {errs}"


if __name__ == "__main__":
    main()
