"""
Modular Real-Estate Scraper for Natal & Rio Grande do Norte (RN).
Supports:
  --mode caixa      (Caixa Econômica Federal RN official CSV catalog)
  --mode agencies   (Local Natal/RN real estate agencies via URL list)
  --mode combined   (Merges both sources into a single real_properties.csv)

Offline agency merge: pass --agencies-csv path/to/file.csv together with --mode combined
(or --mode agencies) to load the existing non-Caixa rows (source != "caixa") from that CSV
instead of re-scraping the live agency websites.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

log = logging.getLogger("scrape_real_properties")


class ScrapeError(Exception):
    """Fatal, user-facing error (bad input file, nothing collected, unwritable output)."""


# ─── DEFAULT LOCAL NATAL / RN AGENCY URLS ───────────────────────────────────
# Add any local real estate agency search/catalog URLs here:
DEFAULT_AGENCY_URLS: list[str] = [
    "https://ibeda.com.br/imoveis/a-venda/natal-rn/ponta-negra",
    "https://ibeda.com.br/imoveis/a-venda/natal-rn",
    "https://www.pereseperes.com.br/imoveis/a-venda/apartamento/natal/ponta-negra",
    "https://www.pereseperes.com.br/imoveis/a-venda/casa/natal",
    "https://www.imoveispontanegra.com.br/",
    "https://suzananeryimoveis.com.br/comprar-ou-alugar/rn/natal/ponta-negra/",
    "https://emobiimobiliaria.com.br/",
    # Land-focused (terreno) listings across Natal and Greater Natal
    "https://www.pereseperes.com.br/imoveis/a-venda/terreno/natal",
    "https://www.gilmaramaralimoveis.com.br/imoveis/a-venda/terreno",
    "https://www.dnnegociosimobiliarios.com.br/imoveis/a-venda/terreno",
    "https://www.marizaldaimoveis.com.br/imoveis/a-venda/terreno/parnamirim",
    "https://portal.kenlo.com.br/imoveis/a-venda/terreno/parnamirim",
    "https://suzananeryimoveis.com.br/comprar/parnamirim/centro/terreno",
    "https://suzananeryimoveis.com.br/comprar/rn/nisia-floresta/centro/terreno/",
    "https://ibeda.com.br/terreno-lote-em-condominio/a-venda/natal-rn/ponta-negra",
    "https://ibeda.com.br/terreno-lote-em-condominio/a-venda/parnamirim-rn/pium-distrito-litoral",
]

CAIXA_RN_CSV_URL = "https://venda-imoveis.caixa.gov.br/listaweb/Lista_imoveis_RN.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
}

KNOWN_RN_CITIES = [
    "Natal",
    "Parnamirim",
    "São Gonçalo do Amarante",
    "Macaíba",
    "Extremoz",
    "Nísia Floresta",
    "São José de Mipibu",
    "Mossoró",
    "Ceará-Mirim",
    "Tibau do Sul",
    "Touros",
    "Caicó",
    "São Miguel do Gostoso",
]

KNOWN_NEIGHBORHOODS = [
    "Ponta Negra",
    "Capim Macio",
    "Tirol",
    "Petrópolis",
    "Lagoa Nova",
    "Candelária",
    "Neópolis",
    "Nova Parnamirim",
    "Emaús",
    "Cidade Verde",
    "Pitimbu",
    "Planalto",
    "Areia Preta",
    "Barro Vermelho",
    "Alecrim",
    "Cidade Alta",
    "Ribeira",
    "Lagoa Seca",
    "Mirassol",
    "Cidade Satélite",
    "Pirangi",
    "Cotovelo",
    "Búzios",
    "Tabatinga",
    "Pium",
    "Redinha",
    "Pajuçara",
    "Nova Descoberta",
    "San Vale",
    # Land / allotment-heavy neighborhoods (appended last so specific names match first)
    "Parque das Nações",
    "Cajupiranga",
    "Parque do Jiqui",
    "Santa Tereza",
    "Bom Fim",
    "Vila de Ponta Negra",
    "Nossa Senhora da Apresentação",
    "Quintas",
    "Igapó",
    "Praia de Pipa",
    "Pirangi do Norte",
    "Vale do Sol",
    "Centro",
]

# Optional domain-specific overrides if you want to pin exact CSS selectors for a new agency
DOMAIN_CONFIGS: dict[str, dict[str, str]] = {
    # Example format:
    # "myagency.com.br": {
    #     "card": "div.property-card",
    #     "title": "h2.property-title",
    #     "price": "span.property-price",
    # }
}


@dataclass
class RawProperty:
    prop_id: str = ""
    source: str = ""
    external_code: str = ""
    title: str = ""
    prop_type: str = ""
    city: str = ""
    neighborhood: str = ""
    price_brl: float | None = None
    appraisal_value_brl: float | None = None
    discount_pct: float | None = None
    area_m2: float | None = None
    bedrooms: int | None = None
    bathrooms: int | None = None
    garage_spots: int | None = None
    condo_fee_brl: float | None = None
    iptu_brl: float | None = None
    sale_modality: str = ""
    accepts_financing: bool | None = None
    url: str = ""


# Caixa publishes RN city names upper-case and without accents.
CAIXA_CITY_MAP: dict[str, str] = {
    "ASSU": "Assú",
    "CARAUBAS": "Caraúbas",
    "CEARA-MIRIM": "Ceará-Mirim",
    "FLORANIA": "Florânia",
    "ITAJA": "Itajá",
    "ITAU": "Itaú",
    "JACANA": "Jaçanã",
    "JOAO CAMARA": "João Câmara",
    "MACAIBA": "Macaíba",
    "MOSSORO": "Mossoró",
    "NISIA FLORESTA": "Nísia Floresta",
    "PENDENCIAS": "Pendências",
    "SANTO ANTONIO": "Santo Antônio",
    "SAO GONCALO DO AMARANTE": "São Gonçalo do Amarante",
    "SAO JOSE DE MIPIBU": "São José de Mipibu",
    "SAO VICENTE": "São Vicente",
    "TAIPU": "Taipu",
    "TANGARA": "Tangará",
}

_PT_PREPOSITIONS = (" Do ", " Da ", " De ", " Dos ", " Das ")


def normalize_city_name(raw_city: str) -> str:
    """Maps Caixa's unaccented upper-case city names to standard Portuguese spelling."""
    key = " ".join(raw_city.strip().upper().split())
    if key in CAIXA_CITY_MAP:
        return CAIXA_CITY_MAP[key]
    city = key.title()
    for prep in _PT_PREPOSITIONS:
        city = city.replace(prep, prep.lower())
    return city


# ─── PARSING HELPERS ────────────────────────────────────────────────────────


def parse_brl_number(text: str | float | int | None) -> float | None:
    """Converts Brazilian number strings like 'R$ 340.000,00' or '1.250' to float."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    if not s:
        return None
    # Isolate number token
    m = re.search(r"(\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)", s)
    if not m:
        return None
    token = m.group(1)
    if "." in token and "," in token:
        token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        # If ends with ,00 or ,5 etc., it's a decimal comma
        parts = token.split(",")
        if len(parts[-1]) <= 2:
            token = token.replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "." in token:
        # In pt-BR, '340.000' has 3 digits after dot -> thousands separator
        parts = token.split(".")
        if len(parts[-1]) == 3:
            token = token.replace(".", "")
    try:
        return round(float(token), 2)
    except ValueError:
        return None


def normalize_prop_type(text: str) -> str | None:
    """Maps raw Portuguese property descriptions to casa, apartamento, terreno, or comercial."""
    t = text.strip().lower()
    # Strip subordinate phrases so "área do terreno" or "sala de estar" don't misclassify
    t_clean = re.sub(
        r"(?:área|area)\s+do\s+terreno|sala\s+(?:de\s+estar|de\s+jantar|para\s+\d|ampla|com\s+varanda)",
        "",
        t,
    )
    comercial_keywords = [
        "sala comercial",
        "sala-conjunto",
        "sala conjunto",
        "loja",
        "galpão",
        "galpao",
        "prédio",
        "comercial",
        "ponto",
    ]
    if any(k in t_clean for k in comercial_keywords) or re.match(r"^sala\b", t_clean):
        return "comercial"
    apt_keywords = ["apartamento", "apto", "flat", "studio", "cobertura", "kitnet", "loft"]
    if any(k in t_clean for k in apt_keywords):
        return "apartamento"
    if any(k in t_clean for k in ["casa", "sobrado", "duplex", "triplex", "mansão", "bangalô"]):
        return "casa"
    if any(k in t_clean for k in ["terreno", "lote", "gleba", "chácara", "chacara"]):
        return "terreno"
    return None


_LAND_WORDS_RE = re.compile(r"\b(?:terrenos?|lotes?)\b", flags=re.IGNORECASE)
_SOLD_RE = re.compile(r"\bvendido\b", flags=re.IGNORECASE)


def is_terreno_url(page_url: str) -> bool:
    """True when a listing page URL is a land (terreno / lote) catalog."""
    u = page_url.lower()
    return "/terreno" in u or "terreno-lote" in u


def resolve_prop_type(
    card_text: str,
    context: str,
    page_url: str,
    default: str = "apartamento",
) -> str:
    """Resolves a listing's type, defaulting to 'terreno' on land catalog URLs.

    On a terreno URL the card is kept as 'terreno' unless its text explicitly says
    casa/apartamento (or comercial) without also mentioning terreno/lote, since land ads
    often say "construa sua casa" or "terreno comercial".
    """
    if is_terreno_url(page_url):
        explicit = normalize_prop_type(card_text)
        if explicit in ("casa", "apartamento", "comercial") and not _LAND_WORDS_RE.search(
            card_text
        ):
            return explicit
        return "terreno"
    return normalize_prop_type(context) or normalize_prop_type(page_url) or default


_CITY_RN_RE = re.compile(
    r"(?:\s[-–]\s+)((?:(?!\s[-–]\s)[^/|]){3,40}?)\s*(?:\s[-–]\s*|/)\s*RN\b"
)
_CODE_IN_TEXT_RE = re.compile(r"\b[A-Z]{1,3}\d{3,6}(?:-[A-Z0-9]{2,6})?\b")
_NON_RN_UF_RE = re.compile(
    r"[-–/]\s*(?:AC|AL|AM|AP|BA|CE|DF|ES|GO|MA|MG|MS|MT|PA|PB|PE|PI|PR|RJ|RO|RR|RS|SC|SE|SP|TO)\b"
)


def detect_foreign_state(text: str) -> bool:
    """True when a listing's text places it in another Brazilian state (e.g. '... - PB')."""
    return bool(_NON_RN_UF_RE.search(_CODE_IN_TEXT_RE.sub(" ", text)))


def _find_known(options: Iterable[str], haystack: str, skip: Iterable[str] = ()) -> str:
    skipped = set(skip)
    for opt in sorted(options, key=len, reverse=True):
        if opt in skipped:
            continue
        if re.search(rf"\b{re.escape(opt)}\b", haystack, flags=re.IGNORECASE):
            return opt
    return ""


def detect_city_and_neighborhood(
    text: str,
    default_city: str = "Natal",
    fallbacks: Sequence[str] = (),
) -> tuple[str, str]:
    """Detects RN city and neighborhood, trusting the listing text over URL hints.

    City tiers: explicit '<City> - RN' / '<City>/RN' in the text, any known city in the
    text, then known cities in each fallback (listing link path, then catalog page URL),
    then ``default_city``. Neighborhood tiers: text, then fallbacks (ignoring the ambiguous
    word 'Centro' in URL hints, since URL slugs reuse it generically).
    """

    def norm(s: str) -> str:
        return s.replace("-", " ").replace("/", " ")

    city = ""
    m = _CITY_RN_RE.search(text)
    if m:
        cand = m.group(1).strip()
        city = _find_known(KNOWN_RN_CITIES, norm(cand))
        if not city and not _find_known(KNOWN_NEIGHBORHOODS, norm(cand)) and len(cand.split()) <= 4:
            city = cand
    city = city or _find_known(KNOWN_RN_CITIES, norm(text))
    for fb in fallbacks:
        if city:
            break
        city = _find_known(KNOWN_RN_CITIES, norm(fb))
    city = city or default_city

    neighborhood = _find_known(KNOWN_NEIGHBORHOODS, norm(text))
    for fb in fallbacks:
        if neighborhood:
            break
        neighborhood = _find_known(KNOWN_NEIGHBORHOODS, norm(fb), skip=("Centro",))

    return city, neighborhood


def extract_metrics_from_text(text: str, prop_type: str | None = None) -> dict[str, Any]:
    """Extracts area_m2, bedrooms, bathrooms, garage_spots, condo_fee_brl, iptu_brl."""
    clean = " ".join(text.replace("-", " ").split())

    # 1. Area in m2 (prefer área privativa -> área total/útil/construída -> área do terreno -> m2)
    area: float | None = None
    area_patterns = [
        r"(\d+(?:[.,]\d+)?)\s*(?:m[²2]\s*)?de\s*área\s*privativa",
        r"área\s*privativa\s*(?:de)?\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*m?[²2]?",
        r"(\d+(?:[.,]\d+)?)\s*(?:m[²2]\s*)?de\s*área\s*(?:total|útil|construída)",
        r"área\s*(?:total|útil|construída)\s*(?:de)?\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*m?[²2]?",
        r"(\d+(?:[.,]\d+)?)\s*(?:m[²2]\s*)?de\s*área\s*do\s*terreno",
        r"área\s*do\s*terreno\s*(?:de)?\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*m?[²2]?",
        r"(\d+(?:[.,]\d+)?)\s*m(?:²|2)\b",
    ]
    for pat in area_patterns:
        for m in re.finditer(pat, clean, flags=re.IGNORECASE):
            val = parse_brl_number(m.group(1))
            if val is not None and 15 <= val <= 500000:
                area = val
                break
        if area is not None:
            break

    # 2. Bedrooms (0 for terreno)
    bedrooms: int | None = None
    if prop_type == "terreno":
        bedrooms = 0
    else:
        m_bed = re.search(
            r"\b(\d{1,2})\s*(?:quartos?|dormit[óo]rios?|qto(?:\(s\))?s?|su[íi]tes?)\b"
            r"|\b(\d)\s*/\s*4\b",
            clean,
            flags=re.IGNORECASE,
        )
        if m_bed:
            b_str = m_bed.group(1) or m_bed.group(2)
            bedrooms = int(b_str)

    # 3. Bathrooms
    bathrooms: int | None = None
    m_bath = re.search(
        r"\b(\d{1,2})\s*(?:banheiros?|wc|bwc|ban\.)\b",
        clean,
        flags=re.IGNORECASE,
    )
    if m_bath:
        bathrooms = int(m_bath.group(1))
    elif re.search(r"\bwc\b", clean, flags=re.IGNORECASE):
        bathrooms = 1

    # 4. Garage spots
    garage: int | None = None
    m_gar = re.search(
        r"\b(\d{1,2})\s*(?:vaga(?:\(s\))?s?(?:\s*de\s*garagem)?|garagens?)\b",
        clean,
        flags=re.IGNORECASE,
    )
    if m_gar:
        garage = int(m_gar.group(1))

    # 5. Condomínio & IPTU
    condo: float | None = None
    m_condo = re.search(
        r"cond(?:om[íi]nio)?\.?\s*[:•-]?\s*R\$\s*([\d\.,]+)",
        clean,
        flags=re.IGNORECASE,
    )
    if m_condo:
        condo = parse_brl_number(m_condo.group(1))

    iptu: float | None = None
    m_iptu = re.search(
        r"iptu\.?\s*[:•-]?\s*R\$\s*([\d\.,]+)",
        clean,
        flags=re.IGNORECASE,
    )
    if m_iptu:
        iptu = parse_brl_number(m_iptu.group(1))

    return {
        "area_m2": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "garage_spots": garage,
        "condo_fee_brl": condo,
        "iptu_brl": iptu,
    }


def _get_row_val(row: pd.Series, col_map: dict[str, str], *key_parts: str) -> str:
    """Safely extracts a string value from a DataFrame row by partial column name."""
    for part in key_parts:
        for k, orig_col in col_map.items():
            if part in k:
                val = row[orig_col]
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    s = str(val).strip()
                    if s:
                        return s
    return ""


# ─── PIPELINE 1: CAIXA ECONÔMICA FEDERAL (RN) ───────────────────────────────


class CaixaScraper:
    """Downloads or reads Caixa's Lista_imoveis_RN.csv and parses RN properties."""

    def __init__(self, local_csv_path: str | Path | None = None):
        self.local_csv_path = local_csv_path
        self.row_errors = 0

    def fetch_raw_csv_text(self) -> str:
        """Reads the local Caixa CSV (preferred) or downloads it. Raises OSError on failure."""
        if self.local_csv_path:
            path = Path(self.local_csv_path)
            if path.is_file():
                print(f"[Caixa] Lendo arquivo local: {path}")
                return path.read_bytes().decode("latin-1", errors="replace")
            log.warning("[Caixa] Arquivo local não encontrado (%s); tentando baixar.", path)

        print(f"[Caixa] Baixando catálogo RN: {CAIXA_RN_CSV_URL}")
        resp = requests.get(CAIXA_RN_CSV_URL, headers=HEADERS, timeout=45)
        resp.raise_for_status()
        return resp.content.decode("latin-1", errors="replace")

    @staticmethod
    def _find_header_index(lines: list[str]) -> int | None:
        for idx, line in enumerate(lines):
            has_header_marker = (
                "N° do imóvel" in line
                or "Nº do imóvel" in line
                or "UF;Cidade;Bairro" in line
                or ("Cidade" in line and "Bairro" in line)
            )
            if has_header_marker and ";" in line:
                return idx
        return None

    def _parse_row(self, row: pd.Series, col_map: dict[str, str]) -> RawProperty | None:
        """Converts one Caixa CSV row into a RawProperty (None when it should be skipped)."""
        ext_code = _get_row_val(row, col_map, "imóvel", "imovel")
        city = normalize_city_name(_get_row_val(row, col_map, "cidade"))
        neighborhood = _get_row_val(row, col_map, "bairro").title()
        address = _get_row_val(row, col_map, "endereço", "endereco")
        price = parse_brl_number(_get_row_val(row, col_map, "preço", "preco"))
        appraisal = parse_brl_number(_get_row_val(row, col_map, "avaliação", "avaliacao"))
        discount = parse_brl_number(_get_row_val(row, col_map, "desconto"))
        desc = _get_row_val(row, col_map, "descrição", "descricao")
        modality = _get_row_val(row, col_map, "modalidade")
        link = _get_row_val(row, col_map, "link")
        financing_raw = _get_row_val(row, col_map, "financiamento").lower()
        accepts_financing: bool | None = None
        if financing_raw == "sim":
            accepts_financing = True
        elif financing_raw in ("não", "nao"):
            accepts_financing = False

        if not price or price < 15000:
            return None

        ptype = normalize_prop_type(desc) or normalize_prop_type(address) or "casa"
        if ptype == "comercial":
            return None

        metrics = extract_metrics_from_text(desc, prop_type=ptype)
        title = (
            address.split(",")[0].strip().title() if address else f"Imóvel Caixa {neighborhood}"
        )
        return RawProperty(
            source="caixa",
            external_code=ext_code,
            title=title[:90],
            prop_type=ptype,
            city=city,
            neighborhood=neighborhood,
            price_brl=price,
            appraisal_value_brl=appraisal,
            discount_pct=discount,
            area_m2=metrics["area_m2"],
            bedrooms=metrics["bedrooms"],
            bathrooms=metrics["bathrooms"],
            garage_spots=metrics["garage_spots"],
            condo_fee_brl=metrics["condo_fee_brl"],
            iptu_brl=metrics["iptu_brl"],
            sale_modality=modality or "Venda Caixa",
            accepts_financing=accepts_financing,
            url=link,
        )

    def run(self) -> list[RawProperty]:
        try:
            raw_text = self.fetch_raw_csv_text()
        except (OSError, requests.RequestException) as e:
            log.error(
                "[Caixa] Falha ao obter o CSV da Caixa (%s: %s). "
                "Dica: baixe pelo navegador e passe --caixa-file Lista_imoveis_RN.csv",
                type(e).__name__,
                e,
            )
            return []

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        header_idx = self._find_header_index(lines)
        if header_idx is None:
            log.error("[Caixa] Cabeçalho CSV não encontrado; o arquivo não parece ser da Caixa.")
            return []

        csv_body = "\n".join(lines[header_idx:])
        try:
            df = pd.read_csv(io.StringIO(csv_body), sep=";", on_bad_lines="skip")
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
            log.error("[Caixa] CSV ilegível (%s: %s).", type(e).__name__, e)
            return []
        df.columns = [c.strip() for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}

        missing = [k for k in ("cidade", "descri") if not any(k in c for c in col_map)]
        if not any(k in c for c in col_map for k in ("preço", "preco")):
            missing.append("preço")
        if missing:
            log.error(
                "[Caixa] Colunas obrigatórias ausentes: %s (colunas: %s)",
                missing,
                list(df.columns),
            )
            return []

        results: list[RawProperty] = []
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                item = self._parse_row(row, col_map)
            except Exception as e:  # one malformed row must not abort the whole catalog
                self.row_errors += 1
                log.warning("[Caixa] Linha %d ignorada (%s: %s).", idx, type(e).__name__, e)
                continue
            if item is not None:
                results.append(item)

        print(f"[Caixa] {len(results)} imóveis residenciais/terrenos extraídos no RN.")
        if self.row_errors:
            log.warning("[Caixa] %d linha(s) ignorada(s) por erro de parsing.", self.row_errors)
        return results


# ─── PIPELINE 2: LOCAL NATAL / RN REAL ESTATE AGENCIES ──────────────────────

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_DETAIL_HREF_RE = re.compile(r"/imovel|/\d{4,}|-id-\d+", flags=re.IGNORECASE)
_JUNK_RE = re.compile(
    r"\b\d+\s+(?:[\wÀ-ÿ]+\s+){0,3}(?:encontrad[oa]s?|resultados?)\b",
    flags=re.IGNORECASE,
)
_GENERIC_TITLES = {"terreno", "casa", "apartamento", "lote", "imóvel", "imovel", "venda"}


class AgencyScraper:
    """Scrapes local Natal/RN real estate agencies via JSON-LD and heuristic card parsing.

    Network errors are isolated per URL: transient failures (timeouts, connection errors,
    HTTP 429/5xx) are retried with exponential backoff; HTTP 401/403 is recorded as a block
    and is never worked around; 404/410 simply end pagination. Every failure is collected in
    ``failed_urls`` so the caller can report it. A malformed card never aborts a page.
    """

    def __init__(
        self,
        urls: Iterable[str],
        max_pages: int = 3,
        delay_s: float = 1.0,
        timeout_s: float = 25.0,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        max_detail_fetches: int = 80,
    ):
        self.urls = [u.strip() for u in urls if u.strip()]
        self.max_pages = max_pages
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)
        self.backoff_s = backoff_s
        self.max_detail_fetches = max_detail_fetches
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.failed_urls: list[tuple[str, str]] = []
        self.card_errors = 0
        self._catalog_urls: set[str] = set()

    # ── networking ──────────────────────────────────────────────────────────

    def _record_failure(self, url: str, reason: str) -> None:
        self.failed_urls.append((url, reason))
        log.warning("Falha em %s: %s", url, reason)

    def _fetch(self, url: str) -> tuple[str, requests.Response | None]:
        """GET with retries. Returns (status, response); status is ok/not_found/blocked/failed."""
        reason = "erro desconhecido"
        for attempt in range(1, self.max_retries + 1):
            wait = self.backoff_s * 2 ** (attempt - 1)
            try:
                resp = self.session.get(url, timeout=self.timeout_s)
            except (requests.Timeout, requests.ConnectionError) as exc:
                reason = type(exc).__name__
            except requests.RequestException as exc:
                self._record_failure(url, f"{type(exc).__name__}: {exc}")
                return "failed", None
            else:
                code = resp.status_code
                if code < 400:
                    return "ok", resp
                if code in (404, 410):
                    return "not_found", None
                if code in (401, 403):
                    self._record_failure(
                        url, f"acesso negado (HTTP {code}); possível bloqueio, não será contornado"
                    )
                    return "blocked", None
                reason = f"HTTP {code}"
                if code not in RETRYABLE_STATUS:
                    self._record_failure(url, reason)
                    return "failed", None
                retry_after = resp.headers.get("Retry-After", "") if resp.headers else ""
                if str(retry_after).isdigit():
                    wait = max(wait, float(retry_after))
            if attempt < self.max_retries:
                wait = min(wait, 60.0)
                log.warning("%s em %s; nova tentativa %d/%d em %.0fs", reason, url, attempt + 1,
                            self.max_retries, wait)
                time.sleep(wait)
        self._record_failure(url, f"{reason} após {self.max_retries} tentativas")
        return "failed", None

    def _paginated_urls(self, base_url: str) -> list[str]:
        """Generates candidate pagination URLs if max_pages > 1.

        suzananeryimoveis.com.br paginates by path (``/pagina-2/``); every other agency
        (Kenlo sites such as pereseperes, gilmaramaralimoveis, dnnegociosimobiliarios,
        marizaldaimoveis, portal.kenlo, and the rest) uses ``?pagina=N``.
        """
        pages = [base_url]
        host = urlparse(base_url).netloc.lower()
        for p in range(2, self.max_pages + 1):
            if "suzananeryimoveis.com.br" in host:
                pages.append(f"{base_url.rstrip('/')}/pagina-{p}/")
            else:
                sep = "&" if "?" in base_url else "?"
                pages.append(f"{base_url}{sep}pagina={p}")
        return pages

    @staticmethod
    def _signature(item: RawProperty) -> tuple[str, float | None]:
        """Identity used to detect pages that repeat already-seen listings."""
        return (item.external_code or item.url or item.title[:40], item.price_brl)

    # ── parsing ─────────────────────────────────────────────────────────────

    def _extract_json_ld(
        self,
        soup: BeautifulSoup,
        domain: str,
        page_url: str,
    ) -> list[RawProperty]:
        items: list[RawProperty] = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (ValueError, TypeError):
                continue
            nodes = data if isinstance(data, list) else [data]
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                if "@graph" in node and isinstance(node["@graph"], list):
                    nodes.extend(node["@graph"])
                    continue
                try:
                    item = self._parse_json_ld_node(node, domain, page_url)
                except Exception as exc:  # malformed node: skip it, keep the page
                    self.card_errors += 1
                    log.debug(
                        "Nó JSON-LD ignorado em %s (%s: %s)", page_url, type(exc).__name__, exc
                    )
                    continue
                if item is not None:
                    items.append(item)
        return items

    def _parse_json_ld_node(
        self,
        node: dict[str, Any],
        domain: str,
        page_url: str,
    ) -> RawProperty | None:
        ntype = str(node.get("@type", "")).lower()
        valid_types = ("product", "residence", "apartment", "house", "listing", "offer")
        if not any(k in ntype for k in valid_types):
            return None
        name = str(node.get("name") or node.get("headline") or "").strip()
        desc = str(node.get("description") or "").strip()
        full_text = f"{name} {desc}"
        if re.search(r"/\s*m[êe]s\b", full_text, flags=re.IGNORECASE):
            return None
        if _SOLD_RE.search(full_text) or detect_foreign_state(full_text):
            return None
        offers = node.get("offers") or {}
        if isinstance(offers, list) and offers:
            offers = offers[0]
        raw_price = offers.get("price") if isinstance(offers, dict) else node.get("price")
        price = parse_brl_number(raw_price)
        if not price or price < 25000:
            return None
        ptype = resolve_prop_type(full_text, full_text, page_url)
        if ptype == "comercial":
            return None
        city, nb = detect_city_and_neighborhood(full_text, fallbacks=[page_url])
        metrics = extract_metrics_from_text(full_text, prop_type=ptype)
        return RawProperty(
            source=domain,
            external_code=str(node.get("sku") or node.get("identifier") or ""),
            title=name[:90] or f"{ptype.title()} em {nb or city}",
            prop_type=ptype,
            city=city,
            neighborhood=nb,
            price_brl=price,
            area_m2=metrics["area_m2"],
            bedrooms=metrics["bedrooms"],
            bathrooms=metrics["bathrooms"],
            garage_spots=metrics["garage_spots"],
            condo_fee_brl=metrics["condo_fee_brl"],
            iptu_brl=metrics["iptu_brl"],
            sale_modality="Venda Padrão",
            url=urljoin(page_url, str(node.get("url") or page_url)),
        )

    @staticmethod
    def _looks_like_container(card: Tag, text: str) -> bool:
        """True for wrapper elements that hold several listing cards.

        Such wrappers carry the prices of every nested card; parsing them as one listing
        yields the highest price in the group paired with the first listing's code.
        """
        codes = {m.split("-")[0] for m in _CODE_IN_TEXT_RE.findall(text)}
        if len(codes) >= 2:
            return True
        ignored_hosts = ("wa.me", "whatsapp.com", "facebook.com", "instagram.com", "youtube.com")
        hrefs: set[str] = set()
        for a in card.find_all("a", href=True):
            href = str(a["href"]).split("#")[0].strip()
            low = href.lower()
            if low.startswith(("tel:", "mailto:", "javascript:")):
                continue
            if any(h in low for h in ignored_hosts):
                continue
            if _DETAIL_HREF_RE.search(href):
                hrefs.add(href)
        return len(hrefs) >= 2

    def _parse_card(self, card: Tag, domain: str, page_url: str) -> RawProperty | None:
        text = " ".join(card.get_text(" ", strip=True).split())
        # Filter out whole-page wrappers or tiny badges
        if len(text) < 35 or len(text) > 1400 or "R$" not in text:
            return None
        # Sold listings, "N imóveis encontrados" headers and multi-card wrappers
        if _SOLD_RE.search(text) or _JUNK_RE.search(text) or self._looks_like_container(card, text):
            return None
        # Listings located outside RN (e.g. "Zona Rural - Bananeiras - PB")
        if detect_foreign_state(text):
            return None

        # Strip explicit monthly rental prices (e.g., "R$ 3.500/mês" or "R$ 4.000 / mês")
        text_no_rent = re.sub(
            r"R\$\s*[\d\.,]+\s*(?:/\s*m[êe]s|por\s*m[êe]s)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        if "R$" not in text_no_rent:
            return None

        # Extract all R$ amounts; the largest >= 25,000 is typically the sale price
        prices = [
            parse_brl_number(m.group(1)) for m in re.finditer(r"R\$\s*([\d\.,]+)", text_no_rent)
        ]
        valid_sale_prices = [p for p in prices if p and 25000 <= p <= 15000000]
        if not valid_sale_prices:
            return None
        price = max(valid_sale_prices)

        # Link
        a_tag = card if card.name == "a" else card.find("a", href=True)
        href = a_tag.get("href") if isinstance(a_tag, Tag) else None
        link = urljoin(page_url, href) if isinstance(href, str) and href else page_url

        # Skip filter/search widget links (e.g., "?vma=200000" / "Pesquisa por Valor")
        if "vma=" in link or "pesquisa por valor" in text.lower():
            return None

        link_path = urlparse(link).path
        combined_context = f"{text} {link_path}"

        ptype = resolve_prop_type(text, combined_context, page_url)
        if ptype == "comercial":
            return None

        # Card text is the most reliable location signal, then the listing link, then the
        # catalog URL (a Parnamirim search page can contain Nísia Floresta listings).
        city, nb = detect_city_and_neighborhood(text, fallbacks=[link_path, page_url])
        metrics = extract_metrics_from_text(combined_context, prop_type=ptype)

        # Extract external listing code if present (e.g., Cód. 552, # 037, AP1880-PEP, id-569)
        ext_code = ""
        m_code = re.search(
            r"(?:c[óo]d\.?|ref\.?|#)\s*[:.]?\s*([A-Z0-9-]{2,14})\b"
            r"|\b([A-Z]{2}\d{3,5}(?:-[A-Z]{2,4})?)\b"
            r"|-id-(\d+)\b"
            r"|/imovel/(\d+)/",
            combined_context,
            flags=re.IGNORECASE,
        )
        if m_code:
            ext_code = (
                m_code.group(1) or m_code.group(2) or m_code.group(3) or m_code.group(4) or ""
            ).strip()

        # Heading or title
        heading = card.find(["h1", "h2", "h3", "h4", "strong"])
        title = heading.get_text(" ", strip=True) if isinstance(heading, Tag) else ""
        if not title or len(title) < 5 or "R$" in title:
            title = f"{ptype.title()} em {nb or city}"
        elif title.strip().lower() in _GENERIC_TITLES:
            title = f"{ptype.title()} em {f'{nb}, ' if nb else ''}{city}"

        return RawProperty(
            source=domain,
            external_code=ext_code,
            title=title[:90],
            prop_type=ptype,
            city=city,
            neighborhood=nb,
            price_brl=price,
            area_m2=metrics["area_m2"],
            bedrooms=metrics["bedrooms"],
            bathrooms=metrics["bathrooms"],
            garage_spots=metrics["garage_spots"],
            condo_fee_brl=metrics["condo_fee_brl"],
            iptu_brl=metrics["iptu_brl"],
            sale_modality="Venda Padrão",
            url=link,
        )

    def _extract_html_cards(
        self,
        soup: BeautifulSoup,
        domain: str,
        page_url: str,
    ) -> list[RawProperty]:
        items: list[RawProperty] = []
        seen_signatures: set[tuple[str, float | None]] = set()

        # Check custom domain selector first, else fallback to common card containers
        cfg = DOMAIN_CONFIGS.get(domain, {})
        if "card" in cfg:
            candidates = soup.select(cfg["card"])
        else:
            candidates = soup.find_all(
                ["article", "div", "li", "a"],
                class_=re.compile(
                    r"(imovel|imóvel|property|card|listing|item|box-imovel|resultado)",
                    re.IGNORECASE,
                ),
            )
            # Fallback: any container holding an "R$" price with bounded text length
            if not candidates:
                candidates = [
                    tag for tag in soup.find_all(["article", "div", "li"]) if "R$" in tag.get_text()
                ]

        for card in candidates:
            if not isinstance(card, Tag):
                continue
            try:
                item = self._parse_card(card, domain, page_url)
            except Exception as exc:  # one malformed card must not abort the page
                self.card_errors += 1
                log.debug("Card ignorado em %s (%s: %s)", page_url, type(exc).__name__, exc)
                continue
            if item is None:
                continue
            sig = self._signature(item)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            items.append(item)
        return items

    # ── detail-page enrichment ──────────────────────────────────────────────

    @staticmethod
    def _detail_text(html: str) -> str:
        """Title/description/heading text of a listing page (avoids 'related listings')."""
        soup = BeautifulSoup(html, "html.parser")
        parts: list[str] = []
        if soup.title and soup.title.string:
            parts.append(soup.title.string)
        meta_queries = (
            {"name": "description"},
            {"property": "og:title"},
            {"property": "og:description"},
        )
        for attrs in meta_queries:
            tag = soup.find("meta", attrs=attrs)
            if isinstance(tag, Tag) and tag.get("content"):
                parts.append(str(tag["content"]))
        parts.extend(h.get_text(" ", strip=True) for h in soup.find_all("h1"))
        return " ".join(parts)

    def enrich_missing_metrics(self, items: list[RawProperty]) -> int:
        """Fills missing area/rooms from each listing's detail page. Returns areas filled.

        Some catalogs (e.g. Suzanery) show only price and code on list cards. Rows pointing
        at a catalog page rather than a specific listing are skipped. Failures never raise.
        """
        if self.max_detail_fetches <= 0:
            return 0
        targets = [
            it
            for it in items
            if it.area_m2 is None and it.url and it.url not in self._catalog_urls
        ]
        if not targets:
            return 0
        # Deduplicate URL fetches while applying metrics to every row sharing the same URL
        unique_urls: list[str] = []
        seen_urls: set[str] = set()
        for it in targets:
            if it.url not in seen_urls:
                seen_urls.add(it.url)
                unique_urls.append(it.url)
                if len(unique_urls) >= self.max_detail_fetches:
                    break

        print(f"[Detalhes] Buscando área em {len(unique_urls)} páginas de anúncio sem metragem ...")
        metrics_by_url: dict[str, dict[str, Any]] = {}
        for url in unique_urls:
            _, resp = self._fetch(url)
            if resp is None:
                continue
            try:
                detail_txt = self._detail_text(resp.text)
                metrics_by_url[url] = extract_metrics_from_text(detail_txt)
            except Exception as exc:
                log.warning("Detalhe ilegível em %s (%s: %s)", url, type(exc).__name__, exc)
                continue
            time.sleep(self.delay_s)

        filled = 0
        for it in targets:
            metrics = metrics_by_url.get(it.url)
            if not metrics:
                continue
            if metrics["area_m2"] is not None:
                it.area_m2 = metrics["area_m2"]
                filled += 1
            for attr in ("bedrooms", "bathrooms", "garage_spots", "condo_fee_brl", "iptu_brl"):
                if attr == "bedrooms" and it.prop_type == "terreno":
                    continue
                if getattr(it, attr) is None and metrics[attr] is not None:
                    setattr(it, attr, metrics[attr])
        print(f"[Detalhes] Área preenchida em {filled}/{len(targets)} anúncios.")
        return filled

    # ── orchestration ───────────────────────────────────────────────────────

    def run(self) -> list[RawProperty]:
        all_items: list[RawProperty] = []
        for base_url in self.urls:
            domain = urlparse(base_url).netloc.replace("www.", "")
            seen_signatures: set[tuple[str, float | None]] = set()
            for page_idx, page_url in enumerate(self._paginated_urls(base_url)):
                self._catalog_urls.add(page_url)
                print(f"[Agência: {domain}] Coletando {page_url} ...")
                status, resp = self._fetch(page_url)
                if resp is None:
                    if status == "not_found" and page_idx > 0:
                        print(f"Fim da paginação (404) em {page_url}")
                    elif status == "not_found":
                        self._record_failure(page_url, "HTTP 404 na URL base")
                    break
                try:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    combined_page = self._extract_json_ld(
                        soup, domain, page_url
                    ) + self._extract_html_cards(soup, domain, page_url)
                except Exception as exc:
                    self._record_failure(page_url, f"HTML ilegível ({type(exc).__name__}: {exc})")
                    break
                if not combined_page:
                    break
                # Stop early when a page adds nothing new (site ignored the page param)
                new_items = []
                for item in combined_page:
                    sig = self._signature(item)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        new_items.append(item)
                if page_idx > 0 and not new_items:
                    print(f"Página {page_idx + 1} sem anúncios novos; encerrando paginação.")
                    break
                all_items.extend(new_items)
                print(f"+{len(new_items)} anúncios novos coletados.")
                time.sleep(self.delay_s)
        self.enrich_missing_metrics(all_items)
        if self.card_errors:
            log.warning("%d card(s) ignorado(s) por erro de parsing.", self.card_errors)
        return all_items


def load_agencies_from_csv(csv_path: str | Path) -> list[RawProperty]:
    """Loads existing non-Caixa rows (source != "caixa") from a CSV as RawProperty objects.

    Used by --agencies-csv for fast, reproducible offline merges instead of re-scraping
    the live agency websites. Missing columns (e.g. accepts_financing in older snapshots)
    are left as None / empty. Raises ScrapeError for unreadable files; malformed rows are
    skipped with a warning.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise ScrapeError(f"--agencies-csv não encontrado: {csv_path}") from e
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise ScrapeError(f"--agencies-csv ilegível ({csv_path}): {type(e).__name__}: {e}") from e
    except OSError as e:
        raise ScrapeError(f"--agencies-csv inacessível ({csv_path}): {e}") from e
    if "source" not in df.columns:
        raise ScrapeError(f"{csv_path}: coluna 'source' ausente; não é um CSV de imóveis válido.")
    df = df[df["source"].astype(str).str.strip().str.lower() != "caixa"]

    def _str(row: pd.Series, col: str) -> str:
        val = row.get(col)
        return "" if val is None or pd.isna(val) else str(val).strip()

    def _float(row: pd.Series, col: str) -> float | None:
        val = row.get(col)
        return None if val is None or pd.isna(val) else float(val)

    def _int(row: pd.Series, col: str) -> int | None:
        val = _float(row, col)
        return None if val is None else int(val)

    def _bool(row: pd.Series, col: str) -> bool | None:
        val = row.get(col)
        if val is None or pd.isna(val):
            return None
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("true", "sim", "1"):
                return True
            if low in ("false", "não", "nao", "0"):
                return False
            return None
        return bool(val)

    items: list[RawProperty] = []
    skipped = 0
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            items.append(
                RawProperty(
                    source=_str(row, "source"),
                    external_code=_str(row, "external_code"),
                    title=_str(row, "title"),
                    prop_type=_str(row, "prop_type"),
                    city=_str(row, "city"),
                    neighborhood=_str(row, "neighborhood"),
                    price_brl=_float(row, "price_brl"),
                    appraisal_value_brl=_float(row, "appraisal_value_brl"),
                    discount_pct=_float(row, "discount_pct"),
                    area_m2=_float(row, "area_m2"),
                    bedrooms=_int(row, "bedrooms"),
                    bathrooms=_int(row, "bathrooms"),
                    garage_spots=_int(row, "garage_spots"),
                    condo_fee_brl=_float(row, "condo_fee_brl"),
                    iptu_brl=_float(row, "iptu_brl"),
                    sale_modality=_str(row, "sale_modality"),
                    accepts_financing=_bool(row, "accepts_financing"),
                    url=_str(row, "url"),
                )
            )
        except (ValueError, TypeError) as e:
            skipped += 1
            log.warning(
                "[Agências] Linha %d de %s ignorada (%s: %s).",
                idx,
                csv_path,
                type(e).__name__,
                e,
            )
    if skipped:
        log.warning("[Agências] %d linha(s) inválida(s) ignorada(s) em %s.", skipped, csv_path)
    print(f"[Agências] {len(items)} imóveis não-Caixa carregados de {csv_path} (modo offline).")
    return items


# ─── VALIDATION, DEDUPLICATION & EXPORT ─────────────────────────────────────

# Agency-prefixed codes such as TE0048-MARI are unique across sites, so the same code on two
# domains is the same ad (e.g. Kenlo's portal re-listing another agency's property).
GLOBAL_CODE_RE = re.compile(r"^[A-Za-z]{2}\d{3,6}-[A-Za-z0-9]{2,6}$")
AGGREGATOR_SOURCES: tuple[str, ...] = ("portal.kenlo.com.br",)


def _invalid_reason(item: RawProperty) -> str | None:
    """Returns why a collected row must be discarded, or None if it is acceptable."""
    price = item.price_brl
    if price is None or pd.isna(price) or price < 15000:
        return "preço ausente/inválido"
    if price > 50_000_000:
        return "preço implausível (> R$ 50 mi)"
    if item.source == "caixa":
        return None
    if detect_foreign_state(item.title):
        return "imóvel fora do RN"
    if _JUNK_RE.search(item.title):
        return "cabeçalho de página (não é anúncio)"
    if item.prop_type == "terreno" and item.area_m2:
        ppm = price / item.area_m2
        if ppm > 50_000 or ppm < 1:
            return "preço/m² implausível"
    return None


def sanitize_items(items: list[RawProperty]) -> list[RawProperty]:
    """Drops invalid rows and nulls implausible areas, logging a summary by reason."""
    kept: list[RawProperty] = []
    dropped: Counter[str] = Counter()
    bad_area = 0
    for item in items:
        reason = _invalid_reason(item)
        if reason:
            dropped[reason] += 1
            continue
        if item.area_m2 is not None and not (15 <= item.area_m2 <= 500000):
            item.area_m2 = None
            bad_area += 1
        kept.append(item)
    for reason, n in dropped.items():
        log.warning("[Validação] %d linha(s) descartada(s): %s", n, reason)
    if bad_area:
        log.warning("[Validação] %d área(s) fora de [15, 500000] m² zerada(s).", bad_area)
    return kept


def deduplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate ads, keeping the richest row (aggregator copies lose ties).

    1. Same (source, external_code) or (source, title, price).
    2. Same agency-prefixed code on different sites (cross-agency duplicates).
    3. Same specific listing URL (catalog/home URLs shared by several prices are ignored).
    4. Card without code/link whose (source, price) matches a row with a specific link.
    """
    before = len(df)
    df = df.copy()
    df["_completeness"] = (
        df["external_code"].astype(bool).astype(int) * 3
        + df["area_m2"].notna().astype(int) * 2
        + df["bedrooms"].notna().astype(int)
        + df["bathrooms"].notna().astype(int)
        + df["garage_spots"].notna().astype(int)
        + df["neighborhood"].astype(bool).astype(int)
    )
    df["_aggregator"] = df["source"].isin(AGGREGATOR_SOURCES).astype(int)
    df = df.sort_values(
        by=["_completeness", "_aggregator"], ascending=[False, True], kind="mergesort"
    )

    df["_dedup_key"] = df.apply(
        lambda r: (
            f"{r['source']}:{r['external_code']}"
            if r["external_code"]
            else f"{r['source']}:{r['title']}:{r['price_brl']}"
        ),
        axis=1,
    )
    df = df.drop_duplicates(subset=["_dedup_key"])

    code = df["external_code"].astype(str).str.strip()
    is_global = code.str.match(GLOBAL_CODE_RE)
    cross_dup = is_global & code.where(is_global).str.upper().duplicated()
    df = df.loc[~cross_dup].copy()

    def _is_specific_url(frame: pd.DataFrame) -> pd.Series:
        prices_by_url: dict[str, set[float]] = {}
        for u, p in zip(frame["url"].astype(str), frame["price_brl"], strict=True):
            if u:
                prices_by_url.setdefault(u, set()).add(float(p))
        generic = {u for u, ps in prices_by_url.items() if len(ps) > 1}
        flags = [
            bool(u) and u not in generic and bool(_DETAIL_HREF_RE.search(urlparse(u).path))
            for u in frame["url"].astype(str)
        ]
        return pd.Series(flags, index=frame.index, dtype=bool)

    specific = _is_specific_url(df)
    url_dup = specific & df["url"].where(specific).duplicated()
    df = df.loc[~url_dup].copy()

    specific = _is_specific_url(df)
    has_code = df["external_code"].astype(str).str.strip() != ""
    anchored = specific | has_code
    anchored_keys = set(
        zip(df.loc[anchored, "source"], df.loc[anchored, "price_brl"], strict=True)
    )
    orphan = (
        ~anchored
        & pd.Series(
            [(s, p) in anchored_keys for s, p in zip(df["source"], df["price_brl"], strict=True)],
            index=df.index,
        )
    )
    df = df.loc[~orphan].copy()

    removed = before - len(df)
    if removed:
        log.info("[Dedup] %d anúncio(s) duplicado(s) removido(s).", removed)
    return df.drop(columns=["_dedup_key", "_completeness", "_aggregator"]).reset_index(drop=True)


def finalize_and_save(items: list[RawProperty], output_path: str) -> pd.DataFrame:
    items = sanitize_items(items)
    if not items:
        print("Nenhum imóvel coletado.")
        return pd.DataFrame()

    df = deduplicate_listings(pd.DataFrame([asdict(x) for x in items]))

    # Land has no bedrooms: enforce 0 for every terreno row (scraped or loaded from CSV)
    df.loc[df["prop_type"] == "terreno", "bedrooms"] = 0

    # Ensure integer columns use nullable Int64 so CSV outputs integers instead of floats
    for int_col in ("bedrooms", "bathrooms", "garage_spots"):
        df[int_col] = df[int_col].astype("Int64")
    # Nullable boolean: True/False for Caixa's Financiamento column, <NA> for agencies
    df["accepts_financing"] = df["accepts_financing"].astype("boolean")

    # Assign sequential IDs r001, r002, ...
    df["prop_id"] = [f"r{i:03d}" for i in range(1, len(df) + 1)]

    # Atomic write: never leave a half-written CSV behind if saving fails midway
    abs_out = os.path.abspath(output_path)
    tmp_out = f"{abs_out}.tmp"
    try:
        os.makedirs(os.path.dirname(abs_out), exist_ok=True)
        df.to_csv(tmp_out, index=False)
        os.replace(tmp_out, abs_out)
    except OSError as e:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        raise ScrapeError(f"Não foi possível salvar {output_path}: {e}") from e

    print("\n" + "═" * 68)
    print(f"Arquivo salvo em: {output_path} ({len(df)} imóveis reais únicos)")
    print("═" * 68)
    print("\nPor Fonte:")
    print(df["source"].value_counts().to_string())
    print("\nPor Tipo de Imóvel:")
    print(df["prop_type"].value_counts().to_string())
    print("\nPor Cidade:")
    print(df["city"].value_counts().head(10).to_string())
    print("\nPreenchimento das Colunas (não-nulos):")
    print(df.notna().sum().to_string())
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scraper Modular de Imóveis Reais no RN")
    parser.add_argument(
        "--mode",
        choices=["caixa", "agencies", "combined"],
        default="combined",
        help="Fonte de coleta: caixa, agencies ou combined",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=[],
        help="URLs adicionais de imobiliárias locais para coletar",
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default=None,
        help="Arquivo .txt com uma URL de imobiliária por linha",
    )
    parser.add_argument(
        "--caixa-file",
        type=str,
        default=str(Path("data/Lista_imoveis_RN.csv")),
        help="Caminho local opcional para Lista_imoveis_RN.csv da Caixa",
    )
    parser.add_argument(
        "--agencies-csv",
        type=str,
        default=None,
        help=(
            "CSV existente cujas linhas não-Caixa (source != 'caixa') serão carregadas no "
            "lugar de re-coletar os sites das imobiliárias (merge offline)"
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="Número máximo de páginas por URL de imobiliária",
    )
    parser.add_argument(
        "--max-detail-fetches",
        type=int,
        default=80,
        help="Máx. de páginas de anúncio visitadas para completar metragem ausente (0 desativa)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/real_properties.csv",
        help="Caminho do CSV de saída (padrão: data/real_properties.csv)",
    )
    return parser


def _read_urls_file(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except (OSError, UnicodeDecodeError) as e:
        raise ScrapeError(f"--urls-file ilegível ({path}): {type(e).__name__}: {e}") from e


def _run(args: argparse.Namespace) -> int:
    agency_urls = list(DEFAULT_AGENCY_URLS) + list(args.urls)
    if args.urls_file:
        agency_urls.extend(_read_urls_file(args.urls_file))

    collected: list[RawProperty] = []
    if args.mode in ("caixa", "combined"):
        caixa_items = CaixaScraper(local_csv_path=args.caixa_file).run()
        if not caixa_items and args.mode == "caixa":
            raise ScrapeError("Nenhum imóvel da Caixa coletado; saída NÃO sobrescrita.")
        if not caixa_items:
            log.warning("Caixa não retornou imóveis; seguindo apenas com as imobiliárias.")
        collected.extend(caixa_items)

    scraper: AgencyScraper | None = None
    if args.mode in ("agencies", "combined"):
        if args.agencies_csv:
            agency_items = load_agencies_from_csv(args.agencies_csv)
        else:
            scraper = AgencyScraper(
                urls=agency_urls,
                max_pages=args.max_pages,
                max_detail_fetches=args.max_detail_fetches,
            )
            agency_items = scraper.run()
        if not agency_items:
            # Typical when offline/blocked: don't replace a good CSV with a Caixa-only one.
            raise ScrapeError(
                "Nenhum anúncio de imobiliária coletado (rede indisponível ou bloqueio?); "
                "saída NÃO sobrescrita."
            )
        collected.extend(agency_items)

    df = finalize_and_save(collected, args.output)
    if df.empty:
        raise ScrapeError("Nada a salvar após a validação; saída NÃO sobrescrita.")

    if scraper is not None and scraper.failed_urls:
        log.warning("\nResumo de falhas (%d):", len(scraper.failed_urls))
        for url, reason in scraper.failed_urls:
            log.warning("  - %s -> %s", url, reason)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    if args.max_pages < 1:
        parser.error("--max-pages deve ser >= 1")
    if args.max_detail_fetches < 0:
        parser.error("--max-detail-fetches deve ser >= 0")
    if args.urls_file and not os.path.isfile(args.urls_file):
        parser.error(f"--urls-file não encontrado: {args.urls_file}")
    try:
        return _run(args)
    except ScrapeError as e:
        log.error("Erro: %s", e)
        return 2
    except KeyboardInterrupt:
        log.error("Interrompido pelo usuário; nenhum arquivo foi gravado.")
        return 130
    except Exception:
        log.exception("Erro inesperado")
        return 1


if __name__ == "__main__":
    sys.exit(main())
