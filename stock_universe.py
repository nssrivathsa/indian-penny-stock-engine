"""
Curated & Dynamic NSE/BSE Penny Stock Universe + All-Price Sensex/Nifty Leaders + Screener.in Live Discovery.

Supports both:
1. Strict Penny / Micro-Cap Universe (e.g. Price <= ₹50 or user slider cap)
2. "Any Price (Irrespective of Price)" Mode — adds top Sensex / Nifty 50 / Nifty Midcap & PSU
   breakout leaders (₹50 to ₹5,000+) so the engine can pick the #1 stock across the entire Indian market.
"""

from __future__ import annotations
import re
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

# Curated high-liquidity NSE/BSE stocks that trade (or frequently trade) in the < ₹50 or near-₹50 penny/micro-cap zone.
CURATED_INDIAN_PENNY_UNIVERSE: List[Dict[str, str]] = [
    # Power, Energy & Utilities
    {"symbol": "RPOWER.NS", "screener_slug": "RPOWER", "name": "Reliance Power", "sector": "Power & Energy"},
    {"symbol": "JPPOWER.NS", "screener_slug": "JPPOWER", "name": "Jaiprakash Power Ventures", "sector": "Power & Energy"},
    {"symbol": "RTNPOWER.NS", "screener_slug": "RTNPOWER", "name": "RattanIndia Power", "sector": "Power & Energy"},
    {"symbol": "ORIENTGREEN.NS", "screener_slug": "ORIENTGREEN", "name": "Orient Green Power", "sector": "Renewable Energy"},
    {"symbol": "URJA.NS", "screener_slug": "URJA", "name": "Urja Global", "sector": "Renewable Energy"},
    {"symbol": "GVKone.NS", "screener_slug": "GVKPIL", "name": "GVK Power & Infrastructure", "sector": "Power & Infra"},
    # Banking & Financial Services / NBFCs
    {"symbol": "YESBANK.NS", "screener_slug": "YESBANK", "name": "Yes Bank", "sector": "Banking"},
    {"symbol": "SOUTHINDZK.NS", "screener_slug": "SOUTHBANK", "name": "South Indian Bank", "sector": "Banking"},
    {"symbol": "UCOBANK.NS", "screener_slug": "UCOBANK", "name": "UCO Bank", "sector": "Public Sector Bank"},
    {"symbol": "CENTRALBK.NS", "screener_slug": "CENTRALBK", "name": "Central Bank of India", "sector": "Public Sector Bank"},
    {"symbol": "IOB.NS", "screener_slug": "IOB", "name": "Indian Overseas Bank", "sector": "Public Sector Bank"},
    {"symbol": "MAHABANK.NS", "screener_slug": "MAHABANK", "name": "Bank of Maharashtra", "sector": "Public Sector Bank"},
    {"symbol": "PSB.NS", "screener_slug": "PSB", "name": "Punjab & Sind Bank", "sector": "Public Sector Bank"},
    {"symbol": "IFCI.NS", "screener_slug": "IFCI", "name": "IFCI Ltd", "sector": "Financial Services"},
    {"symbol": "PAISALO.NS", "screener_slug": "PAISALO", "name": "Paisalo Digital", "sector": "NBFC / Fintech"},
    {"symbol": "UGROCAP.NS", "screener_slug": "UGROCAP", "name": "Ugro Capital", "sector": "NBFC"},
    {"symbol": "SALASAR.NS", "screener_slug": "SALASAR", "name": "Salasar Techno Engineering", "sector": "Engineering & Infra"},
    # Infrastructure, EPC & Real Estate
    {"symbol": "HCC.NS", "screener_slug": "HCC", "name": "Hindustan Construction Co", "sector": "Infrastructure & EPC"},
    {"symbol": "PATELENG.NS", "screener_slug": "PATELENG", "name": "Patel Engineering", "sector": "Infrastructure & Hydro"},
    {"symbol": "IRB.NS", "screener_slug": "IRB", "name": "IRB Infrastructure Developers", "sector": "Roads & Highways"},
    {"symbol": "GMRINFRA.NS", "screener_slug": "GMRAIRPORT", "name": "GMR Airports Infrastructure", "sector": "Airports & Infra"},
    {"symbol": "UNITECH.NS", "screener_slug": "UNITECH", "name": "Unitech", "sector": "Real Estate"},
    {"symbol": "NBCC.NS", "screener_slug": "NBCC", "name": "NBCC (India)", "sector": "Infrastructure"},
    {"symbol": "VASCONEQ.NS", "screener_slug": "VASCONEQ", "name": "Vascon Engineers", "sector": "EPC & Real Estate"},
    # Metals, Steel & Mining
    {"symbol": "RAMASTEEL.NS", "screener_slug": "RAMASTEEL", "name": "Rama Steel Tubes", "sector": "Steel & Tubes"},
    {"symbol": "LLOYDSENGG.NS", "screener_slug": "LLOYDSENGG", "name": "Lloyds Engineering Works", "sector": "Heavy Engineering"},
    {"symbol": "NAVKARCORP.NS", "screener_slug": "NAVKARCORP", "name": "Navkar Corporation", "sector": "Logistics & ICD"},
    {"symbol": "HARRMALAYA.NS", "screener_slug": "HARRMALAYA", "name": "Harrisons Malayalam", "sector": "Agri & Plantations"},
    # Sugar, Ethanol & Agri
    {"symbol": "BAJAJHIND.NS", "screener_slug": "BAJAJHIND", "name": "Bajaj Hindusthan Sugar", "sector": "Sugar & Ethanol"},
    {"symbol": "RENUKA.NS", "screener_slug": "RENUKA", "name": "Shree Renuka Sugars", "sector": "Sugar & Ethanol"},
    {"symbol": "DAVANGERE.NS", "screener_slug": "DAVANGERE", "name": "Davangere Sugar Company", "sector": "Sugar & Ethanol"},
    {"symbol": "KMSUGAR.NS", "screener_slug": "KMSUGAR", "name": "K M Sugar Mills", "sector": "Sugar & Ethanol"},
    # Textiles, Apparel & Consumer
    {"symbol": "ALOKINDS.NS", "screener_slug": "ALOKINDS", "name": "Alok Industries", "sector": "Textiles"},
    {"symbol": "TRIDENT.NS", "screener_slug": "TRIDENT", "name": "Trident Ltd", "sector": "Textiles & Paper"},
    {"symbol": "FILATEX.NS", "screener_slug": "FILATEX", "name": "Filatex India", "sector": "Synthetic Textiles"},
    {"symbol": "VIPULLTD.NS", "screener_slug": "VIPULLTD", "name": "Vipul Ltd", "sector": "Real Estate"},
    {"symbol": "SYNCOMF.NS", "screener_slug": "SYNCOMF", "name": "Syncom Formulations (India)", "sector": "Pharmaceuticals"},
    {"symbol": "MOREPENLAB.NS", "screener_slug": "MOREPENLAB", "name": "Morepen Laboratories", "sector": "Pharmaceuticals"},
    # Telecom, Media & Tech
    {"symbol": "IDEA.NS", "screener_slug": "IDEA", "name": "Vodafone Idea", "sector": "Telecom"},
    {"symbol": "TTML.NS", "screener_slug": "TTML", "name": "Tata Teleservices (Maharashtra)", "sector": "Telecom"},
    {"symbol": "HFCL.NS", "screener_slug": "HFCL", "name": "HFCL Ltd", "sector": "Telecom Equipment"},
    {"symbol": "GTLINFRA.NS", "screener_slug": "GTLINFRA", "name": "GTL Infrastructure", "sector": "Telecom Towers"},
    {"symbol": "ZEELEARN.NS", "screener_slug": "ZEELEARN", "name": "Zee Learn", "sector": "Education"},
    {"symbol": "DISHTV.NS", "screener_slug": "DISHTV", "name": "Dish TV India", "sector": "Media & DTH"},
    {"symbol": "HATHWAY.NS", "screener_slug": "HATHWAY", "name": "Hathway Cable & Datacom", "sector": "Broadband & Cable"},
    {"symbol": "DEN.NS", "screener_slug": "DEN", "name": "DEN Networks", "sector": "Broadband & Cable"},
    {"symbol": "NETWORK18.NS", "screener_slug": "NETWORK18", "name": "Network18 Media & Investments", "sector": "Media"},
    {"symbol": "INFIBEAM.NS", "screener_slug": "INFIBEAM", "name": "Infibeam Avenues", "sector": "Fintech & Payments"},
    {"symbol": "VAKRANGEE.NS", "screener_slug": "VAKRANGEE", "name": "Vakrangee Ltd", "sector": "IT & E-Governance"},
    {"symbol": "FCSOFT.NS", "screener_slug": "FCSSOFT", "name": "FCS Software Solutions", "sector": "IT Services"},
    {"symbol": "BRIGHTCOM.NS", "screener_slug": "BCG", "name": "Brightcom Group", "sector": "AdTech"},
    # Specialty Manufacturing & Chemicals
    {"symbol": "MISHTANN.NS", "screener_slug": "MISHTANN", "name": "Mishtann Foods", "sector": "FMCG & Agri"},
    {"symbol": "SARVESHWAR.NS", "screener_slug": "SARVESHWAR", "name": "Sarveshwar Foods", "sector": "FMCG & Agri"},
    {"symbol": "SUZLON.NS", "screener_slug": "SUZLON", "name": "Suzlon Energy", "sector": "Renewable Wind Energy"},
    {"symbol": "PARADEEP.NS", "screener_slug": "PARADEEP", "name": "Paradeep Phosphates", "sector": "Fertilizers"},
    {"symbol": "NFL.NS", "screener_slug": "NFL", "name": "National Fertilizers", "sector": "Fertilizers"},
    {"symbol": "RCF.NS", "screener_slug": "RCF", "name": "Rashtriya Chemicals & Fertilizers", "sector": "Fertilizers"},
]

# Sensex / Nifty 50 / Midcap & PSU High-Conviction Leaders (₹50 to ₹4,000+) included when "Ignore Price Cap" is checked
SENSEX_NIFTY_ALL_PRICE_LEADERS: List[Dict[str, str]] = [
    {"symbol": "IRFC.NS", "screener_slug": "IRFC", "name": "Indian Railway Finance Corp", "sector": "Railway PSU Finance"},
    {"symbol": "RVNL.NS", "screener_slug": "RVNL", "name": "Rail Vikas Nigam", "sector": "Railway Infrastructure"},
    {"symbol": "IREDA.NS", "screener_slug": "IREDA", "name": "Indian Renewable Energy Dev Agency", "sector": "Green Finance PSU"},
    {"symbol": "HUDCO.NS", "screener_slug": "HUDCO", "name": "Housing & Urban Dev Corp", "sector": "Infra Finance PSU"},
    {"symbol": "RECLTD.NS", "screener_slug": "RECLTD", "name": "REC Ltd", "sector": "Power Finance PSU"},
    {"symbol": "PFC.NS", "screener_slug": "PFC", "name": "Power Finance Corporation", "sector": "Power Finance PSU"},
    {"symbol": "BEL.NS", "screener_slug": "BEL", "name": "Bharat Electronics", "sector": "Defence & Aerospace"},
    {"symbol": "BHEL.NS", "screener_slug": "BHEL", "name": "Bharat Heavy Electricals", "sector": "Power Equipment"},
    {"symbol": "TATASTEEL.NS", "screener_slug": "TATASTEEL", "name": "Tata Steel", "sector": "Metals & Steel"},
    {"symbol": "NATIONALUM.NS", "screener_slug": "NATIONALUM", "name": "National Aluminium Co (NALCO)", "sector": "Metals & Mining"},
    {"symbol": "NMDC.NS", "screener_slug": "NMDC", "name": "NMDC Ltd", "sector": "Iron Ore & Mining"},
    {"symbol": "COALINDIA.NS", "screener_slug": "COALINDIA", "name": "Coal India", "sector": "Mining & Energy"},
    {"symbol": "NTPC.NS", "screener_slug": "NTPC", "name": "NTPC Ltd", "sector": "Power Generation"},
    {"symbol": "TATAPOWER.NS", "screener_slug": "TATAPOWER", "name": "Tata Power Company", "sector": "Power & EV"},
    {"symbol": "ONGC.NS", "screener_slug": "ONGC", "name": "Oil & Natural Gas Corp", "sector": "Oil & Gas"},
    {"symbol": "IOC.NS", "screener_slug": "IOC", "name": "Indian Oil Corporation", "sector": "Oil & Refining"},
    {"symbol": "PNB.NS", "screener_slug": "PNB", "name": "Punjab National Bank", "sector": "Public Sector Bank"},
    {"symbol": "CANBK.NS", "screener_slug": "CANBK", "name": "Canara Bank", "sector": "Public Sector Bank"},
    {"symbol": "BANKBARODA.NS", "screener_slug": "BANKBARODA", "name": "Bank of Baroda", "sector": "Public Sector Bank"},
    {"symbol": "SBIN.NS", "screener_slug": "SBIN", "name": "State Bank of India", "sector": "Public Sector Bank"},
    {"symbol": "FEDERALBNK.NS", "screener_slug": "FEDERALBNK", "name": "Federal Bank", "sector": "Private Bank"},
    {"symbol": "IDFCFIRSTB.NS", "screener_slug": "IDFCFIRSTB", "name": "IDFC First Bank", "sector": "Private Bank"},
    {"symbol": "ITC.NS", "screener_slug": "ITC", "name": "ITC Ltd", "sector": "FMCG & Consumer"},
    {"symbol": "ZOMATO.NS", "screener_slug": "ETERNAL", "name": "Eternal (Zomato)", "sector": "Consumer Tech"},
    {"symbol": "ASHOKLEY.NS", "screener_slug": "ASHOKLEY", "name": "Ashok Leyland", "sector": "Commercial Vehicles"},
    {"symbol": "TATAMOTORS.NS", "screener_slug": "TATAMOTORS", "name": "Tata Motors", "sector": "Automobiles"},
    {"symbol": "SUNPHARMA.NS", "screener_slug": "SUNPHARMA", "name": "Sun Pharmaceutical", "sector": "Pharmaceuticals"},
    {"symbol": "TCS.NS", "screener_slug": "TCS", "name": "Tata Consultancy Services", "sector": "IT Services"},
    {"symbol": "INFY.NS", "screener_slug": "INFY", "name": "Infosys", "sector": "IT Services"},
    {"symbol": "RELIANCE.NS", "screener_slug": "RELIANCE", "name": "Reliance Industries", "sector": "Conglomerate & Energy"},
    {"symbol": "HAL.NS", "screener_slug": "HAL", "name": "Hindustan Aeronautics", "sector": "Defence & Aerospace"},
]


def discover_screener_penny_stocks(limit: int = 15) -> List[Dict[str, str]]:
    """
    Attempts to scrape Screener.in public penny stock screens for additional live NSE/BSE symbols.
    Returns a list of symbol dicts compatible with CURATED_INDIAN_PENNY_UNIVERSE.
    """
    discovered: List[Dict[str, str]] = []
    urls = [
        "https://www.screener.in/screens/292119/penny-stocks-below-50-rs/",
        "https://www.screener.in/screens/699416/debt-free-penny-stocks/",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    seen_slugs = {item["screener_slug"] for item in CURATED_INDIAN_PENNY_UNIVERSE}

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, verify=False, timeout=6)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for a_tag in soup.select("table.data-table tr td a[href*='/company/']"):
                href = a_tag.get("href", "")
                match = re.search(r"/company/([^/]+)/", href)
                if not match:
                    continue
                slug = match.group(1).upper().strip()
                if slug.isdigit() or slug in seen_slugs:
                    continue
                name = a_tag.get_text(strip=True)
                seen_slugs.add(slug)
                discovered.append({
                    "symbol": f"{slug}.NS",
                    "screener_slug": slug,
                    "name": name,
                    "sector": "Screener Live Discovery"
                })
                if len(discovered) >= limit:
                    return discovered
        except Exception:
            continue

    return discovered


def get_full_penny_universe(include_live_discovery: bool = True, include_all_prices: bool = True) -> List[Dict[str, str]]:
    """
    Returns deduplicated list of NSE/BSE stock candidates to scan.
    By default includes both the Penny/Micro-cap universe AND the Sensex/Nifty All-Price Leaders
    so that cached scans can instantaneously filter between `Price <= X` or `Any Price (Irrespective of Price)`.
    """
    base = list(CURATED_INDIAN_PENNY_UNIVERSE)
    if include_all_prices:
        base.extend(SENSEX_NIFTY_ALL_PRICE_LEADERS)

    unique: Dict[str, Dict[str, str]] = {}
    for item in base:
        slug = item["screener_slug"]
        if slug not in unique:
            unique[slug] = item

    if include_live_discovery:
        extra = discover_screener_penny_stocks(limit=12)
        for item in extra:
            slug = item["screener_slug"]
            if slug not in unique:
                unique[slug] = item

    return list(unique.values())
