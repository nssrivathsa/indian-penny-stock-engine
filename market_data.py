"""
Market Data, Screener.in Complete Financial Statement Enrichment & Google News RSS Fetcher
for NSE/BSE Penny Stocks.

Uses direct Yahoo Finance v8 Chart JSON API + Screener.in full HTML financial statements
(Top Ratios, P&L TTM Sales/Net Profit/EPS, Balance Sheet Debt-to-Equity, Cash Flow OCF,
Shareholding Promoters/FII/DII, and Pros/Cons) with `verify=False` so Windows SSL proxy
or curl_cffi issues never block live NSE/BSE execution.
"""

from __future__ import annotations
import math
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Any, List, Optional
import urllib3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Computes standard 14-day RSI from closing price series."""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss):
        return 50.0
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _parse_num_list(tr) -> List[float]:
    nums: List[float] = []
    for td in tr.select("td"):
        txt = td.get_text(strip=True).replace(",", "").replace("%", "")
        try:
            nums.append(float(txt))
        except ValueError:
            pass
    return nums


def fetch_screener_enrichment(screener_slug: str) -> Dict[str, Any]:
    """
    Scrapes Screener.in company page (/company/<SLUG>/consolidated/ or /company/<SLUG>/)
    to extract complete Indian fundamentals:
      - Market Cap (Cr), Stock P/E, Book Value (₹), ROCE %, ROE %
      - Shareholding: Promoter %, FII %, DII %
      - P&L: TTM Sales (Cr), TTM Net Profit (Cr), TTM EPS (₹), YoY Sales & Profit Growth %
      - Balance Sheet: Equity + Reserves vs Borrowings -> Debt-to-Equity ratio
      - Cash Flow: Operating Cash Flow (Cr)
      - Screener.in Pros, Cons & Company Profile
    """
    result: Dict[str, Any] = {
        "screener_url": f"https://www.screener.in/company/{screener_slug}/",
        "roce_pct": None,
        "roe_pct_screener": None,
        "book_value_screener": None,
        "pe_screener": None,
        "market_cap_cr_screener": None,
        "promoter_holding_pct_screener": None,
        "institutional_holding_pct_screener": None,
        "sales_ttm_cr": None,
        "net_profit_ttm_cr": None,
        "eps_ttm": None,
        "rev_growth_pct": None,
        "earn_growth_pct": None,
        "profit_margin_pct": None,
        "debt_to_equity": None,
        "operating_cashflow_cr": None,
        "pros": [],
        "cons": [],
        "about_company": "",
    }
    urls_to_try = [
        f"https://www.screener.in/company/{screener_slug}/consolidated/",
        f"https://www.screener.in/company/{screener_slug}/",
    ]
    for url in urls_to_try:
        try:
            resp = requests.get(url, headers=HEADERS, verify=False, timeout=8)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")

            # 1. Top ratios (#top-ratios li)
            for li in soup.select("#top-ratios li"):
                name_el = li.select_one(".name")
                val_el = li.select_one(".number")
                if not name_el or not val_el:
                    continue
                label = name_el.get_text(" ", strip=True).lower()
                raw_val = val_el.get_text(strip=True).replace(",", "")
                try:
                    num = float(raw_val)
                except ValueError:
                    continue

                if "market cap" in label:
                    result["market_cap_cr_screener"] = num
                elif "stock p/e" in label:
                    result["pe_screener"] = num
                elif "book value" in label:
                    result["book_value_screener"] = num
                elif "roce" in label:
                    result["roce_pct"] = num
                elif "roe" in label:
                    result["roe_pct_screener"] = num

            # 2. Pros & Cons
            pros = [li.get_text(" ", strip=True) for li in soup.select(".pros ul li")]
            cons = [li.get_text(" ", strip=True) for li in soup.select(".cons ul li")]
            if pros:
                result["pros"] = pros
            if cons:
                result["cons"] = cons

            # 3. About summary
            about_el = soup.select_one(".company-profile .about p")
            if about_el:
                result["about_company"] = about_el.get_text(" ", strip=True)

            # 4. Shareholding pattern (Promoters, FIIs, DIIs)
            fii_latest = 0.0
            dii_latest = 0.0
            for tr in soup.select("#shareholding table tr"):
                row_text = tr.get_text(" ", strip=True)
                nums = _parse_num_list(tr)
                if not nums:
                    continue
                if "Promoters" in row_text:
                    result["promoter_holding_pct_screener"] = nums[-1]
                elif "FIIs" in row_text or "FPIs" in row_text:
                    fii_latest = nums[-1]
                elif "DIIs" in row_text:
                    dii_latest = nums[-1]
            if fii_latest > 0 or dii_latest > 0:
                result["institutional_holding_pct_screener"] = round(fii_latest + dii_latest, 2)

            # 5. Profit & Loss table (#profit-loss table tr)
            for tr in soup.select("#profit-loss table tr"):
                first_cell = tr.select_one("td, th")
                if not first_cell:
                    continue
                label = first_cell.get_text(" ", strip=True).lower()
                nums = _parse_num_list(tr)
                if not nums:
                    continue
                if label.startswith("sales") or label.startswith("revenue") or label.startswith("interest earned"):
                    result["sales_ttm_cr"] = nums[-1]
                    if len(nums) >= 3 and nums[-3] > 0:
                        result["rev_growth_pct"] = round(((nums[-1] / nums[-3]) - 1.0) * 100.0, 1)
                    elif len(nums) >= 2 and nums[-2] > 0:
                        result["rev_growth_pct"] = round(((nums[-1] / nums[-2]) - 1.0) * 100.0, 1)
                elif label.startswith("net profit"):
                    result["net_profit_ttm_cr"] = nums[-1]
                    if len(nums) >= 3 and nums[-3] > 0:
                        result["earn_growth_pct"] = round(((nums[-1] / nums[-3]) - 1.0) * 100.0, 1)
                    elif len(nums) >= 2 and nums[-2] > 0:
                        result["earn_growth_pct"] = round(((nums[-1] / nums[-2]) - 1.0) * 100.0, 1)
                elif "eps in rs" in label:
                    result["eps_ttm"] = nums[-1]

            if result["sales_ttm_cr"] and result["sales_ttm_cr"] > 0 and result["net_profit_ttm_cr"] is not None:
                result["profit_margin_pct"] = round((result["net_profit_ttm_cr"] / result["sales_ttm_cr"]) * 100.0, 2)

            # 6. Balance Sheet (#balance-sheet table tr) -> Debt / Equity
            eq_cap = 0.0
            reserves = 0.0
            borrowings = None
            for tr in soup.select("#balance-sheet table tr"):
                first_cell = tr.select_one("td, th")
                if not first_cell:
                    continue
                label = first_cell.get_text(" ", strip=True).lower()
                nums = _parse_num_list(tr)
                if not nums:
                    continue
                if "equity capital" in label or "share capital" in label:
                    eq_cap = nums[-1]
                elif "reserves" in label:
                    reserves = nums[-1]
                elif "borrowings" in label:
                    borrowings = nums[-1]

            total_eq = eq_cap + reserves
            if borrowings is not None and total_eq > 0:
                result["debt_to_equity"] = round(borrowings / total_eq, 2)

            # 7. Cash Flow (#cash-flow table tr)
            for tr in soup.select("#cash-flow table tr"):
                first_cell = tr.select_one("td, th")
                if not first_cell:
                    continue
                label = first_cell.get_text(" ", strip=True).lower()
                nums = _parse_num_list(tr)
                if nums and "operating activity" in label:
                    result["operating_cashflow_cr"] = nums[-1]
                    break

            result["screener_url"] = url
            if result["market_cap_cr_screener"] is not None or result["book_value_screener"] is not None:
                break
        except Exception:
            continue

    return result


def fetch_company_news_rss(company_name: str, symbol: str, max_items: int = 8) -> List[Dict[str, str]]:
    """
    Fetches recent Indian financial news & corporate announcements for a company via Google News RSS.
    """
    clean_sym = symbol.replace(".NS", "").replace(".BO", "")
    query = f'"{company_name}" OR "{clean_sym}" share price NSE BSE order profit results'
    encoded = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"

    articles: List[Dict[str, str]] = []
    try:
        resp = requests.get(rss_url, headers=HEADERS, verify=False, timeout=7)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "xml")
            for item in soup.find_all("item")[:max_items]:
                title = item.title.get_text(strip=True) if item.title else ""
                link = item.link.get_text(strip=True) if item.link else ""
                pub_date = item.pubDate.get_text(strip=True) if item.pubDate else ""
                source = item.source.get_text(strip=True) if item.source else "Financial Media"
                if title:
                    articles.append({
                        "title": title,
                        "link": link,
                        "published": pub_date,
                        "source": source,
                    })
    except Exception:
        pass
    return articles


def fetch_yahoo_v8_chart(ticker: str, range_str: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetches daily OHLCV history directly from Yahoo Finance v8 Chart JSON endpoint
    with SSL verification bypassed for Windows proxy resilience.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?range={range_str}&interval={interval}"
    try:
        resp = requests.get(url, headers=HEADERS, verify=False, timeout=8)
        if resp.status_code != 200:
            return None
        data = resp.json()
        res_list = data.get("chart", {}).get("result")
        if not res_list:
            return None
        res = res_list[0]
        timestamps = res.get("timestamp", [])
        quote_list = res.get("indicators", {}).get("quote", [])
        if not timestamps or not quote_list:
            return None
        q = quote_list[0]
        df = pd.DataFrame({
            "Date": [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps],
            "Open": q.get("open", []),
            "High": q.get("high", []),
            "Low": q.get("low", []),
            "Close": q.get("close", []),
            "Volume": q.get("volume", []),
        })
        df = df.dropna(subset=["Close"]).reset_index(drop=True)
        return df
    except Exception:
        return None


def fetch_single_stock_snapshot(
    candidate: Dict[str, str],
    max_price: float = 50.0,
) -> Optional[Dict[str, Any]]:
    """
    Fetches live price and 1-year technical history from Yahoo Finance v8 API
    for a single NSE/BSE stock candidate. Returns None if price > max_price or delisted/no data.
    """
    symbol = candidate["symbol"]
    screener_slug = candidate["screener_slug"]

    tickers_to_try = [symbol]
    if symbol.endswith(".NS"):
        tickers_to_try.append(symbol.replace(".NS", ".BO"))

    for tk_sym in tickers_to_try:
        df = fetch_yahoo_v8_chart(tk_sym, range_str="1y", interval="1d")
        if df is None or len(df) < 20:
            continue

        close_series = df["Close"].astype(float)
        vol_series = df["Volume"].fillna(0).astype(float)

        current_price = float(close_series.iloc[-1])
        if current_price <= 0.5 or current_price > max_price:
            return None

        sma_50 = float(close_series.tail(50).mean()) if len(close_series) >= 50 else current_price
        sma_200 = float(close_series.tail(200).mean()) if len(close_series) >= 200 else sma_50
        high_52w = float(close_series.max())
        low_52w = float(close_series.min())
        rsi_14 = compute_rsi(close_series, 14)
        avg_vol_30d = float(vol_series.tail(30).mean()) if not vol_series.empty else 0.0

        ret_1m = float((current_price / close_series.iloc[-21] - 1.0) * 100.0) if len(close_series) >= 22 else 0.0
        ret_3m = float((current_price / close_series.iloc[-63] - 1.0) * 100.0) if len(close_series) >= 64 else 0.0
        ret_6m = float((current_price / close_series.iloc[-126] - 1.0) * 100.0) if len(close_series) >= 127 else ret_3m
        daily_returns = close_series.pct_change().dropna()
        annualized_vol_pct = float(daily_returns.std() * math.sqrt(252) * 100.0) if len(daily_returns) > 10 else 45.0

        hist_6m = df.tail(130)
        chart_records = []
        for _, row in hist_6m.iterrows():
            chart_records.append({
                "date": str(row["Date"]),
                "open": round(float(row["Open"] or row["Close"]), 2),
                "high": round(float(row["High"] or row["Close"]), 2),
                "low": round(float(row["Low"] or row["Close"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"] or 0),
            })

        return {
            "symbol": tk_sym,
            "screener_slug": screener_slug,
            "name": candidate["name"],
            "sector": candidate["sector"],
            "industry": candidate["sector"],
            "current_price": round(current_price, 2),
            "market_cap_cr": None,
            "pe_ratio": None,
            "pb_ratio": None,
            "book_value": None,
            "eps": None,
            "roe_pct": None,
            "roce_pct": None,
            "profit_margin_pct": None,
            "rev_growth_pct": None,
            "earn_growth_pct": None,
            "debt_to_equity": None,
            "promoter_holding_pct": None,
            "institutional_holding_pct": 0.0,
            "total_revenue_cr": None,
            "net_profit_cr": None,
            "operating_cashflow_cr": None,
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "rsi_14": round(rsi_14, 1),
            "avg_vol_30d": int(avg_vol_30d),
            "ret_1m_pct": round(ret_1m, 2),
            "ret_3m_pct": round(ret_3m, 2),
            "ret_6m_pct": round(ret_6m, 2),
            "annualized_vol_pct": round(annualized_vol_pct, 1),
            "business_summary": "",
            "chart_history": chart_records,
            "screener_pros": [],
            "screener_cons": [],
            "screener_url": f"https://www.screener.in/company/{screener_slug}/",
        }

    return None


def enrich_stock_with_screener(stock: Dict[str, Any]) -> Dict[str, Any]:
    """Merges Screener.in full Indian financial statement metrics into stock dict."""
    sc = fetch_screener_enrichment(stock["screener_slug"])
    if sc.get("roce_pct") is not None:
        stock["roce_pct"] = sc["roce_pct"]
    if sc.get("roe_pct_screener") is not None:
        stock["roe_pct"] = sc["roe_pct_screener"]
    if sc.get("book_value_screener") is not None:
        stock["book_value"] = sc["book_value_screener"]
        if stock["book_value"] and stock["book_value"] > 0:
            stock["pb_ratio"] = round(stock["current_price"] / stock["book_value"], 2)
    if sc.get("pe_screener") is not None:
        stock["pe_ratio"] = sc["pe_screener"]
    if sc.get("market_cap_cr_screener") is not None:
        stock["market_cap_cr"] = sc["market_cap_cr_screener"]
    if sc.get("promoter_holding_pct_screener") is not None:
        stock["promoter_holding_pct"] = sc["promoter_holding_pct_screener"]
    if sc.get("institutional_holding_pct_screener") is not None:
        stock["institutional_holding_pct"] = sc["institutional_holding_pct_screener"]
    if sc.get("sales_ttm_cr") is not None:
        stock["total_revenue_cr"] = sc["sales_ttm_cr"]
    if sc.get("net_profit_ttm_cr") is not None:
        stock["net_profit_cr"] = sc["net_profit_ttm_cr"]
    if sc.get("eps_ttm") is not None:
        stock["eps"] = sc["eps_ttm"]
        if stock["pe_ratio"] is None and stock["eps"] > 0:
            stock["pe_ratio"] = round(stock["current_price"] / stock["eps"], 2)
    if sc.get("rev_growth_pct") is not None:
        stock["rev_growth_pct"] = sc["rev_growth_pct"]
    if sc.get("earn_growth_pct") is not None:
        stock["earn_growth_pct"] = sc["earn_growth_pct"]
    if sc.get("profit_margin_pct") is not None:
        stock["profit_margin_pct"] = sc["profit_margin_pct"]
    if sc.get("debt_to_equity") is not None:
        stock["debt_to_equity"] = sc["debt_to_equity"]
    if sc.get("operating_cashflow_cr") is not None:
        stock["operating_cashflow_cr"] = sc["operating_cashflow_cr"]
    if sc.get("pros"):
        stock["screener_pros"] = sc["pros"]
    if sc.get("cons"):
        stock["screener_cons"] = sc["cons"]
    if sc.get("about_company"):
        stock["business_summary"] = sc["about_company"]
    if sc.get("screener_url"):
        stock["screener_url"] = sc["screener_url"]
    return stock


def fetch_benchmark_context() -> Dict[str, Any]:
    """Fetches Sensex (^BSESN) and Nifty 50 (^NSEI) live levels and 1M/6M trend for macro context."""
    benchmarks = {
        "nifty50": {"symbol": "^NSEI", "name": "Nifty 50", "price": None, "change_1m_pct": 0.0, "change_6m_pct": 0.0},
        "sensex": {"symbol": "^BSESN", "name": "BSE Sensex", "price": None, "change_1m_pct": 0.0, "change_6m_pct": 0.0},
    }
    for key, b in benchmarks.items():
        df = fetch_yahoo_v8_chart(b["symbol"], range_str="6mo", interval="1d")
        if df is not None and not df.empty:
            closes = df["Close"].astype(float)
            curr = float(closes.iloc[-1])
            b["price"] = round(curr, 2)
            if len(closes) >= 21:
                b["change_1m_pct"] = round(float((curr / closes.iloc[-21] - 1.0) * 100.0), 2)
            b["change_6m_pct"] = round(float((curr / closes.iloc[0] - 1.0) * 100.0), 2)
    return benchmarks


def scan_penny_universe(
    candidates: List[Dict[str, str]],
    max_price: float = 50.0,
    max_workers: int = 12,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """
    Concurrently fetches live market & technical data for all penny stock candidates,
    filters strictly for Price <= max_price, and enriches all qualifying stocks with Screener.in financial statements.
    """
    valid_stocks: List[Dict[str, Any]] = []
    total = len(candidates)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cand = {
            executor.submit(fetch_single_stock_snapshot, cand, max_price): cand
            for cand in candidates
        }
        for future in as_completed(future_to_cand):
            completed += 1
            if progress_callback:
                progress_callback(completed, total, future_to_cand[future]["name"])
            try:
                snap = future.result()
                if snap is not None:
                    valid_stocks.append(snap)
            except Exception:
                pass

    # Enrich valid sub-₹50 stocks with Screener.in full financial statements concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(enrich_stock_with_screener, st) for st in valid_stocks]
        enriched = []
        for f in as_completed(futures):
            try:
                enriched.append(f.result())
            except Exception:
                pass

    return enriched
