"""
Main Recommendation Engine & Daily Pick Persistence Tracker.

Coordinates:
1. Online Criteria Learning & Printing
2. NSE/BSE Penny Stock Universe Scan (< ₹50)
3. Multi-Factor Scoring & Sorting
4. News & Company Report Deep Synthesis to finalize the #1 Stock of the Day
5. Repeat-Stock Conviction Tracking across days (6-Month Horizon)
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from criteria_learner import learn_criteria_online, format_criteria_for_terminal
from stock_universe import get_full_penny_universe
from market_data import scan_penny_universe, fetch_benchmark_context
from scorer_and_ai import score_single_stock, synthesize_top_candidates_with_news


HISTORY_FILE = os.path.join(os.path.dirname(__file__), "data", "daily_picks_history.json")


def load_picks_history() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def record_daily_pick(winner: Dict[str, Any], benchmarks: Dict[str, Any]) -> Dict[str, Any]:
    """
    Saves today's #1 stock pick into `data/daily_picks_history.json` and computes whether
    it is a repeat conviction pick (and how many consecutive/total times it has been #1).
    """
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    history = load_picks_history()
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Count previous appearances of this symbol
    prev_entries = [h for h in history if h.get("date") != today_str]
    same_stock_prev = [h for h in prev_entries if h.get("symbol") == winner["symbol"]]
    is_repeat_last_day = bool(prev_entries and prev_entries[-1].get("symbol") == winner["symbol"])

    first_recommended_price = same_stock_prev[0]["current_price"] if same_stock_prev else winner["current_price"]
    first_recommended_date = same_stock_prev[0]["date"] if same_stock_prev else today_str

    entry = {
        "date": today_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": winner["symbol"],
        "name": winner["name"],
        "sector": winner["sector"],
        "current_price": winner["current_price"],
        "target_price_6m": winner["target_price_6m"],
        "bull_target_6m": winner["bull_target_6m"],
        "bear_target_6m": winner["bear_target_6m"],
        "expected_upside_6m_pct": winner["expected_upside_6m_pct"],
        "final_score": winner["final_score"],
        "is_repeat_pick": len(same_stock_prev) > 0,
        "is_consecutive_repeat": is_repeat_last_day,
        "total_days_picked": len(same_stock_prev) + 1,
        "first_recommended_date": first_recommended_date,
        "first_recommended_price": first_recommended_price,
        "return_since_first_pick_pct": round(((winner["current_price"] / first_recommended_price) - 1.0) * 100.0, 2),
        "nifty50": benchmarks.get("nifty50", {}).get("price"),
        "sensex": benchmarks.get("sensex", {}).get("price"),
    }

    # Replace today's entry if re-run on the same day, else append
    updated_history = [h for h in history if h.get("date") != today_str]
    updated_history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_history, f, indent=2)

    winner["repeat_tracker"] = entry
    return entry


def run_recommendation_pipeline(
    max_price: float = 50.0,
    ignore_price: bool = False,
    horizon_months: int = 6,
    ignore_timeline: bool = False,
    include_live_discovery: bool = True,
    api_key: Optional[str] = None,
    force_refresh_criteria: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end Indian Stock Recommendation Engine:
    1. Learns & formats criteria online
    2. Fetches Sensex & Nifty 50 context
    3. Scans all NSE/BSE candidates (both sub-₹50 penny stocks and Sensex/Nifty leaders)
    4. Filters & ranks stocks for the chosen price cap (or Any Price) and timeline (1M to 60M / 5Y)
    5. Reads news & company reports to pick the single #1 Stock of the Day
    """
    from scorer_and_ai import filter_and_rerank_universe

    # Step 1: Learn Criteria
    criteria = learn_criteria_online(api_key=api_key, force_refresh=force_refresh_criteria)

    # Step 2: Benchmark context
    benchmarks = fetch_benchmark_context()

    # Step 3: Universe & Market Scan (scan all prices up to ₹1,00,000 so toggling 'Any Price' is instant!)
    candidates = get_full_penny_universe(include_live_discovery=include_live_discovery, include_all_prices=True)
    raw_stocks = scan_penny_universe(
        candidates=candidates,
        max_price=100000.0,
        progress_callback=progress_callback,
    )

    # Step 4: Score all stocks
    eff_h = 60 if ignore_timeline else horizon_months
    all_scored_stocks = [score_single_stock(st, horizon_months=eff_h) for st in raw_stocks]

    if not all_scored_stocks:
        raise RuntimeError("No valid NSE/BSE stocks found. Check network connectivity.")

    # Step 5: Enrich top candidates across both sub-₹50 and all-price leaders with Google News RSS
    synthesize_top_candidates_with_news(
        top_stocks=all_scored_stocks,
        benchmarks=benchmarks,
        api_key=api_key,
        horizon_months=eff_h,
    )

    # Filter and rank according to current price cap & timeline mode
    active_ranked = filter_and_rerank_universe(
        all_scored_stocks,
        max_price=max_price,
        ignore_price=ignore_price,
        horizon_months=eff_h,
        ignore_timeline=ignore_timeline,
    )
    if not active_ranked:
        active_ranked = all_scored_stocks

    winner = active_ranked[0]

    # Step 6: Record daily pick & repeat tracking
    repeat_meta = record_daily_pick(winner, benchmarks)

    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "max_price_filter": max_price,
        "ignore_price": ignore_price,
        "horizon_months": eff_h,
        "ignore_timeline": ignore_timeline,
        "total_candidates_scanned": len(candidates),
        "qualifying_under_max_price": len(active_ranked),
        "benchmarks": benchmarks,
        "learned_criteria": criteria.model_dump(),
        "winner": winner,
        "repeat_meta": repeat_meta,
        "ranked_stocks": active_ranked,
        "all_scanned_stocks": all_scored_stocks,
        "history": load_picks_history(),
    }

    # Cache latest full scan result for instant dashboard loading
    latest_cache = os.path.join(os.path.dirname(__file__), "data", "latest_scan_result.json")
    os.makedirs(os.path.dirname(latest_cache), exist_ok=True)
    with open(latest_cache, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    print("🚀 Starting NSE/BSE Penny Stock Recommendation Engine (< ₹50 | 6-Month Horizon)...")
    res = run_recommendation_pipeline(max_price=50.0)
    from criteria_learner import LearnedCriteriaFramework
    crit_obj = LearnedCriteriaFramework(**res["learned_criteria"])
    print("\n" + format_criteria_for_terminal(crit_obj))

    w = res["winner"]
    dive = w["ai_deep_dive"]
    rep = res["repeat_meta"]
    print("\n" + "=" * 88)
    print(f"🏆 STOCK OF THE DAY ({rep['date']}): {w['name']} ({w['symbol']}) — Sector: {w['sector']}")
    print("=" * 88)
    repeat_str = (
        f"YES (Picked {rep['total_days_picked']} time(s), first at ₹{rep['first_recommended_price']} on {rep['first_recommended_date']})"
        if rep["is_repeat_pick"]
        else "NEW PICK TODAY (Day #1 of conviction)"
    )
    print(f"• Repeat Pick Status       : {repeat_str}")
    print(f"• Current Market Price     : ₹{w['current_price']:.2f}")
    print(f"• Expected 6M Target Price : ₹{w['target_price_6m']:.2f}  (+{w['expected_upside_6m_pct']:.1f}% Expected Upside)")
    print(f"• 6M Scenario Range        : Bear ₹{w['bear_target_6m']:.2f}  |  Base ₹{w['target_price_6m']:.2f}  |  Bull ₹{w['bull_target_6m']:.2f}")
    print(f"• Composite Conviction     : {w['final_score']}/100  (Quant: {w['quant_score']}/100 | Sentiment: {dive['news_sentiment']})")
    print("-" * 88)
    print(f"📌 KEY FUNDAMENTALS:")
    print(
        f"   Market Cap: ₹{w.get('market_cap_cr') or 'N/A'} Cr | P/E: {w.get('pe_ratio') or 'N/A'}x | "
        f"Book Value: ₹{w.get('book_value') or 'N/A'} (P/B: {w.get('pb_ratio') or 'N/A'}x) | "
        f"Promoter: {w.get('promoter_holding_pct') or 'N/A'}% | ROE: {w.get('roe_pct') or 'N/A'}% | "
        f"ROCE: {w.get('roce_pct') or 'N/A'}% | D/E: {w.get('debt_to_equity') or 'N/A'}x"
    )
    print("-" * 88)
    print(f"🧠 EXECUTIVE INVESTMENT THESIS (6-MONTH TIMELINE):\n   {dive['executive_thesis']}")
    print("\n📑 COMPANY REPORT & FINANCIAL INSIGHTS:")
    for ins in dive["company_report_insights"]:
        print(f"   ✓ {ins}")
    print("\n📰 RECENT NEWS & CATALYSTS:")
    for cat in dive["recent_news_catalysts"]:
        print(f"   • {cat}")
    print("\n🎯 TARGET PRICE RATIONALE:")
    print(f"   {dive['target_price_rationale']}")
    print("=" * 88)
