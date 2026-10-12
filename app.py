"""
Interactive Streamlit Web Dashboard for the NSE/BSE Stock Recommendation Engine.

Features:
1. Flexible Price & Timeline Controls:
   - Price Slider (`₹10` to `₹500`) OR **"Any Price (Irrespective of Price)"** checkbox to include all Sensex/Nifty leaders.
   - Timeline Slider (`1 Month` to `60 Months / 5 Years`) OR **"Any Timeline (5-Year Structural Compounder)"** checkbox.
2. Hero Card: Today's #1 Stock Pick, Repeat-Conviction Tracker, Current Price, Dynamic Target Price, and Composite Score.
3. Deep-Dive Analysis: Executive Investment Thesis, Company Report & Balance Sheet Insights, Live Google News RSS Catalysts, Screener.in Pros/Cons, and Candlestick + Target Projection Chart.
4. Learned Criteria Explorer & Full Leaderboard.
"""

from __future__ import annotations
import os
import json
import copy
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import importlib
import criteria_learner
import scorer_and_ai
import engine

importlib.reload(scorer_and_ai)
importlib.reload(engine)

from criteria_learner import LearnedCriteriaFramework, format_criteria_for_terminal
from engine import run_recommendation_pipeline, load_picks_history
from scorer_and_ai import (
    compute_horizon_targets,
    filter_and_rerank_universe,
    get_horizon_regime_profile,
    format_timeline_short,
    format_timeline_long,
)


st.set_page_config(
    page_title="NSE/BSE Penny & Sensex/Nifty Alpha Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional Styling
st.markdown(
    """
    <style>
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 28px;
        color: #f8fafc;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
    }
    .badge-repeat {
        background: #f59e0b;
        color: #0f172a;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 8px;
    }
    .badge-new {
        background: #10b981;
        color: #052e16;
        padding: 4px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 8px;
    }
    .badge-sector {
        background: #334155;
        color: #e2e8f0;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 6px;
    }
    .criteria-card {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #f1f5f9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_cached_scan() -> dict | None:
    cache_path = os.path.join(os.path.dirname(__file__), "data", "latest_scan_result.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


TIMELINE_STEPS: list[tuple[str, float]] = (
    [("1 Week", 0.25), ("2 Weeks", 0.50), ("3 Weeks", 0.75)]
    + [(f"{m} Month{'s' if m > 1 else ''}", float(m)) for m in range(1, 61)]
)
TIMELINE_LABELS = [t[0] for t in TIMELINE_STEPS]
TIMELINE_MAP = dict(TIMELINE_STEPS)


# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Engine Controls")
    st.caption("Sensex / Nifty Stock Recommendation Engine")

    st.markdown("#### 💰 1. Share Price Filter")
    ignore_price = st.checkbox(
        "🌐 Any Share Price (Irrespective of Price)",
        value=False,
        help="Check this to ignore the penny stock price cap and pick the #1 stock across all Sensex/Nifty price levels (from ₹2 to ₹4,000+).",
    )
    max_price = st.slider(
        "Max Stock Price (₹)",
        min_value=10.0,
        max_value=500.0,
        value=50.0,
        step=5.0,
        disabled=ignore_price,
        help="Strictly filters NSE/BSE universe for stocks trading at or below this price (disabled when 'Any Share Price' is checked).",
    )

    st.markdown("#### ⏱️ 2. Investment Timeline")
    ignore_timeline = st.checkbox(
        "♾️ Any / Long-Term Future (5-Year Compounder Mode)",
        value=False,
        help="Check this when you don't care about a short/fixed exit date and want the #1 long-term wealth compounder over the next 5 years (60 months).",
    )
    selected_timeline_label = st.select_slider(
        "Target Investment Timeline (1 Week to 5 Years)",
        options=TIMELINE_LABELS,
        value="6 Months",
        disabled=ignore_timeline,
        help="Slide from 1 Week, 2 Weeks, 3 Weeks, 1 Month ... all the way to 60 Months (5 Years). Factor weights, #1 stock pick, and target prices adapt dynamically!",
    )
    horizon_months = TIMELINE_MAP[selected_timeline_label]

    effective_horizon = 60.0 if ignore_timeline else horizon_months
    horizon_tag = format_timeline_short(effective_horizon, ignore_timeline=ignore_timeline)
    horizon_long = format_timeline_long(effective_horizon, ignore_timeline=ignore_timeline)
    price_tag = "Any Price (All Sensex/Nifty)" if ignore_price else f"Price ≤ ₹{max_price:.0f}"

    st.divider()
    include_discovery = st.checkbox(
        "Include Live Screener.in Discovery",
        value=True,
        help="Scrapes public Screener.in screens in addition to the curated NSE/BSE universe.",
    )
    force_relearn = st.checkbox(
        "Refresh Online Selection Criteria",
        value=False,
        help="Forces the engine to refresh the online selection criteria rubric.",
    )

    run_btn = st.button("🚀 Run Full Live Market Scan Now", type="primary", use_container_width=True)

    with st.expander("🔑 Optional: AI Studio Developer Key", expanded=False):
        st.caption(
            "Not required! The engine works 100% out-of-the-box using Screener.in financial audits & Google News RSS. "
            "(Note: Consumer Gemini Pro subscriptions don't issue external Python SDK keys; you can optionally paste a free AI Studio key from aistudio.google.com/app/api-keys)."
        )
        gemini_key_input = st.text_input(
            "GEMINI_API_KEY (Optional)",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
        )

    st.divider()
    st.markdown("### 📐 Active Filter Mode")
    st.markdown(
        f"""
        - **Price Filter**: `{price_tag}`
        - **Timeline Mode**: `{horizon_tag}`
        """
    )


# --- Load or Trigger Scan ---
scan_data = load_cached_scan()
# If the cached scan was created before `all_scanned_stocks` existed, trigger a fresh scan once
if run_btn or scan_data is None or "all_scanned_stocks" not in scan_data:
    progress_bar = st.progress(0, text="Initializing NSE/BSE Market Scan (Penny + Sensex/Nifty Leaders)...")
    status_text = st.empty()

    def _on_progress(done: int, total: int, name: str):
        pct = min(1.0, done / max(1, total))
        progress_bar.progress(pct, text=f"Scanning NSE/BSE Universe ({done}/{total}): {name}...")

    with st.spinner("Scanning NSE/BSE Penny Stocks + Sensex/Nifty Leaders, enriching from Screener.in, and reading news reports..."):
        scan_data = run_recommendation_pipeline(
            max_price=max_price,
            ignore_price=ignore_price,
            horizon_months=effective_horizon,
            ignore_timeline=ignore_timeline,
            include_live_discovery=include_discovery,
            api_key=gemini_key_input.strip() or None,
            force_refresh_criteria=force_relearn,
            progress_callback=_on_progress,
        )
    progress_bar.empty()
    status_text.empty()
    st.toast("Live NSE/BSE Market Scan Completed!", icon="✅")


# --- Dynamically Filter by Price (or Any Price) & Re-Rank by Timeline (1M to 60M / 5Y) ---
base_pool = copy.deepcopy(scan_data.get("all_scanned_stocks") or scan_data.get("ranked_stocks") or [])
active_ranked = filter_and_rerank_universe(
    base_pool,
    max_price=max_price,
    ignore_price=ignore_price,
    horizon_months=effective_horizon,
    ignore_timeline=ignore_timeline,
)
if not active_ranked:
    st.warning(f"No stocks found under ₹{max_price:.0f}. Showing full universe instead.")
    active_ranked = filter_and_rerank_universe(
        base_pool,
        ignore_price=True,
        horizon_months=effective_horizon,
        ignore_timeline=ignore_timeline,
    )

winner = active_ranked[0]
regime_profile = get_horizon_regime_profile(effective_horizon, ignore_timeline=ignore_timeline)
dive = winner.get("ai_deep_dive", {})
rep = scan_data.get("repeat_meta", {})

dynamic_rationale = (
    f"**Active Strategy Regime: {regime_profile['regime_name']}**  |  **Price Filter: {price_tag}**\n\n"
    f"{regime_profile['description']}\n\n"
    f"For your selected **{horizon_tag} timeline**, **{winner['name']} ({winner['symbol']})** "
    f"has a Base Target Price of **₹{winner['target_price_6m']:.2f} (+{winner['expected_upside_6m_pct']:.1f}% upside)**, blending:\n"
    f"- **Earnings & Book Value Compounding**: ~{winner.get('compounding_annual_rate_pct', 12.0):.1f}% annualized internal compounding on "
    f"trailing EPS of ₹{winner.get('eps') or 'N/A'} and Book Value of ₹{winner.get('book_value') or 'N/A'}/share over {effective_horizon} month(s).\n"
    f"- **Valuation & Technical Alignment**: P/B ({winner.get('pb_ratio') or 'N/A'}x), P/E ({winner.get('pe_ratio') or 'N/A'}x), "
    f"50-DMA (₹{winner['sma_50']:.2f}), and 200-DMA (₹{winner['sma_200']:.2f}).\n"
    f"- **Scenario Band ({horizon_tag})**: Downside Support Floor at **₹{winner['bear_target_6m']:.2f}** | "
    f"Base Target at **₹{winner['target_price_6m']:.2f}** | Bull Case at **₹{winner['bull_target_6m']:.2f}**."
)


# --- Header & Sensex/Nifty Benchmark Bar ---
st.title("🇮🇳 Sensex & Nifty Stock Recommendation Engine")
benchmarks = scan_data.get("benchmarks", {})
nifty = benchmarks.get("nifty50", {})
sensex = benchmarks.get("sensex", {})

bcol1, bcol2, bcol3, bcol4 = st.columns(4)
with bcol1:
    st.metric(
        "Nifty 50 (^NSEI)",
        f"₹{nifty.get('price', 0):,.2f}" if nifty.get("price") else "Live NSE",
        f"{nifty.get('change_1m_pct', 0):+.2f}% (1M)",
    )
with bcol2:
    st.metric(
        "BSE Sensex (^BSESN)",
        f"₹{sensex.get('price', 0):,.2f}" if sensex.get("price") else "Live BSE",
        f"{sensex.get('change_1m_pct', 0):+.2f}% (1M)",
    )
with bcol3:
    st.metric(
        f"Qualifying Stocks ({price_tag})",
        f"{len(active_ranked)} / {len(base_pool)}",
        f"Timeline: {horizon_tag}",
    )
with bcol4:
    st.metric(
        "Last Scan Timestamp",
        scan_data.get("generated_at", "")[-8:],
        scan_data.get("generated_at", "")[:10],
    )

# --- Winner Hero Section ---
is_repeat = rep.get("is_repeat_pick", False) and rep.get("symbol") == winner["symbol"]
badge_html = (
    f'<span class="badge-repeat">🔁 REPEAT CONVICTION PICK (Day #{rep.get("total_days_picked", 1)} • First picked at ₹{rep.get("first_recommended_price", winner["current_price"])})</span>'
    if is_repeat
    else f'<span class="badge-new">✨ #1 PICK FOR {horizon_tag.upper()} ({price_tag.upper()})</span>'
)

st.markdown(
    f"""
    <div class="hero-card">
        <div style="margin-bottom: 10px;">
            {badge_html}
            <span class="badge-sector">🏛️ {winner.get('sector', 'NSE/BSE')}</span>
            <span class="badge-sector">💰 {price_tag}</span>
            <span class="badge-sector">⏱️ {horizon_tag}</span>
        </div>
        <h1 style="margin: 4px 0 8px 0; font-size: 2.2rem; color: #ffffff;">
            🏆 {winner['name']} <span style="color: #38bdf8; font-size: 1.5rem;">({winner['symbol']})</span>
        </h1>
        <p style="font-size: 1.06rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 12px;">
            {dive.get('executive_thesis', '')}
        </p>
        <div style="font-size: 0.95rem; color: #38bdf8; font-weight: 600;">
            📌 Active Configuration ({price_tag} • {horizon_tag}): Current Price ₹{winner['current_price']:.2f} ➔ Expected {horizon_tag} Target ₹{winner['target_price_6m']:.2f} (+{winner['expected_upside_6m_pct']:.1f}% Expected Upside | Bull Case: ₹{winner['bull_target_6m']:.2f})
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Key Price & Dynamic Horizon Target Metrics Row
pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
with pcol1:
    st.metric(
        "Current Market Price",
        f"₹{winner['current_price']:.2f}",
        f"{winner.get('ret_1m_pct', 0):+.1f}% (1M) | {winner.get('ret_3m_pct', 0):+.1f}% (3M)",
    )
with pcol2:
    st.metric(
        f"🎯 Expected {horizon_tag} Target",
        f"₹{winner['target_price_6m']:.2f}",
        f"+{winner['expected_upside_6m_pct']:.1f}% Expected Upside",
    )
with pcol3:
    st.metric(
        f"🐂 {horizon_tag} Bull Case Target",
        f"₹{winner['bull_target_6m']:.2f}",
        f"+{((winner['bull_target_6m'] / winner['current_price']) - 1) * 100:.1f}% Upside",
    )
with pcol4:
    st.metric(
        f"🛡️ {horizon_tag} Bear / Support Floor",
        f"₹{winner['bear_target_6m']:.2f}",
        f"Book Value: ₹{winner.get('book_value') or 'N/A'}",
    )
with pcol5:
    st.metric(
        "Horizon Conviction Score",
        f"{winner['final_score']:.1f} / 100",
        f"News Sentiment: {dive.get('news_sentiment', 'Bullish')}",
    )

st.divider()

# --- Navigation Tabs ---
tab_pick, tab_criteria, tab_leaderboard, tab_history = st.tabs([
    f"📊 #1 Stock Deep-Dive ({horizon_tag} Target & Reports)",
    "📚 Learned Selection Criteria (Printed & Explained)",
    f"📋 Ranked Leaderboard ({len(active_ranked)} Stocks • {price_tag})",
    "🗓️ Daily Pick & Repeat Conviction Log",
])

with tab_pick:
    col_left, col_right = st.columns([1.35, 1.0])

    with col_left:
        st.subheader(f"📈 {winner['name']} ({winner['symbol']}) — Price Action & {horizon_tag} Target Projection")
        chart_hist = winner.get("chart_history", [])
        if chart_hist:
            df_chart = pd.DataFrame(chart_hist)
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df_chart["date"],
                    open=df_chart["open"],
                    high=df_chart["high"],
                    low=df_chart["low"],
                    close=df_chart["close"],
                    name="Daily Price (₹)",
                )
            )
            fig.add_hline(
                y=winner["target_price_6m"],
                line_dash="dash",
                line_color="#10b981",
                annotation_text=f"{horizon_tag} Base Target: ₹{winner['target_price_6m']:.2f} (+{winner['expected_upside_6m_pct']:.1f}%)",
                annotation_position="top left",
            )
            fig.add_hline(
                y=winner["bull_target_6m"],
                line_dash="dot",
                line_color="#38bdf8",
                annotation_text=f"{horizon_tag} Bull Case: ₹{winner['bull_target_6m']:.2f}",
                annotation_position="top right",
            )
            fig.add_hline(
                y=winner["sma_50"],
                line_dash="solid",
                line_color="#f59e0b",
                annotation_text=f"50-DMA: ₹{winner['sma_50']:.2f}",
                annotation_position="bottom left",
            )
            fig.update_layout(
                height=430,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title="Price (₹ INR)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Multi-Horizon Trajectory Table (1W, 2W, 1M, 3M, 6M, 1Y, 2Y, 5Y)
        st.subheader("🗓️ Multi-Timeline Target Price Curve (1 Week to 5 Years)")
        curve_horizons = [0.25, 0.50, 1.0, 3.0, 6.0, 12.0, 24.0, 60.0]
        if effective_horizon not in curve_horizons:
            curve_horizons = sorted(curve_horizons + [effective_horizon])
        curve_rows = []
        for hm in curve_horizons:
            tc = compute_horizon_targets(winner, horizon_months=hm, ignore_timeline=(ignore_timeline and hm == 60.0))
            lbl = format_timeline_long(hm, ignore_timeline=(ignore_timeline and hm == 60.0))
            if abs(hm - effective_horizon) < 1e-4:
                lbl += " ⭐ (Selected)"
            curve_rows.append({
                "Timeline": lbl,
                "Bear Floor (₹)": f"₹{tc['bear_target']:.2f}",
                "Expected Base Target (₹)": f"₹{tc['target_price']:.2f}",
                "Expected Upside (%)": f"+{tc['expected_upside_pct']:.1f}%",
                "Bull Case Target (₹)": f"₹{tc['bull_target']:.2f}",
            })
        st.dataframe(pd.DataFrame(curve_rows), use_container_width=True, hide_index=True)

        st.subheader("📑 Company Report, Balance Sheet & Financial Audit Insights")
        for insight in dive.get("company_report_insights", []):
            st.markdown(f"- ✅ {insight}")

        if winner.get("business_summary"):
            with st.expander("🏢 Read Full Company Business Profile & Operations"):
                st.write(winner["business_summary"])
                st.markdown(f"🔗 [View Full Financial Statements on Screener.in]({winner.get('screener_url', 'https://www.screener.in')})")

        st.subheader(f"🎯 {horizon_tag} Target Price & Regime Valuation Rationale")
        st.info(dynamic_rationale)

    with col_right:
        st.subheader("🧮 7-Pillar Base Score Breakdown")
        breakdown = winner.get("score_breakdown", {})
        if breakdown:
            bd_df = pd.DataFrame([
                {"Pillar": k, "Points Earned": v} for k, v in breakdown.items()
            ])
            fig_bar = go.Figure(
                go.Bar(
                    x=bd_df["Points Earned"],
                    y=bd_df["Pillar"],
                    orientation="h",
                    marker_color="#38bdf8",
                    text=bd_df["Points Earned"],
                    textposition="auto",
                )
            )
            fig_bar.update_layout(
                height=280,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Score Contribution",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📌 Core Fundamental Ratios")
        f1, f2, f3 = st.columns(3)
        f1.metric("Promoter Holding", f"{winner.get('promoter_holding_pct') or 'N/A'}%")
        f2.metric("P/E Ratio", f"{winner.get('pe_ratio') or 'N/A'}x")
        f3.metric("Book Value", f"₹{winner.get('book_value') or 'N/A'}")

        f4, f5, f6 = st.columns(3)
        f4.metric("P/B Ratio", f"{winner.get('pb_ratio') or 'N/A'}x")
        f5.metric("ROE / ROCE", f"{winner.get('roe_pct') or winner.get('roce_pct') or 'N/A'}%")
        f6.metric("Debt-to-Equity", f"{winner.get('debt_to_equity') if winner.get('debt_to_equity') is not None else 'Bank/Fin'}")

        f7, f8, f9 = st.columns(3)
        f7.metric("Market Cap", f"₹{winner.get('market_cap_cr') or 'N/A'} Cr")
        f8.metric("Net Profit (TTM)", f"₹{winner.get('net_profit_cr') or 'N/A'} Cr")
        f9.metric("14-Day RSI", f"{winner.get('rsi_14', 50):.1f}")

        st.subheader("📰 Recent News Reports & Corporate Catalysts")
        for cat in dive.get("recent_news_catalysts", []):
            st.markdown(f"- 📰 {cat}")

        raw_news = winner.get("recent_news_items", [])
        if raw_news:
            with st.expander(f"🔗 View {len(raw_news)} Live Google News RSS Links for {winner['name']}"):
                for n in raw_news:
                    st.markdown(f"- [{n['title']}]({n['link']}) — *{n['source']}* ({n['published'][:16]})")

        st.subheader(f"⚠️ Key Risks to Monitor ({horizon_tag} Horizon)")
        for r in dive.get("key_risks", []):
            st.markdown(f"- ⚠️ {r}")

with tab_criteria:
    st.subheader("📚 Learned Online Stock Selection Criteria & Multi-Factor Framework")
    st.caption(
        "The engine researches Indian equity screening standards online (Screener.in + Gemini Search Grounding) "
        "and adapts pillar weights dynamically based on your selected Price Cap and Investment Timeline."
    )
    crit_dict = scan_data["learned_criteria"]
    crit_obj = LearnedCriteriaFramework(**crit_dict)

    st.markdown(f"**Source Mode:** `{crit_obj.source_mode}`  |  **Last Learned:** `{crit_obj.last_updated}`")
    st.info(f"**Active Timeline Regime ({horizon_tag}):** {regime_profile['regime_name']} — {regime_profile['description']}")

    st.markdown("#### 🔍 Learned Screener.in Filter Query")
    st.code(crit_obj.screener_in_query, language="sql")

    st.markdown("#### ⚖️ Weighted Quantitative & Qualitative Rules")
    for idx, rule in enumerate(crit_obj.rules, 1):
        st.markdown(
            f"""
            <div class="criteria-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong>{idx}. [{rule.category}] {rule.name}</strong>
                    <span style="background:#0284c7; padding:3px 10px; border-radius:999px; font-size:0.82rem; font-weight:700;">
                        Base Weight: {rule.weight_pct:.0f}%
                    </span>
                </div>
                <div style="margin-top:6px; color:#38bdf8; font-size:0.92rem;">
                    <strong>Ideal Threshold:</strong> {rule.ideal_threshold}
                </div>
                <div style="margin-top:4px; color:#cbd5e1; font-size:0.9rem;">
                    {rule.rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 🚩 Automatic Disqualification & Penalty Red Flags")
    for flag in crit_obj.red_flags_to_reject:
        st.markdown(f"- 🛑 {flag}")

    with st.expander("🖨️ View Printable Plain-Text Criteria Report"):
        st.code(format_criteria_for_terminal(crit_obj), language="text")

with tab_leaderboard:
    st.subheader(
        f"📋 Ranked NSE/BSE Leaderboard ({price_tag}) • Target Horizon: {horizon_tag}"
    )
    rows = []
    for rank_idx, st_item in enumerate(active_ranked, 1):
        rows.append({
            "Rank": rank_idx,
            "Symbol": st_item["symbol"],
            "Company Name": st_item["name"],
            "Sector": st_item["sector"],
            "Current Price (₹)": st_item["current_price"],
            f"{horizon_tag} Target (₹)": st_item["target_price_6m"],
            f"{horizon_tag} Upside (%)": st_item["expected_upside_6m_pct"],
            f"{horizon_tag} Bull (₹)": st_item["bull_target_6m"],
            "Horizon Score (/100)": st_item["final_score"],
            "Promoter (%)": st_item.get("promoter_holding_pct"),
            "FII/DII (%)": st_item.get("institutional_holding_pct"),
            "P/E (x)": st_item.get("pe_ratio"),
            "Book Value (₹)": st_item.get("book_value"),
            "P/B (x)": st_item.get("pb_ratio"),
            "ROE (%)": st_item.get("roe_pct"),
            "ROCE (%)": st_item.get("roce_pct"),
            "Debt/Eq (x)": st_item.get("debt_to_equity"),
            "Market Cap (₹ Cr)": st_item.get("market_cap_cr"),
            "RSI (14D)": st_item.get("rsi_14"),
            "30D Avg Volume": st_item.get("avg_vol_30d"),
        })
    df_rank = pd.DataFrame(rows)
    st.dataframe(df_rank, use_container_width=True, height=520)

    st.markdown("#### 🔍 Inspect Any Candidate from the Active Universe")
    selected_sym = st.selectbox(
        "Choose a stock from the ranked table to inspect its positives & red flags:",
        options=[
            f"{s['symbol']} — {s['name']} (Score: {s['final_score']}/100 | ₹{s['current_price']} ➔ {horizon_tag} Target: ₹{s['target_price_6m']})"
            for s in active_ranked
        ],
    )
    if selected_sym:
        chosen_code = selected_sym.split(" — ")[0]
        chosen_stock = next((s for s in active_ranked if s["symbol"] == chosen_code), active_ranked[0])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**✅ Positive Highlights for {chosen_stock['name']}**")
            for h in chosen_stock.get("positive_highlights", []):
                st.markdown(f"- 🟢 {h}")
            for p in chosen_stock.get("screener_pros", []):
                st.markdown(f"- 📘 *Screener Pro:* {p}")
        with c2:
            st.markdown(f"**⚠️ Red Flags / Watch Items for {chosen_stock['name']}**")
            flags = chosen_stock.get("red_flags", []) + chosen_stock.get("screener_cons", [])
            if flags:
                for fl in flags:
                    st.markdown(f"- 🟠 {fl}")
            else:
                st.markdown("- ✨ No major balance-sheet red flags triggered.")

with tab_history:
    st.subheader("🗓️ Daily Pick Persistence & Repeat-Conviction Tracker")
    st.caption(
        "The engine picks 1 stock per day. If the same stock continues to be the highest-conviction opportunity "
        "across fundamentals and news, it repeats the pick and tracks cumulative return since the first recommendation."
    )
    history = load_picks_history()
    if history:
        df_hist = pd.DataFrame(history)[::-1]
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No historical picks recorded yet. Run a scan to log today's pick!")
