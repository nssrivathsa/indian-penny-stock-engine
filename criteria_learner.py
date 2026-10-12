"""
Dynamic Criteria Learner & Research Framework for Indian Penny Stocks (NSE / BSE).

This module searches online / uses Gemini 3.8 Flash with Google Search grounding (or a
comprehensive institutional framework fallback) to learn, structure, and print out the
exact multi-factor rules used to evaluate penny stocks (< ₹50) for a 6-month horizon.
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CriterionRule(BaseModel):
    category: str = Field(description="Category: Valuation, Profitability, Governance & Promoter, Solvency & Cashflow, Technical & Momentum, or News & Catalysts")
    name: str = Field(description="Name of the financial or qualitative factor")
    metric_key: str = Field(description="Internal metric identifier")
    ideal_threshold: str = Field(description="Ideal value or threshold for Indian penny stocks (< ₹50)")
    weight_pct: float = Field(description="Percentage weight in the composite score (sum to 100)")
    rationale: str = Field(description="Why this factor is critical specifically for Indian NSE/BSE penny stocks over a 6-month horizon")


class LearnedCriteriaFramework(BaseModel):
    last_updated: str = Field(description="ISO timestamp when the criteria were learned/updated")
    source_mode: str = Field(description="Whether learned live via Gemini Google Search Grounding or Institutional Quantitative Baseline")
    market_regime_note: str = Field(description="Summary of current Sensex/Nifty micro-cap & penny stock regime and risk posture")
    screener_in_query: str = Field(description="Equivalent Screener.in query string learned from online Indian equity research")
    rules: List[CriterionRule] = Field(description="List of quantitative and qualitative evaluation rules")
    red_flags_to_reject: List[str] = Field(description="Automatic disqualification / severe penalty red flags (e.g., high promoter pledge, shell company low volume, negative operating cash flow)")


DEFAULT_LEARNED_FRAMEWORK = LearnedCriteriaFramework(
    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    source_mode="Institutional Quantitative & Screener.in Empirical Research Framework",
    market_regime_note=(
        "In the NSE/BSE < ₹50 universe, 80%+ of stocks suffer from illiquidity, circular debt, or governance traps. "
        "To target a realistic 6-month rerating, the engine prioritizes 'Quality Value + Turnaround Momentum': "
        "high unpledged promoter skin-in-the-game, positive trailing earnings, low P/BV asset backing, manageable debt, "
        "and verifiable corporate catalysts (order book wins, debt resolution, or sector tailwinds)."
    ),
    screener_in_query=(
        "Current price < 50 AND Market Capitalization > 50 AND "
        "Return on equity > 10 AND Return on capital employed > 10 AND "
        "Debt to equity < 0.75 AND Promoter holding > 35 AND Pledged percentage < 10 AND "
        "Price to book value < 2.5 AND EPS > 0 AND Volume 1month average > 50000"
    ),
    rules=[
        CriterionRule(
            category="Governance & Promoter",
            name="Promoter Skin-in-the-Game & Unpledged Holding",
            metric_key="promoter_holding_pct",
            ideal_threshold="Promoter Holding > 40% (or strong institutional/FII/DII backing) with Pledged Shares < 5%",
            weight_pct=18.0,
            rationale="In Indian micro/penny caps, promoter conviction is the #1 safeguard against pump-and-dump operators. High promoter pledging risks forced lender liquidation during market dips."
        ),
        CriterionRule(
            category="Valuation",
            name="Trailing P/E vs Sector & Earnings Yield",
            metric_key="pe_ratio",
            ideal_threshold="0 < P/E < 22x (Positive EPS; discounted relative to Nifty Smallcap 250 median)",
            weight_pct=15.0,
            rationale="Eliminates loss-making story stocks while avoiding value traps. A profitable company trading at 8x–18x P/E has room for multiple expansion over 6 months."
        ),
        CriterionRule(
            category="Valuation",
            name="Price-to-Book Value (Asset Backing)",
            metric_key="pb_ratio",
            ideal_threshold="0.3x <= P/BV <= 2.2x (Book Value per share provides downside floor)",
            weight_pct=12.0,
            rationale="Penny stocks in infrastructure, banking, power, and manufacturing need tangible net worth (Book Value) backing so you aren't paying purely for speculative hope."
        ),
        CriterionRule(
            category="Profitability",
            name="Return on Equity (ROE) & Capital Efficiency",
            metric_key="roe_pct",
            ideal_threshold="ROE > 12% and Positive Net Profit Margin (> 5%)",
            weight_pct=15.0,
            rationale="Confirms the business generates genuine economic returns on shareholder equity rather than surviving on equity dilution."
        ),
        CriterionRule(
            category="Profitability",
            name="Revenue & Quarterly Profit Growth Trajectory",
            metric_key="profit_growth_pct",
            ideal_threshold="YoY Revenue Growth > 10% and YoY/QoQ Earnings Growth > 15%",
            weight_pct=12.0,
            rationale="A 6-month target price requires near-term earnings momentum (upcoming 1–2 quarterly earnings releases) to trigger institutional and retail re-rating."
        ),
        CriterionRule(
            category="Solvency & Cashflow",
            name="Debt-to-Equity & Liquidity Safety",
            metric_key="debt_to_equity",
            ideal_threshold="Debt-to-Equity < 0.6x (for non-financials) & Positive Operating Cash Flow",
            weight_pct=13.0,
            rationale="High interest burdens destroy sub-₹50 companies. Low leverage or active deleveraging turns operating cash flow directly into bottom-line EPS growth."
        ),
        CriterionRule(
            category="Technical & Momentum",
            name="6-Month Relative Strength, RSI & Volume Accumulation",
            metric_key="technical_momentum",
            ideal_threshold="Price > 50-DMA or 200-DMA support, RSI between 42–68 (not overbought), Avg Daily Volume > 1,00,000 shares",
            weight_pct=15.0,
            rationale="Avoids 'falling knives' and illiquid upper/lower-circuit operator traps. Healthy delivery/trading volume ensures easy entry and exit."
        ),
    ],
    red_flags_to_reject=[
        "Share Price >= ₹50.00 (Outside user's strict < ₹50 penny stock universe)",
        "Average Daily Volume < 25,000 shares (Severe illiquidity / circuit-trap risk)",
        "Deeply negative Book Value coupled with shrinking revenues (Insolvency / NCLT wipeout risk)",
        "Auditor adverse opinions, SEBI show-cause notices, or >50% Promoter Pledge",
        "RSI > 82 on low fundamental backing (Parabolic pump phase)"
    ]
)


def learn_criteria_online(api_key: Optional[str] = None, force_refresh: bool = False) -> LearnedCriteriaFramework:
    """
    Uses Gemini 3.8 Flash with Google Search grounding to research the latest Indian
    NSE/BSE penny stock quantitative & fundamental screening practices, or falls back
    gracefully to the institutional research framework.
    """
    cache_path = os.path.join(os.path.dirname(__file__), "data", "learned_criteria.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return LearnedCriteriaFramework(**data)
        except Exception:
            pass

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            prompt = """
            Search online for the best quantitative, fundamental, and technical criteria used by top Indian equity analysts
            and Screener.in power users to pick fundamentally strong penny stocks (Price < ₹50 on NSE/BSE Sensex/Nifty universe)
            for a 6-month investment horizon.

            Include traditional factors (Promoter Holding %, Pledged %, P/E ratio, Book Value & P/B ratio, PAT & Profit growth,
            ROE/ROCE, Debt-to-Equity) as well as institutional factors (Operating Cash Flow, Order Book/News Catalysts,
            RSI/Moving Average trend, and Liquidity/Volume filters).

            Synthesize your findings into the exact JSON schema provided, ensuring rule weights sum to 100.0.
            """
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt,
                tools=[{"type": "google_search"}],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": LearnedCriteriaFramework.model_json_schema(),
                },
            )
            if interaction and interaction.output_text:
                parsed = LearnedCriteriaFramework.model_validate_json(interaction.output_text)
                parsed.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                parsed.source_mode = "Live Gemini 3.8 Flash + Google Search Grounding"
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(parsed.model_dump_json(indent=2))
                return parsed
        except Exception as e:
            print(f"[CriteriaLearner] Gemini live search fallback triggered: {e}")

    # Save and return default institutional framework
    framework = DEFAULT_LEARNED_FRAMEWORK.model_copy(deep=True)
    framework.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(framework.model_dump_json(indent=2))
    return framework


def format_criteria_for_terminal(framework: LearnedCriteriaFramework) -> str:
    """Formats the learned criteria into a clean, readable report for CLI/UI."""
    lines = [
        "=" * 88,
        "📚 LEARNED PENNY STOCK SELECTION CRITERIA (NSE / BSE < ₹50 UNIVERSE | 6M HORIZON)",
        "=" * 88,
        f"• Source / Mode   : {framework.source_mode}",
        f"• Last Learned    : {framework.last_updated}",
        f"• Market Regime   : {framework.market_regime_note}",
        f"• Screener Query  : {framework.screener_in_query}",
        "-" * 88,
        "📊 WEIGHTED MULTI-FACTOR SCORING RUBRIC:",
    ]
    for idx, rule in enumerate(framework.rules, 1):
        lines.append(
            f"  {idx}. [{rule.category}] {rule.name} (Weight: {rule.weight_pct:.0f}%)\n"
            f"     ├─ Ideal Threshold : {rule.ideal_threshold}\n"
            f"     └─ Rationale       : {rule.rationale}"
        )
    lines.append("-" * 88)
    lines.append("🚩 AUTOMATIC PENALTIES & RED-FLAG FILTERS:")
    for flag in framework.red_flags_to_reject:
        lines.append(f"  ✖ {flag}")
    lines.append("=" * 88)
    return "\n".join(lines)
