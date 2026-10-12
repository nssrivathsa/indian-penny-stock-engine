# 🇮🇳 Sensex & Nifty Penny Stock Recommendation Engine

An institutional-grade **Indian Stock & Penny Stock Recommendation Engine (`NSE` / `BSE`)** built with Python and Streamlit. It scans the Indian market, learns and prints out multi-factor quantitative & qualitative screening criteria, enriches fundamentals from **Screener.in** financial statements, reads live **Google News RSS** corporate catalysts, and selects **1 High-Conviction Stock of the Day** (with repeat-conviction tracking across days).

---

## ✨ Key Features

1. **Dynamic Criteria Learner & Printer (`criteria_learner.py`)**
   - Automatically researches and structures a **7-Pillar Weighted Scoring Rubric** tailored to Indian NSE/BSE stocks:
     - **Governance & Promoter Skin-in-the-Game (18%)**: Promoter holding %, FII/DII institutional participation, and unpledged equity safety.
     - **P/E Valuation & Earnings Yield (15%)**: Positive trailing EPS and discounted P/E relative to sector peers.
     - **Book Value & P/B Asset Backing (12%)**: Tangible net worth floor (`Price-to-Book`).
     - **Profitability — ROE, ROCE & Net Margin (15%)**: Capital efficiency and bottom-line profitability.
     - **Revenue & Quarterly Profit Growth (12%)**: Top-line and PAT acceleration.
     - **Solvency, Debt-to-Equity & Operating Cash Flow (13%)**: Balance-sheet leverage and positive cash generation.
     - **Technical Momentum, Moving Averages, RSI & Liquidity (15%)**: 50-DMA / 200-DMA trend, 14-day Wilder RSI, and 30-day average trading volume.
2. **Flexible Price & Timeline Controls (`app.py`)**
   - **Share Price Slider (`₹10` to `₹500`)** + **`🌐 Any Share Price (Irrespective of Price)`** toggle to switch seamlessly between strict sub-₹50 penny stocks and all-price Sensex/Nifty leaders (`₹2` to `₹4,000+`).
   - **Investment Timeline Slider (`1 Week` to `5 Years / 60 Months`)** + **`♾️ Any / Long-Term Future (5-Year Compounder Mode)`** toggle.
   - **Horizon-Aware Re-Ranking (`scorer_and_ai.py`)**:
     - *1–3 Weeks (`Tactical Weekly Breakout`)* & *1–3 Months (`Short-Term Swing`)* prioritize active technical breakouts (`Price > 50-DMA & 200-DMA`), relative strength, and volume velocity.
     - *4–9 Months (`Medium-Term Re-Rating`)* balances deep valuation discounts (`P/B`, `P/E`) with upcoming quarterly earnings catalysts.
     - *1–5 Years (`Multi-Year Structural Compounder`)* prioritizes high `ROCE`/`ROE`, zero/low debt (`Debt/Equity`), and long-term earnings compounding.
3. **Zero-Cost, Out-of-the-Box Data Pipeline (`market_data.py`)**
   - **Yahoo Finance v8 Market Feed**: Live NSE (`.NS`) & BSE (`.BO`) prices, 1-year OHLCV candlestick history, 50/200-DMA, 52-week High/Low, and 14-day RSI.
   - **Screener.in Financial Statement Audit**: Scrapes consolidated/standalone Indian financial statements (`P&L`, `Balance Sheet`, `Cash Flow`, `Shareholding Pattern`, and `Pros/Cons`).
   - **Google News RSS (`en-IN`)**: Fetches live Indian financial news and corporate announcements.
   - **Optional Gemini 3.8 Flash + Google Search Grounding**: Works 100% out-of-the-box without any API key, or optionally accepts a `GEMINI_API_KEY` for live LLM search grounding.
4. **Daily Repeat-Conviction Persistence (`engine.py`)**
   - Logs each day's #1 pick and automatically flags if the same stock repeats as the #1 conviction pick across multiple days, tracking cumulative return since first recommendation.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nssrivathsa/indian-penny-stock-engine.git
cd indian-penny-stock-engine
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Interactive Streamlit Web Dashboard (Recommended)
```bash
streamlit run app.py
```
Then open **`http://localhost:8501`** in your browser.

### 4. Run the Command-Line (CLI) Report
```bash
python -X utf8 engine.py
```

---

## 🗂️ Project Structure

```text
├── app.py                  # Interactive Streamlit Web Dashboard (Sliders, Candlestick Charts, Leaderboard)
├── engine.py               # Core Pipeline Orchestrator, CLI Runner & Daily Repeat-Pick Tracker
├── criteria_learner.py     # Online Criteria Learner, 7-Pillar Rubric & Screener.in Query Generator
├── stock_universe.py       # Curated NSE/BSE Penny Universe + Sensex/Nifty Leaders + Live Screener Discovery
├── market_data.py          # Yahoo Finance v8 OHLCV + Screener.in Financial Statement Parser + Google News RSS
├── scorer_and_ai.py        # Multi-Factor Scorer, Horizon-Aware Re-Ranker (1W to 5Y) & Target Price Model
├── requirements.txt        # Python package dependencies
└── data/
    └── learned_criteria.json # Default & learned multi-factor selection criteria rubric
```

---

## ⚠️ Disclaimer
This software is built strictly for **educational, quantitative research, and informational purposes**. It does not constitute financial or investment advice. Penny and micro-cap stocks carry high volatility and liquidity risks; always consult a SEBI-registered financial advisor before making investment decisions.
