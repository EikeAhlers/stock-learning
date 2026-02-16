# Stock Mover Analysis Framework

## Goal
Study every past stock that was a "biggest daily gainer" to understand the real reasons behind big moves, then use those patterns to forecast future big movers.

## Project Structure

```
Stock Learning/
├── src/
│   ├── data_collector.py    # Pull historical stock data & identify top movers
│   ├── move_analyzer.py     # Analyze context around each big move
│   ├── pattern_discovery.py # Find repeating pre-move patterns
│   └── utils.py             # Shared helpers
├── data/                    # Cached data (auto-created)
├── analysis.ipynb           # Main interactive analysis notebook
├── requirements.txt
└── README.md
```

## Key Concepts

### Move Categories (why stocks move big)
1. **Earnings Surprise** — Beat/miss expectations significantly
2. **Revenue Growth Acceleration** — Topline growth speeding up  
3. **Sector/Industry Momentum** — Whole sector rotating in
4. **Technical Breakout** — Breaking key resistance on volume
5. **Short Squeeze** — High short interest + catalyst
6. **News Catalyst** — FDA approval, contract win, M&A, etc.
7. **Pre-Earnings Run** — Anticipation buying before report
8. **Mean Reversion** — Oversold bounce after sharp decline

### Pre-Move Signals We Track
- Unusual volume (vs 20-day average)
- Price compression (low volatility → breakout)
- Relative strength vs sector/market
- Distance from moving averages (20, 50, 200 SMA)
- Recent momentum (5-day, 10-day returns)
- Volume trend (increasing on up days)
- Gap analysis (gap up/down frequency)
- Float & market cap characteristics

## Setup
```bash
pip install -r requirements.txt
```
Then open `analysis.ipynb` to start exploring.
