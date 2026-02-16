#!/usr/bin/env python3
"""
Paper Trader — Track daily picks, log outcomes, measure real performance.

Logs every recommendation with entry/exit prices so you can track
how the system performs in real paper trading over time.

Log files:
  logs/picks_YYYY-MM-DD.json     — Daily picks with metadata
  logs/trades.csv                — All trades with entry/exit/P&L
  logs/portfolio.json            — Current portfolio state
  logs/performance_summary.csv   — Rolling performance metrics
"""
import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _trades_path():
    return os.path.join(LOG_DIR, "trades.csv")


def _portfolio_path():
    return os.path.join(LOG_DIR, "portfolio.json")


def _picks_path(date_str):
    return os.path.join(LOG_DIR, f"picks_{date_str}.json")


def _performance_path():
    return os.path.join(LOG_DIR, "performance_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO STATE
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolio():
    """Load current portfolio state."""
    path = _portfolio_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "cash": 100000,
        "start_capital": 100000,
        "positions": [],
        "last_updated": None,
    }


def save_portfolio(portfolio):
    """Save portfolio state."""
    portfolio["last_updated"] = datetime.now().isoformat()
    with open(_portfolio_path(), "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
#  DAILY PICKS LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def log_picks(date_str, picks, model_info=None):
    """Log daily picks to JSON file."""
    record = {
        "scan_date": date_str,
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info or {},
        "picks": picks,
        "n_picks": len(picks),
    }
    path = _picks_path(date_str)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[PaperTrader] Logged {len(picks)} picks for {date_str} → {path}")
    return path


def load_picks(date_str):
    """Load picks for a specific date."""
    path = _picks_path(date_str)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  TRADE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

def open_trade(ticker, entry_date, entry_price, prob, shares, reason="signal"):
    """Record a new trade entry."""
    portfolio = load_portfolio()

    trade = {
        "ticker": ticker,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "shares": shares,
        "prob": prob,
        "reason": reason,
        "days_held": 0,
        "current_value": shares * entry_price,
        "unrealized_pnl": 0,
    }

    portfolio["positions"].append(trade)
    portfolio["cash"] -= shares * entry_price * 1.0015  # include 15bps cost

    save_portfolio(portfolio)
    print(f"[PaperTrader] OPENED {ticker}: {shares} shares @ ${entry_price:.2f}")
    return trade


def close_trade(ticker, exit_date, exit_price, reason="hold_expiry"):
    """Close a position and log the trade."""
    portfolio = load_portfolio()

    closed = None
    remaining = []
    for pos in portfolio["positions"]:
        if pos["ticker"] == ticker and closed is None:
            closed = pos
        else:
            remaining.append(pos)

    if closed is None:
        print(f"[PaperTrader] WARNING: No open position for {ticker}")
        return None

    portfolio["positions"] = remaining
    proceeds = closed["shares"] * exit_price * (1 - 0.0015)  # exit cost
    portfolio["cash"] += proceeds
    save_portfolio(portfolio)

    # Calculate P&L
    cost_basis = closed["shares"] * closed["entry_price"] * 1.0015
    net_pnl = proceeds - cost_basis
    pnl_pct = (exit_price * (1 - 0.0015)) / (closed["entry_price"] * 1.0015) - 1

    # Log to trades.csv
    trade_record = {
        "ticker": ticker,
        "entry_date": closed["entry_date"],
        "exit_date": exit_date,
        "entry_price": closed["entry_price"],
        "exit_price": exit_price,
        "shares": closed["shares"],
        "prob": closed.get("prob", 0),
        "pnl_dollars": round(net_pnl, 2),
        "pnl_pct": round(pnl_pct * 100, 2),
        "days_held": closed.get("days_held", 0),
        "exit_reason": reason,
    }

    trades_file = _trades_path()
    file_exists = os.path.exists(trades_file)
    with open(trades_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trade_record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_record)

    print(f"[PaperTrader] CLOSED {ticker}: ${net_pnl:+.2f} ({pnl_pct*100:+.1f}%) — {reason}")
    return trade_record


def update_positions(current_prices: dict, today_str: str):
    """Update position values and check stop-losses."""
    portfolio = load_portfolio()
    stop_loss_pct = -7.0  # from strategy config

    to_close = []
    for pos in portfolio["positions"]:
        ticker = pos["ticker"]
        if ticker in current_prices:
            current = current_prices[ticker]
            pos["current_value"] = pos["shares"] * current
            pos["unrealized_pnl"] = (current / pos["entry_price"] - 1) * 100
            pos["days_held"] = pos.get("days_held", 0) + 1

            # Stop-loss check
            if pos["unrealized_pnl"] <= stop_loss_pct:
                to_close.append((ticker, current, "stop_loss"))

    save_portfolio(portfolio)

    # Execute stop-losses
    for ticker, price, reason in to_close:
        close_trade(ticker, today_str, price, reason)

    return portfolio, to_close


# ═══════════════════════════════════════════════════════════════════════════
#  PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def get_performance_summary():
    """Calculate rolling performance metrics from trade log."""
    trades_file = _trades_path()
    if not os.path.exists(trades_file):
        return {"total_trades": 0, "message": "No trades yet"}

    import pandas as pd
    trades = pd.read_csv(trades_file)

    if len(trades) == 0:
        return {"total_trades": 0, "message": "No trades yet"}

    n = len(trades)
    wins = (trades["pnl_pct"] > 0).sum()
    total_pnl = trades["pnl_dollars"].sum()
    avg_pnl = trades["pnl_pct"].mean()
    best = trades.loc[trades["pnl_pct"].idxmax()]
    worst = trades.loc[trades["pnl_pct"].idxmin()]

    # Recent performance (last 20 trades)
    recent = trades.tail(20)
    recent_wr = (recent["pnl_pct"] > 0).mean() * 100
    recent_avg = recent["pnl_pct"].mean()

    portfolio = load_portfolio()
    total_value = portfolio["cash"] + sum(
        p.get("current_value", 0) for p in portfolio["positions"]
    )
    total_ret = (total_value / portfolio["start_capital"] - 1) * 100

    return {
        "total_trades": n,
        "win_rate": wins / n * 100,
        "avg_pnl_pct": avg_pnl,
        "total_pnl_dollars": total_pnl,
        "best_trade": f"{best['ticker']} {best['pnl_pct']:+.1f}%",
        "worst_trade": f"{worst['ticker']} {worst['pnl_pct']:+.1f}%",
        "recent_20_win_rate": recent_wr,
        "recent_20_avg_pnl": recent_avg,
        "portfolio_value": total_value,
        "portfolio_return": total_ret,
        "open_positions": len(portfolio["positions"]),
    }


def format_performance_message():
    """Format performance for Telegram."""
    s = get_performance_summary()
    if s["total_trades"] == 0:
        return "📊 No trades logged yet."

    msg = "📊 <b>PERFORMANCE REPORT</b>\n\n"
    msg += f"📈 Portfolio: ${s['portfolio_value']:,.0f} ({s['portfolio_return']:+.1f}%)\n\n"
    msg += f"🔢 Total trades: {s['total_trades']}\n"
    msg += f"✅ Win rate: {s['win_rate']:.0f}%\n"
    msg += f"📊 Avg P&L: {s['avg_pnl_pct']:+.2f}%\n"
    msg += f"💰 Total P&L: ${s['total_pnl_dollars']:+,.0f}\n\n"
    msg += f"🏆 Best: {s['best_trade']}\n"
    msg += f"💀 Worst: {s['worst_trade']}\n\n"
    msg += f"📉 Recent 20: {s['recent_20_win_rate']:.0f}% WR, {s['recent_20_avg_pnl']:+.2f}% avg\n"
    msg += f"📦 Open positions: {s['open_positions']}"
    return msg


if __name__ == "__main__":
    # Test
    print("=== Paper Trader Test ===")
    p = load_portfolio()
    print(f"Portfolio: ${p['cash']:,.0f} cash, {len(p['positions'])} positions")
    s = get_performance_summary()
    print(f"Performance: {s}")
