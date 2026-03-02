#!/usr/bin/env python3
"""
Telegram Notifier — Send daily stock picks to Telegram.

Setup instructions:
  1. Open Telegram, search for @BotFather
  2. Send /newbot, follow prompts, get your BOT_TOKEN
  3. Start a chat with your bot, then visit:
     https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
  4. Find your chat_id in the response
  5. Put both in deploy/config.json
"""
import json
import requests
import os
from datetime import datetime

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def send_telegram(message: str, parse_mode: str = "HTML") -> bool:
    """Send a message via Telegram bot."""
    cfg = load_config()
    tg = cfg.get("telegram", {})

    if not tg.get("enabled", False):
        print("[Telegram] Disabled in config. Message not sent.")
        print(f"[Telegram] Would have sent:\n{message[:500]}")
        return False

    token = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")

    if "YOUR_" in token or "YOUR_" in chat_id:
        print("[Telegram] Bot token or chat_id not configured!")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print("[Telegram] Message sent successfully")
            return True
        else:
            print(f"[Telegram] Error {resp.status_code}: {resp.text[:200]}")
            # If HTML parse fails, retry as plain text
            if parse_mode == "HTML":
                payload["parse_mode"] = ""
                resp2 = requests.post(url, json=payload, timeout=10)
                if resp2.status_code == 200:
                    print("[Telegram] Sent as plain text (HTML failed)")
                    return True
            return False
    except Exception as e:
        print(f"[Telegram] Failed: {e}")
        return False


def format_daily_picks(picks: list, scan_date: str, model_info: dict = None,
                       portfolio: dict = None, hold_days: int = 5) -> str:
    """Format daily picks into a quick, clear Telegram message.
    
    User wants: what to buy, when to sell, what we hold, overall performance.
    Nothing extra.
    """
    from datetime import datetime, timedelta

    # Buy today at close (scan runs 30 min before close, orders fill before 4pm)
    buy_date = datetime.strptime(scan_date, "%Y-%m-%d")
    # If scan_date is weekend (manual run), shift to Monday
    while buy_date.weekday() >= 5:
        buy_date += timedelta(days=1)
    
    # Sell date = hold_days trading days later
    sell_date = buy_date
    trading_days = 0
    while trading_days < hold_days:
        sell_date += timedelta(days=1)
        if sell_date.weekday() < 5:
            trading_days += 1

    msg = f"📊 <b>SCAN {scan_date}</b>\n\n"

    # ── BUY ──
    if not picks:
        msg += "No picks today — nothing passed filters.\n"
    else:
        msg += f"<b>BUY NOW</b> (before close):\n"
        for p in picks:
            prob = p.get("prob", 0)
            msg += f"  → <b>{p['ticker']}</b>  ${p['close']:.2f}  ({prob:.0%})\n"
        msg += f"\n<b>SELL</b> {sell_date.strftime('%a %b %d')} at close\n"

    # ── HOLDING ──
    if portfolio and portfolio.get("positions"):
        positions = portfolio["positions"]
        msg += f"\n<b>HOLDING ({len(positions)})</b>\n"
        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0)
            days = pos.get("days_held", 0)
            left = max(0, hold_days - days)
            icon = "🟢" if pnl >= 0 else "🔴"
            msg += f"  {icon} {pos['ticker']}  {pnl:+.1f}%  ({left}d left)\n"

    # ── PERFORMANCE ──
    if portfolio:
        cash = portfolio.get("cash", 0)
        pos_val = sum(p.get("current_value", 0) for p in portfolio.get("positions", []))
        total = cash + pos_val
        start = portfolio.get("start_capital", 100000)
        ret = (total / start - 1) * 100
        msg += f"\n💰 <b>${total:,.0f}</b>  ({ret:+.1f}%)\n"

    # Closed trade stats
    trades = portfolio.get("trades", []) if portfolio else []
    if trades:
        wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
        wr = wins / len(trades) * 100 if trades else 0
        avg_pnl = sum(t.get("pnl_pct", 0) for t in trades) / len(trades) if trades else 0
        msg += f"📈 {len(trades)} trades | WR {wr:.0f}% | Avg {avg_pnl:+.1f}%"

    return msg


def format_portfolio_update(portfolio: dict) -> str:
    """Format paper trading portfolio status."""
    msg = "📈 <b>PORTFOLIO UPDATE</b>\n\n"

    capital = portfolio.get("cash", 0)
    positions = portfolio.get("positions", [])
    total_value = capital + sum(p.get("current_value", 0) for p in positions)
    start = portfolio.get("start_capital", 100000)
    total_ret = (total_value / start - 1) * 100

    msg += f"💰 Total: ${total_value:,.0f} ({total_ret:+.1f}%)\n"
    msg += f"💵 Cash: ${capital:,.0f}\n"
    msg += f"📦 Positions: {len(positions)}\n\n"

    for p in positions:
        pnl = p.get("unrealized_pnl", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        msg += f"{emoji} {p['ticker']}: {pnl:+.1f}% ({p.get('days_held', 0)}d)\n"

    return msg


if __name__ == "__main__":
    # Test
    test_picks = [
        {"ticker": "NVDA", "prob": 0.78, "close": 134.50, "momentum_20d": 12.3, "pred_return": 6.2, "ev": 4.8, "sector": "Technology"},
    ]
    test_portfolio = {
        "cash": 50000, "start_capital": 100000,
        "positions": [
            {"ticker": "FIX", "unrealized_pnl": 5.2, "days_held": 3, "current_value": 35000},
            {"ticker": "MPWR", "unrealized_pnl": -1.1, "days_held": 1, "current_value": 33000},
        ],
    }
    msg = format_daily_picks(test_picks, "2026-02-27", {"auc": 0.939, "hold_days": 5},
                             portfolio=test_portfolio, hold_days=5)
    print(msg)
    send_telegram(msg)
