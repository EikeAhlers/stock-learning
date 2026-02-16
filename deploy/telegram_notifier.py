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


def format_daily_picks(picks: list, scan_date: str, model_info: dict = None) -> str:
    """Format daily stock picks into a Telegram message."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    msg = f"🔔 <b>STOCK SCANNER — {scan_date}</b>\n"
    msg += f"📅 Scanned: {now}\n\n"

    if model_info:
        msg += f"📊 Model: AUC {model_info.get('auc', 'N/A'):.3f}, "
        msg += f"trained on {model_info.get('train_rows', 'N/A'):,} rows\n\n"

    if not picks:
        msg += "⚠️ <b>No tradeable picks today.</b>\n"
        msg += "Filters: momentum > 0, prob 50-85%\n"
        return msg

    msg += f"🎯 <b>TOP {len(picks)} PICKS</b> (buy tomorrow at open, hold {model_info.get('hold_days', 5)}d)\n\n"

    for i, p in enumerate(picks):
        rank = i + 1
        prob = p.get("prob", 0)
        ticker = p.get("ticker", "???")
        price = p.get("close", 0)
        mom = p.get("momentum_20d", 0)
        sector = p.get("sector", "")

        # Confidence emoji
        if prob >= 0.75:
            conf = "🔥"
        elif prob >= 0.65:
            conf = "✅"
        else:
            conf = "📊"

        msg += f"{conf} <b>#{rank} {ticker}</b>  —  {prob:.0%} prob\n"
        msg += f"   💰 ${price:.2f}  |  Mom: {mom:+.1f}%"
        if sector:
            msg += f"  |  {sector}"
        msg += "\n\n"

    msg += "─────────────────\n"
    msg += "⚠️ Paper trading only. Not financial advice.\n"
    msg += f"🤖 Walk-forward XGBoost | P-cap 85% | SL -7%"
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
        {"ticker": "NVDA", "prob": 0.78, "close": 134.50, "momentum_20d": 12.3, "sector": "Technology"},
        {"ticker": "NRG", "prob": 0.71, "close": 98.20, "momentum_20d": 8.5, "sector": "Utilities"},
    ]
    msg = format_daily_picks(test_picks, "2026-02-15", {"auc": 0.869, "train_rows": 126000, "hold_days": 5})
    print(msg)
    send_telegram(msg)
