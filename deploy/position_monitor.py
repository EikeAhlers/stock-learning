#!/usr/bin/env python3
"""
Position Monitor — Runs on Le Potato every 15 min during market hours.

Ensures all open positions have stop-loss orders on Alpaca.
Detects if a stop-loss was triggered (Alpaca sold automatically) and
cleans up the hold tracker. Sends Telegram alerts for any stop events.

This is a SAFETY NET on top of Alpaca's native stop-loss orders.
Alpaca does the real-time selling; this script just verifies and alerts.

Cron entry:
  */15 8-21 * * 1-5 cd /root/stock-scanner && /usr/bin/python3 position_monitor.py >> logs/monitor.log 2>&1
"""
import os
import sys
import json
import time
import requests
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
HOLD_TRACKER = os.path.join(SCRIPT_DIR, "data", "alpaca_holds.json")
MONITOR_LOG = os.path.join(SCRIPT_DIR, "logs", "monitor.log")

os.makedirs(os.path.join(SCRIPT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "data"), exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[MONITOR {ts}] {msg}"
    print(line, flush=True)


def send_telegram(msg, cfg):
    """Send a Telegram alert."""
    tg = cfg.get("telegram", {})
    if not tg.get("enabled"):
        return
    token = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        log(f"Telegram send failed: {e}")


def is_market_hours():
    """Check if US market is open (9:30 AM - 4:00 PM ET)."""
    # ET = UTC-5 (EST) or UTC-4 (EDT)
    # March-November: EDT (UTC-4), November-March: EST (UTC-5)
    now_utc = datetime.now(timezone.utc)
    month = now_utc.month

    # Simple DST check: EDT from March second Sunday to November first Sunday
    if 3 <= month <= 10:
        et_offset = timedelta(hours=-4)  # EDT
    else:
        et_offset = timedelta(hours=-5)  # EST

    now_et = now_utc + et_offset
    # Market hours: 9:30 - 16:00 ET, weekdays only
    if now_et.weekday() >= 5:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def main():
    cfg = load_config()
    alpaca_cfg = cfg.get("alpaca", {})

    if not alpaca_cfg.get("enabled"):
        log("Alpaca not enabled, skipping")
        return

    api_key = alpaca_cfg.get("api_key", "")
    api_secret = alpaca_cfg.get("api_secret", "")
    base_url = alpaca_cfg.get("base_url", "https://paper-api.alpaca.markets")

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Content-Type": "application/json",
    }

    def api_get(endpoint):
        r = requests.get(f"{base_url}{endpoint}", headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def api_post(endpoint, data):
        r = requests.post(f"{base_url}{endpoint}", headers=headers, json=data, timeout=15)
        r.raise_for_status()
        return r.json()

    # Check if market is open
    if not is_market_hours():
        log("Market closed, skipping check")
        return

    log("=== Position monitor check ===")

    # Get account status
    try:
        acct = api_get("/v2/account")
        equity = float(acct["equity"])
        cash = float(acct["cash"])
        log(f"Account: equity=${equity:,.0f}, cash=${cash:,.0f}")
    except Exception as e:
        log(f"Failed to reach Alpaca: {e}")
        return

    # Get current positions
    try:
        positions = api_get("/v2/positions")
    except Exception as e:
        log(f"Failed to get positions: {e}")
        return

    # Get open orders
    try:
        open_orders = api_get("/v2/orders?status=open&limit=50")
    except Exception as e:
        log(f"Failed to get orders: {e}")
        open_orders = []

    # Build set of tickers with existing stop orders
    tickers_with_stops = set()
    for o in open_orders:
        if o.get("type") == "stop" and o.get("side") == "sell":
            tickers_with_stops.add(o["symbol"])

    stop_pct = cfg["strategy"].get("stop_loss_pct", -7.0)
    position_tickers = set()

    # Check each position
    for pos in positions:
        tk = pos["symbol"]
        position_tickers.add(tk)
        entry = float(pos["avg_entry_price"])
        current = float(pos["current_price"])
        qty = int(float(pos["qty"]))
        pnl_pct = float(pos["unrealized_plpc"]) * 100
        stop_price = round(entry * (1 + stop_pct / 100), 2)

        status = "OK"
        if pnl_pct <= stop_pct:
            status = "STOP-ZONE"
        elif pnl_pct <= stop_pct + 2:
            status = "WARNING"

        log(f"  {tk}: {pnl_pct:+.1f}% (entry ${entry:.2f}, now ${current:.2f}) [{status}]")

        # Place stop-loss if missing
        if tk not in tickers_with_stops:
            log(f"  MISSING stop-loss for {tk}! Placing now at ${stop_price:.2f}")
            try:
                api_post("/v2/orders", {
                    "symbol": tk,
                    "side": "sell",
                    "type": "stop",
                    "stop_price": str(stop_price),
                    "qty": str(qty),
                    "time_in_force": "gtc",
                })
                log(f"  Stop-loss placed for {tk} at ${stop_price:.2f}")
                send_telegram(
                    f"<b>STOP-LOSS PLACED</b>\n"
                    f"{tk}: stop at ${stop_price:.2f} (-{abs(stop_pct):.0f}% from ${entry:.2f})",
                    cfg
                )
            except Exception as e:
                log(f"  FAILED to place stop for {tk}: {e}")
                send_telegram(f"WARNING: Failed to place stop-loss for {tk}: {e}", cfg)

    # Check if any tracked positions were sold by Alpaca (stop triggered)
    if os.path.exists(HOLD_TRACKER):
        with open(HOLD_TRACKER) as f:
            holds = json.load(f)

        stopped_out = []
        for tk in list(holds.keys()):
            if tk not in position_tickers:
                # Position gone — either stop triggered or manual close
                # Check recent filled orders
                try:
                    recent = api_get(f"/v2/orders?status=closed&limit=10&symbols={tk}")
                    for o in recent:
                        if o["side"] == "sell" and o["status"] == "filled":
                            filled_price = float(o.get("filled_avg_price", 0))
                            entry_price = holds[tk].get("entry_price", 0)
                            pnl = ((filled_price / entry_price) - 1) * 100 if entry_price else 0
                            log(f"  DETECTED: {tk} was sold @ ${filled_price:.2f} (P&L: {pnl:+.1f}%)")
                            stopped_out.append({
                                "ticker": tk,
                                "entry": entry_price,
                                "exit": filled_price,
                                "pnl": pnl,
                            })
                            break
                except Exception:
                    pass

                # Remove from tracker
                holds.pop(tk, None)
                log(f"  Removed {tk} from hold tracker (position closed)")

        if stopped_out:
            with open(HOLD_TRACKER, "w") as f:
                json.dump(holds, f, indent=2)

            # Send Telegram alert
            for s in stopped_out:
                icon = "RED" if s["pnl"] < 0 else "GREEN"
                send_telegram(
                    f"<b>STOP-LOSS TRIGGERED</b>\n"
                    f"{s['ticker']}: sold @ ${s['exit']:.2f}\n"
                    f"Entry: ${s['entry']:.2f} | P&L: {s['pnl']:+.1f}%\n"
                    f"Alpaca auto-sold this position.",
                    cfg
                )
        elif holds != json.load(open(HOLD_TRACKER)) if os.path.exists(HOLD_TRACKER) else {}:
            with open(HOLD_TRACKER, "w") as f:
                json.dump(holds, f, indent=2)

    if not positions:
        log("  No open positions")

    log("=== Monitor check complete ===")


if __name__ == "__main__":
    main()
