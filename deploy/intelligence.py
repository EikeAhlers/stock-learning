#!/usr/bin/env python3
"""
Market Intelligence Service — Runs throughout the trading day on Le Potato.

Modes (via cron):
  --morning     9:00 AM ET  Pre-market briefing: gap detection, overnight news
  --midday      12:00 PM ET  Position health check, market regime update
  --prescan     3:00 PM ET  Pre-buy validation: verify market isn't crashing
  --evening     6:00 PM ET  Post-close early scan: generate tomorrow's watchlist
  --cleanup     Check if Alpaca auto-sold any positions (stop triggered)

Cron schedule (all UTC, Mon-Fri):
  0  14 * * 1-5  intelligence.py --morning
  0  17 * * 1-5  intelligence.py --midday
  0  20 * * 1-5  intelligence.py --prescan
  0  23 * * 1-5  intelligence.py --evening
  */30 14-21 * * 1-5  intelligence.py --cleanup
"""
import os
import sys
import json
import time
import pickle
import argparse
import requests
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
HOLD_TRACKER = os.path.join(SCRIPT_DIR, "data", "alpaca_holds.json")
WATCHLIST_PATH = os.path.join(SCRIPT_DIR, "data", "watchlist.json")
INTEL_LOG = os.path.join(SCRIPT_DIR, "logs", "intelligence.log")

os.makedirs(os.path.join(SCRIPT_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "data"), exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[INTEL {ts}] {msg}"
    # Only print to stdout — cron redirects to intelligence.log
    print(line, flush=True)


def send_telegram(msg, cfg=None):
    """Send a Telegram alert."""
    if cfg is None:
        cfg = load_config()
    tg = cfg.get("telegram", {})
    if not tg.get("enabled"):
        return False
    token = tg.get("bot_token", "")
    chat_id = tg.get("chat_id", "")
    if not token or not chat_id:
        return False
    try:
        # Tag all messages with project name
        tagged = f"[Stock Learning]\n{msg}"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={
            "chat_id": chat_id,
            "text": tagged,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log(f"Telegram failed: {e}")
        return False


class AlpacaAPI:
    """Lightweight Alpaca REST client for intelligence checks."""

    def __init__(self, cfg):
        ac = cfg.get("alpaca", {})
        self.base = ac.get("base_url", "https://paper-api.alpaca.markets")
        self.headers = {
            "APCA-API-KEY-ID": ac.get("api_key", ""),
            "APCA-API-SECRET-KEY": ac.get("api_secret", ""),
        }

    def get(self, endpoint):
        r = requests.get(f"{self.base}{endpoint}", headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def account(self):
        return self.get("/v2/account")

    def positions(self):
        return self.get("/v2/positions")

    def orders(self, status="open"):
        return self.get(f"/v2/orders?status={status}&limit=50")


def get_market_snapshot():
    """Get broad market health indicators using free Yahoo Finance data."""
    import yfinance as yf

    indicators = {}

    # SPY (S&P 500), QQQ (Nasdaq), VIX
    tickers = ["SPY", "QQQ", "^VIX", "IWM"]
    try:
        data = yf.download(tickers, period="5d", progress=False, threads=True)
        close = data["Close"]

        for tk in tickers:
            clean = tk.replace("^", "")
            if tk in close.columns:
                prices = close[tk].dropna()
                if len(prices) >= 2:
                    last = float(prices.iloc[-1])
                    prev = float(prices.iloc[-2])
                    chg = (last / prev - 1) * 100
                    indicators[clean] = {"price": last, "change_pct": chg}
    except Exception as e:
        log(f"Market snapshot failed: {e}")

    return indicators


# ────────────────────────────────────────────────────────────────
# MODE: MORNING BRIEFING (9:00 AM ET)
# ────────────────────────────────────────────────────────────────

def morning_briefing():
    """Pre-market analysis: check overnight gaps, market health, held positions."""
    log("=== MORNING BRIEFING ===")
    cfg = load_config()
    api = AlpacaAPI(cfg)

    # 1. Account snapshot
    try:
        acct = api.account()
        equity = float(acct["equity"])
        cash = float(acct["cash"])
        log(f"Account: ${equity:,.0f} equity, ${cash:,.0f} cash")
    except Exception as e:
        log(f"Alpaca unreachable: {e}")
        return

    # 2. Check held positions
    positions = api.positions()
    if not positions:
        log("No open positions")
        send_telegram("Morning: No open positions. Cash ready for today's scan.", cfg)
        return

    alerts = []
    position_lines = []

    for pos in positions:
        tk = pos["symbol"]
        entry = float(pos["avg_entry_price"])
        current = float(pos["current_price"])
        pnl_pct = float(pos["unrealized_plpc"]) * 100
        qty = int(float(pos["qty"]))

        icon = "+" if pnl_pct >= 0 else ""
        position_lines.append(f"  {tk}: {icon}{pnl_pct:.1f}% (${current:.2f})")

        # Flag big overnight moves
        if pnl_pct <= -5:
            alerts.append(f"WARNING: {tk} down {pnl_pct:.1f}% - approaching stop-loss")
        elif pnl_pct >= 8:
            alerts.append(f"STRONG: {tk} up {pnl_pct:.1f}% - consider if trailing stop makes sense")

    # 3. Market health
    market = get_market_snapshot()
    market_lines = []
    market_warning = False

    for name in ["SPY", "QQQ", "VIX", "IWM"]:
        if name in market:
            m = market[name]
            icon = "+" if m["change_pct"] >= 0 else ""
            market_lines.append(f"  {name}: ${m['price']:.2f} ({icon}{m['change_pct']:.1f}%)")
            if name == "VIX" and m["price"] > 30:
                alerts.append(f"HIGH VIX: {m['price']:.1f} - elevated fear, be cautious")
                market_warning = True
            if name == "SPY" and m["change_pct"] < -2:
                alerts.append(f"MARKET SELL-OFF: SPY {m['change_pct']:.1f}% - consider pausing buys")
                market_warning = True

    # 4. Build Telegram message
    msg = "<b>MORNING BRIEFING</b>\n\n"
    msg += f"Account: ${equity:,.0f}\n\n"

    msg += "<b>Positions:</b>\n"
    msg += "\n".join(position_lines) + "\n"

    if market_lines:
        msg += "\n<b>Market:</b>\n"
        msg += "\n".join(market_lines) + "\n"

    if alerts:
        msg += "\n<b>ALERTS:</b>\n"
        for a in alerts:
            msg += f"  ! {a}\n"

    send_telegram(msg, cfg)
    log("Morning briefing sent")


# ────────────────────────────────────────────────────────────────
# MODE: MIDDAY CHECK (12:00 PM ET)
# ────────────────────────────────────────────────────────────────

def midday_check():
    """Midday position health: check intraday moves, volume anomalies."""
    log("=== MIDDAY CHECK ===")
    cfg = load_config()
    api = AlpacaAPI(cfg)

    positions = api.positions()
    if not positions:
        log("No positions to check")
        return

    import yfinance as yf

    alerts = []

    for pos in positions:
        tk = pos["symbol"]
        entry = float(pos["avg_entry_price"])
        current = float(pos["current_price"])
        pnl_pct = float(pos["unrealized_plpc"]) * 100
        change_today = float(pos.get("change_today", 0)) * 100

        log(f"  {tk}: P&L {pnl_pct:+.1f}%, today {change_today:+.1f}%")

        # Check for big intraday moves
        if change_today <= -4:
            alerts.append(f"{tk} dropping hard today ({change_today:+.1f}%) - stop at ${entry * 0.93:.2f}")
        elif change_today >= 5:
            alerts.append(f"{tk} surging today ({change_today:+.1f}%) - looking good")

        # Check volume vs average
        try:
            info = yf.Ticker(tk).history(period="5d")
            if len(info) >= 2:
                avg_vol = info["Volume"].iloc[:-1].mean()
                today_vol = info["Volume"].iloc[-1]
                if avg_vol > 0:
                    vol_ratio = today_vol / avg_vol
                    if vol_ratio > 3:
                        alerts.append(f"{tk} unusual volume ({vol_ratio:.1f}x average) - something happening")
        except Exception:
            pass

    if alerts:
        msg = "<b>MIDDAY ALERT</b>\n\n"
        for a in alerts:
            msg += f"! {a}\n"
        send_telegram(msg, cfg)
        log(f"Midday alerts sent: {len(alerts)}")
    else:
        log("Nothing unusual at midday")


# ────────────────────────────────────────────────────────────────
# MODE: PRE-SCAN VALIDATION (3:00 PM ET, 30 min before scanner)
# ────────────────────────────────────────────────────────────────

def prescan_validation():
    """Check market conditions before the scanner runs at 3:30 PM.
    
    Creates a skip_file if conditions are too dangerous to buy.
    The daily scanner can check for this file and skip buying.
    """
    log("=== PRE-SCAN VALIDATION ===")
    cfg = load_config()
    skip_file = os.path.join(SCRIPT_DIR, "data", "skip_buy_today.json")

    # Remove yesterday's skip file if present
    if os.path.exists(skip_file):
        os.remove(skip_file)

    market = get_market_snapshot()
    reasons = []

    # Check for market crash
    spy = market.get("SPY", {})
    vix = market.get("VIX", {})

    if spy.get("change_pct", 0) < -3:
        reasons.append(f"SPY crashed {spy['change_pct']:.1f}% today")

    if vix.get("price", 0) > 35:
        reasons.append(f"VIX at {vix['price']:.1f} (extreme fear)")

    qqq = market.get("QQQ", {})
    if qqq.get("change_pct", 0) < -4:
        reasons.append(f"QQQ crashed {qqq['change_pct']:.1f}% today")

    if reasons:
        skip_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "reasons": reasons,
            "market": market,
        }
        with open(skip_file, "w") as f:
            json.dump(skip_data, f, indent=2)

        msg = "<b>PRE-SCAN WARNING</b>\n\n"
        msg += "Today's scan may skip buying:\n"
        for r in reasons:
            msg += f"  ! {r}\n"
        msg += "\nScanner will still run but won't buy in this environment."
        send_telegram(msg, cfg)
        log(f"Skip file created: {reasons}")
    else:
        log("Market conditions OK for buying")

        # Send a brief all-clear
        lines = []
        for name in ["SPY", "QQQ", "VIX"]:
            if name in market:
                m = market[name]
                icon = "+" if m["change_pct"] >= 0 else ""
                lines.append(f"{name} {icon}{m['change_pct']:.1f}%")
        log(f"Market: {', '.join(lines)}")


# ────────────────────────────────────────────────────────────────
# MODE: EVENING PRE-SCAN (6:00 PM ET)
# ────────────────────────────────────────────────────────────────

def evening_prescan():
    """Run the model on today's close data to generate tomorrow's watchlist.
    
    This gives us overnight due diligence time. If a pick has bad news
    by tomorrow, the pre-scan validation can flag it.
    """
    log("=== EVENING PRE-SCAN ===")
    cfg = load_config()

    # Import scanner functions
    try:
        sys.path.insert(0, SCRIPT_DIR)
        from daily_scanner import (
            get_sp500_tickers, download_fresh_data, add_features,
            load_model
        )
    except ImportError as e:
        log(f"Could not import scanner functions: {e}")
        return

    import numpy as np
    import pandas as pd

    today = datetime.now().strftime("%Y-%m-%d")
    strategy = cfg.get("strategy", {})
    min_prob = strategy.get("min_prob", 0.60)
    max_prob = strategy.get("max_prob", 0.85)

    try:
        # Download fresh data
        log("Downloading prices...")
        tickers = get_sp500_tickers()
        prices = download_fresh_data(tickers)
        log(f"Downloaded {len(prices)} stock-days")

        # Engineer features (returns tuple: DataFrame, feature_list)
        log("Engineering features...")
        df, feature_list = add_features(prices)
        log(f"Features ready: {len(df)} rows, {len(feature_list)} features")

        # Load existing model (don't retrain, just score)
        model_data = load_model()
        if model_data is None:
            log("No model found — skipping evening scan")
            return

        model = model_data["model"]
        scaler = model_data.get("scaler")
        feature_cols = model_data.get("features", [])

        if not feature_cols:
            log("No feature columns in model — skipping")
            return

        # Score today's stocks
        latest = df.groupby("Ticker").tail(1).copy()

        # Only keep rows that have all required features
        available = [c for c in feature_cols if c in latest.columns]
        if len(available) < len(feature_cols) * 0.8:
            log(f"Only {len(available)}/{len(feature_cols)} features available — skipping")
            return

        scoreable = latest.dropna(subset=available, how="all")
        X = scoreable[available].fillna(0).values

        # Apply scaler if available (model was trained on scaled features)
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception as e:
                log(f"Scaler transform failed ({e}), using raw features")

        scoreable = scoreable.copy()
        scoreable["prob"] = model.predict_proba(X)[:, 1]

        # Get momentum from Prev_Return_20d
        if "Prev_Return_20d" in scoreable.columns:
            scoreable["momentum_20d"] = scoreable["Prev_Return_20d"]
        else:
            scoreable["momentum_20d"] = 0

        # Apply filters
        filtered = scoreable[
            (scoreable["prob"] >= min_prob) &
            (scoreable["prob"] <= max_prob) &
            (scoreable["momentum_20d"] > 0)
        ].copy()
        filtered = filtered.sort_values("prob", ascending=False)

        top = filtered.head(10)
        watchlist = []
        for _, row in top.iterrows():
            watchlist.append({
                "ticker": row["Ticker"],
                "prob": round(float(row["prob"]), 3),
                "close": round(float(row["Close"]), 2),
                "momentum": round(float(row.get("momentum_20d", 0)), 1),
            })

        # Save watchlist
        wl_data = {
            "date": today,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "candidates": watchlist,
        }
        with open(WATCHLIST_PATH, "w") as f:
            json.dump(wl_data, f, indent=2)

        log(f"Watchlist saved: {len(watchlist)} candidates")
        for w in watchlist[:5]:
            log(f"  {w['ticker']}: P={w['prob']:.0%} ${w['close']:.2f} mom={w['momentum']:+.1f}%")

        # Send tomorrow's preview
        if watchlist:
            msg = "<b>TOMORROW'S WATCHLIST</b>\n\n"
            msg += "Top candidates for next scan:\n"
            for i, w in enumerate(watchlist[:5], 1):
                msg += f"  {i}. <b>{w['ticker']}</b> P={w['prob']:.0%} ${w['close']:.2f}\n"
            msg += f"\n{len(watchlist)} total candidates passed filters."
            msg += "\nFinal picks confirmed at 3:30 PM ET tomorrow."
            send_telegram(msg, cfg)

    except Exception as e:
        log(f"Evening scan failed: {e}")
        import traceback
        traceback.print_exc()


# ────────────────────────────────────────────────────────────────
# MODE: CLEANUP (check for Alpaca auto-sells)
# ────────────────────────────────────────────────────────────────

def cleanup_check():
    """Detect if Alpaca auto-sold any positions via stop-loss.
    
    Checks if tracked positions no longer exist on Alpaca.
    Cleans up hold tracker and sends alert.
    """
    cfg = load_config()
    api = AlpacaAPI(cfg)

    if not os.path.exists(HOLD_TRACKER):
        return

    with open(HOLD_TRACKER) as f:
        holds = json.load(f)

    if not holds:
        return

    try:
        positions = api.positions()
    except Exception as e:
        log(f"Cleanup: can't reach Alpaca: {e}")
        return

    position_tickers = {p["symbol"] for p in positions}
    removed = []

    for tk in list(holds.keys()):
        if tk not in position_tickers:
            entry_price = holds[tk].get("entry_price", 0)
            entry_date = holds[tk].get("entry_date", "?")
            log(f"DETECTED: {tk} no longer held (was from {entry_date} @ ${entry_price:.2f})")

            # Try to find the fill price from recent orders
            fill_price = None
            try:
                orders = api.get(f"/v2/orders?status=closed&symbols={tk}&limit=5")
                for o in orders:
                    if o["side"] == "sell" and o["status"] == "filled":
                        fill_price = float(o.get("filled_avg_price", 0))
                        break
            except Exception:
                pass

            pnl = ((fill_price / entry_price) - 1) * 100 if fill_price and entry_price else None
            removed.append({
                "ticker": tk,
                "entry_price": entry_price,
                "exit_price": fill_price,
                "pnl": pnl,
                "entry_date": entry_date,
            })
            holds.pop(tk)

    if removed:
        with open(HOLD_TRACKER, "w") as f:
            json.dump(holds, f, indent=2)

        for r in removed:
            pnl_str = f"{r['pnl']:+.1f}%" if r['pnl'] is not None else "unknown"
            exit_str = f"${r['exit_price']:.2f}" if r['exit_price'] else "unknown"
            msg = (
                f"<b>POSITION CLOSED</b>\n\n"
                f"{r['ticker']} sold automatically\n"
                f"Entry: ${r['entry_price']:.2f} ({r['entry_date']})\n"
                f"Exit: {exit_str}\n"
                f"P&L: {pnl_str}"
            )
            send_telegram(msg, cfg)
            log(f"Alert sent: {r['ticker']} closed, P&L {pnl_str}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Intelligence Service")
    parser.add_argument("--morning", action="store_true", help="Pre-market briefing")
    parser.add_argument("--midday", action="store_true", help="Midday position check")
    parser.add_argument("--prescan", action="store_true", help="Pre-buy market validation")
    parser.add_argument("--evening", action="store_true", help="Post-close early scan")
    parser.add_argument("--cleanup", action="store_true", help="Check for auto-sold positions")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    args = parser.parse_args()

    if args.all or args.morning:
        morning_briefing()
    if args.all or args.midday:
        midday_check()
    if args.all or args.prescan:
        prescan_validation()
    if args.all or args.evening:
        evening_prescan()
    if args.all or args.cleanup:
        cleanup_check()

    if not any([args.morning, args.midday, args.prescan, args.evening, args.cleanup, args.all]):
        print("Usage: python intelligence.py [--morning|--midday|--prescan|--evening|--cleanup|--all]")
