#!/usr/bin/env python3
"""
Alpaca Paper Trading — Auto-execute picks on Alpaca paper account.

Handles:
  - Buying picks at market open (market orders)
  - Selling positions when hold period expires
  - Stop-loss monitoring
  - Portfolio sync with paper_trader.py

Requires: pip install alpaca-trade-api
"""
import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[ALPACA {timestamp}] {msg}"
    print(line)
    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"alpaca_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, "a") as f:
        f.write(line + "\n")


class AlpacaTrader:
    """Simple Alpaca paper trading client using REST API directly."""

    def __init__(self):
        cfg = load_config()
        alpaca_cfg = cfg.get("alpaca", {})
        self.api_key = alpaca_cfg.get("api_key", "")
        self.api_secret = alpaca_cfg.get("api_secret", "")
        self.base_url = alpaca_cfg.get("base_url", "https://paper-api.alpaca.markets")
        self.enabled = alpaca_cfg.get("enabled", False)

        if not self.api_key or "YOUR_" in self.api_key:
            log("Alpaca not configured — missing API key")
            self.enabled = False

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    def _get(self, endpoint):
        """GET request to Alpaca API."""
        url = f"{self.base_url}{endpoint}"
        resp = requests.get(url, headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint, data):
        """POST request to Alpaca API."""
        url = f"{self.base_url}{endpoint}"
        resp = requests.post(url, headers=self.headers, json=data, timeout=15)
        if resp.status_code not in (200, 201):
            log(f"POST {endpoint} failed: {resp.status_code} {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()

    def _delete(self, endpoint):
        """DELETE request to Alpaca API."""
        url = f"{self.base_url}{endpoint}"
        resp = requests.delete(url, headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp

    # ── Account ──────────────────────────────────────────────────

    def get_account(self):
        """Get account info (equity, cash, buying power)."""
        return self._get("/v2/account")

    def get_equity(self):
        """Get current account equity."""
        acct = self.get_account()
        return float(acct["equity"])

    def get_cash(self):
        """Get available cash."""
        acct = self.get_account()
        return float(acct["cash"])

    # ── Positions ────────────────────────────────────────────────

    def get_positions(self):
        """Get all open positions."""
        return self._get("/v2/positions")

    def get_position(self, ticker):
        """Get position for a specific ticker. Returns None if not held."""
        try:
            return self._get(f"/v2/positions/{ticker}")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    # ── Orders ───────────────────────────────────────────────────

    def buy_market(self, ticker, qty=None, notional=None):
        """Place a market buy order.
        
        Args:
            ticker: Stock symbol
            qty: Number of shares (integer)
            notional: Dollar amount (for fractional shares)
        """
        order = {
            "symbol": ticker,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }
        if notional and not qty:
            order["notional"] = str(round(notional, 2))
        else:
            order["qty"] = str(int(qty))

        log(f"BUY {ticker}: qty={qty}, notional=${notional}")
        result = self._post("/v2/orders", order)
        log(f"  Order placed: {result['id']} status={result['status']}")
        return result

    def sell_market(self, ticker, qty=None):
        """Sell all or specific qty of a position."""
        if qty is None:
            # Sell entire position
            pos = self.get_position(ticker)
            if pos is None:
                log(f"SELL {ticker}: No position found")
                return None
            qty = int(float(pos["qty"]))

        order = {
            "symbol": ticker,
            "side": "sell",
            "type": "market",
            "qty": str(int(qty)),
            "time_in_force": "day",
        }
        log(f"SELL {ticker}: qty={qty}")
        result = self._post("/v2/orders", order)
        log(f"  Order placed: {result['id']} status={result['status']}")
        return result

    def close_position(self, ticker):
        """Close an entire position (sell all shares)."""
        try:
            resp = self._delete(f"/v2/positions/{ticker}")
            log(f"CLOSE {ticker}: position closed")
            return True
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                log(f"CLOSE {ticker}: no position to close")
                return False
            raise

    def get_orders(self, status="open"):
        """Get orders filtered by status."""
        return self._get(f"/v2/orders?status={status}&limit=50")

    # ── High-level trading logic ─────────────────────────────────

    def execute_picks(self, picks, max_positions=3):
        """Execute daily picks: sell expired, buy new.
        
        Args:
            picks: List of pick dicts with ticker, close, prob
            max_positions: Maximum simultaneous positions
        """
        if not self.enabled:
            log("Alpaca trading disabled in config")
            return {"bought": [], "sold": [], "held": []}

        results = {"bought": [], "sold": [], "held": []}

        try:
            acct = self.get_account()
            equity = float(acct["equity"])
            cash = float(acct["cash"])
            log(f"Account: equity=${equity:,.0f}, cash=${cash:,.0f}")
        except Exception as e:
            log(f"Failed to get account: {e}")
            return results

        # Get current positions
        positions = self.get_positions()
        held_tickers = [p["symbol"] for p in positions]
        log(f"Currently holding: {held_tickers if held_tickers else 'none'}")

        for p in positions:
            pnl_pct = float(p["unrealized_plpc"]) * 100
            results["held"].append({
                "ticker": p["symbol"],
                "pnl_pct": pnl_pct,
                "qty": int(float(p["qty"])),
                "market_value": float(p["market_value"]),
            })

        # Buy new picks (skip if already held)
        open_slots = max_positions - len(positions)
        if open_slots <= 0:
            log(f"No open slots ({len(positions)}/{max_positions} positions)")
            return results

        buys = []
        for pick in picks:
            tk = pick["ticker"]
            if tk in held_tickers:
                log(f"  Skip {tk} — already held")
                continue
            buys.append(pick)

        buys = buys[:open_slots]

        if not buys:
            log("No new buys needed")
            return results

        # Position sizing: deploy only max_portfolio_pct of equity (default 50%)
        cfg = load_config()
        max_pct = cfg["strategy"].get("max_portfolio_pct", 0.50)
        deploy_amount = equity * max_pct
        pos_size = deploy_amount / max_positions
        log(f"Position sizing: {max_pct:.0%} of ${equity:,.0f} = ${deploy_amount:,.0f}, ${pos_size:,.0f}/pick")

        for pick in buys:
            tk = pick["ticker"]
            price = pick["close"]
            qty = int(pos_size / (price * 1.001))  # slight buffer for slippage

            if qty <= 0 or cash < qty * price:
                log(f"  Skip {tk} — insufficient cash (need ${qty*price:,.0f}, have ${cash:,.0f})")
                continue

            try:
                self.buy_market(tk, qty=qty)
                cash -= qty * price
                results["bought"].append({"ticker": tk, "qty": qty, "price": price})
            except Exception as e:
                log(f"  FAILED to buy {tk}: {e}")

        return results

    def sell_expired(self, hold_tracker_path=None):
        """Sell positions that have exceeded hold period.
        
        Uses a JSON file to track when positions were opened.
        """
        cfg = load_config()
        hold_days = cfg["strategy"].get("hold_days", 5)

        # Load hold tracker
        if hold_tracker_path is None:
            hold_tracker_path = os.path.join(SCRIPT_DIR, "data", "alpaca_holds.json")

        if os.path.exists(hold_tracker_path):
            with open(hold_tracker_path) as f:
                holds = json.load(f)
        else:
            holds = {}

        today = datetime.now()
        sold = []

        positions = self.get_positions()
        for pos in positions:
            tk = pos["symbol"]
            entry_date_str = holds.get(tk, {}).get("entry_date")

            if not entry_date_str:
                # First time seeing this — record it
                holds[tk] = {
                    "entry_date": today.strftime("%Y-%m-%d"),
                    "entry_price": float(pos["avg_entry_price"]),
                }
                continue

            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d")

            # Count trading days held
            trading_days = 0
            d = entry_date
            while d < today:
                d += timedelta(days=1)
                if d.weekday() < 5:
                    trading_days += 1

            # Check stop loss
            pnl_pct = float(pos["unrealized_plpc"]) * 100
            stop_loss = cfg["strategy"].get("stop_loss_pct", -7.0)

            if pnl_pct <= stop_loss:
                log(f"STOP LOSS {tk}: {pnl_pct:+.1f}% (limit: {stop_loss}%)")
                try:
                    self.close_position(tk)
                    sold.append({"ticker": tk, "reason": "stop_loss", "pnl": pnl_pct})
                    holds.pop(tk, None)
                except Exception as e:
                    log(f"  Failed to close {tk}: {e}")
                continue

            # Check hold period expiry
            if trading_days >= hold_days:
                log(f"HOLD EXPIRY {tk}: {trading_days}d held, P&L={pnl_pct:+.1f}%")
                try:
                    self.close_position(tk)
                    sold.append({"ticker": tk, "reason": "hold_expiry", "pnl": pnl_pct})
                    holds.pop(tk, None)
                except Exception as e:
                    log(f"  Failed to close {tk}: {e}")

        # Save updated tracker
        os.makedirs(os.path.dirname(hold_tracker_path), exist_ok=True)
        with open(hold_tracker_path, "w") as f:
            json.dump(holds, f, indent=2)

        return sold

    def record_buys(self, picks, hold_tracker_path=None):
        """Record today's buys in the hold tracker."""
        if hold_tracker_path is None:
            hold_tracker_path = os.path.join(SCRIPT_DIR, "data", "alpaca_holds.json")

        if os.path.exists(hold_tracker_path):
            with open(hold_tracker_path) as f:
                holds = json.load(f)
        else:
            holds = {}

        today_str = datetime.now().strftime("%Y-%m-%d")
        for pick in picks:
            tk = pick["ticker"]
            if tk not in holds:
                holds[tk] = {
                    "entry_date": today_str,
                    "entry_price": pick["close"],
                }

        os.makedirs(os.path.dirname(hold_tracker_path), exist_ok=True)
        with open(hold_tracker_path, "w") as f:
            json.dump(holds, f, indent=2)

    def get_summary(self):
        """Get a summary string for Telegram."""
        if not self.enabled:
            return ""

        try:
            acct = self.get_account()
            equity = float(acct["equity"])
            cash = float(acct["cash"])
            positions = self.get_positions()

            lines = [f"\n🏦 <b>ALPACA PAPER</b>"]
            lines.append(f"  Equity: ${equity:,.0f}  |  Cash: ${cash:,.0f}")

            if positions:
                for p in positions:
                    pnl = float(p["unrealized_plpc"]) * 100
                    icon = "🟢" if pnl >= 0 else "🔴"
                    lines.append(f"  {icon} {p['symbol']}  {pnl:+.1f}%  ({p['qty']} shares)")

            return "\n".join(lines)
        except Exception as e:
            log(f"Summary failed: {e}")
            return f"\n🏦 ALPACA: Error — {str(e)[:60]}"


# ── CLI for testing ──────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = AlpacaTrader()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "status":
            acct = trader.get_account()
            print(f"Equity:  ${float(acct['equity']):,.2f}")
            print(f"Cash:    ${float(acct['cash']):,.2f}")
            print(f"Buying:  ${float(acct['buying_power']):,.2f}")
            print(f"Day trades: {acct['daytrade_count']}")
            print()
            positions = trader.get_positions()
            if positions:
                print("Positions:")
                for p in positions:
                    pnl = float(p["unrealized_plpc"]) * 100
                    print(f"  {p['symbol']:6s}  {int(float(p['qty'])):4d} shares"
                          f"  avg=${float(p['avg_entry_price']):.2f}"
                          f"  mkt=${float(p['current_price']):.2f}"
                          f"  P&L={pnl:+.1f}%")
            else:
                print("No positions")

        elif cmd == "sell-expired":
            sold = trader.sell_expired()
            print(f"Sold {len(sold)} positions")
            for s in sold:
                print(f"  {s['ticker']}: {s['reason']} ({s['pnl']:+.1f}%)")

        elif cmd == "test":
            # Just verify connection
            acct = trader.get_account()
            print(f"✅ Connected to Alpaca paper account")
            print(f"   Account ID: {acct['id']}")
            print(f"   Equity: ${float(acct['equity']):,.2f}")
            print(f"   Status: {acct['status']}")
    else:
        print("Usage: python alpaca_trader.py [status|sell-expired|test]")
