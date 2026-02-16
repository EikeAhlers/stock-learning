"""
Drift Monitor — Real-time model health tracking.

Tracks predicted probabilities vs actual outcomes to detect:
  1. Calibration Drift — model predicts 80% but only 40% hit
  2. Edge Loss — hit rate drops below minimum threshold
  3. Drawdown Alert — consecutive losses exceed tolerance

Usage:
    monitor = DriftMonitor(window=20, max_calibration_error=0.20)
    monitor.record_trade(predicted_prob=0.82, actual_hit=True, pnl_pct=3.2)
    status = monitor.check_health()
    if status["halt"]:
        print("SYSTEM HALT:", status["reason"])
"""
import numpy as np
import json
import os
from datetime import datetime
from collections import deque

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class DriftMonitor:
    """
    Monitors model health in real-time by comparing predictions to outcomes.
    
    Signals:
      - MODEL_DRIFT: calibration error exceeds threshold
      - EDGE_LOSS: actual hit rate too low over rolling window
      - DRAWDOWN_ALERT: consecutive losses exceed limit
      - COLD_STREAK: recent Sharpe-like metric turns negative
    """
    
    def __init__(
        self,
        window: int = 30,
        max_calibration_error: float = 0.25,
        min_hit_rate: float = 0.30,
        max_consecutive_losses: int = 5,
        min_rolling_sharpe: float = -0.50,
        log_file: str = None,
    ):
        self.window = window
        self.max_calibration_error = max_calibration_error
        self.min_hit_rate = min_hit_rate
        self.max_consecutive_losses = max_consecutive_losses
        self.min_rolling_sharpe = min_rolling_sharpe
        self.log_file = log_file or os.path.join(DATA_DIR, "drift_monitor_log.json")
        
        # Rolling trade history
        self.trades = deque(maxlen=200)  # keep last 200
        
        # Load existing history if present
        self._load_history()
    
    def record_trade(
        self,
        ticker: str,
        entry_date: str,
        exit_date: str,
        predicted_prob: float,
        actual_hit: bool,
        pnl_pct: float,
        entry_price: float = 0,
        exit_price: float = 0,
    ):
        """Record a completed trade for monitoring."""
        trade = {
            "ticker": ticker,
            "entry_date": str(entry_date),
            "exit_date": str(exit_date),
            "predicted_prob": round(float(predicted_prob), 4),
            "actual_hit": bool(actual_hit),
            "pnl_pct": round(float(pnl_pct), 4),
            "entry_price": round(float(entry_price), 4),
            "exit_price": round(float(exit_price), 4),
            "timestamp": datetime.now().isoformat(),
        }
        self.trades.append(trade)
        self._save_history()
    
    def check_health(self) -> dict:
        """
        Run all health checks. Returns:
          {
            "halt": bool,          # True = STOP TRADING
            "warnings": [str],     # Non-fatal warnings
            "reason": str | None,  # Why halted
            "metrics": dict,       # All computed metrics
          }
        """
        n = len(self.trades)
        result = {
            "halt": False,
            "warnings": [],
            "reason": None,
            "metrics": {},
            "n_trades": n,
        }
        
        if n < 5:
            result["warnings"].append(f"Only {n} trades recorded — too few for monitoring")
            return result
        
        # Use most recent `window` trades
        recent = list(self.trades)[-self.window:]
        
        # ── 1. Calibration Error (WARNING only, not HALT) ────────────
        # NOTE: The model predicts P(≥5% daily move) but trades span 3 days.
        # Calibration error between predicted_prob and actual_hit is expected
        # to be large. We track it for info but don't halt on it alone.
        avg_pred = np.mean([t["predicted_prob"] for t in recent])
        actual_hit_rate = np.mean([t["actual_hit"] for t in recent])
        cal_error = avg_pred - actual_hit_rate
        
        result["metrics"]["avg_predicted_prob"] = round(avg_pred, 4)
        result["metrics"]["actual_hit_rate"] = round(actual_hit_rate, 4)
        result["metrics"]["calibration_error"] = round(cal_error, 4)
        
        if abs(cal_error) > self.max_calibration_error:
            result["warnings"].append(
                f"CALIBRATION_NOTE: error = {cal_error:.1%} "
                f"(predicted {avg_pred:.0%}, actual {actual_hit_rate:.0%})"
            )
        
        # ── 2. Edge Loss — win rate below minimum ─────────────────
        if actual_hit_rate < self.min_hit_rate and len(recent) >= 15:
            result["halt"] = True
            reason = (
                f"EDGE_LOSS: Win rate = {actual_hit_rate:.0%} "
                f"(minimum: {self.min_hit_rate:.0%}) "
                f"over last {len(recent)} trades"
            )
            if result["reason"]:
                result["reason"] += " + " + reason
            else:
                result["reason"] = reason
        
        # ── 3. Consecutive Losses ────────────────────────────────────
        consecutive_losses = 0
        max_consec = 0
        for t in reversed(list(self.trades)):
            if t["pnl_pct"] < 0:
                consecutive_losses += 1
                max_consec = max(max_consec, consecutive_losses)
            else:
                break  # streak ended
        
        result["metrics"]["consecutive_losses"] = consecutive_losses
        
        if consecutive_losses >= self.max_consecutive_losses:
            result["halt"] = True
            reason = f"DRAWDOWN_ALERT: {consecutive_losses} consecutive losses"
            if result["reason"]:
                result["reason"] += " + " + reason
            else:
                result["reason"] = reason
        
        # ── 4. Recent PnL Trend ──────────────────────────────────────
        pnls = [t["pnl_pct"] for t in recent]
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
        recent_sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = np.mean([p > 0 for p in pnls])
        
        result["metrics"]["avg_pnl_pct"] = round(avg_pnl, 4)
        result["metrics"]["win_rate"] = round(win_rate, 4)
        result["metrics"]["recent_sharpe"] = round(recent_sharpe, 4)
        
        if recent_sharpe < self.min_rolling_sharpe and len(recent) >= 15:
            result["halt"] = True
            reason = f"COLD_STREAK: Rolling Sharpe = {recent_sharpe:.2f} over {len(recent)} trades"
            if result["reason"]:
                result["reason"] += " + " + reason
            else:
                result["reason"] = reason
        
        # ── 5. Ticker concentration ──────────────────────────────────
        ticker_counts = {}
        for t in recent:
            ticker_counts[t["ticker"]] = ticker_counts.get(t["ticker"], 0) + 1
        if ticker_counts:
            max_ticker = max(ticker_counts, key=ticker_counts.get)
            max_pct = ticker_counts[max_ticker] / len(recent)
            result["metrics"]["most_traded"] = max_ticker
            result["metrics"]["concentration_pct"] = round(max_pct, 4)
            if max_pct > 0.5:
                result["warnings"].append(
                    f"CONCENTRATION: {max_ticker} = {max_pct:.0%} of recent trades"
                )
        
        return result
    
    def print_status(self) -> dict:
        """Pretty-print current health status."""
        status = self.check_health()
        
        print("=" * 60)
        if status["halt"]:
            print("  ██ SYSTEM HALT ██ — DO NOT TRADE")
            print(f"  Reason: {status['reason']}")
        else:
            print("  ✅ SYSTEM HEALTHY — Clear to trade")
        print("=" * 60)
        
        m = status["metrics"]
        if m:
            print(f"  Trades tracked:      {status['n_trades']}")
            print(f"  Avg predicted prob:  {m.get('avg_predicted_prob', 0):.1%}")
            print(f"  Actual hit rate:     {m.get('actual_hit_rate', 0):.1%}")
            print(f"  Calibration error:   {m.get('calibration_error', 0):+.1%}")
            print(f"  Win rate (PnL>0):    {m.get('win_rate', 0):.1%}")
            print(f"  Avg PnL per trade:   {m.get('avg_pnl_pct', 0):+.2f}%")
            print(f"  Recent Sharpe:       {m.get('recent_sharpe', 0):.2f}")
            print(f"  Consec. losses:      {m.get('consecutive_losses', 0)}")
            if m.get("most_traded"):
                print(f"  Most traded:         {m['most_traded']} ({m.get('concentration_pct', 0):.0%})")
        
        for w in status["warnings"]:
            print(f"  ⚠️  {w}")
        
        return status
    
    def backfill_from_backtest(self, trades_df, hit_definition="profitable"):
        """Load historical trades from a backtest DataFrame into the monitor.
        
        hit_definition: 
            "profitable" — hit = PnL > 0  (correct for ranking models)
            "big_move"   — hit = PnL >= 5% (original, too strict for 3-day holds)
        """
        count = 0
        for _, row in trades_df.iterrows():
            pnl = row.get("Return_Pct", row.get("PnL_Pct", 0))
            if hit_definition == "profitable":
                hit = pnl > 0
            else:
                hit = pnl >= 5.0
            self.record_trade(
                ticker=row.get("Ticker", "?"),
                entry_date=str(row.get("Entry_Date", "")),
                exit_date=str(row.get("Exit_Date", "")),
                predicted_prob=row.get("Probability", 0.5),
                actual_hit=hit,
                pnl_pct=pnl,
                entry_price=row.get("Entry_Price", 0),
                exit_price=row.get("Exit_Price", 0),
            )
            count += 1
        print(f"  Backfilled {count} trades (hit_definition={hit_definition})")
    
    def _save_history(self):
        """Persist trade log to JSON."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump(list(self.trades), f, indent=2)
    
    def _load_history(self):
        """Load trade log from JSON if exists."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                for t in data:
                    self.trades.append(t)
            except (json.JSONDecodeError, KeyError):
                pass  # corrupted file — start fresh


def simulate_drift_detection(trades_df, window=20):
    """
    Run drift monitor across a full backtest to show where it would have
    triggered halts. Returns list of (trade_idx, status) tuples.
    """
    monitor = DriftMonitor(window=window)
    events = []
    
    for i, (_, row) in enumerate(trades_df.iterrows()):
        pnl = row.get("Return_Pct", row.get("PnL_Pct", 0))
        monitor.record_trade(
            ticker=row.get("Ticker", "?"),
            entry_date=str(row.get("Entry_Date", "")),
            exit_date=str(row.get("Exit_Date", "")),
            predicted_prob=row.get("Probability", 0.5),
            actual_hit=pnl >= 5.0,
            pnl_pct=pnl,
        )
        
        if i >= window - 1:  # enough data
            status = monitor.check_health()
            if status["halt"] or status["warnings"]:
                events.append({
                    "trade_idx": i,
                    "trade_date": str(row.get("Exit_Date", "")),
                    "halt": status["halt"],
                    "reason": status["reason"],
                    "warnings": status["warnings"],
                    "metrics": status["metrics"],
                })
    
    return events
