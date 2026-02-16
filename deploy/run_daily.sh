#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_daily.sh — Wrapper for cron to run the daily scanner
# ═══════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Timestamp
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Stock Scanner — $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════"

# Run scanner
/usr/bin/python3 "$SCRIPT_DIR/daily_scanner.py" 2>&1

echo "Done at $(date '+%H:%M:%S')"
