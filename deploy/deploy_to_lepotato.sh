#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  deploy_to_lepotato.sh — Deploy stock scanner to Le Potato
#  Run from Windows: bash deploy_to_lepotato.sh  (or via SSH commands)
# ═══════════════════════════════════════════════════════════════════════════
set -e

LEPOTATO_IP="192.168.178.52"
LEPOTATO_USER="root"
REMOTE_DIR="/root/stock-scanner"

echo "═══════════════════════════════════════════════════════════════"
echo "  Deploying Stock Scanner to Le Potato ($LEPOTATO_IP)"
echo "═══════════════════════════════════════════════════════════════"

# Step 1: Create remote directory
echo "[1/5] Creating remote directory..."
ssh -o StrictHostKeyChecking=no ${LEPOTATO_USER}@${LEPOTATO_IP} "mkdir -p ${REMOTE_DIR}/{logs,data,models}"

# Step 2: Copy files
echo "[2/5] Copying files..."
scp -o StrictHostKeyChecking=no \
    deploy/config.json \
    deploy/daily_scanner.py \
    deploy/telegram_notifier.py \
    deploy/paper_trader.py \
    deploy/requirements.txt \
    deploy/run_daily.sh \
    ${LEPOTATO_USER}@${LEPOTATO_IP}:${REMOTE_DIR}/

# Step 3: Install dependencies
echo "[3/5] Installing Python packages (this may take a few minutes on ARM)..."
ssh ${LEPOTATO_USER}@${LEPOTATO_IP} "cd ${REMOTE_DIR} && pip3 install --break-system-packages -r requirements.txt"

# Step 4: Make scripts executable
echo "[4/5] Setting permissions..."
ssh ${LEPOTATO_USER}@${LEPOTATO_IP} "chmod +x ${REMOTE_DIR}/run_daily.sh ${REMOTE_DIR}/daily_scanner.py"

# Step 5: Set up cron
echo "[5/5] Setting up cron job..."
ssh ${LEPOTATO_USER}@${LEPOTATO_IP} << 'CRON_EOF'
# Remove old stock-scanner cron if exists
crontab -l 2>/dev/null | grep -v "stock-scanner" > /tmp/cron_clean 2>/dev/null || true

# Add new cron job: Run at 9:30 PM UTC (4:30 PM ET) Mon-Fri after market close
echo "30 21 * * 1-5 cd /root/stock-scanner && /bin/bash run_daily.sh >> /root/stock-scanner/logs/cron.log 2>&1" >> /tmp/cron_clean

# Sunday weekly performance report
echo "0 18 * * 0 cd /root/stock-scanner && /usr/bin/python3 daily_scanner.py --status >> /root/stock-scanner/logs/cron.log 2>&1" >> /tmp/cron_clean

crontab /tmp/cron_clean
rm /tmp/cron_clean

echo "Cron jobs installed:"
crontab -l | grep stock-scanner
CRON_EOF

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT COMPLETE!"
echo ""
echo "  Le Potato IP: $LEPOTATO_IP"
echo "  Project dir:  $REMOTE_DIR"
echo "  Schedule:     Mon-Fri 9:30 PM UTC (4:30 PM ET)"
echo ""
echo "  Next steps:"
echo "    1. Set up Telegram bot (see instructions below)"
echo "    2. Test: ssh root@$LEPOTATO_IP 'cd $REMOTE_DIR && python3 daily_scanner.py --manual'"
echo "    3. Check logs: ssh root@$LEPOTATO_IP 'tail -50 $REMOTE_DIR/logs/cron.log'"
echo "═══════════════════════════════════════════════════════════════"
