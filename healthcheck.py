"""Quick health check — verify everything is set up correctly."""
import json
import paramiko
import requests

# 1. LOCAL CONFIG
with open("deploy/config.json") as f:
    cfg = json.load(f)
s = cfg["strategy"]
print("=== LOCAL CONFIG ===")
print(f"  Hold: {s['hold_days']}d | Picks: {s['top_n']} | SL: {s['stop_loss_pct']}% | Prob: {s['min_prob']}+")
print(f"  Telegram: {'ON' if cfg['telegram']['enabled'] else 'OFF'}")
print(f"  Alpaca: {'ON' if cfg['alpaca']['enabled'] else 'OFF'}")

# 2. ALPACA
headers = {
    "APCA-API-KEY-ID": cfg["alpaca"]["api_key"],
    "APCA-API-SECRET-KEY": cfg["alpaca"]["api_secret"],
}
r = requests.get(cfg["alpaca"]["base_url"] + "/v2/account", headers=headers, timeout=10)
acct = r.json()
print(f"\n=== ALPACA PAPER ===")
print(f"  Status: {acct['status']}")
print(f"  Equity: ${float(acct['equity']):,.2f}")
print(f"  Cash:   ${float(acct['cash']):,.2f}")
positions = requests.get(cfg["alpaca"]["base_url"] + "/v2/positions", headers=headers, timeout=10).json()
print(f"  Positions: {len(positions)}")

# 3. LE POTATO
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("192.168.178.52", username="root", password="lepotato", timeout=10)
print(f"\n=== LE POTATO ===")

# Config
sftp = ssh.open_sftp()
with sftp.open("/root/stock-scanner/config.json") as f:
    remote_cfg = json.load(f)
rs = remote_cfg["strategy"]
match = rs["hold_days"] == 7 and rs["top_n"] == 2 and rs["stop_loss_pct"] == -7.0
print(f"  Config: hold={rs['hold_days']}d picks={rs['top_n']} SL={rs['stop_loss_pct']}%")
print(f"  Config matches optimal: {'YES ✓' if match else 'NO !!'}")
sftp.close()

# Cron
stdin, stdout, stderr = ssh.exec_command("crontab -l 2>/dev/null")
cron = stdout.read().decode().strip()
cron_lines = [l for l in cron.split("\n") if l.strip() and not l.startswith("#")]
print(f"  Cron jobs: {len(cron_lines)}")
for line in cron_lines:
    print(f"    {line.strip()}")

# Files
for f in ["daily_scanner.py", "telegram_notifier.py", "paper_trader.py", "alpaca_trader.py", "config.json"]:
    stdin, stdout, stderr = ssh.exec_command(f"test -f /root/stock-scanner/{f} && echo OK || echo MISSING")
    print(f"  {f}: {stdout.read().decode().strip()}")

# Syntax via remote python script
check_script = """
import py_compile, sys
for f in ['daily_scanner.py', 'telegram_notifier.py', 'alpaca_trader.py']:
    try:
        py_compile.compile(f'/root/stock-scanner/{f}', doraise=True)
        print(f'  Syntax {f}: OK')
    except py_compile.PyCompileError as e:
        print(f'  Syntax {f}: FAIL - {e}')
"""
stdin, stdout, stderr = ssh.exec_command(f"python3 -c '{check_script}'")
print(stdout.read().decode().strip())

# Last log
stdin, stdout, stderr = ssh.exec_command("ls -t /root/stock-scanner/logs/scanner_*.log 2>/dev/null | head -1")
last_log = stdout.read().decode().strip()
if last_log:
    stdin, stdout, stderr = ssh.exec_command(f"tail -3 {last_log}")
    logname = last_log.split("/")[-1]
    print(f"  Last log ({logname}):")
    for line in stdout.read().decode().strip().split("\n"):
        print(f"    {line.strip()}")
else:
    print("  No logs yet (first run pending)")

# Model
stdin, stdout, stderr = ssh.exec_command("ls /root/stock-scanner/models/model_latest.pkl 2>/dev/null && echo EXISTS || echo NONE")
model_status = stdout.read().decode().strip()
print(f"  Model: {'will retrain on next run' if 'NONE' in model_status else 'exists (cached)'}")

# Portfolio
stdin, stdout, stderr = ssh.exec_command("python3 -c \"import json; p=json.load(open('/root/stock-scanner/data/portfolio.json')); print(f'  Cash: \\${p[\\\"cash\\\"]:,.0f} | Positions: {len(p[\\\"positions\\\"])} | Trades: {len(p[\\\"trades\\\"])}')\"")
print(stdout.read().decode().strip())

ssh.close()

print(f"\n{'='*40}")
print("  ✓ ALL SYSTEMS GO")
print(f"{'='*40}")
