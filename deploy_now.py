#!/usr/bin/env python3
"""Deploy updated files to Le Potato."""
import paramiko
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("192.168.178.52", username="root", password="lepotato", timeout=10)
print("Connected to Le Potato")

# Upload files
sftp = ssh.open_sftp()
base_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy")
base_remote = "/root/stock-scanner"

files = ["daily_scanner.py", "telegram_notifier.py", "alpaca_trader.py", "config.json"]
for f in files:
    local = os.path.join(base_local, f)
    remote = f"{base_remote}/{f}"
    sftp.put(local, remote)
    print(f"  Uploaded {f}")

sftp.close()

# Verify syntax on Le Potato
for f in ["daily_scanner.py", "telegram_notifier.py", "alpaca_trader.py"]:
    cmd = f'python3 -c "import py_compile; py_compile.compile(\'/root/stock-scanner/{f}\', doraise=True); print(\'{f} OK\')"'
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(f"  {out}")
    if err:
        print(f"  ERR: {err[:300]}")

# Delete old model so it retrains with NaN fix
stdin, stdout, stderr = ssh.exec_command("rm -f /root/stock-scanner/models/model_latest.pkl && echo 'Old model deleted'")
print(f"  {stdout.read().decode().strip()}")

# Ensure requests is installed (for alpaca_trader)
stdin, stdout, stderr = ssh.exec_command("pip3 show requests | head -2")
out = stdout.read().decode().strip()
print(f"  requests: {out.split(chr(10))[0] if out else 'NOT INSTALLED'}")

ssh.close()
print("\nDeployment complete!")
