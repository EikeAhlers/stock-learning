"""Update cron to 20:30 UTC (3:30 PM ET, 30 min before market close)."""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("192.168.178.52", username="root", password="lepotato", timeout=10)

# Read current cron
stdin, stdout, stderr = ssh.exec_command("crontab -l")
old_cron = stdout.read().decode().strip()
print("OLD:")
print(old_cron)

# Replace: 21:05 -> 20:30 (3:30 PM ET = 30 min before close)
new_cron = old_cron.replace("5 21", "30 20")
print("\nNEW:")
print(new_cron)

cmd = 'echo "' + new_cron.replace("\n", "\\n") + '" | crontab -'
stdin, stdout, stderr = ssh.exec_command(cmd)
err = stderr.read().decode()
if err:
    print(f"Error: {err}")
else:
    # Verify
    stdin, stdout, stderr = ssh.exec_command("crontab -l")
    verified = stdout.read().decode().strip()
    if "30 20" in verified:
        print("\n  Cron updated to 20:30 UTC (3:30 PM ET)")
    else:
        print(f"\nVerification: {verified}")

ssh.close()
