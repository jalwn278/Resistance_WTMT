#!/usr/bin/env python3
"""Paramiko-based remote command runner for the GPU server."""
import sys
import paramiko

HOST = "i.easy-ai.cloud"
PORT = 32092
USER = "root"
PASSWORD = "T5t8mk7b"

BACKUP_HOSTS = ["b.easy-ai.cloud"]
BACKUP_USERS = ["easyai"]


def connect(host, user, password):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname=host,
        port=PORT,
        username=user,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    return cli


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo ok"
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    cli = None
    last_err = None
    # Try primary host, then backups with fallback users.
    candidates = [(HOST, USER, PASSWORD)] + [
        (h, u, PASSWORD) for h in BACKUP_HOSTS for u in BACKUP_USERS
    ]
    for host, user, password in candidates:
        try:
            cli = connect(host, user, password)
            print(f"[connected] {user}@{host}:{PORT}", file=sys.stderr)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[fail] {user}@{host}: {e}", file=sys.stderr)
            continue
    if cli is None:
        print(f"ALL_CONNECT_FAILED: {last_err}", file=sys.stderr)
        sys.exit(2)

    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    sys.stdout.write(out)
    if err.strip():
        sys.stderr.write(err)
    cli.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
