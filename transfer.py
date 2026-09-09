#!/usr/bin/env python3
"""SFTP transfer helper for the GPU server (put / get / getdir)."""
import os
import sys

import paramiko

HOST = "i.easy-ai.cloud"
PORT = 32092
USER = "root"
PASSWORD = "T5t8mk7b"


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname=HOST, port=PORT, username=USER, password=PASSWORD,
        timeout=30, banner_timeout=30, auth_timeout=30,
        allow_agent=False, look_for_keys=False,
    )
    return cli


def main():
    if len(sys.argv) < 3:
        print("usage: transfer.py put <local> <remote> | get <remote> <local> | getdir <remote_dir> <local_dir>")
        sys.exit(2)

    op = sys.argv[1]
    cli = connect()
    sftp = cli.open_sftp()

    if op == "put":
        local, remote = sys.argv[2], sys.argv[3]
        sftp.put(local, remote)
        print(f"put {local} -> {remote}")
    elif op == "get":
        remote, local = sys.argv[2], sys.argv[3]
        sftp.get(remote, local)
        print(f"get {remote} -> {local}")
    elif op == "getdir":
        remote_dir, local_dir = sys.argv[2], sys.argv[3]
        os.makedirs(local_dir, exist_ok=True)
        n = 0
        for name in sftp.listdir(remote_dir):
            rp = remote_dir.rstrip("/") + "/" + name
            lp = os.path.join(local_dir, name)
            try:
                st = sftp.stat(rp)
                if stat_is_file(st):
                    sftp.get(rp, lp)
                    n += 1
                    print(f"get {name} ({st.st_size} bytes)")
            except OSError as e:
                print(f"skip {name}: {e}")
        print(f"downloaded {n} files")
    else:
        print(f"unknown op: {op}")
        sys.exit(2)

    sftp.close()
    cli.close()


def stat_is_file(st):
    import stat
    return stat.S_ISREG(st.st_mode)


if __name__ == "__main__":
    main()
