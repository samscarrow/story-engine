#!/usr/bin/env python3
"""
Cross-platform launcher for the Cloud SQL Auth Proxy (macOS/Linux/Windows).

It discovers the proxy binary (cloud-sql-proxy) on PATH or via env, accepts
common flags, and can read defaults from .env.

Environment (optional):
  CLOUDSQL_INSTANCE  - Instance connection name (project:region:instance)
  CLOUDSQL_PORT      - Local listen port (default: 5433)
  CLOUDSQL_ADDRESS   - Local listen address (default: 127.0.0.1)
  CLOUDSQL_PROXY_BIN - Explicit path to proxy binary

Examples:
  # Using env values from .env
  python scripts/start_cloud_sql_proxy.py

  # Explicit args
  python scripts/start_cloud_sql_proxy.py \
    --connection-name my-proj:us-central1:pg-prod --port 5432 --iam

Notes:
  - Requires you to be authenticated with Google Cloud (ADC), e.g.:
      gcloud auth application-default login
  - Ensure you have the Cloud SQL Auth Proxy installed locally.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    # Load CLOUDSQL_* from .env if available
    from story_engine.core.common.dotenv_loader import load_dotenv_keys  # type: ignore
    load_dotenv_keys(keys_prefixes=("CLOUDSQL_",))
except Exception:
    # Best-effort only
    pass


def find_proxy_binary(explicit: str | None = None) -> str | None:
    """Locate the Cloud SQL Auth Proxy binary across platforms.

    Order:
      1) Explicit path
      2) $CLOUDSQL_PROXY_BIN
      3) cloud-sql-proxy on PATH
      4) cloud_sql_proxy on PATH (legacy name)
      5) ./cloud-sql-proxy(.exe) in CWD
    """
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env_bin = os.getenv("CLOUDSQL_PROXY_BIN")
    if env_bin:
        candidates.append(env_bin)
    # Common names
    for name in ("cloud-sql-proxy", "cloud_sql_proxy"):
        hit = shutil.which(name)
        if hit:
            candidates.append(hit)
    # Current directory fallbacks
    for local in ("cloud-sql-proxy", "cloud-sql-proxy.exe"):
        p = Path.cwd() / local
        if p.exists() and p.is_file():
            candidates.append(str(p))

    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def build_command(proxy_bin: str, connection_name: str, port: int, address: str, iam: bool) -> list[str]:
    cmd = [proxy_bin, "--port", str(port), "--address", address]
    if iam:
        cmd.append("--auto-iam-authn")
    cmd.append(connection_name)
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Start Cloud SQL Auth Proxy (cross-platform)")
    p.add_argument("--connection-name", dest="connection_name", default=os.getenv("CLOUDSQL_INSTANCE"),
                   help="Cloud SQL instance connection name (project:region:instance)")
    p.add_argument("--port", type=int, default=int(os.getenv("CLOUDSQL_PORT", "5433")),
                   help="Local port to listen on (default: 5433)")
    p.add_argument("--address", default=os.getenv("CLOUDSQL_ADDRESS", "127.0.0.1"),
                   help="Local address to bind (default: 127.0.0.1)")
    p.add_argument("--iam", action="store_true", help="Enable automatic IAM authentication")
    p.add_argument("--proxy-bin", default=None, help="Explicit path to cloud-sql-proxy binary")
    p.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args = p.parse_args()

    if not args.connection_name:
        print("Error: --connection-name not provided and CLOUDSQL_INSTANCE not set.", file=sys.stderr)
        sys.exit(2)

    proxy = find_proxy_binary(args.proxy_bin)
    if not proxy:
        print("cloud-sql-proxy not found. Install it and/or set CLOUDSQL_PROXY_BIN.", file=sys.stderr)
        print("Download: https://cloud.google.com/sql/docs/postgres/sql-proxy", file=sys.stderr)
        sys.exit(1)

    cmd = build_command(proxy, args.connection_name, args.port, args.address, args.iam)
    if args.dry_run:
        print("Command:", " ".join(cmd))
        return

    print(f"Starting Cloud SQL Auth Proxy on {args.address}:{args.port} for {args.connection_name}...")
    try:
        # Inherit stdio; user stops with Ctrl-C
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Proxy exited with error (code {e.returncode}).", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()


