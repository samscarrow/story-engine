#!/usr/bin/env python3
"""Simple Oracle connection test to isolate the issue."""

import os
from pathlib import Path
from dotenv import load_dotenv
import oracledb


def test_connection():
    """Test Oracle connection with minimal configuration."""

    # Load environment
    load_dotenv(".env.oracle")

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    dsn = os.getenv("DB_DSN")

    print(f"Testing connection to: {dsn} as {user}")

    # Set wallet path
    wallet_path = Path("./oracle_wallet").resolve()
    print(f"Wallet path: {wallet_path}")

    # Method 1: Direct config_dir
    print("\n1. Testing with config_dir parameter...")
    try:
        conn = oracledb.connect(
            user=user, password=password, dsn=dsn, config_dir=str(wallet_path)
        )

        cursor = conn.cursor()
        cursor.execute("SELECT 'Success!' FROM DUAL")
        result = cursor.fetchone()[0]
        print(f"✅ Connection successful: {result}")

        cursor.close()
        conn.close()
        return True

    except oracledb.Error as e:
        error_code = str(e).split(":")[0] if ":" in str(e) else "Unknown"
        print(f"❌ Failed with error {error_code}: {e}")

        # Specific error analysis
        if "12506" in str(e):
            print("   → Database is PAUSED - resume in OCI Console")
        elif "28000" in str(e):
            print("   → Account locked or password expired")
        elif "12154" in str(e):
            print("   → TNS name resolution issue")
        elif "12170" in str(e):
            print("   → Connection timeout")

        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
