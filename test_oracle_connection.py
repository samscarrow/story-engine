#!/usr/bin/env python3
"""Simple Oracle connection test"""

import os
import sys

sys.path.insert(0, "/home/sam/claude-workspace/story-engine")

from story_engine.core.storage.database import OracleConnection


def test_oracle():
    print("Testing Oracle connection...")

    # Get config from environment
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    dsn = os.getenv("DB_DSN")
    wallet_location = os.getenv("DB_WALLET_LOCATION")

    print(f"User: {user}")
    print(f"DSN: {dsn}")
    print(f"Wallet: {wallet_location}")
    print(f"Password set: {'Yes' if password else 'No'}")

    try:
        conn = OracleConnection(
            user=user, password=password, dsn=dsn, wallet_location=wallet_location
        )

        print("Connecting...")
        conn.connect()
        print("✅ Connection successful!")

        print("Testing table creation...")
        conn._create_table()
        print("✅ Table creation successful!")

        conn.disconnect()
        print("✅ Disconnected successfully!")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_oracle()
