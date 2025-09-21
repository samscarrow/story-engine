#!/usr/bin/env python3
"""
Test Oracle configuration without connecting to database.
This validates configuration files and settings.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))

from story_engine.core.core.common.settings import get_db_settings

def test_oracle_config():
    """Test Oracle configuration setup."""
    print("🧪 Oracle Configuration Test")
    print("=" * 40)

    # Load Oracle environment
    load_dotenv(".env.oracle")

    # Test settings loading
    try:
        settings = get_db_settings(env_file=".env.oracle")
        print("✅ Database settings loaded successfully")
        print(f"   DB Type: {settings['db_type']}")
        print(f"   User: {settings['user']}")
        print(f"   DSN: {settings['dsn']}")
        print(f"   Wallet: {settings['wallet_location']}")
        print(f"   Use Pool: {settings['use_pool']}")
        print(f"   Pool Min/Max: {settings['pool_min']}/{settings['pool_max']}")
        print(f"   Retry Attempts: {settings['retry_attempts']}")
    except Exception as e:
        print(f"❌ Error loading database settings: {e}")
        return False

    # Test wallet files
    wallet_path = Path(settings['wallet_location']).resolve()
    print("\n📁 Wallet Configuration:")
    print(f"   Path: {wallet_path}")

    required_files = [
        "cwallet.sso",
        "tnsnames.ora",
        "sqlnet.ora",
        "ojdbc.properties"
    ]

    all_files_exist = True
    for file_name in required_files:
        file_path = wallet_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   ✅ {file_name}: {size} bytes")
        else:
            print(f"   ❌ {file_name}: MISSING")
            all_files_exist = False

    # Test TNS configuration
    tns_file = wallet_path / "tnsnames.ora"
    if tns_file.exists():
        with open(tns_file, 'r') as f:
            content = f.read()
            if settings['dsn'] in content:
                print(f"   ✅ DSN '{settings['dsn']}' found in tnsnames.ora")
            else:
                print(f"   ❌ DSN '{settings['dsn']}' NOT found in tnsnames.ora")
                all_files_exist = False

    # Test OracleConnection class instantiation (without connecting)
    try:
        from story_engine.core.core.storage.database import OracleConnection

        conn = OracleConnection(
            user=settings['user'],
            password=settings['password'],
            dsn=settings['dsn'],
            wallet_location=settings['wallet_location'],
            use_pool=settings['use_pool'],
            retry_attempts=1  # Don't retry for this test
        )
        print("\n✅ OracleConnection object created successfully")
        print("   Configuration appears valid")

    except Exception as e:
        print(f"\n❌ Error creating OracleConnection: {e}")
        return False

    print("\n📋 Summary:")
    if all_files_exist:
        print("✅ All configuration files present")
        print("✅ Oracle connection should work once database is resumed")
        print("\nTo resume database:")
        print("  python scripts/resume_oracle_db.py")
        print("  # OR manually in Oracle Cloud Console")
        print("\nTo test connection after resume:")
        print("  python scripts/oracle_healthcheck.py")
        return True
    else:
        print("❌ Configuration issues found")
        print("   Check wallet files and DSN configuration")
        return False

if __name__ == "__main__":
    success = test_oracle_config()
    sys.exit(0 if success else 1)
