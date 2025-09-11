#!/usr/bin/env python3
"""
Comprehensive Oracle connection diagnostics
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    import oracledb
except ImportError:
    print("❌ oracledb not available - run: pip install oracledb")
    sys.exit(1)


def check_wallet_files():
    """Check if all required wallet files exist and have proper permissions."""
    print("🔍 Checking wallet files...")

    wallet_path = Path("./oracle_wallet").resolve()
    print(f"📂 Wallet directory: {wallet_path}")

    required_files = [
        "cwallet.sso",
        "ewallet.p12",
        "ewallet.pem",
        "tnsnames.ora",
        "sqlnet.ora",
        "ojdbc.properties",
    ]

    all_exist = True
    for file_name in required_files:
        file_path = wallet_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {file_name}: {size} bytes")
        else:
            print(f"  ❌ {file_name}: MISSING")
            all_exist = False

    return all_exist, wallet_path


def check_environment_vars():
    """Check environment variables and configuration."""
    print("\n🔧 Checking environment configuration...")

    load_dotenv(".env.oracle")

    vars_to_check = [
        "DB_USER",
        "DB_PASSWORD",
        "DB_DSN",
        "DB_WALLET_LOCATION",
        "DB_WALLET_PASSWORD",
    ]

    config = {}
    for var in vars_to_check:
        value = os.getenv(var)
        if value:
            if "PASSWORD" in var:
                print(f"  ✅ {var}: {'*' * len(value)}")
            else:
                print(f"  ✅ {var}: {value}")
            config[var] = value
        else:
            print(f"  ❌ {var}: NOT SET")

    return config


def test_tns_admin():
    """Test TNS_ADMIN configuration."""
    print("\n🗂️  Testing TNS_ADMIN configuration...")

    wallet_path = Path("./oracle_wallet").resolve()
    os.environ["TNS_ADMIN"] = str(wallet_path)

    print(f"  📍 Set TNS_ADMIN to: {os.environ['TNS_ADMIN']}")

    # Check if tnsnames.ora is accessible
    tns_file = wallet_path / "tnsnames.ora"
    if tns_file.exists():
        print(f"  ✅ tnsnames.ora accessible at: {tns_file}")
        with open(tns_file, "r") as f:
            content = f.read()
            if "mainbase_high" in content.lower():
                print("  ✅ mainbase_high service found in tnsnames.ora")
            else:
                print("  ❌ mainbase_high service NOT found in tnsnames.ora")
    else:
        print(f"  ❌ tnsnames.ora not found at: {tns_file}")


def test_connection_methods():
    """Test different connection approaches."""
    print("\n🔌 Testing connection methods...")

    config = check_environment_vars()
    wallet_path = Path("./oracle_wallet").resolve()

    # Method 1: Basic connection with config_dir
    print("\n1️⃣ Testing with config_dir parameter...")
    try:
        conn = oracledb.connect(
            user=config["DB_USER"],
            password=config["DB_PASSWORD"],
            dsn=config["DB_DSN"],
            config_dir=str(wallet_path),
        )
        print("  ✅ Connection successful with config_dir!")

        cursor = conn.cursor()
        cursor.execute(
            "SELECT 'Connected to ' || USER || ' on ' || SYS_CONTEXT('USERENV', 'DB_NAME') FROM DUAL"
        )
        result = cursor.fetchone()[0]
        print(f"  🎉 Result: {result}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")

    # Method 2: Connection with TNS_ADMIN environment variable
    print("\n2️⃣ Testing with TNS_ADMIN environment variable...")
    try:
        os.environ["TNS_ADMIN"] = str(wallet_path)

        conn = oracledb.connect(
            user=config["DB_USER"], password=config["DB_PASSWORD"], dsn=config["DB_DSN"]
        )
        print("  ✅ Connection successful with TNS_ADMIN!")

        cursor = conn.cursor()
        cursor.execute("SELECT 'Connected!' FROM DUAL")
        result = cursor.fetchone()[0]
        print(f"  🎉 Result: {result}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"  ❌ Failed: {e}")

        # Analyze specific error types
        error_str = str(e)
        if "12506" in error_str:
            print("  💡 Error 12506 = Database is likely PAUSED")
            print("     → Resume database in OCI Console")
        elif "12154" in error_str:
            print("  💡 Error 12154 = TNS could not resolve service name")
            print("     → Check tnsnames.ora and DSN configuration")
        elif "28040" in error_str:
            print("  💡 Error 28040 = No matching authentication protocol")
            print("     → Check wallet files and SSL configuration")
        elif "12170" in error_str:
            print("  💡 Error 12170 = TNS connect timeout")
            print("     → Check network connectivity and firewall")

    return False


def test_oci_cli():
    """Test if OCI CLI is available and configured."""
    print("\n🛠️  Testing OCI CLI availability...")

    try:
        result = os.system("oci --version > nul 2>&1")  # Windows null redirect
        if result == 0:
            print("  ✅ OCI CLI is installed")

            # Test OCI configuration
            config_result = os.system("oci iam region list > nul 2>&1")
            if config_result == 0:
                print("  ✅ OCI CLI is configured")
                return True
            else:
                print("  ❌ OCI CLI not configured")
                print("     → Run: oci setup config")
        else:
            print("  ❌ OCI CLI not installed")
            print(
                "     → Install from: https://docs.oracle.com/en-us/iaas/tools/oci-cli/"
            )
    except Exception as e:
        print(f"  ❌ Error checking OCI CLI: {e}")

    return False


def check_database_status_with_oci():
    """Use OCI CLI to check database status."""
    print("\n🔍 Checking database status with OCI CLI...")

    try:
        # Get database OCID from connection string
        # GFCA71B2AACCE62 appears to be the database identifier
        db_id = "ocid1.compartment.oc1..aaaaaaaasmfbe4prr46zz6dxbv7mu2abnoiomqjunbn5pmsfsiq2zhxavtwa"
        print(f"  🔍 Looking for database with ID containing: {db_id}")

        # This would need the actual OCID, but let's try a general approach
        result = os.system(
            "oci db autonomous-database list --compartment-id ocid1.compartment.oc1..aaaaaaaasmfbe4prr46zz6dxbv7mu2abnoiomqjunbn5pmsfsiq2zhxavtwa 2>&1"
        )
        if result != 0:
            print("  ❌ Could not query database status")
            print("     → Need proper OCI configuration with compartment ID")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def main():
    """Run comprehensive diagnostics."""
    print("🔬 Oracle Connection Diagnostics")
    print("=" * 50)

    # 1. Check wallet files
    wallet_ok, wallet_path = check_wallet_files()

    # 2. Check environment
    config = check_environment_vars()

    # 3. Test TNS_ADMIN
    test_tns_admin()

    # 4. Test connections
    connection_ok = test_connection_methods()

    # 5. Check OCI CLI
    oci_ok = test_oci_cli()

    # 6. Try database status check
    if oci_ok:
        check_database_status_with_oci()

    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)

    if wallet_ok:
        print("✅ Wallet files: Complete")
    else:
        print("❌ Wallet files: Missing files")

    if config:
        print("✅ Environment: Configured")
    else:
        print("❌ Environment: Missing variables")

    if connection_ok:
        print("✅ Database connection: Working")
        print("🎉 Your Oracle setup is functional!")
    else:
        print("❌ Database connection: Failed")
        print("💡 Most likely cause: Database is paused")
        print("   → Go to OCI Console and resume MAINBASE database")

    print("\n🚀 Next steps:")
    if not connection_ok:
        print("1. Resume database in Oracle Cloud Console")
        print("2. Wait 2-3 minutes for database to become available")
        print("3. Re-run this diagnostic script")
    else:
        print("1. Run: python setup_story_schema.py")
        print("2. Test story engine database operations")
        print("3. Set up APEX applications")


if __name__ == "__main__":
    main()
