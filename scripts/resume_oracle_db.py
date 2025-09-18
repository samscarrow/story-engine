#!/usr/bin/env python3
"""
Resume Oracle Autonomous Database using OCI CLI
This script helps automate database resumption when it's paused.
"""

import os
import subprocess
import sys
import json
import time

def check_oci_cli():
    """Check if OCI CLI is installed and configured."""
    try:
        result = subprocess.run(['oci', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ OCI CLI is installed")
            print(f"   Version: {result.stdout.strip()}")
            return True
        else:
            print("❌ OCI CLI is not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ OCI CLI is not installed")
        print("   Install from: https://docs.oracle.com/en-us/iaas/tools/oci-cli/")
        return False

def find_mainbase_database():
    """Find the MAINBASE database OCID."""
    try:
        # Try to list databases to find MAINBASE
        cmd = [
            'oci', 'db', 'autonomous-database', 'list',
            '--all',
            '--lifecycle-state', 'AVAILABLE,STOPPED',
            '--query', 'data[?contains("display-name", `MAINBASE`) || contains("db-name", `mainbase`)].{name:"display-name", id:id, state:"lifecycle-state"}',
            '--output', 'json'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            databases = json.loads(result.stdout)
            if databases:
                print(f"Found {len(databases)} matching database(s):")
                for db in databases:
                    print(f"  - {db['name']} ({db['state']}): {db['id']}")
                return databases[0]['id'] if databases else None
            else:
                print("No databases found matching 'MAINBASE' or 'mainbase'")
                return None
        else:
            print(f"Error listing databases: {result.stderr}")
            return None

    except Exception as e:
        print(f"Error finding database: {e}")
        return None

def resume_database(db_ocid):
    """Resume the specified database."""
    try:
        print(f"Resuming database: {db_ocid}")

        cmd = [
            'oci', 'db', 'autonomous-database', 'start',
            '--autonomous-database-id', db_ocid,
            '--wait-for-state', 'AVAILABLE',
            '--max-wait-seconds', '300'  # 5 minutes timeout
        ]

        print("Starting database resume operation...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=400)

        if result.returncode == 0:
            print("✅ Database resume operation completed successfully!")
            return True
        else:
            print(f"❌ Error resuming database: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Database resume operation timed out (but may still be in progress)")
        print("   Check Oracle Cloud Console for status")
        return False
    except Exception as e:
        print(f"❌ Error resuming database: {e}")
        return False

def main():
    """Main script to resume Oracle database."""
    print("🔄 Oracle Database Resume Script")
    print("=" * 40)

    # Check OCI CLI
    if not check_oci_cli():
        print("\n❌ Cannot proceed without OCI CLI")
        print("Please install and configure OCI CLI first:")
        print("1. Install: https://docs.oracle.com/en-us/iaas/tools/oci-cli/")
        print("2. Configure: oci setup config")
        return 1

    # Find database
    print("\n🔍 Searching for MAINBASE database...")
    db_ocid = find_mainbase_database()

    if not db_ocid:
        print("\n❌ Could not find MAINBASE database")
        print("Options:")
        print("1. Check database name in Oracle Cloud Console")
        print("2. Manually resume database in console.oracle.com")
        print("3. Update this script with correct database name")
        return 1

    # Resume database
    print(f"\n🚀 Found database, attempting to resume...")
    success = resume_database(db_ocid)

    if success:
        print("\n🎉 Database resumption completed!")
        print("\nNext steps:")
        print("1. Wait 1-2 minutes for full availability")
        print("2. Test connection: python scripts/oracle_healthcheck.py")
        print("3. Run story engine operations")
        return 0
    else:
        print("\n❌ Database resumption failed")
        print("\nManual steps:")
        print("1. Go to console.oracle.com")
        print("2. Navigate to Autonomous Database > MAINBASE")
        print("3. Click 'Start' button")
        print("4. Wait for status to become 'Available'")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())