#!/usr/bin/env python3
"""Simple Oracle connection test."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    import oracledb
except ImportError:
    print("oracledb not available")
    sys.exit(1)

def test_simple_connection():
    """Test simple Oracle connection without custom class."""
    
    # Load environment variables
    load_dotenv('.env.oracle')
    
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    dsn = os.getenv('DB_DSN')
    wallet_location = os.getenv('DB_WALLET_LOCATION')
    
    print("Testing simple Oracle connection...")
    print(f"User: {user}")
    print(f"DSN: {dsn}")
    
    # Set wallet location
    wallet_path = str(Path(wallet_location).resolve())
    os.environ['TNS_ADMIN'] = wallet_path
    print(f"TNS_ADMIN: {wallet_path}")
    
    # Check files exist
    tns_file = Path(wallet_path) / "tnsnames.ora"
    sqlnet_file = Path(wallet_path) / "sqlnet.ora"
    cwallet_file = Path(wallet_path) / "cwallet.sso"
    
    print(f"tnsnames.ora exists: {tns_file.exists()}")
    print(f"sqlnet.ora exists: {sqlnet_file.exists()}")  
    print(f"cwallet.sso exists: {cwallet_file.exists()}")
    
    try:
        print("\n1. Attempting connection...")
        
        # Try with config_dir parameter
        conn = oracledb.connect(
            user=user,
            password=password,
            dsn=dsn,
            config_dir=wallet_path
        )
        
        print("✓ Connected successfully!")
        
        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT 'Hello Oracle!' FROM DUAL")
        result = cursor.fetchone()
        print(f"✓ Test query result: {result[0]}")
        
        # Check user and schema
        cursor.execute("SELECT USER, SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL")
        user_info = cursor.fetchone()
        print(f"✓ Connected as: {user_info[0]}, Schema: {user_info[1]}")
        
        cursor.close()
        conn.close()
        
        print("\n=== CONNECTION SUCCESSFUL ===")
        return True
        
    except oracledb.Error as e:
        print(f"✗ Connection failed: {e}")
        
        # Check if database might be paused
        error_str = str(e)
        if "12506" in error_str or "refused" in error_str.lower():
            print("! This error often indicates the database is paused or unavailable")
            print("  Try resuming the database in Oracle Cloud Console")
        elif "12154" in error_str:
            print("! TNS could not resolve the connect identifier")
            print("  Check the tnsnames.ora file and DSN configuration")
        elif "28040" in error_str:
            print("! No matching authentication protocol")
            print("  Check wallet files and configuration")
            
        print("\n=== CONNECTION FAILED ===")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print("\n=== CONNECTION FAILED ===")
        return False

if __name__ == "__main__":
    success = test_simple_connection()
    sys.exit(0 if success else 1)