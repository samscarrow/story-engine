import sqlite3
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

# Note: You will need to install psycopg2-binary to use the PostgreSQL connection
# pip install psycopg2-binary
try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    import oracledb
except ImportError:
    oracledb = None


class DatabaseConnection(ABC):
    """Abstract base class for database connections."""

    @abstractmethod
    def connect(self):
        """Connect to the database."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the database."""
        pass

    @abstractmethod
    def store_output(self, workflow_name: str, output_data: Dict[str, Any]):
        """Store workflow output."""
        pass

    @abstractmethod
    def get_outputs(self, workflow_name: str) -> List[Dict[str, Any]]:
        """Get all outputs for a given workflow."""
        pass


class SQLiteConnection(DatabaseConnection):
    """SQLite database connection."""

    def __init__(self, db_name: str = "workflow_outputs.db"):
        self.db_name = db_name
        self.conn = None

    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_name)
            self._create_table()
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        """Disconnect from the SQLite database."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _create_table(self):
        """Create the outputs table if it doesn't exist."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_outputs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workflow_name TEXT NOT NULL,
                        output_data TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Error creating table: {e}")
                raise

    def store_output(self, workflow_name: str, output_data: Dict[str, Any]):
        """Store workflow output in the database."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO workflow_outputs (workflow_name, output_data)
                    VALUES (?, ?)
                """,
                    (workflow_name, json.dumps(output_data)),
                )
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Error storing output: {e}")
                raise

    def get_outputs(self, workflow_name: str) -> List[Dict[str, Any]]:
        """Get all outputs for a given workflow."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    SELECT output_data FROM workflow_outputs WHERE workflow_name = ?
                """,
                    (workflow_name,),
                )
                rows = cursor.fetchall()
                return [json.loads(row[0]) for row in rows]
            except sqlite3.Error as e:
                print(f"Error getting outputs: {e}")
                raise
        return []


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection."""

    def __init__(
        self,
        db_name: str,
        user: str,
        password: str | None,
        host: str,
        port: int = 5432,
        sslmode: str | None = None,
        sslrootcert: str | None = None,
        sslcert: str | None = None,
        sslkey: str | None = None,
    ):
        if not psycopg2:
            raise ImportError(
                "psycopg2-binary is not installed. Please run 'pip install psycopg2-binary'"
            )
        self.db_name = db_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        # SSL params (optional)
        self.sslmode = sslmode
        self.sslrootcert = sslrootcert
        self.sslcert = sslcert
        self.sslkey = sslkey

    def connect(self):
        """Connect to the PostgreSQL database."""
        try:
            conn_kwargs = dict(
                dbname=self.db_name,
                user=self.user,
                host=self.host,
                port=self.port,
            )
            # Password may be None if using IAM/other auth, psycopg2 allows missing
            if self.password is not None:
                conn_kwargs["password"] = self.password
            # Attach SSL parameters if provided
            if self.sslmode:
                conn_kwargs["sslmode"] = self.sslmode
            if self.sslrootcert:
                conn_kwargs["sslrootcert"] = self.sslrootcert
            if self.sslcert:
                conn_kwargs["sslcert"] = self.sslcert
            if self.sslkey:
                conn_kwargs["sslkey"] = self.sslkey

            self.conn = psycopg2.connect(**conn_kwargs)
            self._create_table()
        except psycopg2.Error as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            raise

    def disconnect(self):
        """Disconnect from the PostgreSQL database."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _create_table(self):
        """Create the outputs table if it doesn't exist."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_outputs (
                        id SERIAL PRIMARY KEY,
                        workflow_name VARCHAR(255) NOT NULL,
                        output_data JSONB NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                self.conn.commit()
                cursor.close()
            except psycopg2.Error as e:
                print(f"Error creating table: {e}")
                raise

    def store_output(self, workflow_name: str, output_data: Dict[str, Any]):
        """Store workflow output in the database."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO workflow_outputs (workflow_name, output_data)
                    VALUES (%s, %s)
                """,
                    (workflow_name, json.dumps(output_data)),
                )
                self.conn.commit()
                cursor.close()
            except psycopg2.Error as e:
                print(f"Error storing output: {e}")
                raise

    def get_outputs(self, workflow_name: str) -> List[Dict[str, Any]]:
        """Get all outputs for a given workflow."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    SELECT output_data FROM workflow_outputs WHERE workflow_name = %s
                """,
                    (workflow_name,),
                )
                rows = cursor.fetchall()
                cursor.close()
                return [row[0] for row in rows]
            except psycopg2.Error as e:
                print(f"Error getting outputs: {e}")
                raise
        return []


class OracleConnection(DatabaseConnection):
    """Oracle database connection with wallet authentication."""

    def __init__(
        self,
        user: str,
        password: str,
        dsn: str,
        wallet_location: str | None = None,
        wallet_password: str | None = None,
    ):
        if not oracledb:
            raise ImportError(
                "oracledb is not installed. Please run 'pip install oracledb'"
            )
        self.user = user
        self.password = password
        self.dsn = dsn
        self.wallet_location = wallet_location
        self.wallet_password = wallet_password
        self.conn = None

    def connect(self):
        """Connect to the Oracle database using wallet authentication."""
        try:
            # Return CLOB/NCLOB as Python str instead of LOB objects
            # to simplify JSON handling on fetch.
            try:
                oracledb.defaults.fetch_lobs = False  # type: ignore[attr-defined]
            except Exception:
                pass
            # Set environment variables for wallet location (Oracle standard)
            if self.wallet_location:
                import os
                from pathlib import Path

                # Ensure TNS_ADMIN is absolute path
                wallet_path = Path(self.wallet_location).resolve()
                os.environ["TNS_ADMIN"] = str(wallet_path)
                print(f"Set TNS_ADMIN to: {wallet_path}")

            # For Oracle Autonomous Database, use connection with config_dir
            # The wallet files handle SSL/TLS automatically
            wallet_path = (
                Path(self.wallet_location).resolve() if self.wallet_location else None
            )

            self.conn = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                config_dir=str(wallet_path) if wallet_path else None,
            )
            self._create_table()
        except oracledb.Error as e:
            print(f"Error connecting to Oracle database: {e}")
            raise

    def disconnect(self):
        """Disconnect from the Oracle database."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _create_table(self):
        """Create the outputs table if it doesn't exist."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    BEGIN
                        EXECUTE IMMEDIATE 'CREATE TABLE workflow_outputs (
                            id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                            workflow_name VARCHAR2(255) NOT NULL,
                            output_data CLOB NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )';
                    EXCEPTION
                        WHEN OTHERS THEN
                            IF SQLCODE != -955 THEN  -- Table already exists
                                RAISE;
                            END IF;
                    END;
                """
                )
                self.conn.commit()
                cursor.close()
            except oracledb.Error as e:
                print(f"Error creating table: {e}")
                raise

    def store_output(self, workflow_name: str, output_data: Dict[str, Any]):
        """Store workflow output in the database."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO workflow_outputs (workflow_name, output_data)
                    VALUES (:workflow_name, :output_data)
                """,
                    {
                        "workflow_name": workflow_name,
                        "output_data": json.dumps(output_data),
                    },
                )
                self.conn.commit()
                cursor.close()
            except oracledb.Error as e:
                print(f"Error storing output: {e}")
                raise

    def get_outputs(self, workflow_name: str) -> List[Dict[str, Any]]:
        """Get all outputs for a given workflow."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    SELECT output_data FROM workflow_outputs 
                    WHERE workflow_name = :workflow_name
                    ORDER BY timestamp DESC
                """,
                    {"workflow_name": workflow_name},
                )
                rows = cursor.fetchall()
                cursor.close()
                results: List[Dict[str, Any]] = []
                for (val,) in rows:
                    # Handle value as str or LOB
                    try:
                        text = val.read() if hasattr(val, "read") else val
                    except Exception:
                        text = val
                    results.append(json.loads(text))
                return results
            except oracledb.Error as e:
                print(f"Error getting outputs: {e}")
                raise
        return []


def get_database_connection(db_type: str, **kwargs) -> DatabaseConnection:
    """Factory function to get a database connection."""
    if db_type == "sqlite":
        return SQLiteConnection(**kwargs)
    elif db_type == "postgresql":
        return PostgreSQLConnection(**kwargs)
    elif db_type == "oracle":
        return OracleConnection(**kwargs)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


# --- Messaging/Outbox utilities (optional) ---
def ensure_message_tables(db: DatabaseConnection) -> None:
    """Create outbox and processed_messages tables if supported.

    Supports SQLite and PostgreSQL; for Oracle this is a no-op placeholder.
    """
    conn = getattr(db, "conn", None)
    if conn is None:
        return
    try:
        if isinstance(db, SQLiteConnection):
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS message_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    idempotency_key TEXT PRIMARY KEY,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        elif psycopg2 and isinstance(db, PostgreSQLConnection):
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS message_outbox (
                    id SERIAL PRIMARY KEY,
                    topic VARCHAR(255) NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    idempotency_key VARCHAR(255) PRIMARY KEY,
                    processed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
            cur.close()
    except Exception:
        # Non-fatal in environments without DB permissions
        pass


def insert_outbox(db: DatabaseConnection, topic: str, payload: Dict[str, Any]) -> None:
    conn = getattr(db, "conn", None)
    if conn is None:
        return
    try:
        if isinstance(db, SQLiteConnection):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO message_outbox (topic, payload) VALUES (?, ?)",
                (topic, json.dumps(payload)),
            )
            conn.commit()
        elif psycopg2 and isinstance(db, PostgreSQLConnection):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO message_outbox (topic, payload) VALUES (%s, %s)",
                (topic, json.dumps(payload)),
            )
            conn.commit()
            cur.close()
    except Exception:
        pass


def mark_processed(db: DatabaseConnection, idempotency_key: str) -> None:
    conn = getattr(db, "conn", None)
    if conn is None:
        return
    try:
        if isinstance(db, SQLiteConnection):
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO processed_messages (idempotency_key) VALUES (?)",
                (idempotency_key,),
            )
            conn.commit()
        elif psycopg2 and isinstance(db, PostgreSQLConnection):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO processed_messages (idempotency_key) VALUES (%s) ON CONFLICT DO NOTHING",
                (idempotency_key,),
            )
            conn.commit()
            cur.close()
    except Exception:
        pass
