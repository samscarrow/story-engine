import sqlite3
import json
import time
import os
import sys
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Structured logging helpers
from llm_observability import get_logger, log_exception, observe_metric

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


def _truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def oracle_env_is_healthy(
    *,
    require_opt_in: bool = False,
    timeout_seconds: float = 1.5,
) -> bool:
    """Best-effort, fast Oracle reachability probe.

    - Returns False unless environment looks minimally configured and a quick
      connection attempt succeeds within ``timeout_seconds``.
    - If ``require_opt_in`` is True, only runs when ``ORACLE_TESTS`` is truthy
      to avoid unexpected connection attempts during generic test runs.

    This probe intentionally uses a subprocess to enforce a hard timeout so a
    misconfigured DSN cannot stall the process.
    """
    if require_opt_in and not _truthy(os.getenv("ORACLE_TESTS")):
        return False

    # Gather env values with Oracle aliases supported in tests
    user = os.getenv("DB_USER") or os.getenv("ORACLE_USER")
    password = os.getenv("DB_PASSWORD") or os.getenv("ORACLE_PASSWORD")
    dsn = (
        os.getenv("DB_DSN")
        or os.getenv("ORACLE_DSN")
        or os.getenv("DB_CONNECT_STRING")
    )
    wallet = os.getenv("DB_WALLET_LOCATION") or os.getenv("ORACLE_WALLET_DIR") or os.getenv("TNS_ADMIN")

    # Minimal config requirements: DSN present, user/password typically required
    if not dsn or not user or not password:
        return False

    # Build a short Python snippet to attempt a real connection and exit fast
    code = (
        "import os, sys\n"
        "import oracledb\n"
        "user=os.getenv('DB_USER') or os.getenv('ORACLE_USER')\n"
        "pwd=os.getenv('DB_PASSWORD') or os.getenv('ORACLE_PASSWORD')\n"
        "dsn=os.getenv('DB_DSN') or os.getenv('ORACLE_DSN') or os.getenv('DB_CONNECT_STRING')\n"
        "# Avoid expensive defaults where possible\n"
        "try:\n"
        "    oracledb.defaults.fetch_lobs=False\n"
        "except Exception:\n"
        "    pass\n"
        "conn=None\n"
        "try:\n"
        "    conn=oracledb.connect(user=user, password=pwd, dsn=dsn)\n"
        "    # lightweight validation: round-trip a no-op\n"
        "    cur=conn.cursor(); cur.execute('select 1 from dual'); cur.fetchone(); cur.close()\n"
        "    print('OK')\n"
        "    sys.exit(0)\n"
        "except Exception as e:\n"
        "    # print minimal error for diagnostics, but exit non-zero\n"
        "    sys.stderr.write(str(e))\n"
        "    sys.exit(2)\n"
        "finally:\n"
        "    try:\n"
        "        conn and conn.close()\n"
        "    except Exception:\n"
        "        pass\n"
    )

    env = os.environ.copy()
    # Ensure TNS_ADMIN is set if wallet dir provided
    if wallet:
        env.setdefault("TNS_ADMIN", wallet)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        return proc.returncode == 0 and (proc.stdout or "").strip() == "OK"
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


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
                # Some test stubs or drivers may not require/implement commit
                if hasattr(self.conn, "commit"):
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
        *,
        use_pool: bool = False,
        pool_min: int = 1,
        pool_max: int = 4,
        pool_increment: int = 1,
        pool_timeout: int = 60,
        wait_timeout: Optional[int] = None,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        ping_on_connect: bool = True,
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
        # Connection stability settings
        self.use_pool = use_pool
        self._pool = None
        self.pool_min = pool_min
        self.pool_max = pool_max
        self.pool_increment = pool_increment
        self.pool_timeout = pool_timeout
        self.wait_timeout = wait_timeout
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.ping_on_connect = ping_on_connect

    def connect(self):
        """Connect to the Oracle database with optional pooling and retries."""
        from pathlib import Path
        import os
        log = get_logger(
            __name__,
            component="db.oracle",
            workflow="connect",
            dsn=self.dsn,
            use_pool=self.use_pool,
            pool_min=self.pool_min,
            pool_max=self.pool_max,
        )

        # Return CLOB/NCLOB as Python str instead of LOB objects
        try:
            oracledb.defaults.fetch_lobs = False  # type: ignore[attr-defined]
        except Exception:
            pass

        # Standard wallet env wiring for ADB
        wallet_path: Optional[Path] = None
        if self.wallet_location:
            wallet_path = Path(self.wallet_location).resolve()
            os.environ["TNS_ADMIN"] = str(wallet_path)
            # Note: using env avoids having to pass config_dir explicitly
            log.debug("set TNS_ADMIN for wallet", extra={"TNS_ADMIN": str(wallet_path)})

        # Create or reuse pool if requested
        if self.use_pool and self._pool is None:
            try:
                pool_kwargs = dict(
                    user=self.user,
                    password=self.password,
                    dsn=self.dsn,
                    min=self.pool_min,
                    max=self.pool_max,
                    increment=self.pool_increment,
                    timeout=self.pool_timeout,
                    homogeneous=True,
                )
                # Some oracledb versions support wait_timeout
                if self.wait_timeout is not None:
                    pool_kwargs["wait_timeout"] = self.wait_timeout
                t0 = time.time()
                self._pool = oracledb.create_pool(**pool_kwargs)  # type: ignore[arg-type]
                log.info(
                    "oracle pool created",
                    extra={"elapsed_ms": int((time.time() - t0) * 1000)},
                )
            except Exception as e:
                log_exception(
                    log,
                    code="DB_CONN_FAIL",
                    component="db.oracle",
                    exc=e,
                    stage="create_pool",
                )
                # Fall back to direct connect attempts
                self._pool = None
                self.use_pool = False

        # Acquire a connection (from pool or direct) with retries
        last_err: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            try:
                t0 = time.time()
                if self._pool is not None:
                    self.conn = self._pool.acquire()
                else:
                    # In thin mode, config_dir can be provided; env TNS_ADMIN typically suffices
                    self.conn = oracledb.connect(
                        user=self.user,
                        password=self.password,
                        dsn=self.dsn,
                        config_dir=str(wallet_path) if wallet_path else None,
                    )
                if self.ping_on_connect and self.conn:
                    cur = self.conn.cursor()
                    try:
                        cur.execute("SELECT 1 FROM DUAL")
                    finally:
                        cur.close()
                # Ensure base table exists
                self._create_table()
                elapsed_ms = int((time.time() - t0) * 1000)
                log.info(
                    "oracle connect ok",
                    extra={"attempt": attempt + 1, "elapsed_ms": elapsed_ms},
                )
                try:
                    # Emit a lightweight timing metric
                    observe_metric("db.oracle.connect_ms", elapsed_ms, dsn=self.dsn, pooled=bool(self._pool))
                except Exception:
                    pass
                return
            except Exception as e:
                last_err = e
                # Common transient/connectivity ORA codes to retry
                msg = str(e)
                retryable = any(
                    code in msg
                    for code in (
                        "ORA-12154",  # TNS: could not resolve the connect identifier
                        "ORA-12514",  # TNS: listener does not currently know of service
                        "ORA-12541",  # TNS: no listener
                        "ORA-12537",  # TNS: connection closed
                        "ORA-12547",  # TNS: lost contact
                        "ORA-12560",  # TNS: protocol adapter error
                        "ORA-12506",  # TNS: listener refused the connection (ADB paused)
                    )
                )
                if attempt < self.retry_attempts - 1 and retryable:
                    backoff = self.retry_backoff_seconds * (2**attempt)
                    # add small jitter
                    try:
                        import random
                        backoff = backoff * (0.8 + 0.4 * random.random())
                    except Exception:
                        pass
                    log_exception(
                        log,
                        code="DB_CONN_FAIL",
                        component="db.oracle",
                        exc=e,
                        attempt=attempt + 1,
                        retry_in_s=round(backoff, 2),
                    )
                    time.sleep(backoff)
                    continue
                log_exception(
                    log,
                    code="DB_CONN_FAIL",
                    component="db.oracle",
                    exc=e,
                    attempt=attempt + 1,
                    terminal=True,
                )
                raise

    def disconnect(self):
        """Disconnect from the Oracle database."""
        if self.conn:
            self.conn.close()
            self.conn = None
        # Do not close the pool here; pool is process-wide and reused.

    def close_pool(self):
        """Close the session pool if one was created."""
        try:
            if self._pool is not None:
                self._pool.close()
                self._pool = None
        except Exception:
            pass

    def healthy(self) -> bool:
        """Lightweight health check for the connection/pool."""
        try:
            if self._pool is not None:
                with self._pool.acquire() as conn:  # type: ignore[attr-defined]
                    cur = conn.cursor()
                    try:
                        cur.execute("SELECT 1 FROM DUAL")
                        _ = cur.fetchone()
                    finally:
                        cur.close()
                return True
            if self.conn is None:
                return False
            cur = self.conn.cursor()
            try:
                cur.execute("SELECT 1 FROM DUAL")
                _ = cur.fetchone()
                return True
            finally:
                cur.close()
        except Exception:
            return False

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
                if hasattr(self.conn, "commit"):
                    self.conn.commit()
                cursor.close()
            except Exception as e:
                get_logger(__name__, component="db.oracle", workflow="create_table").error(
                    "error creating workflow_outputs",
                    extra={"error": str(e)},
                )
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
                if hasattr(self.conn, "commit"):
                    self.conn.commit()
                cursor.close()
            except Exception as e:
                get_logger(__name__, component="db.oracle", workflow="store_output").error(
                    "error storing output",
                    extra={"error": str(e)},
                )
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
                get_logger(__name__, component="db.oracle", workflow="get_outputs").error(
                    "error getting outputs",
                    extra={"error": str(e)},
                )
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
