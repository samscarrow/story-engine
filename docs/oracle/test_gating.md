# Oracle Test Gating and Fallback

This project includes pragmatic defaults to keep local and CI runs fast and reliable when Oracle is not available.

- To run Oracle-dependent tests, explicitly opt in:
  - Set `ORACLE_TESTS=1` in your environment.
  - Ensure `DB_USER`, `DB_PASSWORD`, and `DB_DSN` (and `DB_WALLET_LOCATION` if using a wallet) are valid.
  - The helper `oracle_env_is_healthy(require_opt_in=True)` performs a fast, real connection probe before enabling tests.

- By default, Oracle tests are skipped. This avoids hangs or failures when the listener is down or the DSN is invalid.

- Acceptance runs and utilities use a health-aware fallback:
  - If `DB_TYPE=oracle` but the health probe indicates Oracle is unreachable, code will fall back to SQLite automatically.
  - To require Oracle (e.g., in a dedicated environment), set `DB_REQUIRE_ORACLE=1` to disable fallback.

Relevant code:
- Probe: `story_engine.core.storage.database.oracle_env_is_healthy()`
- Settings: `story_engine.core.core.common.settings.get_db_settings()`
