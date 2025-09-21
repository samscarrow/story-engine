# Oracle ADB via TNS Alias (mainbase_high)

Use the wallet’s `tnsnames.ora` alias for cleaner configuration.

## Requirements
- Wallet directory contains `tnsnames.ora` with an entry named `mainbase_high` (downloaded from Oracle ADB console).
- `TNS_ADMIN` (or `DB_WALLET_LOCATION`) points to that wallet directory.
- Credentials for the target schema (`DB_USER`, `DB_PASSWORD`).

## Minimal `.env.oracle`
```
DB_TYPE=oracle
DB_USER=STORY_DB
DB_PASSWORD=... # your schema password
DB_DSN=mainbase_high
TNS_ADMIN=./oracle_wallet
# Optional: mirror vars for external scripts
ORACLE_USER=$DB_USER
ORACLE_PASSWORD=$DB_PASSWORD
```

That’s it—the client resolves `mainbase_high` via `tnsnames.ora` and connects securely using the wallet.

## Notes
- Do not mix a full `tcps://...` connect string and a TNS alias in the same file; keep one source of truth.
- If you see `ORA-12154` (cannot resolve connect identifier), verify:
  - `TNS_ADMIN` points to the correct wallet dir
  - `tnsnames.ora` contains the `mainbase_high` alias
- For local XE, set `DB_DSN=localhost/XEPDB1` and you can omit wallet variables.

