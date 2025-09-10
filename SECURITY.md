# Security Policy

This project takes protection of credentials and user data seriously. The following policies and practices apply.

## Reporting a Vulnerability

If you discover a vulnerability or exposed secret, please report it privately to the maintainers. Do not open a public issue containing sensitive details.

## Secrets Management

- Do not commit secrets to the repository. This includes API keys, database passwords, TLS certificates, and Oracle wallet files.
- Local development may use a `.env` file; production and CI must use a secret manager (e.g., Vault or cloud-native services).
- TLS and wallet materials must be provided as runtime mounts or secret volumes, not checked into git.

## Git Hygiene

- `.gitignore` blocks `*.pem`, `*.jks`, `*.p12`, `.env*`, and wallet directories.
- Pre-commit hooks are configured. To enable locally:
  - `pip install pre-commit detect-secrets`
  - `pre-commit install`
  - `detect-secrets scan > .secrets.baseline`

## Credential Rotation & Remediation

- If a secret is committed, rotate it immediately and remove it from history with a tool like `git-filter-repo`.
- For Oracle wallets, rotate and reissue the wallet; do not reuse compromised material.

## Logging & Redaction

- Never log sensitive values (e.g., `DB_PASSWORD`, wallet paths). Use structured logging and centralized redaction filters.

