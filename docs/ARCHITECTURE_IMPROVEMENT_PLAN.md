# Architecture Improvement Plan

This document outlines the recommended steps to improve the architecture of the Story Engine project, focusing on security, maintainability, scalability, and reliability.

## 1. Critical Security Vulnerability (Priority: Immediate)

**Goal:** Remove all sensitive credentials from the repository and implement a secure secrets management strategy.

- **Step 1.1: Prevent Future Commits of Secrets**
  - **Action:** Add the following rules to the root `.gitignore` file.
  ```gitignore
  # Ignore sensitive files and directories
  *.pem
  *.jks
  *.p12
  .env
  /wallet/
  ```

- **Step 1.2: Purge Secrets from Git History**
  - **Action:** Use a tool like `git-filter-repo` to rewrite history and remove sensitive files. This is a destructive action that requires team coordination.
  - **Command:** 
  ```bash
  git filter-repo --invert-paths --path 'path/to/your/wallet/' --path 'path/to/your/key.pem'
  ```

- **Step 1.3: Implement a Secrets Manager**
  - **Action:** Modify the application code (`core/common/config.py`, `core/storage/database.py`) to fetch secrets from environment variables or a dedicated secrets management service (e.g., HashiCorp Vault, AWS/GCP/Azure Secrets Manager) instead of from files in the repository.

## 2. Import Fragility (Priority: High)

**Goal:** Stabilize the codebase by creating a proper, installable Python package and eliminating `sys.path` manipulations.

- **Step 2.1: Create `pyproject.toml`**
  - **Action:** Create a `pyproject.toml` file in the project root to define project metadata and dependencies.

- **Step 2.2: Adopt a `src` Layout**
  - **Action:** Create a `src` directory and move the `core` and `poml` packages into it. This standardizes the project structure.

- **Step 2.3: Install in Editable Mode**
  - **Action:** Use `pip install -e .` during development. This makes the project importable as a package while allowing code changes to be reflected immediately.

- **Step 2.4: Refactor All Imports**
  - **Action:** Systematically replace all relative or path-based imports with absolute imports based on the new package structure.
  - **Example:** `from common.config import ...` becomes `from core.common.config import ...`.

## 3. High Central Coupling (Priority: Medium)

**Goal:** Decouple the `core/story_engine` from its dependencies to improve modularity and ease of testing.

- **Step 3.1: Apply Inversion of Control (IoC)**
  - **Action:** Modify the `StoryEngine` and other core classes to receive their dependencies (like an orchestrator or database connection) in their constructor (`__init__`) instead of creating them internally.

- **Step 3.2: Introduce an In-Process Event Bus**
  - **Action:** As a first step towards a distributed architecture, implement a simple publish/subscribe event bus. Instead of making direct calls between major components, have them emit events (e.g., `PlotGeneratedEvent`) that other components can subscribe to.

## 4. Synchronous Bottlenecks & State Management (Priority: Medium)

**Goal:** Improve the reliability and performance of long-running story generation pipelines.

- **Step 4.1: Ensure All I/O is Asynchronous**
  - **Action:** Audit all network and database calls to ensure they use non-blocking, `async`-compatible libraries (e.g., `aiohttp`, `psycopg` for async Postgres).

- **Step 4.2: Externalize Pipeline State**
  - **Action:** Modify the pipeline logic to persist the story's state (e.g., the `NarrativeGraph`) to the database after each significant generation step. This prevents data loss if the process fails and is a prerequisite for distributed workers.

## 5. Limited Observability (Priority: Low, but high impact)

**Goal:** Improve debugging and monitoring capabilities.

- **Step 5.1: Implement Structured Logging**
  - **Action:** Use a library like `structlog` to output logs as JSON objects, including relevant context like a `story_id`.

- **Step 5.2: Introduce a Correlation ID**
  - **Action:** Generate a unique ID for each end-to-end story generation request. Pass this ID through all subsequent function calls, events, and log messages to allow for easy tracing.

- **Step 5.3: Add Basic Metrics**
  - **Action:** Instrument the code with a library like `prometheus-client` to track key metrics, such as the number of stories generated, the error rate of LLM calls, and the latency of different pipeline stages.
