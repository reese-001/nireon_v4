## **7. CI/CD and Operational Practices**

This section defines the CI/CD strategy for NIREON V4, ensuring code quality, correctness, and adherence to architectural principles.

---

### **7.1. CI Job Overview**

The Continuous Integration (CI) pipeline for NIREON includes several jobs designed to ensure the system is correct, robust, and maintainable.

| CI Job                   | Purpose                                                                                | Tools / Notes                                                               |
| :----------------------- | :------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| **Lint & Formatting**    | Enforce style consistency, typing rules, and import order.                             | `ruff` (for linting, formatting, and import sorting)                        |
| **Type Safety**          | Verify adherence to Protocols and strict type checking.                                | `mypy --strict`                                                             |
| **Architectural Hygiene**| Block violations of modular import boundaries defined in `subsystem_mapping.yaml`.       | `tach check` (using `tach.toml` for configuration)                          |
| **Test Suite**           | Run all unit, component, and integration tests. Measure code coverage.                 | `pytest`, `coverage.py`                                                     |
| **Smoke Run**            | Bootstrap the full system and run an end-to-end generative loop.                       | `run_explorer_test.py` (or a similar end-to-end script)                     |
| **Rule Inspection**      | (Currently for diagnostics) Ensure Reactor rules are loadable and syntactically correct. | `scripts/inspect_rules.py` (planned to enforce signal coverage)             |
| **Diagram Generation**   | Auto-create architecture diagrams from system metadata.                                | `scripts/gen_diagram.py` (uploads artifact)                                 |

---

### **7.2. Required CI Invariants**

The CI pipeline must enforce the following invariants, blocking Pull Requests that violate them:

*   **No Layering Violations:** The import graph of the codebase must adhere to the dependency rules defined in `tach.toml` (which is based on `subsystem_mapping.yaml`). For example, a `domain` module cannot import from an `infrastructure` module. This is enforced by `tach check`.
*   **Protocol Adherence:** All components must correctly implement their respective `Port` or `NireonBaseComponent` interfaces. This is enforced by `mypy`.
*   **Smoke Test Success:** The primary smoke test (`run_explorer_test.py`) must complete successfully. This proves that the system can be bootstrapped, a generative loop can be initiated via the Reactor, and the core mechanisms (Explorer, Sentinel) can interact to produce and assess ideas.
*   **Full Test Suite Passage:** All unit and integration tests must pass with a code coverage percentage above a defined threshold (e.g., 90%).

---

### **7.3. Core CI Scripts Explained**

#### **Architectural Hygiene (`validation/tach/check_architecture.py`)**

This script is a simple runner for the `tach` tool, which is the primary enforcer of our architectural layering.

*   **How it works:** `tach` reads the `tach.toml` configuration file, which defines the modules within each subsystem (e.g., "Kernel", "Domain_Model"). It then analyzes the Python import graph to ensure that no module imports from a module it is not supposed to depend on. For example, it will fail the build if a file in `domain/` tries to `import` a file from `infrastructure/`.
*   **CI Usage:** `tach check` is run in the pipeline. A non-zero exit code fails the build, preventing architectural violations from being merged.

#### **Rule & Signal Coverage (`scripts/inspect_rules.py`)**

This script is vital for maintaining the health of the Reactor's logic.

*   **Current Function:** It loads all YAML rule files from `configs/reactor/rules/` to ensure they are syntactically valid and can be parsed by the `RuleLoader`. This catches basic errors in rule definitions.
*   **Future Enforcement (CI Invariant):** The script's role will be expanded to enforce that every `EpistemicSignal` defined in `signals/` is handled by at least one Reactor rule. When run with a future `--enforce` flag in CI, it will fail the build if any "unmapped signals" are detected, ensuring the system's "nervous system" has no dead ends.

#### **Smoke Test (`run_explorer_test.py`)**

This is not a simple unit test; it is a minimal but complete end-to-end run of the NIREON system.

*   **How it works:**
    1.  It calls `bootstrap_nireon_system` to load the `standard.yaml` manifest and bring the entire application online.
    2.  It creates and publishes an initial `SeedSignal`.
    3.  It listens on the event bus for the subsequent `IdeaGeneratedSignal` and `TrustAssessmentSignal` events that are orchestrated by the Reactor.
    4.  It may even listen for the final `GenerativeLoopFinishedSignal`, which indicates a full reasoning cycle has completed.
    5.  It captures the results and saves them to a file in `runtime/` for inspection.
*   **Purpose:** This test validates that the bootstrap process, event bus, reactor, and core mechanisms are all integrated and functioning correctly. A failure here indicates a critical, system-level problem.

---

### **7.4. GitHub Actions Workflow (`.github/workflows/ci.yaml`)**

This is a suggested layout for the main CI workflow file.

```yaml
name: NIREON V4 CI

on: [push, pull_request]

jobs:
  lint-and-validate:
    name: Lint, Type & Architecture Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.12
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install poetry tach
          poetry install --no-root
      - name: Run Ruff (Lint & Format Check)
        run: poetry run ruff check . && poetry run ruff format --check .
      - name: Run MyPy (Strict Type Check)
        run: poetry run mypy nireon_v4/
      - name: Run Tach (Architecture Check)
        run: tach check

  unit-and-integration-tests:
    name: Test Suite & Coverage
    runs-on: ubuntu-latest
    needs: lint-and-validate
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.12
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install Dependencies
        run: |
          pip install poetry
          poetry install --no-root
      - name: Run tests with coverage
        run: poetry run pytest --cov=nireon_v4 --cov-report=xml
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage.xml

  smoke-test:
    name: System Smoke Run
    runs-on: ubuntu-latest
    needs: unit-and-integration-tests
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.12
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install Dependencies
        run: |
          pip install poetry
          poetry install --no-root
      - name: Execute end-to-end smoke run
        run: python run_explorer_test.py --iterations 1
      - name: Upload smoke run artifacts
        uses: actions/upload-artifact@v4
        with:
          name: smoke-run-results
          path: runtime/idea_evolution_*.json
```

---

### **7.5. Future CI Jobs & Enhancements**

As the system matures, the CI pipeline can be extended:

*   **Performance Regression:** Use `pytest-benchmark` to run performance tests and fail the build if key operations (e.g., bootstrap time, signal processing latency) regress beyond a certain threshold.
*   **Security Scanning:** Integrate tools like `pip-audit` and `bandit` to automatically scan for vulnerable dependencies and common security issues in the code.
*   **Live DAG Tracer:** Enhance the smoke test to capture the idea graph generated during the run and compare it against a "golden" fixture to detect unintended changes in the system's reasoning path.
*   **Bootstrap Footprint:** Monitor the memory usage and time taken by the bootstrap process to prevent bloat over time.