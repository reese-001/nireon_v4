## **1. Developer Onboarding & Getting Started**

This section provides a practical guide for developers to set up, run, and begin contributing to the NIREON V4 project.

---

### **1.1. Prerequisites**

Before you begin, ensure you have the following installed on your system:

*   **Python:** Version 3.12 or higher.
*   **Poetry:** The recommended dependency and environment manager for this project.
*   **Git:** For version control.
*   **Docker:** (Recommended) For running components in a sandboxed environment, especially the `ProtoEngine`.
*   **(Optional) An OpenAI API Key:** To use the default LLM configurations, you will need an OpenAI API key.

---

### **1.2. Initial Setup**

Follow these steps to get your local development environment running.

1.  **Clone the Repository:**
    ```sh
    git clone <your_repository_url>
    cd nireon_v4
    ```

2.  **Install Dependencies:**
    Use Poetry to create a virtual environment and install all required packages from `pyproject.toml`.
    ```sh
    poetry install
    ```

3.  **Activate the Virtual Environment:**
    To work with the installed dependencies, activate the Poetry shell.
    ```sh
    poetry shell
    ```

4.  **Set Up Environment Variables:**
    The system uses environment variables for secrets like API keys. Create a `.env` file in the project root (`nireon_v4/`) to manage them locally.
    ```sh
    # In your new .env file
    OPENAI_API_KEY="sk-..."
    ```
    The application will automatically load this file on startup.

---

### **1.3. Running Core Tests**

Before making any changes, verify that the system is in a good state by running the core test suites.

1.  **Run All Unit & Integration Tests:**
    From the project root (`nireon_v4/`), run `pytest`.
    ```sh
    pytest
    ```
    All tests should pass. This confirms that the core logic and component interactions are working as expected.

2.  **Check Architectural Hygiene:**
    NIREON V4 uses `tach` to enforce its layered architecture. This check ensures that no module improperly imports from another, maintaining clean dependencies.
    ```sh
    tach check
    ```
    This command should exit with a "checks passed" message.

---

### **1.4. First Run: The Explorer Test Script**

The easiest way to see the full system in action is to run the end-to-end explorer test script. This script bootstraps the entire application, injects a seed idea, and runs a full generative loop through the Reactor, Explorer, and Sentinel.

```sh
# From the nireon_v4/ directory
python run_explorer_test.py
```

**What to Expect:**

1.  Logs showing the **Bootstrap** process initializing all components.
2.  Logs from the **Reactor** matching the initial `SeedSignal` to a rule.
3.  Logs from the **ExplorerMechanism** indicating it's generating idea variations.
4.  Logs from the **SentinelMechanism** showing it's assessing the newly generated ideas.
5.  If a high-trust idea is generated, you may see logs from the **QuantifierAgent** and **ProtoEngine** as they are triggered.
6.  A final JSON result file will be saved in the `runtime/` directory, containing a tree of the generated ideas and their trust scores.

This script is your primary tool for verifying that the core epistemic loop is functioning correctly after you've made changes.

---

### **1.5. Using the LLM Test CLI**

To test specific LLM models or routes without running the full system, use the `llm_test_cli.py` script. This is invaluable for debugging prompts and validating your `llm_config.yaml`.

*   **Validate your LLM config:**
    ```sh
    python -m tests.llm_subsystem.llm_test_cli validate-config
    ```

*   **Test connectivity to a specific model:**
    ```sh
    python -m tests.llm_subsystem.llm_test_cli test-connectivity --model nano_default
    ```

*   **Send an interactive prompt:**
    ```sh
    python -m tests.llm_subsystem.llm_test_cli interactive
    ```
    This will drop you into a shell where you can test different models and prompts directly.

---

### **1.6. Key Directories for Developers**

*   `bootstrap/`: The system startup logic. Modify this if you need to change the initialization sequence.
*   `configs/`: All declarative configuration.
    *   `manifests/standard.yaml`: The primary file for registering components.
    *   `reactor/rules/`: Where all signal-to-action logic lives.
    *   `default/llm_config.yaml`: The central hub for all LLM configuration.
*   `components/`: The home for all core mechanisms, observers, and managers. Start here when adding new agents.
*   `domain/`: The heart of the application's business logic, containing data models (`Idea`, `Frame`) and abstract interfaces (`Ports`).
*   `infrastructure/`: Concrete implementations of the `Ports` defined in the domain (e.g., `MemoryEventBus`, `LLMRouter`, `IdeaRepository`).
*   `signals/`: Definitions for all `EpistemicSignal` types that flow through the system.
*   `tests/`: All unit and integration tests. New code should be accompanied by new tests here.

---

**(The rest of the documentation would follow, starting with Section 2: NIREON V4 Subsystem Contracts & Responsibilities)**