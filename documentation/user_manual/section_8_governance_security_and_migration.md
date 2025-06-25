## **8. Governance, Security, and Migration**

This section defines the processes for managing change, securing the system, and understanding the architectural evolution of NIREON V4.

---

### **8.1. API Governance (NIP Process - Planned)**

To ensure that changes to NIREON's core contracts are well-documented, reviewed, and aligned with the project's goals, a **NIREON Improvement Proposal (NIP)** process is planned for the future.

#### **8.1.1. Scope of "API"**

In the context of NIREON V4, "API" refers to several key contracts:

*   **Python API:** The public methods of core classes like `NireonBaseComponent`, `MechanismGateway`, and core services.
*   **Signal Schemas:** The structure and fields of `EpistemicSignal` subclasses.
*   **Configuration Schemas:** The structure of YAML files (manifests, rule files, component configs).
*   **Data Models:** The structure of core domain objects like `Idea` and `Frame`.

#### **8.1.2. The NIP Process (Future)**

The NIP process will provide a formal lifecycle for proposing and implementing significant changes.

1.  **Draft:** An initial proposal is created as a GitHub issue or dedicated document.
2.  **Discussion:** A community feedback period for review and debate.
3.  **Review & Decision:** Core maintainers evaluate the proposal's technical merit, architectural alignment, and community feedback, then vote to accept, reject, or request changes.
4.  **Implementation:** An accepted NIP is implemented via a Pull Request that references the proposal.

#### **8.1.3. Versioning & Deprecation Policy**

NIREON V4 adheres to **Semantic Versioning (X.Y.Z)** for its core contracts.

*   **MAJOR (X):** Breaking changes to the public API (e.g., removing a method from `NireonBaseComponent`, significantly altering the `Frame` schema).
*   **MINOR (Y):** New, backward-compatible features (e.g., adding a new optional method to a Port, adding a new field to a signal with a default value).
*   **PATCH (Z):** Backward-compatible bug fixes and non-functional improvements.

A clear deprecation policy will be established, providing runtime warnings and detailed release notes for any feature planned for removal in a future major version.

---

### **8.2. Security & Role-Based Access Control (RBAC)**

NIREON V4 includes a foundational RBAC system to control actions within the application. This system is loaded during bootstrap and is used to govern component-level permissions.

*   **🔑 Key Concepts:** `RBACPolicyEngine`, `RBACRule`, `@requires_permission` decorator.
*   **📄 Key Files:**
    *   `configs/default/bootstrap_rbac.yaml`: The central policy definition file.
    *   `bootstrap/phases/rbac_setup_phase.py`: The bootstrap phase that loads policies.
    *   `security/rbac_engine.py`: The engine that evaluates permissions.
    *   `security/decorators.py`: The decorators used to protect methods.

#### **8.2.1. RBAC Design & Flow**

The current RBAC system focuses on **intra-system authorization**. It determines whether a given *subject* (e.g., a component ID like `explorer_instance_01`) is allowed to perform an *action* (e.g., `write`) on a *resource* (e.g., `ideas`). It does not currently handle end-user authentication (AuthN).

**Permission Check Flow:**

```mermaid
graph TD
    A[Component calls a protected method] --> B{Decorator @requires_permission('resource', 'action')};
    B --> C{Get current subject (e.g., component_id from context)};
    C --> D{Get RBACPolicyEngine from registry};
    D --> E{engine.is_allowed(subject, resource, action)?};
    E -- Yes --> F[Execute Method];
    E -- No --> G[Raise RBACPermissionError];
```

#### **8.2.2. Policy Definition (`bootstrap_rbac.yaml`)**

Policies are defined in a simple YAML format. Each rule grants or denies permissions.

```yaml
# In configs/default/bootstrap_rbac.yaml
version: "1.0"
rules:
  - id: "system_admin_full_access"
    subjects: ["system_admin", "admin", "system"]
    resources: ["*"]
    actions: ["*"]
    effect: "allow"
    description: "System administrators have full access to all resources."

  - id: "explorer_permissions"
    subjects: ["explorer", "explorer_mechanism"] # Matches component ID or its tags
    resources: ["ideas", "exploration"]
    actions: ["read", "write", "execute"]
    effect: "allow"
    description: "Explorer mechanisms can read/write ideas and execute explorations."
```

*   `subjects`: A list of roles or component IDs. Wildcards (`*`) are supported.
*   `resources`: The target resource (e.g., `ideas`, `components`, `config`).
*   `actions`: The verb (e.g., `read`, `write`, `execute`, `delete`).
*   `effect`: Must be `allow` or `deny`. **`deny` rules always take precedence.**

#### **8.2.3. Credential Management**

*   **Current Practice:** All secrets, especially LLM API keys, **must** be managed through environment variables. The configuration files reference these variables (e.g., `${OPENAI_API_KEY}`).
*   **Future Direction:** For production deployments, integrating with a dedicated secrets manager like HashiCorp Vault or a cloud provider's KMS is the recommended path.

---

### **8.3. NIREON V4 Migration Summary**

The move from NIREON V3 to V4 was a strategic refactoring to establish a more robust, modular, and scalable architecture. The V3 codebase was frozen, and the `nireon_v4/` directory was created to house the new system.

#### **8.3.1. Key Architectural Shifts**

The migration introduced several fundamental changes that define the V4 architecture:

1.  **Strict Layering and Domain-Driven Design:** The codebase was strictly segregated into `domain`, `application`, `infrastructure`, and `core` layers. The `domain` layer contains pure business logic and abstract ports, making it independent of any specific technology. This is enforced by the `tach` architecture linter in CI.
2.  **The A➜F➜CE Model and Mechanism Gateway:** Direct service calls from mechanisms were eliminated. All mechanism interactions are now mediated by the `MechanismGateway` using the **Agent ➜ Frame ➜ Cognitive Event** model. This provides a single point for policy enforcement, context management, and observability.
3.  **Declarative Reactor Engine:** The V3's programmatic orchestration logic was replaced by the V4 Reactor. System behavior is now defined declaratively in YAML rule files (`configs/reactor/rules/`), which map signals to component actions. This makes the system's control flow more transparent and easier to modify.
4.  **Centralized Bootstrap Process:** The startup sequence was formalized into a series of distinct, ordered phases managed by the `BootstrapOrchestrator`. This ensures a predictable and reliable system initialization.
5.  **Pydantic-Driven Configuration:** All component configurations are now defined by Pydantic models, providing automatic validation, type safety, and better self-documentation.

#### **8.3.2. Migration Outcome**

The result of the migration is a system that is:

*   **More Modular:** Clear separation between components and layers reduces coupling and makes the system easier to maintain and extend.
*   **More Testable:** The use of ports and a central gateway allows for easier mocking and isolated testing of components.
*   **More Flexible:** The declarative Reactor allows for complex behaviors to be rewired by changing YAML files instead of Python code.
*   **More Robust:** Centralized policy enforcement via the Gateway and a predictable bootstrap process improve system reliability and governance.