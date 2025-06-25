### **Section 11: Troubleshooting & Integration**

This new section provides solutions to common problems and guides for advanced integration.

#### **11.1. Common Errors & Troubleshooting Guide**

*   **Error: `Bootstrap failed: Component ... not found in registry`**
    *   **Cause:** A component declared a dependency that was not successfully registered.
    *   **Solution:** Check the bootstrap logs for earlier errors. Did the missing component fail to instantiate? Is it correctly defined in `standard.yaml`? Is it marked `enabled: true`?

*   **Error: `Reactor rule ... did not trigger`**
    *   **Cause:** The conditions for the rule were not met.
    *   **Solution:**
        1.  Check the signal type: Does `signal_type_match` in your rule exactly match the `signal_type` of the signal being emitted?
        2.  Debug the `payload_expression`: Add logging inside `reactor/expressions/rel_engine.py` to print the expression and the context it's being evaluated against.
        3.  Check rule priority: Is another, higher-priority rule catching the signal first and stopping propagation?

*   **Error: `MechanismGateway: BudgetExceededError`**
    *   **Cause:** A `Frame` has exhausted its allocated resources (e.g., `llm_calls`).
    *   **Solution:** Increase the `resource_budget` for the `Frame` being created in the component that initiates the task (e.g., in the `ExplorerMechanism`'s `_process_impl` method).

*   **Error: `ProtoEngine: ImageNotFound`**
    *   **Cause:** The Docker image required for a Proto-Block's dialect (e.g., `nireon-proto-math:latest`) has not been built locally.
    *   **Solution:** Navigate to the `docker/` directory for that dialect and run `docker build -t <image_name> .`.

#### **11.2. Integration with External Data Sources**

To integrate a new data source (e.g., a SQL database, a REST API), follow the Port-and-Adapter pattern:

1.  **Define the Port:** In `domain/ports/`, create a new protocol defining the methods you need (e.g., `class UserDataPort(Protocol): def get_user_profile(user_id: str) -> dict: ...`).
2.  **Create the Adapter:** In `infrastructure/`, create a new directory (e.g., `infrastructure/user_data/`) and implement a concrete class that adheres to your new Port (e.g., `class SqlUserDataAdapter(UserDataPort): ...`). This class will contain the specific logic for connecting to and querying your data source.
3.  **Register in Manifest:** In `standard.yaml`, add your new adapter as a `shared_service`.
    ```yaml
    shared_services:
      UserDataPort: # Register it by its Port type
        class: "infrastructure.user_data.sql_adapter:SqlUserDataAdapter"
        config:
          connection_string: "${USER_DB_URL}"
    ```
4.  **Inject and Use:** In any component's `_initialize_impl` method, resolve your service via the registry and use it.
    ```python
    async def _initialize_impl(self, context: NireonExecutionContext):
        self.user_data_service = context.component_registry.get_service_instance(UserDataPort)
    ```

#### **11.3. Migration Guide for Custom Components**

When a new MAJOR version of NIREON is released, core contracts may change. To migrate your custom components:

1.  **Review the Changelog:** The release notes will detail all breaking changes to `NireonBaseComponent`, core signals, and essential `Ports`.
2.  **Update `NireonBaseComponent` Methods:** If method signatures in the base class have changed (e.g., a new parameter was added to `_process_impl`), update your component's implementation to match.
3.  **Update Signal Usage:** If `EpistemicSignal` schemas have changed, update the code that creates or consumes those signals.
4.  **Update Pydantic Configs:** Use Pydantic's `@validator(..., pre=True)` to create backward-compatible configuration models that can load older YAML files by renaming or transforming fields on the fly.
5.  **Run Tests:** The CI test suite is the best way to catch integration issues after a migration. Run `pytest` and fix any failures related to your component.