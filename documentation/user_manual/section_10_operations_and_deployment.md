### **Section 10: Operations & Deployment**

This new section provides guidance on deploying, scaling, and monitoring a NIREON V4 instance in a production environment.

#### **10.1. Deployment Strategy**

NIREON V4 is designed to be deployed as a containerized application.

*   **Recommended Environment:** A Docker or Kubernetes-based environment.
*   **Dockerfile:** A production-ready `Dockerfile` should be used to build the application image. It should:
    1.  Start from a stable Python base image (e.g., `python:3.12-slim`).
    2.  Install `poetry` and use it to install only production dependencies (`poetry install --no-dev`).
    3.  Copy the `nireon_v4/` application code into the image.
    4.  Set the working directory and a non-root user for security.
    5.  Define the `CMD` or `ENTRYPOINT` to launch the main application (e.g., via a master script that calls `bootstrap_nireon_system`).
*   **Configuration Management:**
    *   **Docker/Compose:** Use a `.env` file passed to `docker-compose.yml` to manage environment variables.
    *   **Kubernetes:** Use `ConfigMaps` for non-sensitive configuration and `Secrets` for API keys and database credentials. These should be mounted into the application container as environment variables.

#### **10.2. Scaling and Concurrency**

The architecture supports scaling at multiple levels:

*   **Vertical Scaling:** Increasing the CPU and memory allocated to the NIREON container will improve the performance of CPU-bound tasks and allow for larger in-memory caches (e.g., for `MemoryEventBus`, `InMemoryVectorStore`).
*   **Horizontal Scaling (Read-Only Workloads - Advanced):** While the default `MemoryEventBus` and `SQLite` repositories are single-instance, the system can be scaled for read-heavy or stateless workloads.
    *   **Stateless Mechanisms:** Multiple instances of NIREON can be run if they connect to a shared, production-grade `EventBus` (like RabbitMQ/Kafka) and `IdeaRepository` (like PostgreSQL).
    *   **Workload Partitioning:** Different NIREON instances could be configured to run only specific mechanisms by enabling/disabling them in their respective manifests, allowing for dedicated "generator" nodes and "evaluator" nodes.
*   **Proto-Engine Scaling:** The `DockerExecutor` for the Proto-Plane naturally scales, as each execution runs in its own isolated container. A powerful host machine can run many sandboxed `ProtoBlock` executions concurrently.

#### **10.3. Monitoring and Observability**

A comprehensive monitoring strategy is essential for production health.

1.  **Logging:**
    *   **Destination:** Configure the Python `logging` module to output structured JSON logs to `stdout`.
    *   **Ingestion:** Use a log aggregator like Fluentd, Logstash, or a cloud provider's native service (e.g., AWS CloudWatch Logs, Google Cloud Logging) to collect, index, and search logs from all running instances.
2.  **Metrics (Prometheus/Grafana):**
    *   **Exposition:** Implement a small web server (e.g., using FastAPI or Flask) within NIREON that exposes a `/metrics` endpoint.
    *   **Collector:** A dedicated metrics collector component would aggregate stats from key services (`LLMRouter`, `IdeaRepository`, `EventBus`) and format them for Prometheus scraping. The `LLMMetricsCollector` is a prime example of this pattern.
    *   **Dashboarding:** Use Grafana to build dashboards that visualize key SLIs (Service Level Indicators) like:
        *   `nireon_llm_calls_total` (with labels for model, success/failure)
        *   `nireon_llm_duration_ms_histogram`
        *   `nireon_event_bus_published_total`
        *   `nireon_idea_repository_count`
        *   `nireon_bootstrap_duration_seconds`
3.  **Health Checks:**
    *   The `health_check()` method on components can be exposed via an HTTP endpoint (e.g., `/health`).
    *   Container orchestrators like Kubernetes can use this endpoint for liveness and readiness probes to automatically manage application health.

#### **10.4. Performance Benchmarks**

While performance is hardware-dependent, the following are baseline expectations for a standard development environment (e.g., a modern laptop with an SSD and 16GB RAM):

*   **Bootstrap Time:** A full system bootstrap should complete in **under 10 seconds**. Times significantly longer may indicate issues with I/O or network dependencies during initialization.
*   **Idea Generation Cycle (Single):** From `SeedSignal` to the first `TrustAssessmentSignal` for a generated idea, the p95 latency should be **under 5 seconds** (heavily dependent on LLM provider latency).
*   **Reactor Throughput:** The Reactor should be able to process several hundred signals per second for simple routing rules.
*   **Embedding Throughput (Local):** The `SentenceTransformerAdapter` should be able to encode **50-100+ embeddings per second** in batch mode.

