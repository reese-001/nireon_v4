Proto Execution Subsystem
Description: An evolution of the original Math Engine, the Proto Execution Subsystem is a powerful, general-purpose service for executing sandboxed code. It processes declarative ProtoBlock objects—self-contained, versioned, and dialect-specific YAML definitions—which describe a computational task. The subsystem routes these blocks to the appropriate secure executor (e.g., Docker, subprocess) based on their specified eidos (dialect).
A key component is the ProtoGenerator, which uses an LLM to translate natural language requests (e.g., "model the impact of a tariff") into fully-formed, executable ProtoBlock YAML, making complex analysis accessible through a simple conversational interface.
Public API / Contracts
proto_engine.service.ProtoEngine: The core component that receives ProtoTaskSignals, validates the contained ProtoBlock, and manages its execution via a sandboxed backend.
proto_generator.service.ProtoGenerator: An LLM-backed service that converts a natural language string into a ProtoBlock. It's the primary "compiler" for the Proto plane.
domain.proto.base_schema.ProtoBlock: The fundamental data structure. A pydantic model defining the schema for a declarative, executable task, including code, inputs, requirements, and resource limits.
proto_engine.executors.DockerExecutor / SubprocessExecutor: Concrete backends that execute the code defined in a ProtoBlock in a secure, isolated environment.
Accepted Signals: ProtoTaskSignal (which contains a ProtoBlock).
Produced Signals: ProtoResultSignal, MathProtoResultSignal (a specialized version for the 'math' dialect), and ProtoErrorSignal.
Dependencies (Imports From)
LLM_Subsystem
Application_Services
Event_and_Signal_System
Domain_Model
Kernel


graph TD
    subgraph ProtoExecution [Proto Execution Subsystem]
        A(proto_generator/service.py) --> B((LLM Subsystem));
        C(proto_engine/service.py) --> D[executors];
        D --> E(docker.py)
        D --> F(subprocess.py)
        G[domain/proto] --> H(base_schema.py)
        G --> I(validation.py)
    end