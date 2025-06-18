# NIREON Development Priorities

## Priority 1: Solidify and Expand the Core Loop

Before adding entirely new systems (like Manager/Observer Gateways), the most valuable next step is to make the existing generative loop more robust and intelligent. You've proven the Explorer → Sentinel → Catalyst flow can technically work; now, let's make it meaningful.

### A. Flesh out the Catalyst Mechanism

**The Goal:** The Catalyst is currently a placeholder that accepts a trigger but doesn't do anything. This is the next logical component in your generative cycle.

**Implementation Steps:**

- **Input Handling:** Implement the `_process_impl` to properly handle the `input_data` coming from the `high_trust_amplification` rule. It should extract the `idea_id` or `idea_content` from the `TrustAssessmentSignal`.

- **Vector Blending:** It needs to fetch the target idea's vector, choose a `cross_domain_vector` (you'll need a way to load these), and perform the vector blending logic.

- **LLM Prompting:** Create a new, synthesized idea by prompting an LLM with the context of the original idea and the blended vector's domain.

- **Output:** Publish its own `IdeaGeneratedSignal` (or a more specific `IdeaSynthesizedSignal`) for the new, blended idea. This new idea will then naturally flow back to the Sentinel for assessment, creating a virtuous cycle.

**Why this first?** Completing the Catalyst closes the primary feedback loop of the system: Generate → Evaluate → Synthesize/Amplify → Evaluate.... This is the engine of innovation for NIREON.

### B. Refine the Sentinel Mechanism

**The Goal:** Make the Sentinel's evaluation more nuanced.

**Implementation Steps:**

- **Load Reference Ideas:** The Sentinel's `_process_impl` currently gets an empty list of `reference_ideas`. It needs to use the `IdeaService` to fetch relevant sibling or parent ideas to properly calculate novelty.

- **Implement Edge Trust:** Add the logic for `enable_edge_trust`. This involves checking the parent ideas' trust scores to give a boost to children of high-trust parents. This introduces the concept of inherited quality.

- **Implement Progression Bonus:** Fully implement the `enable_progression_adjustment` logic. This will reward ideas that show iterative improvement over time.

## Priority 2: Enhance the Reactor's Intelligence

With a working generative loop, the next step is to make the system's "brain" smarter. The Reactor is currently quite simple; it's time to unlock its potential.

### A. Implement More Sophisticated Rules

**The Goal:** Move beyond simple "Signal A triggers Component B" rules.

**Implementation Steps:**

- **Stagnation Rule:** The `stagnation_intervention` rule currently points to a non-existent `explorer_diverse`. Create a rule that, upon detecting stagnation, modifies the configuration of the existing `explorer_instance_01`. For example, it could trigger an `AdaptationAction` to temporarily increase its `creativity_factor`.

- **Context-Aware Rules:** Write rules that use more complex expressions. For example: "If a `TrustAssessmentSignal` has `trust_score > 0.8` AND `novelty_score < 0.2` (i.e., it's good but boring), trigger the Explorer with a high `divergence_strength` to make it more creative."

- **Negative Feedback Rules:** Implement the `low_trust_quarantine` rule. This is a crucial negative feedback loop that prevents the system from wasting resources on bad ideas.

### B. Introduce a "Metronome" Signal

**The Goal:** Give the system a "heartbeat" so it can perform periodic tasks without needing an external trigger.

**Implementation Steps:**

- Create a simple `SystemManager` component. Its only job is to run a continuous `asyncio` loop that, every N seconds, publishes a `SystemTickSignal`.

- You can then create reactor rules that trigger on `SystemTickSignal` to perform maintenance, analysis, or health checks (e.g., "On every 10th `SystemTickSignal`, run `analyze()` on all mechanisms and report health").

## Priority 3: Onboard New Components (Observers & Managers)

Now that the core engine is robust and intelligent, you can safely expand the system's capabilities.

### A. Implement an Observer (e.g., LineageTracker)

**The Goal:** Add passive monitoring and data collection. Observers listen to signals but don't trigger other components, making them very safe to add.

**Implementation Steps:**

- Create the `LineageTracker` component.
- Have it subscribe to `IdeaGeneratedSignal` and `TrustAssessmentSignal`.
- When it receives a signal, it doesn't trigger anything. It simply writes the relationship (e.g., "Idea B was generated from Idea A," "Idea C was assessed with score X") to its own database or log file.

**Why this is important:** This builds the "memory" and "history" of the system, which is invaluable for later analysis and understanding how the system arrived at a particular outcome.

### B. Implement a Manager (e.g., FlowCoordinator or a Resource Manager)

**The Goal:** Introduce top-down control and system-wide policy enforcement.

**Implementation Steps:**

- Instead of just having a `MechanismGateway`, you can now consider a `ManagerGateway` for higher-level commands.
- A `ResourceManager` could listen to the `SystemTickSignal` and check the `BudgetManager`. If resources are low, it could publish a `ThrottleComponentsSignal`.
- Mechanisms like Explorer would then subscribe to the `ThrottleComponentsSignal` and adjust their `application_rate` or `max_variations_per_level` accordingly.

## Summary of Recommendations

### Strengthen the Core (Highest Priority)
- **Finish Catalyst:** Close the main generative feedback loop.
- **Improve Sentinel:** Make evaluation more meaningful with novelty and history.

### Evolve the Brain
- **Smarter Reactor Rules:** Move from simple triggers to context-aware, adaptive decision-making.
- **Add a SystemTickSignal:** Give the system a heartbeat for proactive, periodic actions.

### Expand the Body
- **Add an Observer:** Start recording the system's history with a LineageTracker.
- **Add a Manager:** Introduce top-down resource management and control.