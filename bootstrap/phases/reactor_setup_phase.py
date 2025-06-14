# nireon_v4/bootstrap/phases/reactor_setup_phase.py
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base_phase import BootstrapPhase, PhaseResult
from reactor.engine.main import MainReactorEngine
from reactor.engine.base import ReactorEngine
from reactor.loader import RuleLoader
from bootstrap.bootstrap_helper.metadata import create_service_metadata
from bootstrap.exceptions import BootstrapError

if TYPE_CHECKING:
    from bootstrap.context.bootstrap_context import BootstrapContext

logger = logging.getLogger(__name__)

class ReactorSetupPhase(BootstrapPhase):
    """
    Bootstrap phase responsible for setting up the Reactor Engine.
    This includes loading rules from YAML files and registering the engine as a core service.
    """
    
    async def execute(self, context: "BootstrapContext") -> PhaseResult:
        """
        Execute the Reactor setup phase.
        
        Args:
            context: The bootstrap context containing registry and configuration.
            
        Returns:
            PhaseResult indicating success or failure.
        """
        logger.info("Executing Reactor Setup Phase...")
        
        try:
            # 1. Create rule loader
            rule_loader = RuleLoader()
            
            # 2. Determine rules path from configuration (relative to project root)
            # A robust way to find the project root might be needed if not in context
            # For now, let's assume a standard structure.
            project_root = getattr(context.config, 'project_root', Path.cwd())
            default_path = project_root / "configs" / "reactor" / "rules"
            
            reactor_config = context.global_app_config.get('reactor', {})
            rules_path_str = reactor_config.get('rules_path', str(default_path))
            rules_path = Path(rules_path_str)
            
            if not rules_path.is_dir():
                 warning_msg = f"Reactor rules directory not found at '{rules_path}'. The reactor will be empty."
                 logger.warning(warning_msg)
                 # In non-strict mode, we can proceed with an empty reactor.
                 if context.strict_mode:
                     raise BootstrapError(warning_msg)
                 loaded_rules = []
            else:
                logger.info(f"Loading reactor rules from: {rules_path}")
                loaded_rules = rule_loader.load_rules_from_directory(rules_path)

            if not loaded_rules:
                logger.warning("No rules were loaded. The reactor will function but won't process any signals.")
            
            # 3. Get max recursion depth from config
            max_recursion_depth = reactor_config.get('max_recursion_depth', 10)
            
            # 4. Instantiate the Reactor Engine with the loaded rules
            engine = MainReactorEngine(
                registry=context.registry,
                rules=loaded_rules,
                max_recursion_depth=max_recursion_depth
            )
            
            logger.info(
                f"MainReactorEngine instantiated with {len(loaded_rules)} rules "
                f"(max recursion depth: {max_recursion_depth})"
            )
            
            # 5. Register the engine itself as a service using the V4 standard method
            engine_metadata = create_service_metadata(
                service_id="reactor_engine_main",
                service_name="Main Reactor Engine",
                category="core_engine",
                description="Central engine for processing signals and executing rules.",
                requires_initialize=False
            )
            context.registry_manager.register_with_certification(engine, engine_metadata)
            
            # Also register by the Protocol type for dependency injection
            context.registry.register_service_instance(ReactorEngine, engine)

            logger.info("ReactorEngine registered as a service with certification.")
            
            # 6. Store reference in context for other phases if needed
            context.reactor_engine = engine
            
            # Log summary of loaded rules by namespace
            namespaces = {}
            for rule in loaded_rules:
                ns = getattr(rule, 'namespace', 'unknown')
                namespaces[ns] = namespaces.get(ns, 0) + 1
            
            namespace_summary = ", ".join(f"{ns}: {count}" for ns, count in namespaces.items())
            
            return PhaseResult.success_result(
                message=f"Reactor engine initialized with {len(loaded_rules)} rules. "
                        f"Rules by namespace: {namespace_summary or 'None'}",
                metadata={'rules_loaded': len(loaded_rules), 'namespaces': list(namespaces.keys())}
            )
            
        except Exception as e:
            error_msg = f"Critical error during Reactor setup: {e}"
            logger.error(error_msg, exc_info=True)
            if context.strict_mode:
                # In strict mode, re-raise as a BootstrapError to halt the process
                raise BootstrapError(error_msg) from e
            
            # In non-strict mode, just report the failure
            return PhaseResult.failure_result(
                message=f"Reactor setup failed: {e}",
                errors=[error_msg]
            )