from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
import reactor
from .base_phase import BootstrapPhase, PhaseResult
from reactor.engine.main import MainReactorEngine
from reactor.engine.base import ReactorEngine
from reactor.loader import RuleLoader
from bootstrap.bootstrap_helper.metadata import create_service_metadata
from bootstrap.exceptions import BootstrapError
from domain.ports.event_bus_port import EventBusPort
from signals import EpistemicSignal
import signals

if TYPE_CHECKING:
    from bootstrap.context.bootstrap_context import BootstrapContext

logger = logging.getLogger(__name__)

class ReactorSetupPhase(BootstrapPhase):
    async def execute(self, context: 'BootstrapContext') -> PhaseResult:
        logger.info('Executing Reactor Setup Phase...')
        try:
            rule_loader = RuleLoader()
            default_path = Path.cwd() / 'configs' / 'reactor' / 'rules'
            reactor_config = context.global_app_config.get('reactor', {})
            rules_path_str = reactor_config.get('rules_path', str(default_path))
            rules_path = Path(rules_path_str)

            if not rules_path.is_dir():
                warning_msg = f"Reactor rules directory not found at '{rules_path}'. The reactor will be empty."
                logger.warning(warning_msg)
                if context.strict_mode:
                    raise BootstrapError(warning_msg)
                loaded_rules = []
            else:
                logger.info('Loading reactor rules from: %s', rules_path)
                loaded_rules = rule_loader.load_rules_from_directory(rules_path)

            if not loaded_rules:
                logger.warning("No rules were loaded. The reactor will function but won't process any signals.")

            max_recursion_depth = reactor_config.get('max_recursion_depth', 10)
            engine = MainReactorEngine(
                registry=context.registry,
                rules=loaded_rules,
                max_recursion_depth=max_recursion_depth
            )
            logger.info(f'MainReactorEngine instantiated with {len(loaded_rules)} rules (max recursion depth: {max_recursion_depth})')

            engine_metadata = create_service_metadata(
                service_id='reactor_engine_main',
                service_name='Main Reactor Engine',
                category='core_engine',
                description='Central engine for processing signals and executing rules.',
                requires_initialize=False
            )
            context.registry_manager.register_with_certification(engine, engine_metadata)
            context.registry.register_service_instance(ReactorEngine, engine)
            logger.info('ReactorEngine registered as a service with certification.')
            context.reactor_engine = engine

            logger.info('Setting up Reactor to listen to EventBus signals...')
            self._bridge_event_bus_to_reactor(engine, context)

            namespaces = {}
            for rule in loaded_rules:
                ns = getattr(rule, 'namespace', 'unknown')
                namespaces[ns] = namespaces.get(ns, 0) + 1
            namespace_summary = ', '.join(f'{ns}: {count}' for ns, count in namespaces.items())

            return PhaseResult.success_result(
                message=f"Reactor engine initialized with {len(loaded_rules)} rules. Rules by namespace: {namespace_summary or 'None'}",
                metadata={'rules_loaded': len(loaded_rules), 'namespaces': list(namespaces.keys())}
            )
        except Exception as e:
            error_msg = f'Critical error during Reactor setup: {e}'
            logger.error(error_msg, exc_info=True)
            if context.strict_mode:
                raise BootstrapError(error_msg) from e
            return PhaseResult.failure_result(message=f'Reactor setup failed: {e}', errors=[error_msg])

    def _bridge_event_bus_to_reactor(self, engine: MainReactorEngine, context: 'BootstrapContext'):
        try:
            event_bus = context.registry.get_service_instance(EventBusPort)
        except Exception as e:
            logger.error(f'Cannot bridge EventBus to Reactor: EventBus not found in registry. Error: {e}')
            return

        signal_class_map = signals.signal_class_map
        logger.debug(f'Bridging event bus with {len(signal_class_map)} known signal types: {list(signal_class_map.keys())}')

        # FIX: Instead of trying to subscribe to every signal name found in the rules,
        # we will iterate through the *known* signal classes and subscribe to them.
        # This prevents warnings for signal names that are just identifiers and not classes.
        for signal_type_str, SignalClass in signal_class_map.items():
            if not issubclass(SignalClass, EpistemicSignal):
                continue

            def create_handler(cls):
                async def handler(payload: Any): # The payload is now the object or a dict
                    try:
                        # FIX: This logic now correctly handles receiving either a
                        # full signal object or a dictionary to reconstruct from.
                        if isinstance(payload, EpistemicSignal):
                            reconstructed_signal = payload
                        elif isinstance(payload, dict):
                            # When coming from an external source or a JSON dump,
                            # the entire payload is the data for the constructor.
                            reconstructed_signal = cls(**payload)
                        else:
                            logger.error(f"Cannot process payload of type {type(payload)} for signal '{signal_type_str}'")
                            return

                        await engine.process_signal(reconstructed_signal)
                    except Exception as e:
                        logger.error(f"Error handling event bus signal '{signal_type_str}': {e}", exc_info=True)
                        logger.error(f'Failed payload was: {payload}')
                return handler
            
            event_bus.subscribe(signal_type_str, create_handler(SignalClass))
            logger.info(f"Reactor is now subscribed to '{signal_type_str}' signals on the EventBus.")