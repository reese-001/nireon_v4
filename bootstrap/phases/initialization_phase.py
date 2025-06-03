from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List

from .base_phase import BootstrapPhase, PhaseResult
from bootstrap.bootstrap_helper.context_helper import build_component_init_context
from application.components.base import NireonBaseComponent

logger = logging.getLogger(__name__)

class ComponentInitializationPhase(BootstrapPhase):
    """
    Component Initialization Phase - Initializes all registered components
    that require initialization according to their metadata
    """
    
    async def execute(self, context) -> PhaseResult:
        logger.info('Executing Component Initialization Phase - Initializing all registered components')
        
        errors = []
        warnings = []
        initialization_stats = {
            'total_components': 0,
            'components_requiring_init': 0,
            'successfully_initialized': 0,
            'initialization_skipped': 0,
            'initialization_failed': 0
        }
        
        try:
            # Get all registered components
            all_component_ids = context.registry.list_components()
            initialization_stats['total_components'] = len(all_component_ids)
            
            if not all_component_ids:
                logger.info('No components registered for initialization')
                return PhaseResult.success_result(
                    message='No components to initialize',
                    metadata=initialization_stats
                )
            
            logger.info(f'Found {len(all_component_ids)} registered components')
            
            # Group components by initialization requirement
            components_to_init = []
            components_to_skip = []
            
            for component_id in all_component_ids:
                try:
                    component = context.registry.get(component_id)
                    metadata = context.registry.get_metadata(component_id)
                    
                    if metadata.requires_initialize:
                        components_to_init.append((component_id, component, metadata))
                    else:
                        components_to_skip.append((component_id, component, metadata))
                        
                except Exception as e:
                    error_msg = f'Failed to access component {component_id} for initialization check: {e}'
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            initialization_stats['components_requiring_init'] = len(components_to_init)
            initialization_stats['initialization_skipped'] = len(components_to_skip)
            
            logger.info(f'Components requiring initialization: {len(components_to_init)}')
            logger.info(f'Components skipping initialization: {len(components_to_skip)}')
            
            # Initialize components that require it
            if components_to_init:
                await self._initialize_components(
                    components_to_init, context, initialization_stats, errors, warnings
                )
            
            # Update health reporter with initialization results
            await self._update_health_reporter(context, components_to_init, components_to_skip, errors)
            
            # Emit initialization signals
            await self._emit_initialization_signals(context, initialization_stats)
            
            success = len(errors) == 0 or not context.strict_mode
            message = f'Component initialization complete - {initialization_stats["successfully_initialized"]}/{initialization_stats["components_requiring_init"]} components initialized'
            
            return PhaseResult(
                success=success,
                message=message,
                errors=errors,
                warnings=warnings,
                metadata=initialization_stats
            )
            
        except Exception as e:
            error_msg = f'Critical error during component initialization: {e}'
            logger.error(error_msg, exc_info=True)
            return PhaseResult.failure_result(
                message='Component initialization failed',
                errors=[error_msg],
                warnings=warnings,
                metadata=initialization_stats
            )

    async def _initialize_components(self, components_to_init: List[tuple], context, 
                                   initialization_stats: dict, errors: list, warnings: list) -> None:
        """Initialize all components that require initialization"""
        
        # Determine initialization approach based on configuration
        concurrent_init = context.global_app_config.get('feature_flags', {}).get('enable_concurrent_initialization', False)
        
        if concurrent_init:
            await self._initialize_components_concurrent(components_to_init, context, initialization_stats, errors, warnings)
        else:
            await self._initialize_components_sequential(components_to_init, context, initialization_stats, errors, warnings)

    async def _initialize_components_sequential(self, components_to_init: List[tuple], context,
                                              initialization_stats: dict, errors: list, warnings: list) -> None:
        """Initialize components sequentially"""
        logger.info('Initializing components sequentially')
        
        for component_id, component, metadata in components_to_init:
            try:
                await self._initialize_single_component(component_id, component, metadata, context)
                initialization_stats['successfully_initialized'] += 1
                logger.info(f'✓ Component {component_id} initialized successfully')
                
            except Exception as e:
                initialization_stats['initialization_failed'] += 1
                error_msg = f'Failed to initialize component {component_id}: {e}'
                
                if context.strict_mode:
                    errors.append(error_msg)
                    logger.error(error_msg)
                else:
                    warnings.append(error_msg)
                    logger.warning(error_msg)

    async def _initialize_components_concurrent(self, components_to_init: List[tuple], context,
                                              initialization_stats: dict, errors: list, warnings: list) -> None:
        """Initialize components concurrently"""
        logger.info('Initializing components concurrently')
        
        # Create initialization tasks
        tasks = []
        for component_id, component, metadata in components_to_init:
            task = asyncio.create_task(
                self._initialize_single_component_safe(component_id, component, metadata, context),
                name=f'init_{component_id}'
            )
            tasks.append((task, component_id))
        
        # Wait for all tasks to complete
        for task, component_id in tasks:
            try:
                success = await task
                if success:
                    initialization_stats['successfully_initialized'] += 1
                    logger.info(f'✓ Component {component_id} initialized successfully (concurrent)')
                else:
                    initialization_stats['initialization_failed'] += 1
                    error_msg = f'Failed to initialize component {component_id} (concurrent)'
                    if context.strict_mode:
                        errors.append(error_msg)
                    else:
                        warnings.append(error_msg)
                        
            except Exception as e:
                initialization_stats['initialization_failed'] += 1
                error_msg = f'Exception during concurrent initialization of {component_id}: {e}'
                if context.strict_mode:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)

    async def _initialize_single_component(self, component_id: str, component: Any, 
                                         metadata, context) -> None:
        """Initialize a single component"""
        logger.debug(f'Initializing component: {component_id}')
        
        # Check if component supports initialization
        if not isinstance(component, NireonBaseComponent):
            if hasattr(component, 'initialize') and callable(component.initialize):
                logger.debug(f'Component {component_id} has initialize method but is not NireonBaseComponent')
            else:
                logger.debug(f'Component {component_id} does not support initialization')
                return
        
        # Check if already initialized
        if hasattr(component, 'is_initialized') and component.is_initialized:
            logger.debug(f'Component {component_id} already initialized')
            return
        
        # Create component-specific initialization context
        init_context = build_component_init_context(component_id, context, {})
        
        # Initialize the component
        await component.initialize(init_context)
        
        logger.debug(f'✓ Component {component_id} initialization completed')

    async def _initialize_single_component_safe(self, component_id: str, component: Any, 
                                               metadata, context) -> bool:
        """Safely initialize a single component (for concurrent execution)"""
        try:
            await self._initialize_single_component(component_id, component, metadata, context)
            return True
        except Exception as e:
            logger.error(f'Component {component_id} initialization failed: {e}')
            return False

    async def _update_health_reporter(self, context, components_to_init: List[tuple], 
                                    components_to_skip: List[tuple], errors: list) -> None:
        """Update health reporter with initialization results"""
        try:
            if not hasattr(context, 'health_reporter'):
                return
            
            health_reporter = context.health_reporter
            
            # Update status for initialized components
            for component_id, component, metadata in components_to_init:
                try:
                    if hasattr(component, 'is_initialized') and component.is_initialized:
                        from bootstrap.health.reporter import ComponentStatus
                        health_reporter.add_component_status(
                            component_id,
                            ComponentStatus.INITIALIZED_OK,
                            metadata,
                            []
                        )
                    else:
                        from bootstrap.health.reporter import ComponentStatus
                        health_reporter.add_component_status(
                            component_id,
                            ComponentStatus.INITIALIZATION_ERROR,
                            metadata,
                            ['Component initialization failed or not verified']
                        )
                except Exception as e:
                    logger.warning(f'Failed to update health status for {component_id}: {e}')
            
            # Update status for skipped components
            for component_id, component, metadata in components_to_skip:
                try:
                    from bootstrap.health.reporter import ComponentStatus
                    health_reporter.add_component_status(
                        component_id,
                        ComponentStatus.INITIALIZATION_SKIPPED_NOT_REQUIRED,
                        metadata,
                        []
                    )
                except Exception as e:
                    logger.warning(f'Failed to update health status for {component_id}: {e}')
                    
        except Exception as e:
            logger.warning(f'Failed to update health reporter: {e}')

    async def _emit_initialization_signals(self, context, initialization_stats: dict) -> None:
        """Emit initialization completion signals"""
        try:
            if hasattr(context, 'signal_emitter'):
                from signals.bootstrap_signals import INITIALIZATION_PHASE_COMPLETE
                await context.signal_emitter.emit_signal(
                    INITIALIZATION_PHASE_COMPLETE,
                    {
                        'initialization_stats': initialization_stats,
                        'phase': 'ComponentInitializationPhase',
                        'success': initialization_stats['initialization_failed'] == 0
                    }
                )
        except Exception as e:
            logger.warning(f'Failed to emit initialization signals: {e}')

    def should_skip_phase(self, context) -> tuple[bool, str]:
        """Check if initialization phase should be skipped"""
        # Skip if no registry available
        if not hasattr(context, 'registry') or not context.registry:
            return (True, 'No component registry available')
        
        # Skip if disabled in configuration
        skip_init = context.global_app_config.get('skip_component_initialization', False)
        if skip_init:
            return (True, 'Component initialization disabled in configuration')
        
        return (False, '')