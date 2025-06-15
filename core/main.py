# # nireon_v4/bootstrap/core/main.py
# from __future__ import absolute_import # Moved to the top
# from __future__ import annotations # Moved to the top

# import asyncio
# import logging
# import os
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence, Union
# import dataclasses 

# from pydantic import BaseModel, Field, ValidationError

# from bootstrap.exceptions import BootstrapError, BootstrapValidationError, BootstrapTimeoutError
# from bootstrap.result_builder import BootstrapResult, BootstrapResultBuilder
# from bootstrap.context.bootstrap_context_builder import create_bootstrap_context
# from bootstrap.config.bootstrap_config import BootstrapConfig
# from bootstrap.context.bootstrap_context import BootstrapContext
# from bootstrap.phases.context_formation_phase import ContextFormationPhase
# from core.registry.component_registry import ComponentRegistry

# # Fixed imports from bootstrap.phases - using correct paths
# try:
#     from bootstrap.phases.abiogenesis_phase import AbiogenesisPhase
# except ImportError as e:
#     logger = logging.getLogger(__name__) 
#     logger.error(f'Failed to import AbiogenesisPhase: {e}')
#     AbiogenesisPhase = None # type: ignore

# try:
#     from bootstrap.phases.registry_setup_phase import RegistrySetupPhase
# except ImportError as e:
#     logger = logging.getLogger(__name__)
#     logger.error(f'Failed to import RegistrySetupPhase: {e}')
#     RegistrySetupPhase = None # type: ignore

# try:
#     from bootstrap.phases.factory_setup_phase import FactorySetupPhase
# except ImportError as e:
#     logger = logging.getLogger(__name__)
#     logger.error(f'Failed to import FactorySetupPhase: {e}')
#     FactorySetupPhase = None # type: ignore

# try:
#     from bootstrap.phases.manifest_processing_phase import ManifestProcessingPhase
# except ImportError as e:
#     logger = logging.getLogger(__name__)
#     logger.error(f'Failed to import ManifestProcessingPhase: {e}')
#     ManifestProcessingPhase = None # type: ignore

# try:
#     from bootstrap.phases.component_initialization_phase import ComponentInitializationPhase
# except ImportError as e:
#     logger = logging.getLogger(__name__)
#     logger.error(f'Failed to import ComponentInitializationPhase: {e}')
#     ComponentInitializationPhase = None # type: ignore

# try:
#     from bootstrap.phases.component_validation_phase import InterfaceValidationPhase
# except ImportError as e:
#     logger = logging.getLogger(__name__)
#     logger.error(f'Failed to import InterfaceValidationPhase from component_validation_phase: {e}')
#     InterfaceValidationPhase = None # type: ignore

# try:
#     from bootstrap.phases.rbac_setup_phase import RBACSetupPhase
# except ImportError as e:
#     logger = logging.getLogger(__name__)
#     logger.error(f'Failed to import RBACSetupPhase: {e}')
#     RBACSetupPhase = None # type: ignore

# try:
#     from bootstrap.phases.late_rebinding_phase import LateRebindingPhase
# except ImportError as e:
#     logger = logging.getLogger(__name__)
#     logger.error(f'Failed to import LateRebindingPhase: {e}')
#     LateRebindingPhase = None # type: ignore

# # Load environment variables
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
#     logger = logging.getLogger(__name__) 
#     logger.debug("Dotenv loaded if .env file exists.")
# except ImportError:
#     logger = logging.getLogger(__name__)
#     logger.debug("Dotenv not installed, .env file (if any) will not be loaded.")
# except Exception as e:
#     logger = logging.getLogger(__name__)
#     logger.warning(f"Error loading dotenv: {e}")


# class BootstrapExecutionConfig(BaseModel):
#     timeout_seconds: float = Field(default=300.0, ge=1.0, le=3600.0, description='Maximum time allowed for bootstrap execution')
#     phase_timeout_seconds: float = Field(default=60.0, ge=1.0, le=300.0, description='Maximum time allowed per bootstrap phase')
#     retry_on_failure: bool = Field(default=False, description='Whether to retry bootstrap on failure (non-strict mode only)')
#     max_retries: int = Field(default=3, ge=0, le=10, description='Maximum number of retry attempts')
#     enable_health_checks: bool = Field(default=True, description='Whether to perform health checks during bootstrap')
    
#     class Config:
#         extra = 'forbid'
#         validate_assignment = True


# class BootstrapOrchestrator:
#     def __init__(self, config: BootstrapConfig, execution_config: Optional[BootstrapExecutionConfig] = None):
#         self.config = config
#         self.execution_config = execution_config or BootstrapExecutionConfig()
#         self.run_id = self._generate_run_id()
#         self.start_time: Optional[datetime] = None
#         self._phases: Optional[List[Any]] = None 
#         logger.info(f'BootstrapOrchestrator initialized - run_id: {self.run_id}')
#         logger.debug(f'Execution config: {self.execution_config.model_dump_json(indent=2)}') 

#     async def execute_bootstrap(self) -> BootstrapResult:
#         self.start_time = datetime.now(timezone.utc)
#         logger.info('=== NIREON V4 System Bootstrap Starting (L0 Abiogenesis) ===')
#         logger.info(f'Run ID: {self.run_id}')
#         logger.info(f'Config Paths: {[str(p) for p in self.config.config_paths]}')
#         logger.info(f"Environment: {self.config.env or 'default'}")
        
#         try:
#             bootstrap_task = self._execute_bootstrap_process()
#             return await asyncio.wait_for(bootstrap_task, timeout=self.execution_config.timeout_seconds)
#         except asyncio.TimeoutError:
#             error_msg = f'Bootstrap timed out after {self.execution_config.timeout_seconds}s'
#             logger.error(error_msg)
#             minimal_result_for_timeout = self._create_minimal_result(BootstrapTimeoutError(error_msg))
#             if hasattr(minimal_result_for_timeout, 'health_reporter') and minimal_result_for_timeout.health_reporter:
#                 minimal_result_for_timeout.health_reporter.add_phase_result(
#                     'BootstrapTimeout', 'failed', error_msg, errors=[error_msg]
#                 )
#             raise BootstrapTimeoutError(error_msg)
#         except Exception as e:
#             logger.critical(f'Unhandled exception during bootstrap execution: {e}', exc_info=True)
#             return await self._handle_bootstrap_failure(e)

#     async def _execute_bootstrap_process(self) -> BootstrapResult:
#         global_config = await self._load_global_configuration()
#         context = await self._create_bootstrap_context(global_config)
        
#         logger.info(f'Effective Strict Mode: {context.strict_mode}')
#         logger.info(f'Health Checks Enabled: {self.execution_config.enable_health_checks}')
        
#         await self._signal_bootstrap_started(context)
#         await self._execute_phases(context)
        
#         if self.execution_config.enable_health_checks:
#             await self._perform_final_health_checks(context)
            
#         await self._signal_bootstrap_completion(context)
#         return self._build_result(context)

#     async def _load_global_configuration(self) -> Dict[str, Any]:
#         try:
#             logger.debug('Loading global configuration...')
#             try:
#                 from configs.config_loader import ConfigLoader
#             except ImportError:
#                 logger.warning('ConfigLoader not available, using minimal config.')
#                 return self._get_minimal_global_config()

#             config_loader = ConfigLoader()
#             loaded_global_config = await config_loader.load_global_config(
#                 env=self.config.env, 
#                 provided_config=self.config.global_app_config 
#             )
            
#             if self.config.global_app_config is None or self.config.global_app_config != loaded_global_config:
#                 if dataclasses.is_dataclass(self.config):
#                     self.config = dataclasses.replace(self.config, global_app_config=loaded_global_config)
#                 else: 
#                     self.config.global_app_config = loaded_global_config

#             logger.info(f"Global configuration loaded for env: {self.config.env or 'default'}")
#             logger.debug(f'Config keys: {list(loaded_global_config.keys())}')
#             return loaded_global_config
#         except Exception as e:
#             logger.error(f'Failed to load global configuration: {e}', exc_info=True)
#             logger.warning('Using minimal configuration due to load failure.')
#             minimal_cfg = self._get_minimal_global_config()
#             if dataclasses.is_dataclass(self.config):
#                 self.config = dataclasses.replace(self.config, global_app_config=minimal_cfg)
#             else:
#                 self.config.global_app_config = minimal_cfg
#             return minimal_cfg

#     def _get_minimal_global_config(self) -> Dict[str, Any]:
#         return {
#             'env': self.config.env or 'default',
#             'bootstrap_strict_mode': self.config.initial_strict_mode_param, 
#             'feature_flags': {
#                 'enable_rbac_bootstrap': False,
#                 'enable_schema_validation': False,
#                 'enable_concurrent_initialization': False,
#             },
#             'llm': {'default_model': 'placeholder', 'timeout_seconds': 30},
#             'embedding': {'default_model': 'placeholder', 'dimensions': 384},
#             'shared_services': {},
#             'mechanisms': {},
#             'observers': {}
#         }

#     async def _create_bootstrap_context(self, global_config: Dict[str, Any]) -> BootstrapContext:
#         try:
#             logger.debug('Creating bootstrap context...')
#             context = await create_bootstrap_context(
#                 run_id=self.run_id,
#                 config=self.config, 
#                 global_config=global_config, 
#             )
#             logger.info('Bootstrap context created successfully.')
#             return context
#         except Exception as e:
#             logger.error(f'Failed to create bootstrap context: {e}', exc_info=True)
#             raise BootstrapError(f'Context creation failed: {e}') from e

#     async def _signal_bootstrap_started(self, context: BootstrapContext) -> None:
#         try:
#             if hasattr(context, 'signal_emitter') and context.signal_emitter:
#                  await context.signal_emitter.emit_bootstrap_started()
#                  logger.debug('Bootstrap started signal emitted.')
#             else:
#                 logger.warning("Signal emitter not found on context. Cannot emit bootstrap_started signal.")
#         except Exception as e:
#             logger.warning(f'Failed to emit bootstrap started signal: {e}')
#             if context.strict_mode:
#                 raise BootstrapError(f'Failed to emit bootstrap started signal: {e}') from e

#     async def _execute_phases(self, context: BootstrapContext) -> None:
#         phases = self._get_phases() 
#         total_phases = len(phases)
#         logger.info(f'Executing {total_phases} bootstrap phases...')

#         for i, phase_instance in enumerate(phases, 1):
#             if phase_instance is None: 
#                 logger.warning(f'Phase {i} is None, skipping.')
#                 continue
            
#             phase_name = phase_instance.__class__.__name__
#             logger.info(f'Executing Phase {i}/{total_phases}: {phase_name}')
            
#             try:
#                 if hasattr(phase_instance, 'execute_with_hooks'):
#                     phase_task = phase_instance.execute_with_hooks(context)
#                 elif hasattr(phase_instance, 'execute'):
#                     phase_task = phase_instance.execute(context)
#                 else:
#                     logger.error(f"Phase {phase_name} has no execute or execute_with_hooks method.")
#                     if context.strict_mode:
#                         raise BootstrapError(f"Phase {phase_name} is not executable.")
#                     continue

#                 from bootstrap.phases.base_phase import PhaseResult 
#                 result: PhaseResult = await asyncio.wait_for(phase_task, timeout=self.execution_config.phase_timeout_seconds)
                
#                 if not result.success:
#                     error_msg = f"Phase {phase_name} failed: {'; '.join(result.errors)}"
#                     if context.strict_mode:
#                         raise BootstrapError(error_msg)
#                     else:
#                         logger.warning(f'{error_msg} (continuing in non-strict mode)')
#                 else:
#                     logger.info(f'✓ Phase {phase_name} completed successfully. Message: {result.message}')

#             except asyncio.TimeoutError:
#                 error_msg = f'Phase {phase_name} timed out after {self.execution_config.phase_timeout_seconds}s'
#                 logger.error(error_msg)
#                 if context.strict_mode:
#                     raise BootstrapError(error_msg) 
#                 else:
#                     logger.warning(f'{error_msg} (continuing in non-strict mode)')
#             except BootstrapError: 
#                 raise
#             except Exception as e: 
#                 error_msg = f'Unexpected error in {phase_name}: {e}'
#                 logger.error(error_msg, exc_info=True)
#                 if context.strict_mode:
#                     raise BootstrapError(error_msg) from e
#                 else:
#                     logger.warning(f'{error_msg} (continuing in non-strict mode)')

#     async def _perform_final_health_checks(self, context: BootstrapContext) -> None:
#         try:
#             logger.info('Performing final health checks...')
#             component_count = 0
#             if context.registry and hasattr(context.registry, 'list_components'):
#                  component_count = len(context.registry.list_components())
            
#             if component_count == 0:
#                 logger.warning('No components registered during bootstrap.')
#             else:
#                 logger.info(f'Registry contains {component_count} components.')

#             if hasattr(context, 'health_reporter') and context.health_reporter:
#                 health_summary = context.health_reporter.generate_summary()
#                 logger.info(f'Final Health Summary:\n{health_summary}')
#             else:
#                 logger.warning("Health reporter not available on context for final health checks.")
            
#             logger.info('Final health checks completed.')
#         except Exception as e:
#             logger.warning(f'Health checks failed: {e}', exc_info=True)
#             if context.strict_mode:
#                 raise BootstrapError(f'Final health checks failed: {e}') from e

#     async def _signal_bootstrap_completion(self, context: BootstrapContext) -> None:
#         try:
#             component_count = 0
#             if context.registry and hasattr(context.registry, 'list_components'):
#                  component_count = len(context.registry.list_components())
#             duration_seconds = self._get_elapsed_seconds()
            
#             if hasattr(context, 'signal_emitter') and context.signal_emitter:
#                 await context.signal_emitter.emit_bootstrap_completed(
#                     component_count=component_count,
#                     duration_seconds=duration_seconds
#                 )
#                 logger.info(f'Bootstrap completion signaled - {component_count} components, {duration_seconds:.2f}s duration.')
#             else:
#                  logger.warning("Signal emitter not found on context. Cannot emit bootstrap_completed signal.")
#         except Exception as e:
#             logger.warning(f'Failed to signal bootstrap completion: {e}')
#             if context.strict_mode:
#                 raise BootstrapError(f'Failed to signal completion: {e}') from e

#     def _build_result(self, context: BootstrapContext) -> BootstrapResult:
#         try:
#             logger.debug('Building bootstrap result...')
#             builder = BootstrapResultBuilder(context)
#             result = builder.build() 
            
#             if result.success:
#                 logger.info(f'✓ NIREON V4 Bootstrap Complete. Run ID: {result.run_id}. Components: {result.component_count}, Duration: {result.bootstrap_duration or 0:.2f}s')
#             else:
#                 logger.error(f'✗ NIREON V4 Bootstrap Failed. Run ID: {result.run_id}. Critical Failures: {result.critical_failure_count}')
#             return result
#         except Exception as e:
#             logger.error(f'Failed to build bootstrap result: {e}', exc_info=True)
#             return self._create_minimal_result(e, context_fallback=context)

#     async def _handle_bootstrap_failure(self, error: Exception) -> BootstrapResult:
#         logger.critical(f'Critical bootstrap failure: {error}', exc_info=True)
        
#         if self.config.effective_strict_mode:
#             if isinstance(error, (BootstrapError, BootstrapTimeoutError, BootstrapValidationError)):
#                 raise error
#             raise BootstrapError(f'Bootstrap system failure: {error}') from error
            
#         logger.warning('Attempting to create minimal result in non-strict mode due to failure...')
#         try:
#             return self._create_minimal_result(error)
#         except Exception as recovery_error:
#             logger.error(f'Recovery (minimal result creation) failed: {recovery_error}', exc_info=True)
#             final_error = BootstrapError(f'Complete bootstrap failure and recovery failed. Original error: {error}')
#             final_error.__cause__ = recovery_error 
#             raise final_error

#     def _create_minimal_result(self, original_error: Exception, context_fallback: Optional[BootstrapContext] = None) -> BootstrapResult:
#         logger.info(f"Creating minimal bootstrap result due to error: {original_error}")
#         registry = self.config.existing_registry or (context_fallback.registry if context_fallback else None) or ComponentRegistry()
        
#         health_reporter = None
#         if context_fallback and hasattr(context_fallback, 'health_reporter'):
#             health_reporter = context_fallback.health_reporter
        
#         if health_reporter is None:
#             try:
#                 from bootstrap.health.reporter import HealthReporter
#                 health_reporter = HealthReporter(registry)
#             except ImportError: 
#                 logger.error("Failed to import HealthReporter for minimal result.")
#                 health_reporter = None # type: ignore
        
#         if health_reporter and hasattr(health_reporter, 'add_phase_result'):
#             health_reporter.add_phase_result(
#                 'BootstrapFailure', 
#                 'failed', 
#                 f'Bootstrap failed critically: {original_error}', 
#                 errors=[str(original_error)]
#             )
#             if hasattr(health_reporter, 'mark_bootstrap_complete'):
#                  health_reporter.mark_bootstrap_complete()

#         validation_data = None
#         if context_fallback and hasattr(context_fallback, 'validation_data_store'):
#             validation_data = context_fallback.validation_data_store
        
#         if validation_data is None:
#             try:
#                 from bootstrap.validation_data import BootstrapValidationData
#                 validation_data = BootstrapValidationData(self.config.global_app_config or {}, self.run_id)
#             except ImportError:
#                 logger.error("Failed to import BootstrapValidationData for minimal result.")
#                 validation_data = None # type: ignore

#         return BootstrapResult(
#             registry=registry,
#             health_reporter=health_reporter,
#             validation_data=validation_data,
#             run_id=self.run_id,
#             bootstrap_duration=self._get_elapsed_seconds(),
#             global_config=self.config.global_app_config
#         )

#     def _get_phases(self) -> List[Any]: 
#         if self._phases is None:
#             phase_classes = [
#                 AbiogenesisPhase,
#                 ContextFormationPhase,
#                 RegistrySetupPhase,
#                 FactorySetupPhase,
#                 ManifestProcessingPhase,
#                 ComponentInitializationPhase,
#                 InterfaceValidationPhase,
#                 RBACSetupPhase,
#                 LateRebindingPhase, 
#             ]
            
#             self._phases = []
#             for phase_cls in phase_classes:
#                 if phase_cls is not None:
#                     try:
#                         # For AbiogenesisPhase, we need to handle the different constructor
#                         if phase_cls.__name__ == 'AbiogenesisPhase':
#                             # We'll modify this phase to work with the standard interface
#                             self._phases.append(phase_cls()) 
#                         else:
#                             self._phases.append(phase_cls()) 
#                     except Exception as e:
#                         logger.error(f"Failed to instantiate phase {phase_cls.__name__}: {e}", exc_info=True)
#                 else:
#                     logger.warning(f"A phase class was None during instantiation, likely due to import error.")
            
#             logger.info(f'Initialized {len(self._phases)} bootstrap phase instances.')
#         return self._phases

#     def _generate_run_id(self) -> str:
#         timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') 
#         process_id = os.getpid()
#         return f'nireon_bs_{timestamp}_{process_id}'

#     def _get_elapsed_seconds(self) -> float:
#         if self.start_time is None:
#             return 0.0
#         return (datetime.now(timezone.utc) - self.start_time).total_seconds()


# async def bootstrap_nireon_system(
#     config_paths: Sequence[Union[str, Path]], 
#     **kwargs: Any 
# ) -> BootstrapResult:
#     try:
#         bootstrap_config_fields = BootstrapConfig.__annotations__.keys() 
#         execution_config_fields = BootstrapExecutionConfig.model_fields.keys()

#         bs_config_kwargs = {k: v for k, v in kwargs.items() if k in bootstrap_config_fields and k != 'config_paths'}
#         exec_config_kwargs = {k: v for k, v in kwargs.items() if k in execution_config_fields}
        
#         config = BootstrapConfig.from_params(list(config_paths), **bs_config_kwargs)

#         execution_config = None
#         if exec_config_kwargs:
#             execution_config = BootstrapExecutionConfig(**exec_config_kwargs)
        
#         orchestrator = BootstrapOrchestrator(config, execution_config)
#         return await orchestrator.execute_bootstrap()
        
#     except ValidationError as e: 
#         logger.error(f'Bootstrap configuration validation failed: {e}', exc_info=True)
#         raise BootstrapValidationError(f'Bootstrap configuration validation failed: {e}') from e
#     except BootstrapError: 
#         raise
#     except Exception as e: 
#         logger.error(f'Unexpected error in bootstrap_nireon_system: {e}', exc_info=True)
#         raise BootstrapError(f'System bootstrap failed with an unexpected error: {e}') from e

# bootstrap = bootstrap_nireon_system 

# def bootstrap_sync(
#     config_paths: Sequence[Union[str, Path]], 
#     **kwargs: Any
# ) -> BootstrapResult:
#     logger.debug('Running V4 bootstrap in synchronous mode.')
#     current_loop: Optional[asyncio.AbstractEventLoop] = None
#     new_loop_created = False
#     try:
#         current_loop = asyncio.get_running_loop()
#     except RuntimeError: 
#         current_loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(current_loop)
#         new_loop_created = True
#         logger.debug('No running event loop, created a new one for sync bootstrap.')
    
#     try:
#         future = bootstrap_nireon_system(config_paths, **kwargs)
#         return current_loop.run_until_complete(future)
#     except Exception as e:
#         logger.error(f'Synchronous bootstrap failed: {e}', exc_info=True)
#         if not isinstance(e, BootstrapError):
#             raise BootstrapError(f'Synchronous bootstrap execution error: {e}') from e
#         raise
#     finally:
#         if new_loop_created and current_loop:
#             current_loop.close()
#             logger.debug('Closed the event loop created for sync bootstrap.')
#             try:
#                 if asyncio.get_event_loop() is current_loop: 
#                     asyncio.set_event_loop(None)
#             except RuntimeError: 
#                  pass


# __all__ = [
#     'BootstrapExecutionConfig', 
#     'BootstrapOrchestrator', 
#     'bootstrap_nireon_system', 
#     'bootstrap', 
#     'bootstrap_sync',
#     'smoke_test', 
#     'validate_bootstrap_config' 
# ]

# async def smoke_test() -> bool:
#     """A basic smoke test for the bootstrap system."""
#     logger.info("Executing bootstrap smoke test...")
#     try:
#         from tempfile import NamedTemporaryFile
#         with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_manifest:
#             tmp_manifest.write("version: 1.0\nmetadata:\n  name: SmokeTestManifest\ncomponents: []\n") # Ensure minimal valid structure
#             manifest_path = tmp_manifest.name
        
#         logger.debug(f"Smoke test using temporary manifest: {manifest_path}")
#         # Use fewer default services for smoke test if possible, by passing minimal global_app_config
#         minimal_global_cfg = {
#             'bootstrap_strict_mode': False, # Smoke test should be lenient
#             'feature_flags': {'enable_rbac_bootstrap': False} # Disable heavy features
#         }
#         result = await bootstrap_nireon_system([Path(manifest_path)], global_app_config=minimal_global_cfg, strict_mode=False)
#         os.remove(manifest_path) 

#         if result.success: # For an empty manifest, 0 components is expected
#             logger.info(f"Bootstrap smoke test passed. Components: {result.component_count}")
#             return True
#         else:
#             logger.error(f"Bootstrap smoke test failed. Success: {result.success}, Components: {result.component_count}, Failures: {result.critical_failure_count}")
#             return False
#     except Exception as e:
#         logger.error(f"Bootstrap smoke test encountered an exception: {e}", exc_info=True)
#         return False

# async def validate_bootstrap_config(config_paths: List[str]) -> Dict[str, Any]:
#     """Validates bootstrap configuration files."""
#     logger.info(f"Validating bootstrap configuration: {config_paths}")
#     errors: List[str] = []
#     warnings: List[str] = []
#     is_valid = True

#     if not config_paths:
#         errors.append("No configuration paths provided for validation.")
#         is_valid = False
#         return {"valid": is_valid, "errors": errors, "warnings": warnings} # Return early
    
#     from runtime.utils import load_yaml_robust # Keep import local if only used here
#     # Potentially import schema validation logic here if it becomes more complex
#     # from jsonschema import validate, ValidationError as SchemaError

#     for path_str in config_paths:
#         path = Path(path_str)
#         if not path.exists():
#             errors.append(f"Configuration file not found: {path}")
#             is_valid = False
#             continue
#         if not path.is_file():
#             errors.append(f"Configuration path is not a file: {path}")
#             is_valid = False
#             continue
#         if path.suffix.lower() not in ['.yaml', '.yml']:
#             warnings.append(f"File {path} does not have a .yaml/.yml extension, but attempting to parse.")
        
#         try:
#             data = load_yaml_robust(path)
#             if not isinstance(data, dict):
#                 errors.append(f"Manifest {path} does not load as a dictionary (root is not a map).")
#                 is_valid = False
#                 continue # Skip further checks on this file
            
#             # Basic structural checks (can be expanded with jsonschema)
#             if 'version' not in data: # Common top-level field
#                 warnings.append(f"Manifest {path} is missing a 'version' field.")
#             # Example: Check for 'components' or 'shared_services' which are common sections
#             # if not ('components' in data or 'shared_services' in data or 'mechanisms' in data):
#             #     warnings.append(f"Manifest {path} does not seem to contain typical component sections like 'components', 'shared_services', or 'mechanisms'.")

#             # TODO: Add jsonschema validation here if a schema is defined and available.
#             # try:
#             #     schema = _load_manifest_schema() # You'd need a helper to load the schema
#             #     if schema:
#             #         validate(instance=data, schema=schema)
#             # except SchemaError as se:
#             #     errors.append(f"Schema validation failed for {path}: {se.message} at path {'.'.join(map(str, se.path))}")
#             #     is_valid = False
#             # except Exception as se_other:
#             #     warnings.append(f"Could not perform schema validation for {path}: {se_other}")

#         except Exception as e: # Catch errors from load_yaml_robust or other parsing issues
#             errors.append(f"Error loading or parsing manifest {path}: {e}")
#             is_valid = False
            
#     return {"valid": is_valid, "errors": errors, "warnings": warnings}