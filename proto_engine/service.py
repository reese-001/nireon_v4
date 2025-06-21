from __future__ import annotations
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult, ComponentHealth
from application.services.frame_factory_service import FrameFactoryService
from application.services.budget_manager import BudgetManagerPort, BudgetExceededError
from signals.core import ProtoTaskSignal, ProtoResultSignal, ProtoErrorSignal, MathProtoResultSignal
from domain.proto import get_validator_for_dialect
from domain.proto.base_schema import ProtoBlock, ProtoMathBlock, AnyProtoBlock
from .config import ProtoEngineConfig
from .executors import DockerExecutor, SubprocessExecutor

if TYPE_CHECKING:
    from domain.context import NireonExecutionContext
    from domain.frames import Frame

logger = logging.getLogger(__name__)

PROTO_ENGINE_METADATA = ComponentMetadata(
    id='proto_engine_default',
    name='ProtoEngine',
    version='1.0.0',
    category='service_core',
    description='Domain-agnostic engine for executing declarative Proto blocks in secure, sandboxed environments.',
    epistemic_tags=['executor', 'sandbox', 'declarative_task_runner', 'proto_plane'],
    requires_initialize=True,
    dependencies={'FrameFactoryService': '*', 'BudgetManagerPort': '*'}
)

class ProtoEngine(NireonBaseComponent):
    METADATA_DEFINITION = PROTO_ENGINE_METADATA
    ConfigModel = ProtoEngineConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: Optional[ComponentMetadata] = None, **kwargs):
        super().__init__(config=config, metadata_definition=metadata_definition or self.METADATA_DEFINITION)
        self.proto_engine_cfg: ProtoEngineConfig = self.ConfigModel(**self.config)
        self._executor = self._initialize_executor()
        self.frame_factory: Optional[FrameFactoryService] = None
        self.budget_manager: Optional[BudgetManagerPort] = None
        logger.info(f"ProtoEngine '{self.component_id}' created in '{self.proto_engine_cfg.execution_mode}' mode.")

    def _initialize_executor(self):
        if self.proto_engine_cfg.execution_mode == 'docker':
            return DockerExecutor(self.proto_engine_cfg)
        return SubprocessExecutor(self.proto_engine_cfg)

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        context.logger.info(f"ProtoEngine '{self.component_id}' initializing.")
        self.frame_factory = context.component_registry.get_service_instance(FrameFactoryService)
        self.budget_manager = context.component_registry.get_service_instance(BudgetManagerPort)
        context.logger.info(f"ProtoEngine '{self.component_id}' linked to FrameFactory and BudgetManager.")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not isinstance(data, dict) or 'proto_block' not in data:
            return ProcessResult(success=False, message="Invalid input: expected dict with 'proto_block'.", component_id=self.component_id)

        proto_data = data['proto_block']
        dialect = proto_data.get('eidos', 'unknown')
        typed_proto = self._expand_proto_type(proto_data)

        if isinstance(typed_proto, str): # Error message was returned
            await self._emit_error_signal(proto_data.get('id', 'unknown'), dialect, 'validation', typed_proto, context)
            return ProcessResult(success=False, message=typed_proto, component_id=self.component_id)

        validator = get_validator_for_dialect(dialect)
        validation_errors = validator.validate(typed_proto)
        if validation_errors:
            error_msg = '; '.join(validation_errors)
            await self._emit_error_signal(typed_proto.id, dialect, 'validation', f'Validation failed: {error_msg}', context)
            return ProcessResult(success=False, message=f'Validation failed: {error_msg}', component_id=self.component_id)

        frame_id = context.metadata.get('current_frame_id')
        if frame_id and self.frame_factory and self.budget_manager:
            if not await self._enforce_budget(typed_proto, frame_id, context):
                return ProcessResult(success=False, message='Budget exceeded.', component_id=self.component_id)
        else:
            context.logger.warning('No frame_id or core services. Skipping budget enforcement for proto %s.', typed_proto.id)

        result_data = await self._executor.execute(typed_proto, context)

        if result_data.get('success'):
            await self._emit_result_signal(typed_proto, result_data, context)
            return ProcessResult(success=True, message=f'Successfully executed Proto block {typed_proto.id}', output_data=result_data, component_id=self.component_id)
        else:
            await self._emit_error_signal(typed_proto.id, dialect, 'execution', result_data.get('error', 'Unknown error'), context)
            return ProcessResult(success=False, message=f"Execution failed: {result_data.get('error')}", output_data=result_data, component_id=self.component_id)

    def _expand_proto_type(self, proto_data: Dict[str, Any]) -> Union[AnyProtoBlock, str]:
        dialect = proto_data.get('eidos')
        if not dialect:
            return "Validation Error: 'eidos' field is missing from Proto block."
        try:
            if dialect == 'math':
                return ProtoMathBlock(**proto_data)
            return ProtoBlock(**proto_data)
        except Exception as e:
            return f"Proto type expansion failed for dialect '{dialect}': {e}"

    async def _enforce_budget(self, proto: ProtoBlock, frame_id: str, context: NireonExecutionContext) -> bool:
        try:
            timeout = proto.limits.get('timeout_sec', self.proto_engine_cfg.default_timeout_sec)
            self.budget_manager.consume(f'{frame_id}:proto_executions', 1.0)
            self.budget_manager.consume(f'{frame_id}:proto_cpu_seconds', float(timeout))
            context.logger.info(f"Consumed budget for proto '{proto.id}' from frame '{frame_id}'.")
            return True
        except BudgetExceededError as e:
            error_msg = f"Budget exceeded for proto '{proto.id}' in frame '{frame_id}': {e}"
            context.logger.error(error_msg)
            await self._emit_error_signal(proto.id, proto.eidos, 'budget_exceeded', str(e), context)
            return False
        except KeyError:
            context.logger.debug(f"No proto budget defined for frame '{frame_id}'. Proceeding without check.")
            return True
        except Exception as e:
            context.logger.error(f"Unexpected error during budget enforcement for proto '{proto.id}': {e}")
            return False

    async def _emit_result_signal(self, proto: ProtoBlock, result: Dict[str, Any], context: NireonExecutionContext):
        bus = context.event_bus
        if not bus:
            return

        base_data = {
            'source_node_id': self.component_id,
            'proto_block_id': proto.id,
            'dialect': proto.eidos,
            'success': True,
            'result': result.get('result'),
            'artifacts': result.get('artifacts', []),
            'execution_time_sec': result.get('execution_time_sec', 0.0)
        }

        if proto.eidos == 'math' and isinstance(proto, ProtoMathBlock):
            # --- FIX: Extract numeric result if possible, otherwise pass None ---
            raw_result = result.get('result')
            numeric_res = None
            if isinstance(raw_result, (float, int)):
                numeric_res = float(raw_result)
            elif isinstance(raw_result, list) and all(isinstance(x, (float, int)) for x in raw_result):
                 numeric_res = [float(x) for x in raw_result]
            elif isinstance(raw_result, dict) and all(isinstance(v, (float, int)) for v in raw_result.values()):
                numeric_res = {k: float(v) for k,v in raw_result.items()}
            # --- END OF FIX ---
            
            signal = MathProtoResultSignal(
                **base_data, 
                equation_latex=proto.equation_latex, 
                numeric_result=numeric_res # Pass the extracted (or None) numeric result
            )
        else:
            signal = ProtoResultSignal(**base_data)

        await asyncio.to_thread(bus.publish, signal.signal_type, signal)


    async def _emit_error_signal(self, proto_id: str, dialect: str, error_type: str, msg: str, context: NireonExecutionContext):
        bus = context.event_bus
        if not bus:
            return
        signal = ProtoErrorSignal(
            source_node_id=self.component_id,
            proto_block_id=proto_id,
            dialect=dialect,
            error_type=error_type,
            error_message=msg
        )
        await asyncio.to_thread(bus.publish, signal.signal_type, signal)

class ProtoGateway(NireonBaseComponent):
    METADATA_DEFINITION = ComponentMetadata(
        id='proto_gateway_main',
        name='ProtoGateway',
        version='1.0.0',
        category='service_gateway',
        description='Routes Proto tasks to specialized engines.',
        epistemic_tags=['router', 'gateway', 'proto_plane'],
        requires_initialize=True
    )

    def __init__(self, config: Dict[str, Any], metadata_definition: Optional[ComponentMetadata]=None, **kwargs):
        super().__init__(config, metadata_definition or self.METADATA_DEFINITION)
        self.dialect_map = self.config.get('dialect_map', {})
        logger.info(f"ProtoGateway '{self.component_id}' configured with dialect map: {self.dialect_map}")

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        signal: ProtoTaskSignal | None = data.get('signal')

        if not isinstance(signal, ProtoTaskSignal):
            return ProcessResult(success=False, message=f'ProtoGateway only processes ProtoTaskSignal, got {type(data).__name__}.', component_id=self.component_id)

        proto = signal.proto_block
        dialect = proto.get('eidos', 'unknown')
        component_id_to_trigger = self.dialect_map.get(dialect)
        
        if not component_id_to_trigger:
            msg = f"No engine configured for dialect '{dialect}'."
            await self._emit_error_signal(proto.get('id', 'unknown'), dialect, 'routing_error', msg, context)
            return ProcessResult(success=False, message=msg, component_id=self.component_id)

        try:
            engine = context.component_registry.get(component_id_to_trigger)
            engine_context = context.with_component_scope(component_id_to_trigger)
            
            if hasattr(signal, 'context_tags'):
                frame_id = signal.context_tags.get('frame_id')
                if frame_id:
                    engine_context = engine_context.with_metadata(current_frame_id=frame_id)
            
            engine_input_payload = {'proto_block': proto, 'dialect': dialect}
            return await engine.process(engine_input_payload, engine_context)
            
        except Exception as e:
            msg = f"Failed to route or process task for dialect '{dialect}': {e}"
            logger.error(f'Error in ProtoGateway: {msg}', exc_info=True)
            await self._emit_error_signal(proto.get('id', 'unknown'), dialect, 'execution_error', msg, context)
            return ProcessResult(success=False, message=msg, component_id=self.component_id)

    async def _emit_error_signal(self, proto_id: str, dialect: str, error_type: str, msg: str, context: NireonExecutionContext):
        bus = context.event_bus
        if not bus:
            return
        signal = ProtoErrorSignal(
            source_node_id=self.component_id,
            proto_block_id=proto_id,
            dialect=dialect,
            error_type=error_type,
            error_message=msg
        )
        await asyncio.to_thread(bus.publish, signal.signal_type, signal.model_dump())