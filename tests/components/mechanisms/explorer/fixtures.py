# tests/components/mechanisms/explorer/fixtures.py
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from domain.ports.mechanism_gateway_port import MechanismGatewayPort
from application.services.frame_factory_service import FrameFactoryService, Frame
from domain.context import NireonExecutionContext
from core.registry import ComponentRegistry
from core.lifecycle import ComponentMetadata
from components.mechanisms.explorer.config import ExplorerConfig
from components.mechanisms.explorer.service import ExplorerMechanism, EXPLORER_METADATA
from domain.ports.llm_port import LLMResponse


@pytest.fixture
def mock_nireon_execution_context():
    registry = ComponentRegistry() # A real registry for context, can be enhanced
    logger_mock = MagicMock(spec=logging.Logger)
    for level in ['info', 'debug', 'warning', 'error', 'critical']:
        setattr(logger_mock, level, MagicMock())

    context = NireonExecutionContext(
        run_id="test_run_123",
        component_id="test_explorer_component",
        component_registry=registry,
        event_bus=AsyncMock(), # Mock event bus
        logger=logger_mock,
        config={},
        feature_flags={"test_mode": True}
    )
    return context

@pytest.fixture
def mock_gateway_port():
    mock = AsyncMock(spec=MechanismGatewayPort)
    # Default behavior for process_cognitive_event if it's an LLM_ASK
    async def default_process_ce(ce):
        if ce.service_call_type == 'LLM_ASK':
            return LLMResponse({"text": f"Mock LLM response to: {ce.payload.prompt[:20]}..."})
        elif ce.service_call_type == 'EVENT_PUBLISH':
            return {"status": "event_published_mock", "event_type": ce.payload.get("event_type")}
        return None # Should not happen for valid CE types Explorer uses
    
    mock.process_cognitive_event = AsyncMock(side_effect=default_process_ce)
    
    # Add component_id and metadata to mock gateway if Explorer uses it
    mock.component_id = "mock_gateway_01"
    mock.metadata = ComponentMetadata(id="mock_gateway_01", name="MockGateway", version="1.0", category="gateway_mock")
    return mock

@pytest.fixture
def mock_frame_factory_service():
    mock = AsyncMock(spec=FrameFactoryService)
    
    async def create_frame_mock(context, name, owner_agent_id, **kwargs):
        frame_id = f"frame_{shortuuid.uuid()[:6]}"
        mock_frame = Frame(
            id=frame_id,
            name=name,
            owner_agent_id=owner_agent_id,
            description=kwargs.get("description", "Mocked frame"),
            created_ts=time.time(),
            updated_ts=time.time(),
            status=kwargs.get("initial_status", "active"),
            # Populate other fields as needed for tests
            epistemic_goals=kwargs.get("epistemic_goals", []),
            llm_policy=kwargs.get("llm_policy", {}),
            context_tags=kwargs.get("context_tags", {"audit_trail":[]}) # Ensure audit_trail exists
        )
        # Simulate frame providing an RNG
        mock_frame.get_rng = MagicMock(return_value=random.Random(12345)) # Predictable RNG
        return mock_frame

    mock.create_frame = AsyncMock(side_effect=create_frame_mock)
    mock.update_frame_status = AsyncMock(return_value=True)
    mock.update_frame_data = AsyncMock(return_value=True) # For updating context_tags
    mock.get_frame_by_id = AsyncMock(return_value=None) # Default to not found
    return mock

@pytest.fixture
def default_explorer_config_dict():
    return ExplorerConfig().model_dump()

@pytest.fixture
def explorer_mechanism(default_explorer_config_dict, mock_gateway_port, mock_frame_factory_service, mock_nireon_execution_context):
    # Ensure dependencies are registered for Explorer's initialize to find
    mock_nireon_execution_context.component_registry.register_service_instance(MechanismGatewayPort, mock_gateway_port)
    mock_nireon_execution_context.component_registry.register_service_instance(FrameFactoryService, mock_frame_factory_service)
    
    explorer = ExplorerMechanism(
        config=default_explorer_config_dict,
        metadata_definition=EXPLORER_METADATA, # Use the actual metadata
        gateway=mock_gateway_port, # Can be pre-injected or resolved via registry
        frame_factory=mock_frame_factory_service
    )
    # Manually run initialize if tests depend on it (or let individual tests do it)
    # await explorer.initialize(mock_nireon_execution_context) # This needs to be async if called here
    return explorer

# Need to import these for the fixtures
import logging
import shortuuid
import time
import random