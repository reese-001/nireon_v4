# nireon_v4/application/services/frame_factory_service.py

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

# Core Nireon imports that FrameFactoryService actually needs
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ComponentHealth, ProcessResult
from domain.context import NireonExecutionContext
from domain.frames import Frame

logger = logging.getLogger(__name__)

# --- Metadata for THIS service ---
FRAME_FACTORY_SERVICE_METADATA = ComponentMetadata(
    id='frame_factory_service',
    name='Frame Factory Service',
    version='1.1.0',
    category='service_core',
    description='Service for creating, managing, and retrieving Frames for epistemic operations.',
    epistemic_tags=['context_manager', 'frame_orchestrator'],
    capabilities={'create_frames', 'manage_frame_lifecycle', 'retrieve_frames'},
    accepts=['FRAME_CREATION_REQUEST', 'FRAME_UPDATE_REQUEST'],
    produces=['FRAME_CREATED', 'FRAME_UPDATED'],
    requires_initialize=True
)

# --- Configuration model for THIS service ---
class FrameFactoryConfig(BaseModel):
    """Configuration model for FrameFactoryService with validation."""
    root_frame_id: str = Field(
        default=f"F-ROOT-{uuid.UUID('00000000-0000-0000-0000-000000000000')}", 
        description="Static ID for the root frame."
    )
    root_frame_name: str = Field(
        default="NIREON_ROOT_FRAME", 
        description="Name of the root frame."
    )
    root_frame_owner_agent_id: str = Field(
        default="system_bootstrap_agent", 
        description="Owner agent ID for the root frame."
    )
    log_frame_operations: bool = Field(
        default=True, 
        description="Log detailed frame operations."
    )
    max_frames: Optional[int] = Field(
        default=10000, 
        description="Maximum number of frames to maintain (for memory management)."
    )
    enable_frame_expiration: bool = Field(
        default=True, 
        description="Enable automatic frame expiration handling."
    )
    cleanup_interval_seconds: int = Field(
        default=300, 
        description="Interval for cleanup operations in seconds."
    )

    class Config:
        extra = "allow"  # Allow additional configuration parameters

# --- Custom Exceptions for THIS service ---
class FrameNotFoundError(Exception):
    """Raised when a requested frame cannot be found."""
    pass

class FrameOperationError(Exception):
    """Raised when a frame operation fails due to business logic constraints."""
    pass

class FrameConfigurationError(Exception):
    """Raised when frame configuration is invalid."""
    pass

# --- The Service Class itself ---
class FrameFactoryService(NireonBaseComponent):
    """
    Service for creating, managing, and retrieving Frames for epistemic operations.
    
    This service provides centralized frame lifecycle management including:
    - Creating new frames with proper inheritance from parent frames
    - Managing frame status and metadata
    - Providing frame retrieval and search capabilities
    - Maintaining the root frame for the system
    """
    
    ConfigModel = FrameFactoryConfig

    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata = FRAME_FACTORY_SERVICE_METADATA):
        super().__init__(config=config, metadata_definition=metadata_definition)
        
        try:
            self.cfg: FrameFactoryConfig = FrameFactoryConfig(**self.config)
        except ValidationError as e:
            logger.error(f"Invalid FrameFactoryService configuration: {e}")
            # Fall back to default configuration
            self.cfg = FrameFactoryConfig()
            
        self._frames: Dict[str, Frame] = {}
        self._root_frame_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._is_initialized = False
        
        logger.info(f"FrameFactoryService '{self.component_id}' created with config: {self.cfg.model_dump()}")

    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        """Initialize the FrameFactoryService and create the root frame."""
        if self._is_initialized:
            context.logger.warning(f"FrameFactoryService '{self.component_id}' already initialized.")
            return
            
        context.logger.info(f"Initializing FrameFactoryService '{self.component_id}'.")
        
        try:
            async with self._lock:
                await self._create_root_frame_internal(context)
                self._is_initialized = True
                
            context.logger.info(
                f"FrameFactoryService '{self.component_id}' initialized successfully. "
                f"Root Frame ID: {self._root_frame_id}"
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize FrameFactoryService '{self.component_id}': {e}"
            context.logger.error(error_msg, exc_info=True)
            raise FrameConfigurationError(error_msg)

    def _generate_frame_id(self) -> str:
        """Generate a unique frame ID."""
        return f"F-{uuid.uuid4()}"

    async def _create_root_frame_internal(self, context: NireonExecutionContext) -> None:
        """Create the root frame if it doesn't exist."""
        if self.cfg.root_frame_id in self._frames:
            self._root_frame_id = self.cfg.root_frame_id
            context.logger.info(f"Root frame '{self._root_frame_id}' already exists.")
            return

        try:
            root_frame = Frame(
                id=self.cfg.root_frame_id,
                name=self.cfg.root_frame_name,
                description="The foundational root frame for all NIREON operations.",
                owner_agent_id=self.cfg.root_frame_owner_agent_id,
                parent_frame_id=None,
                epistemic_goals=["SYSTEM_STABILITY", "GLOBAL_CONTEXT"],
                trust_basis={"system_core": 1.0},
                llm_policy={
                    'default_temperature': 0.5, 
                    'preferred_route': 'default_balanced'
                },
                resource_budget=None,
                domain_rules=[],
                context_tags={
                    "scope": "global", 
                    "type": "system_root",
                    "created_by": "frame_factory_service"
                },
                status="active",
                created_ts=time.time(),
                updated_ts=time.time(),
                expires_at=None,
                schema_version="1.0"
            )
            
            self._frames[root_frame.id] = root_frame
            self._root_frame_id = root_frame.id
            
            if self.cfg.log_frame_operations:
                context.logger.info(f"Root Frame '{root_frame.id}' created and stored successfully.")
                
        except Exception as e:
            error_msg = f"Failed to create root frame: {e}"
            context.logger.error(error_msg, exc_info=True)
            raise FrameConfigurationError(error_msg)

    async def create_frame(
        self,
        context: NireonExecutionContext,
        name: str,
        owner_agent_id: str,
        description: str,
        parent_frame_id: Optional[str] = None,
        epistemic_goals: Optional[List[str]] = None,
        trust_basis: Optional[Dict[str, float]] = None,
        llm_policy: Optional[Dict[str, Any]] = None,
        resource_budget: Optional[Dict[str, Any]] = None,
        domain_rules: Optional[List[Any]] = None,
        context_tags: Optional[Dict[str, Any]] = None,
        initial_status: str = "active",
        expires_at: Optional[float] = None,
        schema_version: str = "1.0"
    ) -> Frame:
        """
        Create a new frame with the specified parameters.
        
        Args:
            context: Execution context for logging and operations
            name: Human-readable name for the frame
            owner_agent_id: ID of the agent that owns this frame
            description: Description of the frame's purpose
            parent_frame_id: Optional parent frame ID for inheritance
            epistemic_goals: List of epistemic goals for the frame
            trust_basis: Trust relationships for the frame
            llm_policy: LLM routing and parameter policies
            resource_budget: Resource constraints and budgets
            domain_rules: Domain-specific rules and constraints
            context_tags: Additional metadata tags
            initial_status: Initial status (default: "active")
            expires_at: Optional expiration timestamp
            schema_version: Frame schema version
            
        Returns:
            Created Frame object
            
        Raises:
            FrameNotFoundError: If parent frame doesn't exist
            FrameOperationError: If frame creation fails
        """
        if not self._is_initialized:
            raise FrameOperationError("FrameFactoryService not initialized")
            
        async with self._lock:
            # Check memory limits
            if self.cfg.max_frames and len(self._frames) >= self.cfg.max_frames:
                await self._cleanup_expired_frames(context)
                if len(self._frames) >= self.cfg.max_frames:
                    raise FrameOperationError(f"Maximum frame limit ({self.cfg.max_frames}) reached")
            
            # Validate parent frame if specified
            if parent_frame_id and parent_frame_id not in self._frames:
                raise FrameNotFoundError(f"Parent frame with ID '{parent_frame_id}' not found.")

            frame_id = self._generate_frame_id()
            current_ts = time.time()

            # Prepare context tags with service metadata
            final_context_tags = context_tags or {}
            final_context_tags.update({
                "created_by": "frame_factory_service",
                "service_version": self.metadata.version
            })

            try:
                new_frame = Frame(
                    id=frame_id,
                    name=name,
                    description=description,
                    owner_agent_id=owner_agent_id,
                    parent_frame_id=parent_frame_id,
                    epistemic_goals=epistemic_goals or [],
                    trust_basis=trust_basis or {},
                    llm_policy=llm_policy or {},
                    resource_budget=resource_budget,
                    domain_rules=domain_rules or [],
                    context_tags=final_context_tags,
                    status=initial_status,
                    created_ts=current_ts,
                    updated_ts=current_ts,
                    expires_at=expires_at,
                    schema_version=schema_version
                )
                
                self._frames[new_frame.id] = new_frame
                
            except Exception as e:
                error_msg = f"Failed to create frame: {e}"
                context.logger.error(error_msg, exc_info=True)
                raise FrameOperationError(error_msg)

        if self.cfg.log_frame_operations:
            context.logger.info(
                f"Frame '{new_frame.id}' (Name: {name}) created by agent '{owner_agent_id}'. "
                f"Parent: {parent_frame_id or 'None'}."
            )
            
        return new_frame

    async def get_frame_by_id(self, context: NireonExecutionContext, frame_id: str) -> Optional[Frame]:
        """
        Retrieve a frame by its ID.
        
        Args:
            context: Execution context
            frame_id: ID of the frame to retrieve
            
        Returns:
            Frame object if found, None otherwise
        """
        async with self._lock:
            frame = self._frames.get(frame_id)

        if self.cfg.log_frame_operations:
            if frame:
                context.logger.debug(f"Frame '{frame_id}' retrieved successfully.")
            else:
                context.logger.debug(f"Frame '{frame_id}' not found.")
                
        return frame

    async def get_or_create_frame(
        self,
        context: NireonExecutionContext,
        *,
        name: str,
        owner_agent_id: str,
        description: str,
        parent_frame_id: Optional[str] = None,
        epistemic_goals: Optional[List[str]] = None,
        trust_basis: Optional[Dict[str, float]] = None,
        llm_policy: Optional[Dict[str, Any]] = None,
        resource_budget: Optional[Dict[str, Any]] = None,
        domain_rules: Optional[List[Any]] = None,
        context_tags: Optional[Dict[str, Any]] = None,
        initial_status: str = "active",
        expires_at: Optional[float] = None,
        schema_version: str = "1.0"
    ) -> Frame:
        """
        Get an existing frame or create a new one if no matching frame exists.
        
        Searches for an active frame with matching name, owner, and parent.
        If found, returns the existing frame. Otherwise, creates a new frame.
        """
        async with self._lock:
            # Search for existing matching frame
            for f_id, f_obj in self._frames.items():
                if (f_obj.name == name and
                    f_obj.owner_agent_id == owner_agent_id and
                    f_obj.parent_frame_id == parent_frame_id and
                    f_obj.status == "active"):
                    if self.cfg.log_frame_operations:
                        context.logger.debug(
                            f"Found existing active frame '{f_id}' matching criteria for 'get_or_create'."
                        )
                    return f_obj

        if self.cfg.log_frame_operations:
            context.logger.debug(
                f"No existing active frame found for name='{name}', owner='{owner_agent_id}'. "
                "Creating new frame."
            )
            
        return await self.create_frame(
            context, name=name, owner_agent_id=owner_agent_id, description=description,
            parent_frame_id=parent_frame_id, epistemic_goals=epistemic_goals,
            trust_basis=trust_basis, llm_policy=llm_policy, resource_budget=resource_budget,
            domain_rules=domain_rules, context_tags=context_tags, initial_status=initial_status,
            expires_at=expires_at, schema_version=schema_version
        )

    async def spawn_sub_frame(
        self,
        context: NireonExecutionContext,
        parent_frame_id: str,
        name_suffix: str,
        description: str,
        goal_overrides: Optional[List[str]] = None,
        trust_basis_updates: Optional[Dict[str, float]] = None,
        llm_policy_overrides: Optional[Dict[str, Any]] = None,
        resource_budget_overrides: Optional[Dict[str, Any]] = None,
        context_tag_updates: Optional[Dict[str, Any]] = None,
        initial_status: str = "active"
    ) -> Frame:
        """
        Create a sub-frame that inherits properties from a parent frame.
        
        Args:
            context: Execution context
            parent_frame_id: ID of the parent frame
            name_suffix: Suffix to append to parent frame name
            description: Description for the new sub-frame
            goal_overrides: Override epistemic goals (None to inherit)
            trust_basis_updates: Updates to apply to inherited trust basis
            llm_policy_overrides: Overrides for LLM policy
            resource_budget_overrides: Overrides for resource budget
            context_tag_updates: Updates to apply to context tags
            initial_status: Initial status for the sub-frame
            
        Returns:
            Created sub-frame
            
        Raises:
            FrameNotFoundError: If parent frame doesn't exist
            FrameOperationError: If parent frame is not active
        """
        parent_frame = await self.get_frame_by_id(context, parent_frame_id)
        if not parent_frame:
            raise FrameNotFoundError(
                f"Parent frame with ID '{parent_frame_id}' not found for spawning sub-frame."
            )

        if not parent_frame.is_active():
            raise FrameOperationError(
                f"Cannot spawn sub-frame from parent '{parent_frame_id}' with status "
                f"'{parent_frame.status}'. Parent must be active."
            )

        # Inherit and override properties
        new_name = f"{parent_frame.name}_{name_suffix}"
        epistemic_goals = goal_overrides if goal_overrides is not None else parent_frame.epistemic_goals.copy()
        
        trust_basis = parent_frame.trust_basis.copy()
        if trust_basis_updates:
            trust_basis.update(trust_basis_updates)
            
        llm_policy = parent_frame.llm_policy.copy()
        if llm_policy_overrides:
            llm_policy.update(llm_policy_overrides)
            
        resource_budget = parent_frame.resource_budget.copy() if parent_frame.resource_budget is not None else {}
        if resource_budget_overrides:
            resource_budget.update(resource_budget_overrides)
            
        context_tags = parent_frame.context_tags.copy()
        context_tags.update({"inherited_from": parent_frame_id})
        if context_tag_updates:
            context_tags.update(context_tag_updates)

        return await self.create_frame(
            context=context,
            name=new_name,
            owner_agent_id=parent_frame.owner_agent_id,
            description=description,
            parent_frame_id=parent_frame_id,
            epistemic_goals=epistemic_goals,
            trust_basis=trust_basis,
            llm_policy=llm_policy,
            resource_budget=resource_budget if resource_budget else None,
            domain_rules=parent_frame.domain_rules.copy(),
            context_tags=context_tags,
            initial_status=initial_status,
            schema_version=parent_frame.schema_version
        )

    async def update_frame_status(self, context: NireonExecutionContext, frame_id: str, new_status: str) -> bool:
        """
        Update the status of a frame.
        
        Args:
            context: Execution context
            frame_id: ID of the frame to update
            new_status: New status to set
            
        Returns:
            True if update was successful
            
        Raises:
            FrameNotFoundError: If frame doesn't exist
        """
        async with self._lock:
            frame = self._frames.get(frame_id)
            if not frame:
                raise FrameNotFoundError(f"Frame with ID '{frame_id}' not found for status update.")
            frame.update_status(new_status)

        if self.cfg.log_frame_operations:
            context.logger.info(f"Frame '{frame_id}' status updated to '{new_status}'.")
        return True

    async def archive_frame(self, context: NireonExecutionContext, frame_id: str) -> bool:
        """
        Archive a frame by setting its status to 'archived'.
        
        Args:
            context: Execution context
            frame_id: ID of the frame to archive
            
        Returns:
            True if archiving was successful, False if frame doesn't exist
        """
        try:
            return await self.update_frame_status(context, frame_id, "archived")
        except FrameNotFoundError:
            context.logger.warning(f"Attempted to archive non-existent frame '{frame_id}'.")
            return False

    async def get_active_frames_by_owner(self, context: NireonExecutionContext, owner_agent_id: str) -> List[Frame]:
        """
        Get all active frames owned by a specific agent.
        
        Args:
            context: Execution context
            owner_agent_id: ID of the owning agent
            
        Returns:
            List of active frames owned by the agent
        """
        async with self._lock:
            active_frames = [
                frame for frame in self._frames.values()
                if frame.owner_agent_id == owner_agent_id and frame.is_active()
            ]
            
        if self.cfg.log_frame_operations:
            context.logger.debug(f"Found {len(active_frames)} active frames for owner '{owner_agent_id}'.")
        return active_frames

    async def get_root_frame(self, context: NireonExecutionContext) -> Optional[Frame]:
        """
        Get the root frame for the system.
        
        Returns:
            Root frame if it exists, None otherwise
        """
        if self._root_frame_id:
            return await self.get_frame_by_id(context, self._root_frame_id)
        return None

    async def _cleanup_expired_frames(self, context: NireonExecutionContext) -> int:
        """
        Clean up expired frames to manage memory usage.
        
        Returns:
            Number of frames cleaned up
        """
        if not self.cfg.enable_frame_expiration:
            return 0
            
        current_time = time.time()
        cleaned_count = 0
        
        async with self._lock:
            expired_frame_ids = [
                frame_id for frame_id, frame in self._frames.items()
                if frame.expires_at is not None and frame.expires_at < current_time
            ]
            
            for frame_id in expired_frame_ids:
                if frame_id != self._root_frame_id:  # Never clean up root frame
                    del self._frames[frame_id]
                    cleaned_count += 1
                    
        if cleaned_count > 0 and self.cfg.log_frame_operations:
            context.logger.info(f"Cleaned up {cleaned_count} expired frames.")
            
        return cleaned_count

    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        """
        Process implementation for NireonBaseComponent interface.
        
        FrameFactoryService primarily provides an API rather than processing data.
        """
        context.logger.debug(
            f"FrameFactoryService '{self.component_id}' received process call, "
            "but it primarily provides an API."
        )
        return ProcessResult(
            success=True,
            component_id=self.component_id,
            message="FrameFactoryService API is available. Use specific methods like create_frame, get_frame_by_id, etc."
        )

    async def health_check(self, context: NireonExecutionContext) -> ComponentHealth:
        """
        Perform a health check on the FrameFactoryService.
        
        Returns:
            ComponentHealth object with current status
        """
        async with self._lock:
            root_frame_status = "UNKNOWN"
            if self._root_frame_id and self._root_frame_id in self._frames:
                root_frame_status = self._frames[self._root_frame_id].status

            details = {
                "total_frames_managed": len(self._frames),
                "root_frame_id": self._root_frame_id,
                "root_frame_status": root_frame_status,
                "is_initialized": self._is_initialized,
                "config": self.cfg.model_dump()
            }

        status = "HEALTHY"
        message = "FrameFactoryService operating normally."

        if not self._is_initialized:
            status = "UNHEALTHY"
            message = "FrameFactoryService not initialized."
        elif not self._root_frame_id or self._root_frame_id not in self._frames:
            status = "DEGRADED"
            message = "Root frame is missing or not initialized."
        elif root_frame_status != "active":
            status = "DEGRADED"
            message = f"Root frame status is '{root_frame_status}', expected 'active'."

        return ComponentHealth(
            component_id=self.component_id,
            status=status,
            message=message,
            details=details
        )