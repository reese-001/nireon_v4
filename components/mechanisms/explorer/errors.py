# nireon_v4/components/mechanisms/explorer/errors.py
from enum import Enum

class ExplorerErrorCode(Enum):
    UNKNOWN_ERROR = "EXP_000"
    INITIALIZATION_FAILURE = "EXP_001"
    GATEWAY_UNAVAILABLE = "EXP_002"
    FRAME_FACTORY_UNAVAILABLE = "EXP_003"
    FRAME_CREATION_FAILED = "EXP_004"
    FRAME_UPDATE_FAILED = "EXP_005"
    LLM_REQUEST_FAILED = "EXP_006"  # General LLM failure
    LLM_BUDGET_EXCEEDED = "EXP_007"  # Specific budget error from Gateway
    LLM_TIMEOUT = "EXP_008"  # LLM call timeout
    LLM_RESPONSE_INVALID = "EXP_009"  # LLM response malformed or unusable
    EVENT_PUBLISH_FAILED = "EXP_010"
    EMBEDDING_REQUEST_FAILED = "EXP_011"  # If Explorer handles embeddings directly
    EMBEDDING_RESPONSE_TIMEOUT = "EXP_012"
    INVALID_INPUT_DATA = "EXP_013"
    CONFIGURATION_ERROR = "EXP_014"
    EXPLORER_PROCESSING_ERROR = "EXP_999"  # General processing error
    # Add more specific codes as needed

    def __str__(self):
        return self.value