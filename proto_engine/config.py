from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import re

class ProtoEngineConfig(BaseModel):
    class Config:
        extra = 'forbid'
    execution_mode: str = Field(default='docker', description="Primary execution mode: 'docker' for sandboxing (recommended) or 'subprocess' for local testing.")
    python_executable: str = Field(default=sys.executable, description="Python executable path for 'subprocess' mode. Defaults to the current Python executable.")
    docker_image_prefix: str = Field(default='nireon-proto', description="Prefix for Docker images (e.g., 'nireon-proto-math').")
    work_directory: str = Field(default='runtime/proto/workspace', description='Base working directory for temporary execution files.')
    artifacts_directory: str = Field(default='runtime/proto/artifacts', description='Directory for storing persistent output artifacts.')
    default_timeout_sec: int = Field(default=10, ge=1, le=30, description='Default execution timeout in seconds.')
    default_memory_mb: int = Field(default=256, ge=64, le=1024, description='Default memory limit in megabytes.')
    max_file_size_mb: int = Field(default=10, ge=1, le=50, description='Maximum size for any single artifact file.')
    dialect_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description='Container for dialect-specific configurations.')
    cleanup_after_execution: bool = Field(default=True, description='If true, delete the temporary workspace directory after execution.')
    retain_artifacts_hours: int = Field(default=24, ge=1, le=168, description='How long to retain generated artifacts.')

    @field_validator('docker_image_prefix')
    @classmethod
    def validate_docker_prefix(cls, v: str) -> str:
        if not v:
            raise ValueError('docker_image_prefix cannot be empty.')
        if v.startswith('-') or v.startswith('_') or v.endswith('-') or v.endswith('_'):
            raise ValueError(f"Invalid docker_image_prefix '{v}': cannot start or end with a separator.")
        if not re.match(r'^[a-z0-9]+([._-][a-z0-9]+)*$', v):
            raise ValueError(f"Invalid docker_image_prefix '{v}': must be lowercase alphanumeric with separators.")
        return v

    @field_validator('work_directory', 'artifacts_directory')
    @classmethod
    def ensure_directory_exists(cls, v: str) -> str:
        # --- FIX: Clean up potential leading path separators from env var defaults ---
        clean_v = v.strip().lstrip('/\\-')
        Path(clean_v).mkdir(parents=True, exist_ok=True)
        return clean_v
        # --- END OF FIX ---

    def get_dialect_config(self, dialect: str) -> Dict[str, Any]:
        return self.dialect_configs.get(dialect, {})

class ProtoMathEngineConfig(ProtoEngineConfig):
    allowed_packages: List[str] = Field(default=['matplotlib', 'numpy', 'pandas', 'scipy', 'sympy', 'seaborn', 'plotly', 'statsmodels'])
    enable_latex_rendering: bool = Field(default=True)
    plot_dpi: int = Field(default=150, ge=72, le=300)

    def __init__(self, **data):
        super().__init__(**data)
        self.dialect_configs['math'] = {
            'allowed_packages': self.allowed_packages,
            'enable_latex': self.enable_latex_rendering,
            'plot_dpi': self.plot_dpi
        }