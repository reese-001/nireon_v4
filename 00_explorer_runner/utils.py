# nireon_v4/00_explorer_runner/utils.py

import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

def find_project_root(markers: List[str] = ['bootstrap', 'domain', 'core', 'configs']) -> Optional[Path]:
    """Find the project root by looking for marker directories.
    
    Returns a Path object, not a string.
    """
    # Start from the script location
    here = Path(__file__).resolve().parent
    
    # When running as a module, we might already be in the project root
    # Check current working directory first
    cwd = Path.cwd()
    if all((cwd / m).is_dir() for m in markers):
        return cwd
    
    # Otherwise, traverse up from the script location
    # Go up one level since we're in 00_explorer_runner
    start_path = here.parent if here.name == '00_explorer_runner' else here
    
    for candidate in [start_path, *start_path.parents]:
        if all((candidate / m).is_dir() for m in markers):
            return candidate
    
    return None

def setup_logging(debug_config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)-30s - %(levelname)-8s - [%(component_id)s] - %(message)s'
    
    log_level = getattr(logging, debug_config.get('log_level', 'INFO'))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%H:%M:%S'
    )
    
    # Quiet specific loggers
    for logger_name in debug_config.get('quiet_loggers', []):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Setup component logging
    _setup_component_logging()
    
    return logging.getLogger('explorer')

def _setup_component_logging():
    """Setup component ID tracking in log records"""
    _default_record_factory = logging.getLogRecordFactory()
    
    def _record_factory(*args, **kwargs):
        record = _default_record_factory(*args, **kwargs)
        record.component_id = getattr(_record_factory, '_current_cid', 'System')
        return record
    
    _record_factory._current_cid = 'System'
    logging.setLogRecordFactory(_record_factory)

def set_component(component_id: str):
    """Set the current component ID for logging"""
    factory = logging.getLogRecordFactory()
    factory._current_cid = component_id

async def wait_for_signal(event_bus, *, signal_name: str, timeout: float = 60.0, condition: Optional[callable] = None):
    """Wait for a specific signal with optional condition"""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    def callback(payload: Any):
        if future.done():
            return
        
        # Try to reconstruct signal from dict
        reconstructed_signal = None
        if isinstance(payload, dict):
            event_data = payload.get('event_data', payload)
            if 'signal_type' in event_data:
                try:
                    from signals import signal_class_map
                    signal_class = signal_class_map.get(event_data['signal_type'])
                    if signal_class:
                        reconstructed_signal = signal_class(**event_data)
                except Exception:
                    logging.error('Failed to reconstruct signal from dict', exc_info=True)
                    return
        elif hasattr(payload, 'signal_type'):
            reconstructed_signal = payload
        
        if reconstructed_signal:
            if condition:
                if condition(reconstructed_signal):
                    future.set_result(reconstructed_signal)
            else:
                future.set_result(reconstructed_signal)
    
    event_bus.subscribe(signal_name, callback)
    
    try:
        return await asyncio.wait_for(future, timeout)
    finally:
        event_bus.unsubscribe(signal_name, callback)

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        minutes = seconds / 60
        return f'{minutes:.1f}m'
    else:
        hours = seconds / 3600
        return f'{hours:.1f}h'

def format_tree(node: Dict[str, Any], indent: int = 0, max_depth: int = 3) -> List[str]:
    """Format a tree structure for display"""
    lines = []
    prefix = '  ' * indent
    
    node_id = node.get('id', 'unknown')
    trust_score = node.get('trust_score')
    trust_str = f'[{trust_score:.2f}]' if trust_score is not None else '[N/A]'
    
    text = node.get('text', '')
    text_preview = text[:50] + '...' if len(text) > 50 else text
    
    lines.append(f'{prefix}├─ {node_id} {trust_str}')
    lines.append(f'{prefix}│  {text_preview}')
    
    if indent < max_depth * 2 and node.get('variations'):
        for child_id, child_node in node['variations'].items():
            lines.extend(format_tree(child_node, indent + 1, max_depth))
    
    return lines

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a text progress bar"""
    if total == 0:
        return '[' + ' ' * width + ']'
    
    percentage = current / total
    filled = int(width * percentage)
    bar = '█' * filled + '░' * (width - filled)
    
    return f'[{bar}] {current}/{total} ({percentage * 100:.1f}%)'

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to be safe for filesystem"""
    # Define safe characters
    safe_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.'
    
    # Replace unsafe characters with underscore
    sanitized = ''.join(c if c in safe_chars else '_' for c in filename)
    
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized

class AsyncTimer:
    """Async context manager for timing operations"""
    
    def __init__(self, name: str = 'Operation'):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    async def __aenter__(self):
        self.start_time = datetime.now()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            logging.info(f'{self.name} completed in {self.duration:.2f}s')
        else:
            logging.error(f'{self.name} failed after {self.duration:.2f}s')

def merge_configs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries"""
    result = base_config.copy()
    
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result