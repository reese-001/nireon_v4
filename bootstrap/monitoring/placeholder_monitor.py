# -*- coding: utf-8 -*-
"""
Placeholder Usage Monitor

Provides runtime monitoring and alerting for placeholder service usage
to help detect configuration issues in production environments.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import asyncio

from bootstrap.registry.registry_manager import RegistryManager
from bootstrap.bootstrap_helper.placeholders import (
    PlaceholderLLMPortImpl, PlaceholderEmbeddingPortImpl,
    PlaceholderEventBusImpl, PlaceholderIdeaRepositoryImpl
)

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels for placeholder detection"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PlaceholderAlert:
    """Represents a placeholder usage alert"""
    service_id: str
    service_type: str
    placeholder_class: str
    detected_at: datetime
    alert_level: AlertLevel
    message: str
    call_count: Optional[int] = None
    recent_operations: Optional[List[str]] = None


class PlaceholderMonitor:
    """
    Monitor for detecting and alerting on placeholder service usage.
    
    This class provides both one-time checks and continuous monitoring
    capabilities to help identify when placeholder services are being
    used in environments where they shouldn't be.
    """
    
    # Known placeholder classes for detection
    PLACEHOLDER_CLASSES = {
        PlaceholderLLMPortImpl,
        PlaceholderEmbeddingPortImpl,
        PlaceholderEventBusImpl,
        PlaceholderIdeaRepositoryImpl,
    }
    
    def __init__(self, alert_callback: Optional[callable] = None):
        self.alert_callback = alert_callback or self._default_alert_handler
        self.monitoring_active = False
        self.detected_placeholders: Dict[str, PlaceholderAlert] = {}
        self.environment_config: Dict[str, Any] = {}
        
    async def scan_for_placeholders(self) -> List[PlaceholderAlert]:
        """
        Perform a one-time scan of the registry for placeholder services.
        
        Returns:
            List of detected placeholder alerts
        """
        alerts = []
        
        try:
            # Get all registered services
            registry = RegistryManager.get_instance()
            all_services = registry.get_all_services()
            
            for service_id, service_instance in all_services.items():
                if self._is_placeholder_instance(service_instance):
                    alert = self._create_placeholder_alert(service_id, service_instance)
                    alerts.append(alert)
                    self.detected_placeholders[service_id] = alert
                    
            logger.info(f'Placeholder scan complete: {len(alerts)} placeholders detected')
            
            # Send alerts if any placeholders found
            for alert in alerts:
                await self.alert_callback(alert)
                
        except Exception as e:
            logger.error(f'Error during placeholder scan: {e}', exc_info=True)
            
        return alerts
    
    def _is_placeholder_instance(self, instance: Any) -> bool:
        """Check if an instance is a placeholder implementation"""
        return any(isinstance(instance, placeholder_class) for placeholder_class in self.PLACEHOLDER_CLASSES)
    
    def _create_placeholder_alert(self, service_id: str, instance: Any) -> PlaceholderAlert:
        """Create an alert for a detected placeholder"""
        
        placeholder_class = instance.__class__.__name__
        service_type = self._get_service_type(instance)
        
        # Determine alert level based on environment and service criticality
        alert_level = self._determine_alert_level(service_id, service_type)
        
        # Get usage statistics if available
        call_count = getattr(instance, 'call_count', None) or getattr(instance, 'operation_count', None)
        recent_ops = self._get_recent_operations(instance)
        
        message = self._create_alert_message(service_id, placeholder_class, call_count)
        
        return PlaceholderAlert(
            service_id=service_id,
            service_type=service_type,
            placeholder_class=placeholder_class,
            detected_at=datetime.now(timezone.utc),
            alert_level=alert_level,
            message=message,
            call_count=call_count,
            recent_operations=recent_ops
        )
    
    def _get_service_type(self, instance: Any) -> str:
        """Determine the service type from instance"""
        class_name = instance.__class__.__name__
        
        if 'LLM' in class_name:
            return 'LLMPort'
        elif 'Embedding' in class_name:
            return 'EmbeddingPort'
        elif 'EventBus' in class_name:
            return 'EventBusPort'
        elif 'Repository' in class_name:
            return 'RepositoryPort'
        else:
            return 'UnknownPort'
    
    def _determine_alert_level(self, service_id: str, service_type: str) -> AlertLevel:
        """Determine appropriate alert level based on context"""
        
        # Check environment
        env = self.environment_config.get('env', 'development').lower()
        
        if env in ['prod', 'production']:
            # In production, placeholders should be treated seriously
            if service_type in ['LLMPort', 'EmbeddingPort']:
                return AlertLevel.CRITICAL  # Core AI services
            else:
                return AlertLevel.ERROR
                
        elif env in ['staging', 'stage']:
            return AlertLevel.WARNING
            
        elif env in ['test', 'testing']:
            return AlertLevel.INFO
            
        else:  # development
            return AlertLevel.INFO
    
    def _create_alert_message(self, service_id: str, placeholder_class: str, call_count: Optional[int]) -> str:
        """Create human-readable alert message"""
        
        base_msg = f"Placeholder service detected: {service_id} ({placeholder_class})"
        
        if call_count is not None and call_count > 0:
            base_msg += f" - {call_count} operations performed"
            
        env = self.environment_config.get('env', 'unknown')
        if env in ['prod', 'production']:
            base_msg += " ⚠️ PRODUCTION ENVIRONMENT - This may indicate misconfiguration!"
            
        return base_msg
    
    def _get_recent_operations(self, instance: Any) -> Optional[List[str]]:
        """Extract recent operation information if available"""
        try:
            # Check for event history (EventBus placeholders)
            if hasattr(instance, 'get_event_history'):
                history = instance.get_event_history()
                return [f"Event: {event.get('event_type', 'unknown')}" for event in history[-5:]]
                
            # Check for cache info (Embedding placeholders)
            elif hasattr(instance, '_cache') and instance._cache:
                cache_keys = list(instance._cache.keys())
                return [f"Cached: {key[:50]}..." for key in cache_keys[-5:]]
                
            # Check for statistics (Repository placeholders)
            elif hasattr(instance, 'get_statistics'):
                stats = instance.get_statistics()
                return [f"Stats: {k}={v}" for k, v in stats.items()]
                
        except Exception:
            pass  # Ignore errors in introspection
            
        return None
    
    async def start_continuous_monitoring(self, check_interval: int = 300) -> None:
        """
        Start continuous monitoring for new placeholder services.
        
        Args:
            check_interval: Time between checks in seconds (default: 5 minutes)
        """
        if self.monitoring_active:
            logger.warning('Placeholder monitoring already active')
            return
            
        self.monitoring_active = True
        logger.info(f'Starting continuous placeholder monitoring (interval: {check_interval}s)')
        
        while self.monitoring_active:
            try:
                await self.scan_for_placeholders()
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f'Error in continuous monitoring: {e}', exc_info=True)
                await asyncio.sleep(min(check_interval, 60))  # Shorter retry on error
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            logger.info('Placeholder monitoring stopped')
    
    async def _default_alert_handler(self, alert: PlaceholderAlert) -> None:
        """Default alert handler that logs alerts"""
        
        log_methods = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }
        
        log_method = log_methods.get(alert.alert_level, logger.info)
        log_method(f'PLACEHOLDER ALERT [{alert.alert_level.value.upper()}]: {alert.message}')
        
        # Log additional details for higher severity alerts
        if alert.alert_level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            if alert.call_count:
                log_method(f'  Usage: {alert.call_count} operations')
            if alert.recent_operations:
                log_method(f'  Recent operations: {alert.recent_operations}')
    
    def configure_environment(self, env_config: Dict[str, Any]) -> None:
        """Configure environment settings for alert level determination"""
        self.environment_config.update(env_config)
    
    def get_placeholder_summary(self) -> Dict[str, Any]:
        """Get summary of detected placeholders"""
        
        total_placeholders = len(self.detected_placeholders)
        alerts_by_level = {}
        
        for alert in self.detected_placeholders.values():
            level = alert.alert_level.value
            alerts_by_level[level] = alerts_by_level.get(level, 0) + 1
        
        placeholder_types = {}
        for alert in self.detected_placeholders.values():
            ptype = alert.service_type
            placeholder_types[ptype] = placeholder_types.get(ptype, 0) + 1
        
        return {
            'total_placeholders': total_placeholders,
            'alerts_by_level': alerts_by_level,
            'placeholder_types': placeholder_types,
            'monitoring_active': self.monitoring_active,
            'environment': self.environment_config.get('env', 'unknown'),
            'detected_services': list(self.detected_placeholders.keys())
        }
    
    @classmethod
    async def quick_check(cls, env_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convenient method for one-time placeholder check.
        
        Args:
            env_config: Environment configuration for alert level determination
            
        Returns:
            Summary dictionary with check results
        """
        monitor = cls()
        
        if env_config:
            monitor.configure_environment(env_config)
            
        alerts = await monitor.scan_for_placeholders()
        summary = monitor.get_placeholder_summary()
        
        # Add quick check specific info
        summary['alerts'] = [
            {
                'service_id': alert.service_id,
                'alert_level': alert.alert_level.value,
                'message': alert.message
            }
            for alert in alerts
        ]
        
        return summary


# Convenience functions for common use cases

async def check_for_placeholders_in_production() -> bool:
    """
    Quick check specifically for production environments.
    
    Returns:
        True if placeholders detected in production, False otherwise
    """
    summary = await PlaceholderMonitor.quick_check({'env': 'production'})
    return summary['total_placeholders'] > 0


async def alert_on_production_placeholders() -> None:
    """
    Check for placeholders in production and log critical alerts if found.
    """
    if await check_for_placeholders_in_production():
        logger.critical(
            '🚨 CRITICAL: Placeholder services detected in production environment! '
            'This indicates potential misconfiguration and should be investigated immediately.'
        )


async def validate_service_configuration(env: str) -> Dict[str, Any]:
    """
    Validate service configuration for a specific environment.
    
    Args:
        env: Environment name (e.g., 'prod', 'staging', 'dev')
        
    Returns:
        Validation results with recommendations
    """
    summary = await PlaceholderMonitor.quick_check({'env': env})
    
    # Add validation recommendations
    recommendations = []
    
    if env.lower() in ['prod', 'production'] and summary['total_placeholders'] > 0:
        recommendations.append('🚨 URGENT: Replace all placeholder services with real implementations')
        recommendations.append('Check manifest configuration and service instantiation')
        
    elif env.lower() in ['staging', 'stage'] and summary['total_placeholders'] > 0:
        recommendations.append('⚠️ WARNING: Staging should mirror production configuration')
        
    elif summary['total_placeholders'] == 0:
        recommendations.append('✅ All services using real implementations')
        
    summary['recommendations'] = recommendations
    summary['validation_status'] = 'FAIL' if (env.lower() in ['prod', 'production'] and summary['total_placeholders'] > 0) else 'PASS'
    
    return summary