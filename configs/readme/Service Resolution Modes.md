# Complete Guide: Fixing Placeholder Leaks

## Problem Summary
The original Abiogenesis phase automatically fell back to placeholder implementations when real services weren't available, potentially masking production misconfigurations. This solution provides comprehensive detection, monitoring, and alerting.

## Solution Components

### 1. Enhanced Abiogenesis Phase
**File**: `bootstrap/phases/enhanced_abiogenesis.py`

**Key Features**:
- Environment-aware service resolution modes
- Detailed tracking of how each service was resolved
- Configurable strictness levels
- Comprehensive logging and alerting

### 2. Placeholder Monitor
**File**: `monitoring/placeholder_monitor.py` 

**Key Features**:
- Runtime detection of placeholder usage
- Continuous monitoring capabilities
- Environment-specific alert levels
- Detailed usage analytics

## Configuration Setup

### 1. Environment-Specific Resolution Modes

Add to your global configuration:

```yaml
# configs/prod/app_config.yaml
env: prod
abiogenesis:
  resolution_mode: production  # Options: strict, production, development, testing

# configs/staging/app_config.yaml  
env: staging
abiogenesis:
  resolution_mode: production

# configs/dev/app_config.yaml
env: development
abiogenesis:
  resolution_mode: development

# configs/test/app_config.yaml
env: testing
abiogenesis:
  resolution_mode: testing
```

### 2. Service Resolution Modes Explained

| Mode | Behavior | Use Case |
|------|----------|----------|
| `strict` | **Fails if any service requires placeholder** | CI/CD validation, critical production |
| `production` | **Warns on placeholders, continues** | Production environments |
| `development` | **Allows placeholders, minimal warnings** | Local development |
| `testing` | **Allows placeholders, logs for debugging** | Test environments |

### 3. Manifest Configuration for Real Services

Update your service manifest to specify real implementations:

```yaml
# shared_services in your manifest
shared_services:
  LLMPort:
    enabled: true
    class: "infrastructure.llm.openai_llm.OpenAILLMAdapter"
    config:
      api_key_env: "OPENAI_API_KEY"
      model: "gpt-4"
      
  EmbeddingPort:
    enabled: true
    class: "infrastructure.embeddings.openai_embeddings.OpenAIEmbeddingAdapter"
    config:
      api_key_env: "OPENAI_API_KEY"
      model: "text-embedding-3-small"
      
  EventBusPort:
    enabled: true
    class: "infrastructure.messaging.redis_event_bus.RedisEventBus"
    config:
      redis_url: "redis://localhost:6379"
      
  IdeaRepositoryPort:
    enabled: true
    class: "infrastructure.persistence.postgres_idea_repo.PostgresIdeaRepository"
    config:
      connection_string: "${DATABASE_URL}"
```

## Implementation Steps

### Step 1: Replace Abiogenesis Phase

```python
# In bootstrap/phases/__init__.py
from .enhanced_abiogenesis import EnhancedAbiogenesisPhase

BOOTSTRAP_PHASES = [
    # ... other phases
    EnhancedAbiogenesisPhase,  # Replace AbiogenesisPhase with this
    # ... remaining phases
]
```

### Step 2: Add Placeholder Monitoring

```python
# In your application startup (after bootstrap)
from monitoring.placeholder_monitor import PlaceholderMonitor, validate_service_configuration

async def startup_validation():
    """Run validation after bootstrap completes"""
    
    # Get environment from config
    env = app_config.get('env', 'development')
    
    # Validate service configuration
    validation_result = await validate_service_configuration(env)
    
    print(f"Service Validation Status: {validation_result['validation_status']}")
    print(f"Placeholders detected: {validation_result['total_placeholders']}")
    
    for recommendation in validation_result['recommendations']:
        print(f"  {recommendation}")
    
    # In production, fail if placeholders detected
    if env == 'prod' and validation_result['total_placeholders'] > 0:
        raise RuntimeError("Production deployment with placeholder services detected!")
    
    # Start continuous monitoring in production/staging
    if env in ['prod', 'staging']:
        monitor = PlaceholderMonitor()
        monitor.configure_environment({'env': env})
        # Start monitoring every 5 minutes
        asyncio.create_task(monitor.start_continuous_monitoring(300))
```

### Step 3: CI/CD Integration

Add validation to your deployment pipeline:

```yaml
# .github/workflows/deploy.yml or similar
- name: Validate Service Configuration
  run: |
    python -c "
    import asyncio
    from monitoring.placeholder_monitor import validate_service_configuration
    
    async def validate():
        result = await validate_service_configuration('${{ env.ENVIRONMENT }}')
        
        if result['validation_status'] == 'FAIL':
            print('❌ Service validation failed!')
            for rec in result['recommendations']:
                print(f'  {rec}')
            exit(1)
        else:
            print('✅ Service validation passed')
            print(f'Real services: {result[\"total_placeholders\"]}')
    
    asyncio.run(validate())
    "
```

## Monitoring and Alerting

### Real-time Monitoring

```python
# Set up custom alert handler
async def production_alert_handler(alert):
    """Custom alert handler for production"""
    
    if alert.alert_level.value in ['error', 'critical']:
        # Send to your alerting system (Slack, PagerDuty, etc.)
        await send_alert_to_slack(
            f"🚨 Placeholder detected in production: {alert.service_id}\n"
            f"Details: {alert.message}"
        )
        
        # Log to monitoring system
        metrics.increment('placeholder.detected', tags={'service': alert.service_id})
    
    # Always log
    logger.warning(f"Placeholder alert: {alert.message}")

# Start monitoring with custom handler
monitor = PlaceholderMonitor(alert_callback=production_alert_handler)
await monitor.start_continuous_monitoring()
```

### Health Check Endpoint

```python
# Add to your health check endpoints
@app.get("/health/services")
async def service_health():
    """Health check that includes placeholder detection"""
    
    summary = await PlaceholderMonitor.quick_check({'env': app_config.get('env')})
    
    return {
        'status': 'healthy' if summary['total_placeholders'] == 0 else 'degraded',
        'placeholder_count': summary['total_placeholders'],
        'placeholder_services': summary['detected_services'],
        'environment': summary['environment']
    }
```

## Example Output and Debugging

### Bootstrap Phase Output

```
INFO - Enhanced L0 Abiogenesis with placeholder leak detection
INFO - Service resolution mode: production
INFO - ✓ FeatureFlagsManager emerged with system feature control
INFO - ✓ ComponentRegistry achieved reflexive self-emergence
INFO - ✓ LLMPort resolved to real implementation: infrastructure.llm.openai_llm.OpenAILLMAdapter
INFO - ✓ EmbeddingPort resolved to real implementation: infrastructure.embeddings.openai_embeddings.OpenAIEmbeddingAdapter
WARNING - ⚠️ EventBusPort resolved to PLACEHOLDER: No manifest entry found for EventBusPort
INFO - ✓ IdeaRepositoryPort resolved to real implementation: infrastructure.persistence.postgres_idea_repo.PostgresIdeaRepository
WARNING - PRODUCTION WARNING: Placeholder services: EventBusPort
WARNING - 🚨 Using placeholder services in production may indicate misconfiguration
WARNING -    - EventBusPort: No manifest entry found for EventBusPort
INFO - Service Resolution Summary:
INFO -   Real implementations: 3
INFO -   Placeholder services: 1
INFO -   Resolution mode: production
INFO -   LLMPort: ✅ REAL (OpenAILLMAdapter)
INFO -   EmbeddingPort: ✅ REAL (OpenAIEmbeddingAdapter)
INFO -   EventBusPort: 🔴 PLACEHOLDER (PlaceholderEventBusImpl)
INFO -   IdeaRepositoryPort: ✅ REAL (PostgresIdeaRepository)
```

### Validation Output

```python
# Quick validation check
summary = await validate_service_configuration('prod')

{
    'total_placeholders': 1,
    'alerts_by_level': {'warning': 1},
    'placeholder_types': {'EventBusPort': 1},
    'validation_status': 'FAIL',
    'recommendations': [
        '🚨 URGENT: Replace all placeholder services with real implementations',
        'Check manifest configuration and service instantiation'
    ],
    'detected_services': ['EventBusPort']
}
```

## Troubleshooting

### Common Issues

1. **"Strict mode: Cannot fallback to placeholder"**
   - **Solution**: Configure real service in manifest or change resolution mode
   - **Check**: Manifest has correct class path and service is enabled

2. **"Failed to instantiate from manifest"**
   - **Solution**: Verify class can be imported and has proper constructor
   - **Check**: Dependencies are available and configuration is valid

3. **"No manifest entry found"**
   - **Solution**: Add service to shared_services in manifest
   - **Check**: Service ID matches exactly (case-sensitive)

### Debugging Commands

```python
# Check current service resolutions
from bootstrap.phases.enhanced_abiogenesis import EnhancedAbiogenesisPhase
phase = EnhancedAbiogenesisPhase()
# (run bootstrap)
for resolution in phase.service_resolutions:
    print(f"{resolution.service_id}: {resolution.implementation_type.value}")

# Manual placeholder scan
from monitoring.placeholder_monitor import PlaceholderMonitor
alerts = await PlaceholderMonitor().scan_for_placeholders()
for alert in alerts:
    print(f"{alert.service_id}: {alert.message}")

# Get detailed service information
summary = await PlaceholderMonitor.quick_check({'env': 'prod'})
print(json.dumps(summary, indent=2))
```

## Production Deployment Checklist

- [ ] All critical services have real implementations in manifest
- [ ] Resolution mode set to `production` or `strict` for prod environment
- [ ] CI/CD pipeline includes service validation step
- [ ] Health check endpoints include placeholder detection
- [ ] Monitoring alerts configured for placeholder detection
- [ ] Service startup validation configured to fail on production placeholders

This comprehensive solution ensures that placeholder services are properly detected, monitored, and prevented from reaching production environments without appropriate warnings.