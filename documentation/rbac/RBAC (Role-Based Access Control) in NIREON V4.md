# RBAC (Role-Based Access Control) in NIREON V4

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Integration Patterns](#integration-patterns)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Overview

NIREON V4's RBAC system provides enterprise-grade access control for ideas, components, mechanisms, and system operations. It integrates seamlessly with the bootstrap process and offers multiple interfaces for permission checking.

### Key Features

- **Declarative Policy Management**: YAML-based policy configuration with environment-specific overrides
- **Type-Safe Implementation**: Pydantic models with validation and JSON schema support
- **Multiple Access Patterns**: Decorators, context managers, direct API calls
- **Performance Optimized**: Built-in caching and efficient rule evaluation
- **Audit Trail**: Detailed logging and audit capabilities
- **Bootstrap Integration**: Automatic initialization during system startup
- **Flexible Subject Resolution**: Multiple ways to determine current user/service
- **Pattern Matching**: Glob-style resource patterns for flexible access control

### Core Components

1. **Policy Engine**: Evaluates permissions against loaded policies
2. **Policy Loader**: Loads and validates RBAC policies from YAML files
3. **Execution Context**: Tracks current subject across operation chains
4. **Decorators**: Easy-to-use permission enforcement
5. **Audit System**: Detailed permission check logging

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NIREON V4 RBAC System                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   Policy Files  │  │  Bootstrap Phase │  │   Engine    │ │
│  │                 │  │                  │  │             │ │
│  │ bootstrap_rbac  ├─►│  RBACSetupPhase  ├─►│ RBACPolicy  │ │
│  │    .yaml        │  │                  │  │   Engine    │ │
│  └─────────────────┘  └──────────────────┘  └──────┬──────┘ │
│                                                     │        │
│  ┌─────────────────┐  ┌──────────────────┐         │        │
│  │ ExecutionContext│  │   Decorators     │         │        │
│  │                 │  │                  │◄────────┘        │
│  │ - Current User  │  │ @requires_       │                  │
│  │ - Session Info  │  │  permission      │                  │
│  │ - Request Data  │  │                  │                  │
│  └─────────────────┘  └──────────────────┘                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Protected Services                         │ │
│  │                                                         │ │
│  │  IdeaService    ComponentManager    MechanismService   │ │
│  │       │               │                   │            │ │
│  │       └───────────────┼───────────────────┘            │ │
│  │                       │                                │ │
│  │               RBAC Enforcement                         │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Setup

```python
# Enable RBAC in configuration
# configs/default/app_config.yaml
feature_flags:
  enable_rbac_bootstrap: true

# Create basic policy file
# configs/default/bootstrap_rbac.yaml
version: "1.0"
rules:
  - id: "admin_access"
    subjects: ["admin@company.com"]
    resources: ["*"]
    actions: ["*"]
    effect: "allow"
    
  - id: "user_read_ideas"
    subjects: ["user@company.com"]
    resources: ["idea", "ideas/*"]
    actions: ["read"]
    effect: "allow"
```

### 2. Protect a Function

```python
from security import requires_permission, set_execution_subject

# Set current user
set_execution_subject("user@company.com")

@requires_permission("idea", "read")
async def get_idea(idea_id: str):
    return await idea_repository.get_by_id(idea_id)
```

### 3. Manual Permission Check

```python
from security import check_permission

if check_permission("user@company.com", "idea", "delete"):
    await delete_idea(idea_id)
else:
    raise PermissionError("Cannot delete idea")
```

## Installation & Setup

### 1. File Structure

Create the RBAC module structure:

```
nireon/
├── security/
│   ├── __init__.py
│   ├── execution_context.py
│   ├── rbac_engine.py
│   └── decorators.py
├── bootstrap/
│   └── phases/
│       └── rbac_setup_phase.py
└── configs/
    ├── default/
    │   └── bootstrap_rbac.yaml
    ├── dev/
    │   └── bootstrap_rbac.yaml
    └── prod/
        └── bootstrap_rbac.yaml
```

### 2. Security Module Initialization

```python
# security/__init__.py
"""NIREON V4 Security Module"""

from .execution_context import (
    get_current_execution_context,
    set_execution_subject,
    ExecutionContextScope,
    ExecutionContext
)
from .rbac_engine import RBACPolicyEngine
from .decorators import (
    requires_permission,
    requires_read,
    requires_write,
    requires_delete,
    requires_execute,
    admin_required,
    RBACContext,
    RBACPermissionError,
    check_permission,
    require_permission
)

__all__ = [
    'get_current_execution_context',
    'set_execution_subject', 
    'ExecutionContextScope',
    'ExecutionContext',
    'RBACPolicyEngine',
    'requires_permission',
    'requires_read',
    'requires_write',
    'requires_delete',
    'requires_execute',
    'admin_required',
    'RBACContext',
    'RBACPermissionError',
    'check_permission',
    'require_permission'
]
```

### 3. Bootstrap Integration

Add RBAC to your bootstrap phases:

```python
# bootstrap/phases/__init__.py
from .rbac_setup_phase import RBACSetupPhase

BOOTSTRAP_PHASES = [
    # ... other phases
    RBACSetupPhase,  # Add after component registry setup
    # ... remaining phases
]
```

## Configuration

### Environment Configuration

```yaml
# configs/default/app_config.yaml
feature_flags:
  enable_rbac_bootstrap: true

env: development

abiogenesis:
  resolution_mode: development

# Optional: Legacy single-file path
bootstrap_rbac_path: "configs/default/bootstrap_rbac.yaml"

# Optional: Additional policy files
rbac:
  additional_policy_files:
    - "configs/custom/module_policies.yaml"
    - "configs/custom/user_policies.yaml"
```

### Policy File Structure

```yaml
# configs/default/bootstrap_rbac.yaml
version: "1.0"

metadata:
  description: "NIREON V4 RBAC Policies"
  author: "Security Team"
  created_at: "2025-06-05T10:00:00Z"

rules:
  # Admin Access
  - id: "system_admin_full_access"
    subjects: ["admin@company.com", "system-admin"]
    resources: ["*"]
    actions: ["*"]
    effect: "allow"
    description: "Full system access for administrators"

  # User Access
  - id: "authenticated_user_read_ideas"
    subjects: ["user@company.com", "standard-user", "*@company.com"]
    resources: ["idea", "ideas/*", "idea/*"]
    actions: ["read", "list"]
    effect: "allow"
    description: "Authenticated users can read ideas"

  - id: "idea_owner_manage"
    subjects: ["idea-owner"]
    resources: ["idea/${user_id}/*"]
    actions: ["read", "write", "delete"]
    effect: "allow"
    description: "Users can manage their own ideas"

  # Service Accounts
  - id: "mechanism_service_access"
    subjects: ["mechanism-service", "nireon-system"]
    resources: ["component/*", "mechanism/*", "idea/process/*"]
    actions: ["read", "write", "execute"]
    effect: "allow"
    description: "Mechanism services can process components and ideas"

  # Security Restrictions
  - id: "deny_sensitive_access"
    subjects: ["*"]
    resources: ["admin/*", "config/secret/*", "system/internal/*"]
    actions: ["*"]
    effect: "deny"
    description: "Explicit denial for sensitive system resources"

  # API Access
  - id: "api_key_access"
    subjects: ["api-key-*"]
    resources: ["api/public/*"]
    actions: ["read", "write"]
    effect: "allow"
    description: "API keys can access public endpoints"
```

### Environment-Specific Policies

```yaml
# configs/prod/bootstrap_rbac.yaml
version: "1.0"

# Inherit from default and add production-specific rules
include: "../default/bootstrap_rbac.yaml"

rules:
  # Production-specific restrictions
  - id: "prod_no_debug_access"
    subjects: ["*"]
    resources: ["debug/*", "test/*", "development/*"]
    actions: ["*"]
    effect: "deny"
    description: "No debug access in production"

  # Production admin access (more restrictive)
  - id: "prod_admin_restricted"
    subjects: ["prod-admin@company.com"]
    resources: ["admin/production/*"]
    actions: ["read", "write"]
    effect: "allow"
    description: "Limited admin access in production"
```

## Usage Guide

### Decorator-Based Protection

```python
from security import (
    requires_permission, requires_read, requires_write, 
    requires_delete, admin_required, set_execution_subject
)

class IdeaService:
    
    @requires_read("idea")
    async def get_idea(self, idea_id: str):
        """Requires read permission on 'idea' resource"""
        return await self.repository.get_by_id(idea_id)
    
    @requires_write("idea")
    async def update_idea(self, idea_id: str, data: dict):
        """Requires write permission on 'idea' resource"""
        return await self.repository.update(idea_id, data)
    
    @requires_delete("idea")
    async def delete_idea(self, idea_id: str):
        """Requires delete permission on 'idea' resource"""
        await self.repository.delete(idea_id)
    
    @requires_permission("idea", "process")
    async def process_idea(self, idea_id: str, mechanism: str):
        """Custom action permission"""
        return await self.mechanism_runner.process(idea_id, mechanism)
    
    @admin_required()
    async def reset_all_ideas(self):
        """Only administrators can call this"""
        await self.repository.clear_all()

class ComponentManager:
    
    @requires_permission("component/sensitive", "read")
    def get_sensitive_component(self, component_id: str):
        """Access to sensitive components"""
        return self.registry.get_component(component_id)
    
    @requires_permission("component", "register", subject="system")
    async def register_component(self, component):
        """Always runs as 'system' user"""
        await self.registry.register(component)
```

### Context Manager Usage

```python
from security import RBACContext, ExecutionContextScope

class DynamicPermissionService:
    
    async def process_batch_operations(self, operations: list, user_id: str):
        """Check permissions for each operation dynamically"""
        
        with RBACContext(user_id) as rbac:
            results = []
            
            for op in operations:
                if rbac.can(op['resource'], op['action']):
                    result = await self.perform_operation(op)
                    results.append({'success': True, 'result': result})
                else:
                    results.append({
                        'success': False, 
                        'error': f"Permission denied: {op['action']} on {op['resource']}"
                    })
            
            return results
    
    async def admin_operation_with_context(self):
        """Temporarily elevate to admin context"""
        
        with ExecutionContextScope(subject="admin@system"):
            # All nested operations run as admin
            await self.perform_admin_tasks()
            
            # Nested decorated functions automatically use admin context
            await self.delete_idea("sensitive-idea-123")  # Works if delete_idea has @requires_delete
```

### Direct Permission Checking

```python
from security import check_permission, require_permission, get_rbac_subject

class FlexibleService:
    
    async def conditional_access(self, resource: str):
        """Different behavior based on permissions"""
        
        current_user = get_rbac_subject()
        
        if check_permission(current_user, resource, "write"):
            return await self.get_full_data(resource)
        elif check_permission(current_user, resource, "read"):
            return await self.get_public_data(resource)
        else:
            raise PermissionError(f"No access to {resource}")
    
    async def strict_access(self, resource: str, action: str):
        """Strict permission enforcement"""
        
        current_user = get_rbac_subject()
        
        # This will raise RBACPermissionError if denied
        require_permission(current_user, resource, action)
        
        # Only executes if permission granted
        return await self.perform_action(resource, action)
    
    async def multi_resource_check(self, resources: list):
        """Check access to multiple resources"""
        
        current_user = get_rbac_subject()
        accessible_resources = []
        
        for resource in resources:
            if check_permission(current_user, resource, "read"):
                accessible_resources.append(resource)
        
        return accessible_resources
```

### Subject Management

```python
from security import set_execution_subject, ExecutionContextScope, SubjectScope

# Method 1: Set subject for entire execution
set_execution_subject("user@company.com")

@requires_permission("idea", "read")
async def get_user_ideas():
    # Automatically uses user@company.com
    return await idea_service.get_user_ideas()

# Method 2: Temporary subject context
with ExecutionContextScope(subject="admin@company.com"):
    await admin_only_operation()

# Method 3: Environment variable (alternative)
import os
os.environ['RBAC_CURRENT_SUBJECT'] = 'service-account@system'

# Method 4: Function-specific subject
@requires_permission("system", "backup", subject="backup-service")
async def system_backup():
    # Always runs as backup-service regardless of current context
    pass

# Method 5: Dynamic subject resolution
def get_current_user_from_jwt():
    # Your logic to extract user from JWT token
    return "user@company.com"

@requires_permission("idea", "write", subject=get_current_user_from_jwt)
async def update_idea_with_jwt():
    # Uses result of get_current_user_from_jwt() as subject
    pass
```

## Integration Patterns

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, Depends
from security import requires_permission, RBACPermissionError, ExecutionContextScope

app = FastAPI()

# Middleware to set execution context from request
@app.middleware("http")
async def rbac_middleware(request, call_next):
    # Extract user from request (customize based on your auth)
    user = extract_user_from_request(request)  # Your implementation
    
    with ExecutionContextScope(subject=user.email if user else None):
        response = await call_next(request)
        return response

# Exception handler for RBAC errors
@app.exception_handler(RBACPermissionError)
async def rbac_exception_handler(request, exc: RBACPermissionError):
    return HTTPException(
        status_code=403,
        detail=f"Access denied: {exc.message}"
    )

# Protected endpoints
@app.get("/ideas/{idea_id}")
@requires_permission("idea", "read")
async def get_idea(idea_id: str):
    return await idea_service.get_idea(idea_id)

@app.delete("/ideas/{idea_id}")
@requires_permission("idea", "delete")
async def delete_idea(idea_id: str):
    await idea_service.delete_idea(idea_id)
    return {"status": "deleted"}

@app.post("/admin/reset")
@admin_required()
async def admin_reset():
    await system_service.reset()
    return {"status": "reset"}

# Manual permission checking in endpoints
@app.get("/ideas")
async def list_ideas(include_sensitive: bool = False):
    ideas = await idea_service.get_public_ideas()
    
    # Add sensitive ideas if user has permission
    if include_sensitive and check_permission(get_rbac_subject(), "idea/sensitive", "read"):
        sensitive_ideas = await idea_service.get_sensitive_ideas()
        ideas.extend(sensitive_ideas)
    
    return ideas
```

### Mechanism Integration

```python
from security import requires_permission, RBACContext

class ExplorerMechanism:
    
    @requires_permission("mechanism/explorer", "execute")
    async def explore_idea(self, idea_id: str, depth: int = 3):
        """Only users with explorer execute permission can run this"""
        
        with RBACContext() as rbac:
            # Check if user can read the source idea
            rbac.require("idea", "read")
            idea = await self.idea_service.get_idea(idea_id)
            
            # Explore and create new ideas
            explored_ideas = []
            for i in range(depth):
                # Check if user can create new ideas
                if rbac.can("idea", "create"):
                    new_idea = await self.generate_exploration(idea, i)
                    explored_ideas.append(new_idea)
                else:
                    break
            
            return explored_ideas

class SynthesizerMechanism:
    
    @requires_permission("mechanism/synthesizer", "execute")
    async def synthesize_ideas(self, idea_ids: list):
        """Synthesize multiple ideas into one"""
        
        current_user = get_rbac_subject()
        
        # Check read access to all source ideas
        accessible_ideas = []
        for idea_id in idea_ids:
            if check_permission(current_user, f"idea/{idea_id}", "read"):
                idea = await self.idea_service.get_idea(idea_id)
                accessible_ideas.append(idea)
        
        if not accessible_ideas:
            raise PermissionError("No accessible ideas for synthesis")
        
        # Synthesize (requires create permission)
        require_permission(current_user, "idea", "create")
        return await self.perform_synthesis(accessible_ideas)
```

### Observer Integration

```python
from security import check_permission, get_rbac_subject

class TrustObserver:
    
    async def observe_mechanism_execution(self, mechanism_id: str, execution_data: dict):
        """Observe mechanism execution with permission-aware logging"""
        
        current_user = get_rbac_subject()
        
        # Basic observation (always allowed)
        basic_metrics = self.calculate_basic_trust_metrics(execution_data)
        
        # Detailed analysis (requires observer permission)
        detailed_metrics = {}
        if check_permission(current_user, "observer/trust", "analyze"):
            detailed_metrics = self.calculate_detailed_metrics(execution_data)
        
        # Sensitive system metrics (admin only)
        system_metrics = {}
        if check_permission(current_user, "system/metrics", "read"):
            system_metrics = self.get_system_metrics()
        
        return {
            'basic': basic_metrics,
            'detailed': detailed_metrics,
            'system': system_metrics,
            'observer': current_user
        }
```

## Best Practices

### 1. Policy Design Patterns

#### Hierarchical Resources
```yaml
# Use hierarchical resource patterns
rules:
  - id: "user_own_ideas"
    subjects: ["${user}"]
    resources: ["idea/${user}/*"]
    actions: ["read", "write", "delete"]
    effect: "allow"
    
  - id: "team_shared_ideas"
    subjects: ["team-member"]
    resources: ["idea/team/shared/*"]
    actions: ["read", "write"]
    effect: "allow"
```

#### Principle of Least Privilege
```yaml
# Start with minimal permissions, add as needed
rules:
  # Default: read-only access
  - id: "default_user_access"
    subjects: ["authenticated-user"]
    resources: ["idea", "component/public/*"]
    actions: ["read"]
    effect: "allow"
    
  # Explicit grants for specific actions
  - id: "idea_creators"
    subjects: ["idea-creator-role"]
    resources: ["idea"]
    actions: ["create"]
    effect: "allow"
```

#### Defense in Depth
```yaml
# Multiple layers of protection
rules:
  # Allow specific access
  - id: "admin_read_sensitive"
    subjects: ["admin"]
    resources: ["sensitive/*"]
    actions: ["read"]
    effect: "allow"
    
  # Explicit denials override allows
  - id: "deny_critical_sensitive"
    subjects: ["*"]
    resources: ["sensitive/critical/*"]
    actions: ["*"]
    effect: "deny"
```

### 2. Code Organization

#### Service Layer Protection
```python
# Protect at service boundaries, not internal methods
class IdeaService:
    
    @requires_permission("idea", "read")
    async def get_idea(self, idea_id: str):
        """Public API - protected"""
        return await self._internal_get_idea(idea_id)
    
    async def _internal_get_idea(self, idea_id: str):
        """Internal method - not protected (called by public API)"""
        return await self.repository.get_by_id(idea_id)
```

#### Resource Naming Conventions
```python
# Use consistent resource naming
RESOURCE_PATTERNS = {
    'ideas': 'idea',
    'user_ideas': 'idea/${user_id}/*',
    'components': 'component/*',
    'sensitive_components': 'component/sensitive/*',
    'mechanisms': 'mechanism/*',
    'admin_functions': 'admin/*',
    'system_config': 'system/config/*'
}

@requires_permission(RESOURCE_PATTERNS['ideas'], "read")
async def get_idea(idea_id: str):
    pass
```

#### Error Handling
```python
from security import RBACPermissionError

try:
    await protected_operation()
except RBACPermissionError as e:
    logger.warning(f"Access denied: {e.message}")
    # Return appropriate HTTP status or handle gracefully
    raise HTTPException(status_code=403, detail="Access denied")
```

### 3. Performance Considerations

#### Batch Permission Checks
```python
# Instead of checking permissions one by one
resources = ["idea/1", "idea/2", "idea/3"]
user = get_rbac_subject()

# Inefficient: multiple engine calls
accessible = []
for resource in resources:
    if check_permission(user, resource, "read"):
        accessible.append(resource)

# Better: use context manager for caching
with RBACContext(user) as rbac:
    accessible = [r for r in resources if rbac.can(r, "read")]
```

#### Cache Warming
```python
# Warm permission cache for common checks
from registry.registry_manager import RegistryManager

async def warm_rbac_cache():
    engine = RegistryManager.get_service("rbac_engine")
    common_checks = [
        ("user@company.com", "idea", "read"),
        ("user@company.com", "idea", "write"),
        ("admin@company.com", "admin", "read"),
    ]
    
    for subject, resource, action in common_checks:
        engine.is_allowed(subject, resource, action)
```

### 4. Security Guidelines

#### Never Trust Client Input
```python
# DON'T: Use client-provided subject
@requires_permission("idea", "delete", subject=request.json.get("user"))

# DO: Use authenticated subject
@requires_permission("idea", "delete")  # Uses current authenticated user
```

#### Audit Critical Operations
```python
@requires_permission("system", "backup", audit=True)
async def system_backup():
    """Audit trail for critical operations"""
    pass
```

#### Environment-Specific Policies
```python
# Use environment-specific policies for different security requirements
# Dev: Relaxed policies for testing
# Staging: Production-like policies
# Prod: Strict policies with minimal access
```

## API Reference

### Decorators

#### `@requires_permission(resource, action, **kwargs)`
```python
@requires_permission(
    resource: str,           # Resource pattern (e.g., "idea", "component/*")
    action: str,             # Action (e.g., "read", "write", "delete")
    subject: str | callable, # Override subject (optional)
    raise_on_deny: bool,     # Raise exception on denial (default: True)
    audit: bool              # Enable detailed audit logging (default: False)
)
```

#### Convenience Decorators
```python
@requires_read(resource)     # Shorthand for read permission
@requires_write(resource)    # Shorthand for write permission
@requires_delete(resource)   # Shorthand for delete permission
@requires_execute(resource)  # Shorthand for execute permission
@admin_required(resource)    # Requires admin access (action="*")
```

### Direct API

#### Permission Checking
```python
check_permission(subject: str, resource: str, action: str) -> bool
require_permission(subject: str, resource: str, action: str) -> None  # Raises on deny
check_current_permission(resource: str, action: str) -> bool  # Uses current subject
require_current_permission(resource: str, action: str) -> None
```

#### Subject Management
```python
set_execution_subject(subject: str) -> None
get_rbac_subject() -> str
ExecutionContextScope(subject: str, **context_data)
SubjectScope(subject: str)  # Environment variable based
```

#### Context Managers
```python
RBACContext(subject: str = None)
# Methods: can(resource, action), require(resource, action), get_allowed_actions(resource), audit(resource, action)

ExecutionContextScope(subject: str, **context_data)
# Sets execution context for nested operations
```

### Engine API

#### Engine Statistics
```python
engine = RegistryManager.get_service("rbac_engine")
stats = engine.get_stats()
# Returns: total_rules, policy_sets, cache_size, cache_hits, cache_misses, cache_hit_ratio, last_updated
```

#### Engine Operations
```python
engine.is_allowed(subject: str, resource: str, action: str) -> bool
engine.get_allowed_actions(subject: str, resource: str) -> Set[str]
engine.get_subjects_with_access(resource: str, action: str) -> List[str]
engine.audit_permission(subject: str, resource: str, action: str) -> Dict
engine.clear_cache() -> None
```

## Troubleshooting

### Common Issues

#### 1. "RBAC engine not available"
```python
# Problem: Engine not registered during bootstrap
# Solution: Check bootstrap logs for RBAC setup phase

# Verify RBAC is enabled
feature_flags:
  enable_rbac_bootstrap: true

# Check bootstrap logs for:
# "✓ RBACPolicyEngine registered with X rules"
```

#### 2. "Permission denied for system operations"
```python
# Problem: No subject set for system operations
# Solution: Set system subject

set_execution_subject("system")
# or
with ExecutionContextScope(subject="system"):
    await system_operation()
```

#### 3. "Policy file not found"
```python
# Problem: Missing policy files
# Solution: Create policy file or check path

# Create: configs/default/bootstrap_rbac.yaml
# Or check custom path in config
bootstrap_rbac_path: "custom/path/to/policies.yaml"
```

#### 4. "No rules matched"
```python
# Problem: Policy rules don't match request
# Solution: Check rule patterns and debug

with RBACContext("user@company.com") as rbac:
    audit = rbac.audit("idea", "read")
    print(f"Matched rules: {audit['matched_rules']}")
    print(f"Decision: {audit['final_decision']}")
```

### Debug Tools

#### Enable RBAC Debug Logging
```python
import logging
logging.getLogger('security.rbac_engine').setLevel(logging.DEBUG)
logging.getLogger('security.decorators').setLevel(logging.DEBUG)
```

#### Manual Engine Testing
```python
from registry.registry_manager import RegistryManager

# Get engine and test directly
engine = RegistryManager.get_service("rbac_engine")

# Test specific permission
result = engine.is_allowed("user@test.com", "idea", "read")
print(f"Permission result: {result}")

# Get detailed audit
audit = engine.audit_permission("user@test.com", "idea", "read")
print(f"Audit details: {audit}")

# Check engine stats
stats = engine.get_stats()
print(f"Engine stats: {stats}")
```

#### Policy Validation
```python
# Validate policy file syntax
from bootstrap.phases.rbac_setup_phase import RBACPolicySet
import yaml

with open('configs/default/bootstrap_rbac.yaml') as f:
    data = yaml.safe_load(f)

try:
    policy_set = RBACPolicySet(**data)
    print(f"✅ Valid policy with {len(policy_set.rules)} rules")
except Exception as e:
    print(f"❌ Invalid policy: {e}")
```

## Examples

### Complete Application Example

```python
# main.py - Complete NIREON application with RBAC
from fastapi import FastAPI, HTTPException
from security import (
    requires_permission, requires_read, requires_write, 
    set_execution_subject, RBACPermissionError, 
    ExecutionContextScope, check_permission
)

app = FastAPI(title="NIREON V4 with RBAC")

# Global exception handler
@app.exception_handler(RBACPermissionError)
async def rbac_exception_handler(request, exc):
    return HTTPException(status_code=403, detail=exc.message)

# Set up authentication middleware (simplified)
@app.middleware("http")
async def auth_middleware(request, call_next):
    # Extract user from Authorization header, JWT, etc.
    user = extract_user_from_request(request)  # Your implementation
    
    with ExecutionContextScope(subject=user.email if user else "anonymous"):
        response = await call_next(request)
        return response

# Protected API endpoints
class IdeaAPI:
    
    @app.get("/ideas")
    @requires_read("idea")
    async def list_ideas():
        """List ideas (requires read permission)"""
        return await idea_service.get_all_ideas()
    
    @app.get("/ideas/{idea_id}")
    @requires_read("idea")
    async def get_idea(idea_id: str):
        """Get specific idea"""
        return await idea_service.get_idea(idea_id)
    
    @app.post("/ideas")
    @requires_permission("idea", "create")
    async def create_idea(idea_data: dict):
        """Create new idea"""
        return await idea_service.create_idea(idea_data)
    
    @app.put("/ideas/{idea_id}")
    @requires_write("idea")
    async def update_idea(idea_id: str, idea_data: dict):
        """Update idea"""
        return await idea_service.update_idea(idea_id, idea_data)
    
    @app.delete("/ideas/{idea_id}")
    @requires_permission("idea", "delete")
    async def delete_idea(idea_id: str):
        """Delete idea"""
        await idea_service.delete_idea(idea_id)
        return {"status": "deleted"}

class MechanismAPI:
    
    @app.post("/mechanisms/explore")
    @requires_permission("mechanism/explorer", "execute")
    async def explore_idea(request: dict):
        """Run exploration mechanism"""
        idea_id = request["idea_id"]
        depth = request.get("depth", 3)
        
        return await explorer_mechanism.explore_idea(idea_id, depth)
    
    @app.post("/mechanisms/synthesize")
    @requires_permission("mechanism/synthesizer", "execute")  
    async def synthesize_ideas(request: dict):
        """Run synthesis mechanism"""
        idea_ids = request["idea_ids"]
        
        return await synthesizer_mechanism.synthesize_ideas(idea_ids)

class AdminAPI:
    
    @app.post("/admin/reset")
    @admin_required()
    async def reset_system():
        """Reset entire system (admin only)"""
        await system_service.reset()
        return {"status": "system reset"}
    
    @app.get("/admin/rbac/stats")
    @requires_permission("admin/rbac", "read")
    async def rbac_stats():
        """Get RBAC engine statistics"""
        from registry.registry_manager import RegistryManager
        engine = RegistryManager.get_service("rbac_engine")
        return engine.get_stats()
    
    @app.post("/admin/rbac/audit")
    @requires_permission("admin/rbac", "audit")
    async def audit_permission(request: dict):
        """Audit specific permission check"""
        from registry.registry_manager import RegistryManager
        engine = RegistryManager.get_service("rbac_engine")
        
        return engine.audit_permission(
            request["subject"],
            request["resource"], 
            request["action"]
        )

# Background services with RBAC
class BackgroundService:
    
    async def scheduled_cleanup(self):
        """Run as system service"""
        with ExecutionContextScope(subject="system-cleanup"):
            
            # Clean up expired ideas (requires delete permission)
            expired_ideas = await self.find_expired_ideas()
            for idea_id in expired_ideas:
                try:
                    # Check if system can delete this idea
                    if check_permission("system-cleanup", f"idea/{idea_id}", "delete"):
                        await idea_service.delete_idea(idea_id)
                except Exception as e:
                    logger.error(f"Failed to delete expired idea {idea_id}: {e}")
    
    async def mechanism_auto_run(self):
        """Auto-run mechanisms on new ideas"""
        with ExecutionContextScope(subject="mechanism-service"):
            
            new_ideas = await self.find_new_ideas()
            for idea in new_ideas:
                # Check if mechanism service can process this idea
                if check_permission("mechanism-service", f"idea/{idea.id}", "process"):
                    await self.auto_process_idea(idea)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

This comprehensive documentation provides everything needed to understand, implement, and use RBAC in NIREON V4 effectively.