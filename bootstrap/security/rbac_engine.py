# -*- coding: utf-8 -*-
"""
RBAC Policy Engine

Simple, efficient engine for checking permissions against loaded RBAC policies.
Integrates with the Combined RBAC Setup Phase for automatic policy loading.
"""

import logging
from typing import List, Dict, Set, Optional, Any
import fnmatch
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RBACPolicyEngine:
    """
    Lightweight RBAC engine for permission checking.
    
    This engine evaluates permissions based on loaded RBAC rules,
    supporting glob patterns for resources and efficient caching.
    """
    
    def __init__(self, policy_sets: List[Any] = None):
        """
        Initialize the RBAC engine with policy sets.
        
        Args:
            policy_sets: List of PolicySetComponent objects from RBAC bootstrap
        """
        self.policy_sets = policy_sets or []
        self.rules = []
        self._permission_cache: Dict[str, bool] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self.last_updated = datetime.now(timezone.utc)
        
        # Extract all rules from policy sets
        self._load_rules_from_policy_sets()
        
        logger.info(f"RBACPolicyEngine initialized with {len(self.rules)} rules from {len(self.policy_sets)} policy sets")
    
    def _load_rules_from_policy_sets(self) -> None:
        """Extract rules from all policy set components"""
        self.rules = []
        
        for policy_set_component in self.policy_sets:
            if hasattr(policy_set_component, 'policy_set') and hasattr(policy_set_component.policy_set, 'rules'):
                self.rules.extend(policy_set_component.policy_set.rules)
                logger.debug(f"Loaded {len(policy_set_component.policy_set.rules)} rules from {policy_set_component.component_id}")
        
        # Clear cache when rules change
        self._permission_cache.clear()
        self.last_updated = datetime.now(timezone.utc)
    
    def add_policy_set(self, policy_set_component: Any) -> None:
        """Add a new policy set to the engine"""
        if policy_set_component not in self.policy_sets:
            self.policy_sets.append(policy_set_component)
            self._load_rules_from_policy_sets()
            logger.info(f"Added policy set {policy_set_component.component_id} to RBAC engine")
    
    def is_allowed(self, subject: str, resource: str, action: str, use_cache: bool = True) -> bool:
        """
        Check if a subject is allowed to perform an action on a resource.
        
        Args:
            subject: The subject (user, role, service account) requesting access
            resource: The resource being accessed (can use glob patterns)
            action: The action being performed (read, write, execute, etc.)
            use_cache: Whether to use permission caching for performance
            
        Returns:
            True if access is allowed, False otherwise
        """
        # Normalize inputs
        subject = subject.lower().strip()
        resource = resource.strip()
        action = action.lower().strip()
        
        # Create cache key
        cache_key = f"{subject}:{resource}:{action}"
        
        # Check cache first
        if use_cache and cache_key in self._permission_cache:
            self._cache_hits += 1
            return self._permission_cache[cache_key]
        
        self._cache_misses += 1
        
        # Evaluate permission
        result = self._evaluate_permission(subject, resource, action)
        
        # Cache result
        if use_cache:
            self._permission_cache[cache_key] = result
        
        logger.debug(f"RBAC check: {subject} -> {resource} ({action}) = {'ALLOW' if result else 'DENY'}")
        return result
    
    def _evaluate_permission(self, subject: str, resource: str, action: str) -> bool:
        """
        Evaluate permission against all rules.
        
        Rules are processed in order, with explicit DENY rules taking precedence.
        """
        # Track if we found any matching allow rules
        has_allow = False
        
        # Process all rules in order
        for rule in self.rules:
            if self._rule_matches(rule, subject, resource, action):
                if rule.effect == 'deny':
                    # Explicit deny always wins
                    logger.debug(f"RBAC: DENY rule matched - {rule.id}")
                    return False
                elif rule.effect == 'allow':
                    has_allow = True
                    logger.debug(f"RBAC: ALLOW rule matched - {rule.id}")
        
        # Return True only if we found at least one allow rule and no deny rules
        return has_allow
    
    def _rule_matches(self, rule: Any, subject: str, resource: str, action: str) -> bool:
        """Check if a rule matches the given subject, resource, and action"""
        
        # Check subject match (exact match or wildcard)
        subject_match = any(
            subject == rule_subject.lower().strip() or 
            rule_subject == '*' or
            fnmatch.fnmatch(subject, rule_subject.lower().strip())
            for rule_subject in rule.subjects
        )
        
        if not subject_match:
            return False
        
        # Check resource match (exact match, wildcard, or glob pattern)
        resource_match = any(
            resource == rule_resource.strip() or
            rule_resource == '*' or
            fnmatch.fnmatch(resource, rule_resource.strip())
            for rule_resource in rule.resources
        )
        
        if not resource_match:
            return False
        
        # Check action match (exact match or wildcard)
        action_match = any(
            action == rule_action.lower().strip() or
            rule_action == '*'
            for rule_action in rule.actions
        )
        
        return action_match
    
    def get_allowed_actions(self, subject: str, resource: str) -> Set[str]:
        """
        Get all actions that a subject is allowed to perform on a resource.
        
        Args:
            subject: The subject requesting access
            resource: The resource being accessed
            
        Returns:
            Set of allowed action strings
        """
        allowed_actions = set()
        
        for rule in self.rules:
            if rule.effect == 'allow' and self._rule_matches_subject_resource(rule, subject, resource):
                # Add all actions from this allow rule
                for action in rule.actions:
                    if action != '*':
                        allowed_actions.add(action.lower().strip())
                    else:
                        # Wildcard - add common actions
                        allowed_actions.update(['read', 'write', 'execute', 'delete'])
        
        # Remove any actions that are explicitly denied
        denied_actions = set()
        for rule in self.rules:
            if rule.effect == 'deny' and self._rule_matches_subject_resource(rule, subject, resource):
                for action in rule.actions:
                    if action != '*':
                        denied_actions.add(action.lower().strip())
                    else:
                        # Wildcard deny - remove all actions
                        return set()
        
        return allowed_actions - denied_actions
    
    def _rule_matches_subject_resource(self, rule: Any, subject: str, resource: str) -> bool:
        """Check if a rule matches subject and resource (used by get_allowed_actions)"""
        subject = subject.lower().strip()
        resource = resource.strip()
        
        # Check subject match
        subject_match = any(
            subject == rule_subject.lower().strip() or 
            rule_subject == '*' or
            fnmatch.fnmatch(subject, rule_subject.lower().strip())
            for rule_subject in rule.subjects
        )
        
        if not subject_match:
            return False
        
        # Check resource match
        resource_match = any(
            resource == rule_resource.strip() or
            rule_resource == '*' or
            fnmatch.fnmatch(resource, rule_resource.strip())
            for rule_resource in rule.resources
        )
        
        return resource_match
    
    def get_subjects_with_access(self, resource: str, action: str) -> List[str]:
        """
        Get all subjects that have access to perform an action on a resource.
        
        Args:
            resource: The resource being accessed
            action: The action being performed
            
        Returns:
            List of subject strings that have access
        """
        allowed_subjects = set()
        
        for rule in self.rules:
            if (rule.effect == 'allow' and 
                self._action_matches(rule, action) and 
                self._resource_matches(rule, resource)):
                allowed_subjects.update(rule.subjects)
        
        # Remove subjects that are explicitly denied
        for rule in self.rules:
            if (rule.effect == 'deny' and 
                self._action_matches(rule, action) and 
                self._resource_matches(rule, resource)):
                for subject in rule.subjects:
                    allowed_subjects.discard(subject)
        
        return list(allowed_subjects)
    
    def _action_matches(self, rule: Any, action: str) -> bool:
        """Check if rule matches the action"""
        return any(
            action.lower().strip() == rule_action.lower().strip() or rule_action == '*'
            for rule_action in rule.actions
        )
    
    def _resource_matches(self, rule: Any, resource: str) -> bool:
        """Check if rule matches the resource"""
        return any(
            resource.strip() == rule_resource.strip() or 
            rule_resource == '*' or
            fnmatch.fnmatch(resource.strip(), rule_resource.strip())
            for rule_resource in rule.resources
        )
    
    def clear_cache(self) -> None:
        """Clear the permission cache"""
        self._permission_cache.clear()
        logger.debug("RBAC permission cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'total_rules': len(self.rules),
            'policy_sets': len(self.policy_sets),
            'cache_size': len(self._permission_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_ratio': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            'last_updated': self.last_updated.isoformat()
        }
    
    def audit_permission(self, subject: str, resource: str, action: str) -> Dict[str, Any]:
        """
        Perform a detailed audit of a permission check.
        
        Returns detailed information about which rules matched and why.
        """
        subject = subject.lower().strip()
        resource = resource.strip()
        action = action.lower().strip()
        
        matched_rules = []
        final_decision = False
        
        for rule in self.rules:
            if self._rule_matches(rule, subject, resource, action):
                rule_info = {
                    'rule_id': rule.id,
                    'effect': rule.effect,
                    'description': getattr(rule, 'description', None),
                    'subjects': rule.subjects,
                    'resources': rule.resources,
                    'actions': rule.actions
                }
                matched_rules.append(rule_info)
                
                # Track final decision
                if rule.effect == 'deny':
                    final_decision = False
                    break  # Deny rules stop processing
                elif rule.effect == 'allow':
                    final_decision = True
        
        return {
            'subject': subject,
            'resource': resource, 
            'action': action,
            'final_decision': final_decision,
            'matched_rules': matched_rules,
            'total_rules_evaluated': len(self.rules),
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
        }