# NIREON V4 Reactor YAML Rules Guide

## Overview

The NIREON V4 Reactor uses YAML-based rules to define system behavior in response to signals. This guide explains how to write and deploy custom rules.

## Rule Structure

Each rule has the following structure:

```yaml
- id: "unique_rule_identifier"
  description: "Human-readable description"
  namespace: "category_name"
  priority: 10  # Lower = higher priority
  enabled: true
  conditions:
    - type: "condition_type"
      # condition-specific fields
  actions:
    - type: "action_type"
      # action-specific fields