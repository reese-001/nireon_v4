# Integration Guide: Unified LLM Response Parser

## Overview
This guide walks through integrating the unified LLM response parser into the NIREON V4 codebase.

## File Structure

```
nireon_v4/
├── components/
│   ├── common/
│   │   └── llm_response_parser.py  # NEW FILE - Unified parser
│   └── mechanisms/
│       ├── sentinel/
│       │   └── assessment_core.py  # UPDATED - Uses unified parser
│       └── analyst_agents/
│           └── quantifier_agent/
│               └── analysis_engine.py  # UPDATED - Uses unified parser
```

## Step-by-Step Integration

### 1. Create the Unified Parser Module

Create the new file: `nireon_v4/components/common/llm_response_parser.py`

This file contains:
- `ParseStatus` enum for different parsing outcomes
- `ParseResult` dataclass for structured results
- Field extractors for different data types
- `LLMResponseParser` class with circuit breaker
- `ParserFactory` for common parsing patterns

### 2. Update Sentinel Assessment Core

In `nireon_v4/components/mechanisms/sentinel/assessment_core.py`:

**Changes:**
1. Remove all regex patterns and parsing logic
2. Import the unified parser
3. Create parser instance in `__init__`
4. Replace `_parse_llm_response` with simplified version using unified parser
5. Update error tracking to use parser's circuit breaker

**Key Benefits:**
- Removes ~100 lines of complex parsing code
- Inherits circuit breaker functionality
- Consistent error handling
- Better observability with ParseResult

### 3. Update Quantifier Analysis Engine

In `nireon_v4/components/mechanisms/analyst_agents/quantifier_agent/analysis_engine.py`:

**Changes:**
1. Remove custom extraction methods
2. Import unified parser components
3. Create parsers for viability and comprehensive analysis
4. Use parser results with confidence scores
5. Simplify error handling

**Key Benefits:**
- Removes duplicate parsing logic
- Standardized field extraction
- Confidence scores for all extractions
- Easier to add new fields

### 4. Additional Components to Update

While not shown in the examples, these components could also benefit:

#### Explorer Mechanism
If it needs to parse structured LLM responses, add:
```python
from components.common.llm_response_parser import parse_llm_response

# In processing method:
result = parse_llm_response(llm_response.text, "exploration", self.component_id)
```

#### Catalyst Mechanism
For any structured prompts that need parsing:
```python
# Create custom field specs for catalyst-specific fields
catalyst_specs = [
    FieldSpec(
        name="blend_suggestion",
        extractor=NumericFieldExtractor("blend", min_val=0.0, max_val=1.0),
        default=0.5,
        required=False
    )
]
```

## Testing Strategy

### 1. Unit Tests for Parser
```python
def test_assessment_parser():
    parser, specs = ParserFactory.create_assessment_parser()
    
    # Test JSON response
    json_response = '{"align_score": 8, "feas_score": 7, "explanation": "Good idea"}'
    result = parser.parse(json_response, specs)
    assert result.is_success
    assert result.data['align_score'] == 8
    
    # Test regex fallback
    text_response = 'align_score: 8\nfeas_score: 7\nexplanation: Good idea'
    result = parser.parse(text_response, specs)
    assert result.is_success
    
    # Test error handling
    error_response = 'Error: Rate limit exceeded'
    result = parser.parse(error_response, specs)
    assert result.status == ParseStatus.RATE_LIMITED
```

### 2. Integration Tests
Test that components handle ParseResult correctly:
```python
async def test_sentinel_with_parser():
    # Mock LLM response
    mock_response = LLMResponse(text='{"align_score": 8, "feas_score": 7}')
    
    # Test assessment
    assessment = await sentinel.assessment_core.perform_assessment(idea, refs, ctx)
    assert assessment.trust_score > 0
    assert 'llm_parsing_status' in assessment.metadata
```

## Migration Checklist

- [ ] Create `components/common/llm_response_parser.py`
- [ ] Update `sentinel/assessment_core.py` imports
- [ ] Replace Sentinel parsing logic
- [ ] Update `quantifier_agent/analysis_engine.py` imports  
- [ ] Replace Quantifier parsing logic
- [ ] Add unit tests for parser
- [ ] Add integration tests for updated components
- [ ] Update any other components with LLM parsing
- [ ] Document any custom field extractors needed
- [ ] Update error monitoring to track new ParseStatus values

## Monitoring and Observability

### New Metrics to Track
1. **Parser Success Rate by Component**
   ```python
   metrics.increment(f'llm_parser.{component_id}.{result.status.value}')
   ```

2. **Circuit Breaker Activations**
   ```python
   if result.status == ParseStatus.CIRCUIT_BREAKER:
       alerts.trigger('llm_parser_circuit_breaker', component_id)
   ```

3. **Confidence Scores**
   ```python
   metrics.gauge(f'llm_parser.confidence.{component_id}', result.confidence)
   ```

### Logging Best Practices
```python
# Log parse results with context
logger.info(
    "LLM parse result",
    extra={
        'component_id': component_id,
        'status': result.status.value,
        'confidence': result.confidence,
        'warnings': len(result.warnings),
        'used_defaults': result.used_defaults
    }
)
```

## Extending the Parser

### Adding New Field Types
```python
class ListFieldExtractor(FieldExtractor):
    """Extract list fields from LLM responses"""
    
    def __init__(self, field_name: str, separator: str = ','):
        self.field_name = field_name
        self.separator = separator
    
    def extract(self, text: str) -> Optional[List[str]]:
        # Implementation
        pass
```

### Creating Component-Specific Parsers
```python
class ExplorerParserFactory:
    @staticmethod
    def create_variation_parser():
        parser = LLMResponseParser()
        specs = [
            FieldSpec(
                name="variations",
                extractor=ListFieldExtractor("variations"),
                default=[],
                required=True
            ),
            FieldSpec(
                name="exploration_depth",
                extractor=NumericFieldExtractor("depth", min_val=0, max_val=10),
                default=0,
                required=False
            )
        ]
        return parser, specs
```

## Performance Considerations

1. **Parser Reuse**: Create parser instances once and reuse them
2. **Circuit Breaker Settings**: Tune based on LLM reliability
3. **Regex Compilation**: Patterns are compiled on first use and cached
4. **Memory Usage**: ParseResult includes raw response - consider trimming for large responses

## Common Pitfalls to Avoid

1. **Don't Parse in Hot Paths**: Cache parser instances
2. **Handle All ParseStatus Values**: Always check `result.used_defaults`
3. **Log Warnings**: ParseResult warnings contain valuable debugging info
4. **Update Stats**: Track new metrics for observability
5. **Test Edge Cases**: Empty responses, malformed JSON, partial extractions

## Future Enhancements

1. **Async Field Extraction**: For complex extractors that need I/O
2. **ML-Based Parsing**: Use small models for robust extraction
3. **Schema Validation**: Integrate with Pydantic for type safety
4. **Response Templates**: Define expected response formats
5. **A/B Testing**: Compare parser strategies in production