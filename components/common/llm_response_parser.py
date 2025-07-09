# nireon_v4/components/common/llm_response_parser.py
"""
Unified LLM Response Parser

Provides consistent parsing logic for LLM responses across all NIREON components.
Includes circuit breaker, multiple extraction strategies, and comprehensive error handling.
"""
from typing import Dict, Any, List, Optional, Union, Tuple, Protocol
from dataclasses import dataclass
from enum import Enum
import json
import re
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ParseStatus(Enum):
    """Status of parsing attempt"""
    JSON_SUCCESS = "json_success"
    REGEX_SUCCESS = "regex_success"
    PARTIAL_SUCCESS = "partial_success"
    PARSE_FAILED = "parse_failed"
    EMPTY_RESPONSE = "empty_response"
    ERROR_PATTERN = "error_pattern"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    QUOTA_EXCEEDED = "quota_exceeded"
    CIRCUIT_BREAKER = "circuit_breaker"
    GATEWAY_ERROR = "gateway_error"
    LLM_ERROR = "llm_error"


@dataclass
class ParseResult:
    """Result of parsing attempt"""
    data: Dict[str, Any]
    status: ParseStatus
    confidence: float  # 0.0 to 1.0
    warnings: List[str]
    raw_response: str
    
    @property
    def is_success(self) -> bool:
        return self.status in [ParseStatus.JSON_SUCCESS, ParseStatus.REGEX_SUCCESS]
    
    @property
    def used_defaults(self) -> bool:
        return self.status not in [ParseStatus.JSON_SUCCESS, ParseStatus.REGEX_SUCCESS, ParseStatus.PARTIAL_SUCCESS]


class FieldExtractor(ABC):
    """Base class for field extractors"""
    
    @abstractmethod
    def extract(self, text: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        pass


class NumericFieldExtractor(FieldExtractor):
    """Extract numeric fields with validation"""
    
    def __init__(self, field_name: str, min_val: float = None, max_val: float = None):
        self.field_name = field_name
        self.min_val = min_val
        self.max_val = max_val
        self.patterns = [
            rf'{field_name}\s*[:=]?\s*["\']?(\d+(?:\.\d+)?)["\']?',
            rf'{field_name}_score\s*[:=]?\s*["\']?(\d+(?:\.\d+)?)["\']?',
            rf'"{field_name}"\s*:\s*(\d+(?:\.\d+)?)'
        ]
    
    def extract(self, text: str) -> Optional[float]:
        for pattern in self.patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            return False, "Value is None"
        
        try:
            num_val = float(value)
            if self.min_val is not None and num_val < self.min_val:
                return False, f"Value {num_val} below minimum {self.min_val}"
            if self.max_val is not None and num_val > self.max_val:
                return False, f"Value {num_val} above maximum {self.max_val}"
            return True, None
        except (ValueError, TypeError):
            return False, f"Cannot convert {value} to float"


class TextFieldExtractor(FieldExtractor):
    """Extract text fields"""
    
    def __init__(self, field_name: str, min_length: int = 0, multiline: bool = False):
        self.field_name = field_name
        self.min_length = min_length
        self.patterns = [
            rf'{field_name}\s*[:=]?\s*["\']([^"\']+)["\']',
            rf'"{field_name}"\s*:\s*"([^"]+)"',
        ]
        if multiline:
            # Add patterns that can capture multiline content
            self.patterns.extend([
                rf'{field_name}\s*[:=]?\s*(.+?)(?=\n[A-Z_]+:|$)',  # Until next section
                rf'{field_name}\s*[:=]\s*\n((?:.*\n)*?)(?=\n[A-Z_]+:|$)',  # Multiline after colon
            ])
        else:
            self.patterns.append(rf'{field_name}\s*[:=]?\s*(.+?)(?:\n|$)')
    
    def extract(self, text: str) -> Optional[str]:
        for pattern in self.patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted
        return None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if not value:
            return False, "Empty value"
        if len(str(value)) < self.min_length:
            return False, f"Length {len(str(value))} below minimum {self.min_length}"
        return True, None


class BooleanFieldExtractor(FieldExtractor):
    """Extract boolean values"""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
        self.true_patterns = [
            rf'{field_name}\s*[:=]?\s*(?:yes|true|viable|can\s+be)',
            rf'(?:^|\n)\s*YES\b',
            rf'VIABILITY:\s*YES',
        ]
        self.false_patterns = [
            rf'{field_name}\s*[:=]?\s*(?:no|false|not\s+viable|cannot)',
            rf'(?:^|\n)\s*NO\b',
            rf'VIABILITY:\s*NO',
        ]
    
    def extract(self, text: str) -> Optional[bool]:
        # Check true patterns first
        for pattern in self.true_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Then check false patterns
        for pattern in self.false_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return None
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            return False, "Value is None"
        if isinstance(value, bool):
            return True, None
        return False, f"Value {value} is not a boolean"


@dataclass
class FieldSpec:
    """Specification for a field to extract"""
    name: str
    extractor: FieldExtractor
    default: Any
    required: bool = True


class LLMResponseParser:
    """Unified LLM response parser with circuit breaker"""
    
    # Common error patterns that indicate LLM issues
    ERROR_PATTERNS = [
        'error:', 'exception:', 'failed:', 'timeout:', 
        'rate limit', 'quota exceeded', 'httpx.', 'overloaded',
        'rate_limit'
    ]
    
    def __init__(self, 
                 circuit_breaker_threshold: int = 3,
                 circuit_breaker_cooldown: int = 300):
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown = circuit_breaker_cooldown
        self._consecutive_failures = 0
        self._circuit_breaker_active = False
        self._circuit_breaker_reset_time = None
        
    def parse(self, 
              raw_response: str, 
              field_specs: List[FieldSpec],
              component_id: str = None) -> ParseResult:
        """Parse LLM response with multiple strategies"""
        
        # Check circuit breaker
        if self._is_circuit_breaker_active():
            return self._circuit_breaker_response(raw_response, field_specs)
        
        # Check for empty response
        if not raw_response or not raw_response.strip():
            self._record_failure()
            return ParseResult(
                data={f.name: f.default for f in field_specs},
                status=ParseStatus.EMPTY_RESPONSE,
                confidence=0.0,
                warnings=["Empty LLM response"],
                raw_response=raw_response
            )
        
        # Check for error patterns
        error_status = self._check_error_patterns(raw_response)
        if error_status:
            self._record_failure()
            return ParseResult(
                data={f.name: f.default for f in field_specs},
                status=error_status,
                confidence=0.0,
                warnings=[f"LLM error detected: {error_status.value}"],
                raw_response=raw_response
            )
        
        # Try JSON extraction first
        json_result = self._try_json_extraction(raw_response, field_specs)
        if json_result.is_success:
            self._record_success()
            return json_result
        
        # Fallback to regex extraction
        regex_result = self._try_regex_extraction(raw_response, field_specs)
        if regex_result.is_success:
            self._record_success()
            return regex_result
        
        # Partial extraction - get what we can
        partial_result = self._try_partial_extraction(raw_response, field_specs)
        if partial_result.data:
            self._record_partial_success()
            return partial_result
        
        # Complete failure
        self._record_failure()
        return ParseResult(
            data={f.name: f.default for f in field_specs},
            status=ParseStatus.PARSE_FAILED,
            confidence=0.0,
            warnings=["Failed to parse response with any strategy"],
            raw_response=raw_response
        )
    
    def _try_json_extraction(self, text: str, field_specs: List[FieldSpec]) -> ParseResult:
        """Try to extract JSON from response"""
        # Look for JSON-like structures
        json_patterns = [
            r'\{[^{}]*\}',  # Simple single-level JSON
            r'\{.*\}',      # Any JSON (greedy)
            r'```json\s*(\{.*?\})\s*```',  # Markdown code blocks
            r'```\s*(\{.*?\})\s*```'       # Generic code blocks
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the match
                    json_str = match.strip()
                    if json_str.startswith('```'):
                        json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
                        json_str = re.sub(r'\s*```$', '', json_str)
                    
                    data = json.loads(json_str)
                    
                    # Extract and validate fields
                    result_data = {}
                    warnings = []
                    all_valid = True
                    
                    for spec in field_specs:
                        value = data.get(spec.name, spec.default)
                        is_valid, warning = spec.extractor.validate(value)
                        
                        if is_valid:
                            result_data[spec.name] = value
                        else:
                            if spec.required:
                                all_valid = False
                            result_data[spec.name] = spec.default
                            if warning:
                                warnings.append(f"{spec.name}: {warning}")
                    
                    if all_valid or len(result_data) > len(field_specs) // 2:
                        return ParseResult(
                            data=result_data,
                            status=ParseStatus.JSON_SUCCESS,
                            confidence=0.9 if all_valid else 0.7,
                            warnings=warnings,
                            raw_response=text
                        )
                    
                except (json.JSONDecodeError, ValueError) as e:
                    continue
        
        return ParseResult(
            data={},
            status=ParseStatus.PARSE_FAILED,
            confidence=0.0,
            warnings=["JSON extraction failed"],
            raw_response=text
        )
    
    def _try_regex_extraction(self, text: str, field_specs: List[FieldSpec]) -> ParseResult:
        """Try regex-based extraction"""
        result_data = {}
        warnings = []
        extracted_count = 0
        required_extracted = 0
        
        for spec in field_specs:
            value = spec.extractor.extract(text)
            if value is not None:
                is_valid, warning = spec.extractor.validate(value)
                if is_valid:
                    result_data[spec.name] = value
                    extracted_count += 1
                    if spec.required:
                        required_extracted += 1
                else:
                    result_data[spec.name] = spec.default
                    if warning:
                        warnings.append(f"{spec.name}: {warning}")
            else:
                result_data[spec.name] = spec.default
                if spec.required:
                    warnings.append(f"{spec.name}: not found in response")
        
        # Determine success based on extraction rate
        required_count = sum(1 for spec in field_specs if spec.required)
        if required_count > 0 and required_extracted == required_count:
            return ParseResult(
                data=result_data,
                status=ParseStatus.REGEX_SUCCESS,
                confidence=0.8,
                warnings=warnings,
                raw_response=text
            )
        elif extracted_count > 0:
            return ParseResult(
                data=result_data,
                status=ParseStatus.PARTIAL_SUCCESS,
                confidence=0.5,
                warnings=warnings,
                raw_response=text
            )
        
        return ParseResult(
            data=result_data,
            status=ParseStatus.PARSE_FAILED,
            confidence=0.0,
            warnings=warnings,
            raw_response=text
        )
    
    def _try_partial_extraction(self, text: str, field_specs: List[FieldSpec]) -> ParseResult:
        """Last resort - extract anything we can find"""
        result_data = {}
        warnings = ["Partial extraction only"]
        
        # Try to find any numbers for numeric fields
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        numeric_specs = [s for s in field_specs if isinstance(s.extractor, NumericFieldExtractor)]
        
        for i, spec in enumerate(numeric_specs):
            if i < len(numbers):
                try:
                    value = float(numbers[i])
                    is_valid, _ = spec.extractor.validate(value)
                    if is_valid:
                        result_data[spec.name] = value
                    else:
                        result_data[spec.name] = spec.default
                except:
                    result_data[spec.name] = spec.default
            else:
                result_data[spec.name] = spec.default
        
        # Extract any quoted text for text fields
        text_specs = [s for s in field_specs if isinstance(s.extractor, TextFieldExtractor)]
        quoted_texts = re.findall(r'"([^"]+)"', text)
        
        for i, spec in enumerate(text_specs):
            if i < len(quoted_texts):
                result_data[spec.name] = quoted_texts[i]
            else:
                result_data[spec.name] = spec.default
        
        return ParseResult(
            data=result_data,
            status=ParseStatus.PARTIAL_SUCCESS,
            confidence=0.2,
            warnings=warnings,
            raw_response=text
        )
    
    def _check_error_patterns(self, text: str) -> Optional[ParseStatus]:
        """Check for known error patterns"""
        text_lower = text.lower().strip()
        
        # Specific error checks
        if 'rate limit' in text_lower or 'rate_limit' in text_lower:
            return ParseStatus.RATE_LIMITED
        if 'timeout' in text_lower:
            return ParseStatus.TIMEOUT
        if 'quota' in text_lower and 'exceeded' in text_lower:
            return ParseStatus.QUOTA_EXCEEDED
        
        # Generic error pattern check
        for pattern in self.ERROR_PATTERNS:
            if text_lower.startswith(pattern) or text_lower == pattern.rstrip(':'):
                return ParseStatus.ERROR_PATTERN
        
        return None
    
    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active"""
        if self._circuit_breaker_active and self._circuit_breaker_reset_time:
            if time.time() > self._circuit_breaker_reset_time:
                self._circuit_breaker_active = False
                self._consecutive_failures = 0
                logger.info("Circuit breaker reset")
        
        return self._circuit_breaker_active
    
    def _record_failure(self):
        """Record a parsing failure"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_breaker_threshold:
            self._circuit_breaker_active = True
            self._circuit_breaker_reset_time = time.time() + self.circuit_breaker_cooldown
            logger.error(f"Circuit breaker activated after {self._consecutive_failures} failures")
    
    def _record_success(self):
        """Record a successful parse"""
        self._consecutive_failures = 0
    
    def _record_partial_success(self):
        """Record a partial success"""
        # Don't reset failures completely, but don't increment either
        pass
    
    def _circuit_breaker_response(self, raw_response: str, field_specs: List[FieldSpec]) -> ParseResult:
        """Response when circuit breaker is active"""
        return ParseResult(
            data={f.name: f.default for f in field_specs},
            status=ParseStatus.CIRCUIT_BREAKER,
            confidence=0.0,
            warnings=["Circuit breaker active - using defaults"],
            raw_response=raw_response
        )


# Factory for common parsing patterns
class ParserFactory:
    """Factory for creating parsers for common patterns"""
    
    @staticmethod
    def create_assessment_parser() -> Tuple[LLMResponseParser, List[FieldSpec]]:
        """Parser for assessment responses (Sentinel)"""
        parser = LLMResponseParser()
        specs = [
            FieldSpec(
                name="align_score",
                extractor=NumericFieldExtractor("align", min_val=1.0, max_val=10.0),
                default=5.0,
                required=True
            ),
            FieldSpec(
                name="feas_score",
                extractor=NumericFieldExtractor("feas", min_val=1.0, max_val=10.0),
                default=5.0,
                required=True
            ),
            FieldSpec(
                name="explanation",
                extractor=TextFieldExtractor("explanation", min_length=10),
                default="No explanation provided",
                required=False
            )
        ]
        return parser, specs
    
    @staticmethod
    def create_viability_parser() -> Tuple[LLMResponseParser, List[FieldSpec]]:
        """Parser for viability checks (Quantifier)"""
        parser = LLMResponseParser()
        specs = [
            FieldSpec(
                name="viable",
                extractor=BooleanFieldExtractor("viability"),
                default=False,
                required=True
            ),
            FieldSpec(
                name="confidence",
                extractor=NumericFieldExtractor("confidence", min_val=0.0, max_val=1.0),
                default=0.5,
                required=False
            )
        ]
        return parser, specs
    
    @staticmethod
    def create_comprehensive_analysis_parser() -> Tuple[LLMResponseParser, List[FieldSpec]]:
        """Parser for comprehensive analysis (Quantifier)"""
        parser = LLMResponseParser()
        specs = [
            FieldSpec(
                name="viable",
                extractor=BooleanFieldExtractor("viability"),
                default=False,
                required=True
            ),
            FieldSpec(
                name="approach",
                extractor=TextFieldExtractor("approach", min_length=10),
                default="Traditional visualization approach",
                required=False
            ),
            FieldSpec(
                name="implementation",
                extractor=TextFieldExtractor("implementation", min_length=50, multiline=True),
                default="",
                required=False
            )
        ]
        return parser, specs


# Convenience function for direct parsing
def parse_llm_response(
    raw_response: str,
    response_type: str = "assessment",
    component_id: str = None
) -> ParseResult:
    """
    Convenience function for parsing LLM responses.
    
    Args:
        raw_response: The raw LLM response text
        response_type: Type of response ("assessment", "viability", "comprehensive")
        component_id: ID of the component doing the parsing (for logging)
    
    Returns:
        ParseResult with extracted data
    """
    factory_methods = {
        "assessment": ParserFactory.create_assessment_parser,
        "viability": ParserFactory.create_viability_parser,
        "comprehensive": ParserFactory.create_comprehensive_analysis_parser,
    }
    
    if response_type not in factory_methods:
        raise ValueError(f"Unknown response type: {response_type}")
    
    parser, specs = factory_methods[response_type]()
    return parser.parse(raw_response, specs, component_id)