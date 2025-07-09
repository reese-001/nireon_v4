"""
Utility functions for the QuantifierAgent subsystem.
"""

import re
from typing import List, Dict, Any, Set

class LibraryExtractor:
    """Utility for extracting Python library requirements from code or text."""
    
    # Common module to package mappings
    MODULE_TO_PACKAGE = {
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn', 
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'bs4': 'beautifulsoup4',
        'requests': 'requests',
        'np': 'numpy',
        'pd': 'pandas',
        'plt': 'matplotlib',
        'sns': 'seaborn'
    }
    
    # Standard library modules to ignore
    STDLIB_MODULES = {
        'sys', 'os', 'math', 'json', 'time', 'datetime', 'random', 
        'collections', 'itertools', 'functools', 'pathlib', 're',
        'logging', 'typing', 'abc', 'dataclasses', 'enum'
    }
    
    @classmethod
    def extract_from_code(cls, code: str) -> List[str]:
        """Extract library requirements from Python code."""
        
        import ast
        requirements = set()
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        requirements.add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        requirements.add(module)
        except SyntaxError:
            # Fallback to regex if AST parsing fails
            requirements.update(cls._extract_with_regex(code))
        
        return cls._map_to_packages(requirements)
    
    @classmethod
    def extract_from_text(cls, text: str, available_libraries: Dict[str, List[str]]) -> List[str]:
        """Extract mentioned libraries from text description."""
        
        all_libs = []
        for category_libs in available_libraries.values():
            all_libs.extend(category_libs)
        
        text_lower = text.lower()
        found_libs = []
        
        for lib in all_libs:
            if lib.lower() in text_lower:
                found_libs.append(lib)
        
        return list(set(found_libs))
    
    @classmethod
    def _extract_with_regex(cls, code: str) -> Set[str]:
        """Extract imports using regex as fallback."""
        
        requirements = set()
        
        # Match import statements
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
        ]
        
        for line in code.split('\n'):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1).split('.')[0]
                    requirements.add(module)
        
        return requirements
    
    @classmethod
    def _map_to_packages(cls, modules: Set[str]) -> List[str]:
        """Map module names to PyPI package names."""
        
        packages = []
        
        for module in modules:
            if module in cls.STDLIB_MODULES:
                continue
                
            if module in cls.MODULE_TO_PACKAGE:
                packages.append(cls.MODULE_TO_PACKAGE[module])
            else:
                packages.append(module)
        
        return sorted(list(set(packages)))

class ResponseParser:
    """Utility for parsing structured LLM responses."""
    
    @staticmethod
    def extract_section(text: str, section_name: str) -> str:
        """Extract a named section from structured text."""
        
        lines = text.split('\n')
        section_lines = []
        in_section = False
        
        section_pattern = f"{section_name.upper()}:"
        
        for line in lines:
            if section_pattern in line.upper():
                in_section = True
                # Include remainder of this line
                remainder = line.split(':', 1)[-1].strip()
                if remainder:
                    section_lines.append(remainder)
                continue
            
            if in_section:
                # Stop at next section or empty line
                if ':' in line and line.strip().isupper():
                    break
                if line.strip():
                    section_lines.append(line.strip())
        
        return '\n'.join(section_lines).strip()
    
    @staticmethod
    def extract_yes_no_decision(text: str) -> tuple[bool, float]:
        """Extract YES/NO decision with confidence from text."""
        
        first_line = text.split('\n')[0].strip().upper()
        
        # Direct matches
        if 'YES' in first_line:
            return True, 0.9
        elif 'NO' in first_line:
            return False, 0.9
        
        # Analyze full text for indicators
        text_lower = text.lower()
        positive_words = ['possible', 'viable', 'feasible', 'can be', 'visualized', 'analyzed']
        negative_words = ['impossible', 'not possible', 'cannot', 'not viable', 'abstract only']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return True, 0.6
        elif negative_count > positive_count:
            return False, 0.6
        else:
            return False, 0.3  # Default to not viable if unclear

class ConfigurationValidator:
    """Utility for validating QuantifierAgent configurations."""
    
    @staticmethod
    def validate_library_config(available_libraries: Dict[str, List[str]]) -> List[str]:
        """Validate library configuration and return any issues."""
        
        issues = []
        
        # Check required categories
        required_categories = ['core_data', 'visualization']
        for category in required_categories:
            if category not in available_libraries:
                issues.append(f"Missing required library category: {category}")
        
        # Check for empty categories
        for category, libs in available_libraries.items():
            if not libs:
                issues.append(f"Empty library category: {category}")
        
        # Check for known problematic combinations
        all_libs = sum(available_libraries.values(), [])
        if 'matplotlib' in all_libs and 'plotly' in all_libs:
            # This is actually fine, just noting
            pass
        
        return issues
    
    @staticmethod
    def estimate_resource_usage(config: 'QuantifierConfig') -> Dict[str, Any]:
        """Estimate resource usage for a given configuration."""
        
        # Rough estimates based on configuration
        calls_per_idea = 1 if config.llm_approach == "single_call" else 3
        tokens_per_call = 2000 if config.llm_approach == "single_call" else 1500
        
        return {
            'llm_calls_per_idea': calls_per_idea,
            'estimated_tokens_per_idea': calls_per_idea * tokens_per_call,
            'visualizations_per_idea': config.max_visualizations,
            'supports_mermaid': config.enable_mermaid_output
        }