from __future__ import annotations
import logging
import yaml
import re
import asyncio
import ast
from typing import Dict, Any, TYPE_CHECKING, Union, List, Set
from core.base_component import NireonBaseComponent
from core.lifecycle import ComponentMetadata
from core.results import ProcessResult
from signals.core import ProtoTaskSignal
from domain.proto.base_schema import ProtoBlock, ProtoMathBlock, AnyProtoBlock
from domain.proto.validation import get_validator_for_dialect
from domain.ports.llm_port import LLMPort
if TYPE_CHECKING:
    from domain.context import NireonExecutionContext
logger = logging.getLogger(__name__)
PROTO_GENERATOR_METADATA = ComponentMetadata(id='proto_generator_main', name='ProtoGenerator', version='1.0.0', category='service', description='Generates Proto blocks from natural language requests using an LLM.', epistemic_tags=['translator', 'declarative_interface', 'proto_compiler'], requires_initialize=True, dependencies={'LLMPort': '*'})

class ProtoGenerator(NireonBaseComponent):
    METADATA_DEFINITION = PROTO_GENERATOR_METADATA
    PROTO_GENERATION_PROMPT_TEMPLATE = '''
You are a Proto block generator for the NIREON V4 system.
Your task is to convert a user's natural language request into a valid, executable Proto block in YAML format.

# Proto Block Schema:
- schema_version (str): Always "proto/1.0".
- eidos (str): The execution dialect. Crucial for routing. Guessed from the request.
- id (str): A unique, descriptive ID (e.g., "proto_math_plot_sine_wave").
- description (str): A brief, one-sentence summary of the Proto's purpose.
- objective (str): A more detailed goal for the analysis.
- function_name (str): The Python function to be called (e.g., "main_function").
- inputs (dict): A dictionary of parameters for the function.
- code (str): A multi-line Python script containing the function and any necessary logic.
- requirements (list[str]): A list of required PyPI packages (e.g., ["numpy", "matplotlib"]). Use an empty list [] if no external packages are needed.

# Available Dialects (`eidos`):
- 'math': For symbolic/numeric math, plotting. Allowed imports: numpy, scipy, matplotlib, sympy, pandas.
- 'graph': For network analysis. Allowed imports: networkx, numpy, matplotlib.

# CRITICAL: IMPORT RULES
- The sandboxed execution environment is VERY STRICT.
- You MUST ONLY import from the "Allowed imports" list for the chosen dialect.
- DO NOT import other modules like `time`, `os`, `sys`, or `memory_profiler`. These imports WILL be rejected and cause a failure.

# CRITICAL: HOW TO CREATE ARTIFACTS IN THE SANDBOX
The execution environment uses a special workaround to create files that become artifacts:
- Import pathlib: `from pathlib import Path`
- Write files using Path().write_text(): `Path('filename.ext').write_text(content)`
- These files will be collected as artifacts after execution
- Do NOT use open() or with statements - they are blocked

# Example patterns:
```python
# For text/Mermaid diagrams:
from pathlib import Path
Path('diagram.mmd').write_text(mermaid_content)
Path('instructions.txt').write_text(instructions)

# For matplotlib plots:
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
# ... create plot ...
plt.savefig('plot.png', dpi=150, bbox_inches='tight')
plt.close()

# Always return a meaningful result too:
return {{"status": "success", "files_created": ["diagram.mmd", "instructions.txt"]}}
```

# User Request:
"{user_request}"

# Task:
1.  Determine the best `eidos` (dialect) for this request.
2.  Create a complete, valid YAML Proto block that fulfills the request.
3.  **CRITICAL CODE QUALITY:** The `code` block must be **complete, executable, and self-contained**. It must define the `function_name`.
4.  **DO NOT use placeholders** like `# your code here`, `pass`, or `# implement logic`. All logic must be fully implemented.
5.  If a requested algorithm is too complex, implement a simpler, but still functional, version. For example, if asked for "Sieve of Atkin", it is acceptable to implement the "Sieve of Eratosthenes" as a functional alternative.
6.  Follow the CRITICAL IMPORT RULES exactly.
7.  Use Path().write_text() to create text files (Mermaid diagrams, CSVs, etc.)
8.  Use plt.savefig() to save plots as PNG files
9.  Always return a dictionary with status and list of created files
10. Infer the `requirements` from the imports in your generated code. IMPORTANT: If no external packages are needed, use an empty list: requirements: []
11. **CRITICAL FOR PLOTTING:** If you create a bar chart, you MUST use numeric positions for x-axis:
    ```python
    labels = ['Category A', 'Category B', 'Category C']
    values = [10, 20, 15]
    x_positions = np.arange(len(labels))  # <-- Use numeric positions
    plt.bar(x_positions, values)          # <-- Pass numeric positions
    plt.xticks(x_positions, labels)      # <-- Set labels separately
    ```
12. **CRITICAL FOR YAML VALIDITY:** Do NOT use LaTeX-style escapes like `\(`, `\pi`, or `\[` inside the YAML `description` or `objective` fields. Write them as plain text (e.g., "pi(x)"). These escape characters will break the YAML parser.
13. **CRITICAL CODE STRUCTURE:** The `code` block should ONLY contain function and class definitions. DO NOT include a top-level call to your main function (e.g., `if __name__ == "__main__":` or a direct call like `my_function()`). The execution harness handles the call.

Proto Block (YAML):
```yaml
{yaml_placeholder}
```
'''
    
    def __init__(self, config: Dict[str, Any], metadata_definition: ComponentMetadata | None=None, **kwargs):
        super().__init__(config, metadata_definition or self.METADATA_DEFINITION)
        self.supported_dialects = self.config.get('supported_dialects', ['math', 'graph'])
        self.default_dialect = self.config.get('default_dialect', 'math')
        self.llm: LLMPort | None = None
    
    async def _initialize_impl(self, context: NireonExecutionContext) -> None:
        self.llm = context.component_registry.get_service_instance(LLMPort)
        context.logger.info(f"ProtoGenerator '{self.component_id}' initialized with LLM port.")
    
    async def _process_impl(self, data: Any, context: NireonExecutionContext) -> ProcessResult:
        if not isinstance(data, dict) or 'natural_language_request' not in data:
            return ProcessResult(success=False, component_id=self.component_id, message="Input must be a dict with 'natural_language_request'.")
        user_request = data['natural_language_request']
        originating_frame_id = context.metadata.get('current_frame_id')
        prompt = self.PROTO_GENERATION_PROMPT_TEMPLATE.format(user_request=user_request, yaml_placeholder='').replace('{yaml_placeholder}', '')
        llm_response = await self.llm.call_llm_async(prompt=prompt, stage='synthesis', role='proto_generator', context=context, settings={'temperature': 0.1, 'max_tokens': 2048})
        try:
            yaml_content = self._extract_yaml(llm_response.text)
            logger.info(f'[{self.component_id}] Extracted YAML from LLM:\n---\n{yaml_content}\n---')
            if not yaml_content:
                raise ValueError('No YAML block found in the LLM response.')
            proto_data = yaml.safe_load(yaml_content)
            
            # Fix the requirements field if it's malformed
            proto_data = self._fix_requirements(proto_data)
            
        except Exception as e:
            return ProcessResult(success=False, component_id=self.component_id, message=f'Failed to parse LLM response as YAML: {e}', output_data={'raw_response': llm_response.text})
        typed_proto = self._expand_proto_type(proto_data)
        if isinstance(typed_proto, str):
            return ProcessResult(success=False, component_id=self.component_id, message=typed_proto)
        validator = get_validator_for_dialect(typed_proto.eidos)
        validation_errors = validator.validate(typed_proto)
        if validation_errors:
            return ProcessResult(success=False, component_id=self.component_id, message=f"Generated Proto block failed validation: {'; '.join(validation_errors)}")
        proto_signal = ProtoTaskSignal(source_node_id=self.component_id, proto_block=typed_proto.model_dump(), dialect=typed_proto.eidos, context_tags={'frame_id': originating_frame_id} if originating_frame_id else {})
        if context.event_bus:
            logger.info(f'[{self.component_id}] Publishing ProtoTaskSignal directly to event bus.')
            context.event_bus.publish(proto_signal.signal_type, proto_signal)
        return ProcessResult(success=True, component_id=self.component_id, message=f"Generated and queued Proto block '{typed_proto.id}'.", output_data=typed_proto.model_dump())
    
    def _fix_requirements(self, proto_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common issues with the requirements field"""
        requirements = proto_data.get('requirements', [])
        
        # If requirements is a list containing a single empty list, fix it
        if isinstance(requirements, list) and len(requirements) == 1 and isinstance(requirements[0], list) and len(requirements[0]) == 0:
            proto_data['requirements'] = []
            logger.info(f'[{self.component_id}] Fixed malformed requirements: [[]] -> []')
        
        # If requirements contains non-string items, filter them out
        elif isinstance(requirements, list):
            fixed_requirements = []
            for item in requirements:
                if isinstance(item, str) and item.strip():  # Only keep non-empty strings
                    fixed_requirements.append(item.strip())
                elif isinstance(item, list):  # Skip nested lists
                    logger.warning(f'[{self.component_id}] Skipping nested list in requirements: {item}')
            proto_data['requirements'] = fixed_requirements
            if len(fixed_requirements) != len(requirements):
                logger.info(f'[{self.component_id}] Fixed requirements from {requirements} to {fixed_requirements}')
        
        return proto_data
    
    def _extract_yaml(self, text: str) -> str:
        match = re.search('```yaml\\n(.*?)\\n```', text, re.DOTALL)
        if match:
            return match.group(1)
        if text.strip().startswith('schema_version:'):
            return text
        logger.warning('Could not find a fenced YAML block in the LLM response. Attempting to parse raw text.')
        return text
    
    def _expand_proto_type(self, proto_data: Dict[str, Any]) -> Union[AnyProtoBlock, str]:
        dialect = proto_data.get('eidos')
        if not dialect:
            return "Validation Error: 'eidos' field is missing from the generated Proto block."
        try:
            if dialect == 'math':
                return ProtoMathBlock(**proto_data)
            return ProtoBlock(**proto_data)
        except Exception as e:
            return f"Proto type expansion failed for dialect '{dialect}': {e}"