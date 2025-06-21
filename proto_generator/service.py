from __future__ import annotations
import logging
import yaml
import re
import asyncio
from typing import Dict, Any, TYPE_CHECKING, Union
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

PROTO_GENERATOR_METADATA = ComponentMetadata(
    id='proto_generator_main',
    name='ProtoGenerator',
    version='1.0.0',
    category='service',
    description='Generates Proto blocks from natural language requests using an LLM.',
    epistemic_tags=['translator', 'declarative_interface', 'proto_compiler'],
    requires_initialize=True,
    dependencies={'LLMPort': '*'}
)

class ProtoGenerator(NireonBaseComponent):
    METADATA_DEFINITION = PROTO_GENERATOR_METADATA
    PROTO_GENERATION_PROMPT_TEMPLATE = """
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
- requirements (list): Optional list of extra pip packages needed.
- limits (dict): Resource limits.

# Available Dialects (`eidos`):
- 'math': For symbolic/numeric math, plotting. Allowed imports: numpy, scipy, matplotlib, sympy, pandas.
- 'graph': For network analysis. Allowed imports: networkx, numpy, matplotlib.
- 'simulate': For simulations. Allowed imports: simpy, numpy.

# User Request:
"{user_request}"

# Task:
1. Determine the best `eidos` (dialect) for this request.
2. Create a complete, valid YAML Proto block that fulfills the request.
3. The `code` block must be self-contained and define the `function_name`.
4. Ensure all code adheres to the security and import restrictions of the chosen dialect.

Proto Block (YAML):
```yaml
{yaml_placeholder}
"""

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
            return ProcessResult(success=False, message="Input must be a dict with 'natural_language_request'.")

        user_request = data['natural_language_request']
        originating_frame_id = context.metadata.get('current_frame_id')

        prompt = self.PROTO_GENERATION_PROMPT_TEMPLATE.format(
            user_request=user_request,
            yaml_placeholder=''
        ).replace('{yaml_placeholder}', '')

        llm_response = await self.llm.call_llm_async(
            prompt=prompt,
            stage='synthesis',
            role='proto_generator',
            context=context,
            settings={'temperature': 0.1, 'max_tokens': 2048}
        )

        try:
            yaml_content = self._extract_yaml(llm_response.text)
            if not yaml_content:
                raise ValueError('No YAML block found in the LLM response.')
            proto_data = yaml.safe_load(yaml_content)
        except Exception as e:
            return ProcessResult(
                success=False,
                message=f'Failed to parse LLM response as YAML: {e}',
                output_data={'raw_response': llm_response.text}
            )

        typed_proto = self._expand_proto_type(proto_data)
        if isinstance(typed_proto, str):
            return ProcessResult(success=False, message=typed_proto)

        validator = get_validator_for_dialect(typed_proto.eidos)
        validation_errors = validator.validate(typed_proto)
        if validation_errors:
            return ProcessResult(success=False, message=f"Generated Proto block failed validation: {'; '.join(validation_errors)}")

        proto_signal = ProtoTaskSignal(
            source_node_id=self.component_id,
            proto_block=typed_proto.model_dump(),
            dialect=typed_proto.eidos,
            context_tags={'frame_id': originating_frame_id} if originating_frame_id else {}
        )

        if context.event_bus:
            await asyncio.to_thread(context.event_bus.publish, proto_signal.signal_type, proto_signal)

        return ProcessResult(
            success=True,
            message=f"Generated and queued Proto block '{typed_proto.id}'.",
            output_data=typed_proto.model_dump()
        )

    def _infer_dialect(self, request: str) -> str:
        request_lower = request.lower()
        if any(w in request_lower for w in ['plot', 'graph', 'equation', 'calculate', 'math', 'integral', 'derivative']):
            return 'math'
        if any(w in request_lower for w in ['network', 'node', 'edge', 'connect']):
            return 'graph'
        return self.default_dialect

    def _extract_yaml(self, text: str) -> str:
        match = re.search(r'```yaml\n(.*?)\n```', text, re.DOTALL)
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