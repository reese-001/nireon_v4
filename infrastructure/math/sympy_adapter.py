# process [math_engine, principia_agent]

import logging
import asyncio
from typing import Any, Dict
import sympy
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from domain.ports.math_port import MathPort

logger = logging.getLogger(__name__)

# This function must be at the top level to be pickled for multiprocessing
def _execute_computation_in_process(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        expression_str = payload.get('expression')
        operations = payload.get('operations', [])
        variables_str = payload.get('variables', 'x')
        if not expression_str:
            raise ValueError("Payload must contain an 'expression'.")
        
        symbols = sympy.symbols(variables_str)
        if not isinstance(symbols, tuple):
            symbols = (symbols,)
            
        local_context = {s.name: s for s in symbols}
        local_context.update({
            'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan, 
            'exp': sympy.exp, 'log': sympy.log, 'pi': sympy.pi,
            'sqrt': sympy.sqrt, 'atan': sympy.atan
        })
        
        expr = sympy.sympify(expression_str, locals=local_context)
        current_result = expr
        steps = []

        for op in operations:
            op_type = op.get('type')
            step_result = {}
            if op_type == 'differentiate':
                # ... (code is the same as before)
                variable_name = op.get('variable', 'x')
                variable_symbol = next((s for s in symbols if s.name == variable_name), None)
                if not variable_symbol:
                    raise ValueError(f"Variable '{variable_name}' for differentiation not defined.")
                current_result = sympy.diff(current_result, variable_symbol)
                step_result = {'operation': 'differentiate', 'variable': variable_name, 'result': str(current_result)}
            
            elif op_type == 'integrate':
                # ... (code is the same as before)
                variable_name = op.get('variable', 'x')
                limits = op.get('limits')
                variable_symbol = next((s for s in symbols if s.name == variable_name), None)
                if not variable_symbol:
                    raise ValueError(f"Variable '{variable_name}' for integration not defined.")
                
                if limits and len(limits) == 2:
                    integral_expr = sympy.integrate(current_result, (variable_symbol, limits[0], limits[1]))
                    if isinstance(integral_expr, sympy.Integral):
                        integral_val = sympy.N(integral_expr)
                        current_result = integral_val
                    else:
                        current_result = integral_expr
                    step_result = {'operation': 'integrate', 'variable': variable_name, 'limits': limits, 'result': str(current_result)}
                else:
                    current_result = sympy.integrate(current_result, variable_symbol)
                    step_result = {'operation': 'integrate', 'variable': variable_name, 'result': str(current_result)}

            elif op_type == 'evaluate':
                # ... (code is the same as before)
                subs_map = op.get('at', {})
                subs_symbols = {local_context[var]: val for var, val in subs_map.items()}
                current_result = current_result.subs(subs_symbols).evalf()
                step_result = {'operation': 'evaluate', 'at': subs_map, 'result': str(current_result)}
            else:
                raise ValueError(f'Unsupported operation type: {op_type}')
            steps.append(step_result)

        return {
            'success': True,
            'final_result': str(current_result),
            'symbolic_expression': str(expr),
            'steps': steps
        }
    except Exception as e:
        return {'success': False, 'error': f"Computation in subprocess failed: {e}", 'final_result': None}


class SymPyAdapter(MathPort):
    # Set a default timeout for the computation itself
    DEFAULT_COMPUTE_TIMEOUT_S = 60.0

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Allow override via config
        self.compute_timeout = self.config.get('max_compute_seconds', self.DEFAULT_COMPUTE_TIMEOUT_S)
        logger.info(f'SymPyAdapter initialized. Computation timeout set to {self.compute_timeout}s.')

    async def compute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.monotonic()
        loop = asyncio.get_running_loop()
        
        # We create a new executor for each call to ensure clean process termination.
        # This is less efficient but far safer for handling timeouts and hangs.
        with ProcessPoolExecutor(max_workers=1) as executor:
            try:
                logger.info(f"[{self.__class__.__name__}] Submitting computation to a new process with a {self.compute_timeout}s limit.")
                
                future = loop.run_in_executor(
                    executor, _execute_computation_in_process, payload
                )
                
                # Wait for the result with an explicit timeout
                result = await asyncio.wait_for(future, timeout=self.compute_timeout)
                
            except TimeoutError:
                logger.error(f"Computation in subprocess timed out after {self.compute_timeout} seconds. The task was cancelled.")
                return {'success': False, 'error': f'Computation timed out after {self.compute_timeout}s.', 'final_result': None}
            except Exception as e:
                logger.error(f'SymPy computation failed in adapter: {e}', exc_info=True)
                return {'success': False, 'error': str(e), 'final_result': None}
        
        duration = (time.monotonic() - start_time) * 1000
        if result.get('success'):
            result['computation_duration_ms'] = duration
            logger.info(f'[{self.__class__.__name__}] Computation finished in {duration:.2f}ms.')
        
        return result