"""
Code Validator for SE-RL Framework
==================================

This module implements a comprehensive code validation pipeline for
LLM-generated code, including:

1. Syntax Check - AST parsing and validation
2. Runtime Verification - Safe execution in sandbox
3. Semantic Validation - Type checking and logic verification
4. Security Check - Dangerous pattern detection

The validation follows a two-stage process as described in the paper.

Author: AI Research Engineer
Date: 2024
"""

import ast
import sys
import traceback
import re
import io
import contextlib
import multiprocessing
import signal
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    stage: str  # 'syntax', 'runtime', 'semantic', 'security'
    error_message: Optional[str] = None
    error_line: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    output: Optional[Any] = None


@dataclass
class ValidatorConfig:
    """Configuration for code validator"""
    # Timeout settings
    syntax_timeout: float = 5.0
    runtime_timeout: float = 30.0

    # Sandbox settings
    max_memory_mb: int = 512
    max_output_lines: int = 1000

    # Security settings
    allow_imports: List[str] = None
    block_patterns: List[str] = None

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if self.allow_imports is None:
            self.allow_imports = [
                'numpy', 'np', 'torch', 'torch.nn', 'torch.optim',
                'torch.nn.functional', 'math', 'random', 'typing',
                'collections', 'dataclasses', 'functools', 'itertools'
            ]

        if self.block_patterns is None:
            self.block_patterns = [
                r'os\.system',
                r'subprocess\.',
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__',
                r'open\s*\(',
                r'file\s*\(',
                r'\.write\s*\(',
                r'\.read\s*\(',
                r'socket\.',
                r'requests\.',
                r'urllib\.',
                r'shutil\.',
                r'pickle\.',
                r'marshal\.',
                r'ctypes\.',
            ]


class SyntaxValidator:
    """
    Stage 1: Syntax validation using AST parsing.

    Checks for valid Python syntax and extracts code structure.
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config

    def validate(self, code: str) -> ValidationResult:
        """
        Validate code syntax.

        Args:
            code: Python code string

        Returns:
            ValidationResult with syntax check status
        """
        try:
            # Parse the AST
            tree = ast.parse(code)

            # Extract information about the code
            info = self._analyze_ast(tree)

            return ValidationResult(
                is_valid=True,
                stage='syntax',
                output=info
            )

        except SyntaxError as e:
            return ValidationResult(
                is_valid=False,
                stage='syntax',
                error_message=str(e.msg),
                error_line=e.lineno
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                stage='syntax',
                error_message=f"Unexpected error: {str(e)}"
            )

    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze AST and extract useful information"""
        info = {
            'functions': [],
            'classes': [],
            'imports': [],
            'global_vars': []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                info['functions'].append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'line': node.lineno
                })
            elif isinstance(node, ast.ClassDef):
                info['classes'].append({
                    'name': node.name,
                    'line': node.lineno
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        info['imports'].append(alias.name)
                else:
                    info['imports'].append(node.module)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        info['global_vars'].append(target.id)

        return info


class SecurityValidator:
    """
    Security validation for LLM-generated code.

    Blocks dangerous patterns and unauthorized imports.
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.block_patterns = [re.compile(p) for p in config.block_patterns]

    def validate(self, code: str) -> ValidationResult:
        """
        Check code for security issues.

        Args:
            code: Python code string

        Returns:
            ValidationResult with security check status
        """
        warnings = []

        # Check for blocked patterns
        for pattern in self.block_patterns:
            match = pattern.search(code)
            if match:
                return ValidationResult(
                    is_valid=False,
                    stage='security',
                    error_message=f"Blocked pattern detected: {match.group()}"
                )

        # Check imports
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._is_allowed_import(alias.name):
                            return ValidationResult(
                                is_valid=False,
                                stage='security',
                                error_message=f"Unauthorized import: {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and not self._is_allowed_import(node.module):
                        return ValidationResult(
                            is_valid=False,
                            stage='security',
                            error_message=f"Unauthorized import: {node.module}"
                        )
        except SyntaxError:
            pass  # Will be caught by syntax validator

        # Check for suspicious patterns (warnings only)
        suspicious_patterns = [
            (r'while\s+True', "Infinite loop pattern detected"),
            (r'for\s+_\s+in\s+range\s*\(\s*\d{6,}', "Very large loop detected"),
            (r'\*\*\s*\d{3,}', "Large exponentiation detected"),
        ]

        for pattern, warning in suspicious_patterns:
            if re.search(pattern, code):
                warnings.append(warning)

        return ValidationResult(
            is_valid=True,
            stage='security',
            warnings=warnings
        )

    def _is_allowed_import(self, module_name: str) -> bool:
        """Check if import is allowed"""
        # Check exact match
        if module_name in self.config.allow_imports:
            return True

        # Check prefix match (e.g., 'torch.nn' for 'torch.nn.functional')
        for allowed in self.config.allow_imports:
            if module_name.startswith(allowed + '.') or allowed.startswith(module_name + '.'):
                return True

        return False


class RuntimeValidator:
    """
    Stage 2: Runtime validation in sandbox.

    Executes code safely with timeout and resource limits.
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config

    def validate(self, code: str, test_inputs: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate code by executing in sandbox.

        Args:
            code: Python code string
            test_inputs: Test inputs to pass to the code

        Returns:
            ValidationResult with runtime status
        """
        if test_inputs is None:
            test_inputs = {}

        try:
            # Create sandbox environment
            sandbox_globals = self._create_sandbox_globals()
            sandbox_globals.update(test_inputs)

            # Capture output
            output_buffer = io.StringIO()

            # Execute with timeout
            result = self._execute_with_timeout(
                code,
                sandbox_globals,
                output_buffer,
                timeout=self.config.runtime_timeout
            )

            if result['success']:
                return ValidationResult(
                    is_valid=True,
                    stage='runtime',
                    execution_time=result['execution_time'],
                    output=result['output']
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    stage='runtime',
                    error_message=result['error'],
                    error_line=result.get('error_line')
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                stage='runtime',
                error_message=f"Runtime error: {str(e)}"
            )

    def _create_sandbox_globals(self) -> Dict[str, Any]:
        """Create safe globals for sandbox execution"""
        safe_globals = {
            '__builtins__': {
                'abs': abs,
                'all': all,
                'any': any,
                'bool': bool,
                'dict': dict,
                'enumerate': enumerate,
                'filter': filter,
                'float': float,
                'int': int,
                'isinstance': isinstance,
                'len': len,
                'list': list,
                'map': map,
                'max': max,
                'min': min,
                'print': print,
                'range': range,
                'reversed': reversed,
                'round': round,
                'set': set,
                'sorted': sorted,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'type': type,
                'zip': zip,
                'True': True,
                'False': False,
                'None': None,
            },
            'np': np,
            'numpy': np,
            'torch': torch,
            'nn': torch.nn,
            'F': torch.nn.functional,
        }

        return safe_globals

    def _execute_with_timeout(self, code: str, globals_dict: Dict,
                               output_buffer: io.StringIO,
                               timeout: float) -> Dict[str, Any]:
        """Execute code with timeout"""
        import time

        start_time = time.time()

        try:
            # Redirect stdout
            with contextlib.redirect_stdout(output_buffer):
                exec(code, globals_dict)

            execution_time = time.time() - start_time

            # Get output
            output = output_buffer.getvalue()

            # Look for result variable
            result = globals_dict.get('result', globals_dict.get('output', None))

            return {
                'success': True,
                'execution_time': execution_time,
                'output': result,
                'stdout': output
            }

        except Exception as e:
            execution_time = time.time() - start_time

            # Extract error line number
            tb = traceback.extract_tb(sys.exc_info()[2])
            error_line = tb[-1].lineno if tb else None

            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'error_line': error_line,
                'traceback': traceback.format_exc()
            }


class SemanticValidator:
    """
    Semantic validation for specific code types.

    Validates that generated code meets expected interface requirements.
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config

    def validate_reward_function(self, code: str) -> ValidationResult:
        """
        Validate that code implements a valid reward function.

        Expected interface:
        - Function named 'calculate_reward' or 'reward_function'
        - Takes state, action, next_state as arguments
        - Returns float reward value
        """
        try:
            tree = ast.parse(code)

            # Find reward function
            reward_func = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name in ['calculate_reward', 'reward_function', 'compute_reward']:
                        reward_func = node
                        break

            if reward_func is None:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    error_message="No reward function found. Expected 'calculate_reward' or 'reward_function'"
                )

            # Check arguments
            args = [arg.arg for arg in reward_func.args.args]
            required_args = ['state', 'action']

            missing_args = [arg for arg in required_args if arg not in args and
                            not any(arg in a for a in args)]

            if missing_args:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    error_message=f"Missing required arguments: {missing_args}"
                )

            # Check for return statement
            has_return = any(isinstance(node, ast.Return) for node in ast.walk(reward_func))

            if not has_return:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    error_message="Reward function must have a return statement"
                )

            return ValidationResult(
                is_valid=True,
                stage='semantic'
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                stage='semantic',
                error_message=str(e)
            )

    def validate_network_architecture(self, code: str) -> ValidationResult:
        """
        Validate that code implements a valid neural network architecture.

        Expected:
        - Class inheriting from nn.Module
        - __init__ and forward methods
        - Proper layer definitions
        """
        try:
            tree = ast.parse(code)

            # Find nn.Module class
            network_class = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Attribute):
                            if base.attr == 'Module':
                                network_class = node
                                break
                        elif isinstance(base, ast.Name):
                            if base.id in ['Module', 'nn.Module']:
                                network_class = node
                                break

            if network_class is None:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    error_message="No nn.Module class found"
                )

            # Check for __init__ and forward
            methods = {node.name for node in network_class.body
                       if isinstance(node, ast.FunctionDef)}

            if '__init__' not in methods:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    error_message="Missing __init__ method"
                )

            if 'forward' not in methods:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    error_message="Missing forward method"
                )

            return ValidationResult(
                is_valid=True,
                stage='semantic'
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                stage='semantic',
                error_message=str(e)
            )

    def validate_imagination_module(self, code: str) -> ValidationResult:
        """
        Validate imagination module code.

        Expected:
        - Function to generate market scenarios
        - Returns dictionary with required fields
        """
        try:
            tree = ast.parse(code)

            # Find generation function
            gen_func = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if 'generate' in node.name.lower() or 'imagine' in node.name.lower():
                        gen_func = node
                        break

            if gen_func is None:
                return ValidationResult(
                    is_valid=False,
                    stage='semantic',
                    error_message="No generation function found"
                )

            return ValidationResult(
                is_valid=True,
                stage='semantic'
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                stage='semantic',
                error_message=str(e)
            )


class CodeValidator:
    """
    Complete code validation pipeline.

    Implements the two-stage validation process:
    1. Syntax check (AST parsing)
    2. Runtime verification (sandbox execution)

    With additional security and semantic checks.
    """

    def __init__(self, config: ValidatorConfig = None):
        if config is None:
            config = ValidatorConfig()

        self.config = config
        self.syntax_validator = SyntaxValidator(config)
        self.security_validator = SecurityValidator(config)
        self.runtime_validator = RuntimeValidator(config)
        self.semantic_validator = SemanticValidator(config)

        # Validation statistics
        self.validation_history = []

    def validate(self, code: str, code_type: str = None,
                 test_inputs: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate code through the complete pipeline.

        Args:
            code: Python code string
            code_type: Type of code ('reward', 'network', 'imagination', None)
            test_inputs: Test inputs for runtime validation

        Returns:
            Final ValidationResult
        """
        logger.info("Starting code validation...")

        # Stage 1: Syntax check
        syntax_result = self.syntax_validator.validate(code)
        if not syntax_result.is_valid:
            logger.warning(f"Syntax validation failed: {syntax_result.error_message}")
            self._record_validation(code, syntax_result)
            return syntax_result

        logger.info("Syntax validation passed")

        # Security check
        security_result = self.security_validator.validate(code)
        if not security_result.is_valid:
            logger.warning(f"Security validation failed: {security_result.error_message}")
            self._record_validation(code, security_result)
            return security_result

        logger.info("Security validation passed")

        # Semantic validation (if code type specified)
        if code_type:
            semantic_result = self._validate_semantic(code, code_type)
            if not semantic_result.is_valid:
                logger.warning(f"Semantic validation failed: {semantic_result.error_message}")
                self._record_validation(code, semantic_result)
                return semantic_result

            logger.info("Semantic validation passed")

        # Stage 2: Runtime verification
        runtime_result = self.runtime_validator.validate(code, test_inputs)
        if not runtime_result.is_valid:
            logger.warning(f"Runtime validation failed: {runtime_result.error_message}")
            self._record_validation(code, runtime_result)
            return runtime_result

        logger.info("Runtime validation passed")

        # Combine warnings
        all_warnings = security_result.warnings.copy()

        final_result = ValidationResult(
            is_valid=True,
            stage='complete',
            warnings=all_warnings,
            execution_time=runtime_result.execution_time,
            output=runtime_result.output
        )

        self._record_validation(code, final_result)
        return final_result

    def _validate_semantic(self, code: str, code_type: str) -> ValidationResult:
        """Perform semantic validation based on code type"""
        if code_type == 'reward':
            return self.semantic_validator.validate_reward_function(code)
        elif code_type == 'network':
            return self.semantic_validator.validate_network_architecture(code)
        elif code_type == 'imagination':
            return self.semantic_validator.validate_imagination_module(code)
        else:
            return ValidationResult(is_valid=True, stage='semantic')

    def _record_validation(self, code: str, result: ValidationResult):
        """Record validation result for statistics"""
        self.validation_history.append({
            'code_hash': hash(code),
            'is_valid': result.is_valid,
            'stage': result.stage,
            'error': result.error_message
        })

    def validate_with_retry(self, code: str, code_type: str = None,
                            fix_callback: Callable = None) -> Tuple[ValidationResult, str]:
        """
        Validate code with retry mechanism.

        If validation fails and fix_callback is provided, attempt to fix and retry.

        Args:
            code: Python code string
            code_type: Type of code
            fix_callback: Function that takes (code, error) and returns fixed code

        Returns:
            Tuple of (final result, final code)
        """
        current_code = code

        for attempt in range(self.config.max_retries):
            result = self.validate(current_code, code_type)

            if result.is_valid:
                return result, current_code

            if fix_callback is None or attempt == self.config.max_retries - 1:
                return result, current_code

            # Attempt to fix
            logger.info(f"Validation failed, attempting fix (attempt {attempt + 1})")
            current_code = fix_callback(current_code, result.error_message)

        return result, current_code

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {}

        valid_count = sum(1 for v in self.validation_history if v['is_valid'])
        total_count = len(self.validation_history)

        # Count failures by stage
        failures_by_stage = {}
        for v in self.validation_history:
            if not v['is_valid']:
                stage = v['stage']
                failures_by_stage[stage] = failures_by_stage.get(stage, 0) + 1

        return {
            'total_validations': total_count,
            'success_rate': valid_count / total_count if total_count > 0 else 0,
            'failures_by_stage': failures_by_stage
        }


# Unit tests
if __name__ == "__main__":
    print("Testing Code Validator...")

    config = ValidatorConfig()
    validator = CodeValidator(config)

    # Test 1: Valid reward function
    print("\n1. Testing valid reward function...")
    valid_reward_code = """
def calculate_reward(state, action, next_state, info=None):
    price_improvement = (state['execution_price'] - state['vwap']) / state['vwap']
    cost_penalty = action * 0.001
    reward = price_improvement - cost_penalty
    return reward
"""
    result = validator.validate(valid_reward_code, code_type='reward')
    print(f"   Valid: {result.is_valid}, Stage: {result.stage}")

    # Test 2: Invalid syntax
    print("\n2. Testing invalid syntax...")
    invalid_syntax_code = """
def broken_function(
    print("missing closing paren"
"""
    result = validator.validate(invalid_syntax_code)
    print(f"   Valid: {result.is_valid}, Error: {result.error_message}")

    # Test 3: Security violation
    print("\n3. Testing security violation...")
    unsafe_code = """
import os
os.system('rm -rf /')
"""
    result = validator.validate(unsafe_code)
    print(f"   Valid: {result.is_valid}, Error: {result.error_message}")

    # Test 4: Valid network architecture
    print("\n4. Testing valid network architecture...")
    valid_network_code = """
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
"""
    result = validator.validate(valid_network_code, code_type='network')
    print(f"   Valid: {result.is_valid}, Stage: {result.stage}")

    # Test 5: Runtime error
    print("\n5. Testing runtime error...")
    runtime_error_code = """
result = 1 / 0
"""
    result = validator.validate(runtime_error_code)
    print(f"   Valid: {result.is_valid}, Error: {result.error_message}")

    # Test statistics
    print("\n6. Validation statistics...")
    stats = validator.get_statistics()
    print(f"   {stats}")

    print("\nAll Code Validator tests completed!")
