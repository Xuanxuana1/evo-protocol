"""Sandboxed code execution with AST-based safety validation and Neural Oracle support."""

from __future__ import annotations

import ast
import signal
import sys
import types
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

SANDBOX_ALLOWED_IMPORTS = {
    "re",
    "json",
    "math",
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
    "copy",
    "random",
    "statistics",
    "string",
    "operator",
    "abc",
    "enum",
    "datetime",
    "pydantic",
    "networkx",
}

SANDBOX_BANNED_BUILTINS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "input",
    "globals",
    "locals",
    "breakpoint",
    "exit",
    "quit",
}

SANDBOX_BANNED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "tempfile",
    "multiprocessing",
    "threading",
}


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""

    success: bool
    output: str = ""
    error: str = ""
    namespace: Optional[dict] = None


def validate_sandbox_code(code: str) -> Optional[str]:
    """AST-based safety check for sandbox code.

    Returns an error message string if unsafe, or None if the code is safe.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntax error at line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in SANDBOX_ALLOWED_IMPORTS:
                    return f"Forbidden import: {alias.name}"
                if root in SANDBOX_BANNED_MODULES:
                    return f"Banned module: {alias.name}"

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root not in SANDBOX_ALLOWED_IMPORTS:
                return f"Forbidden import-from: {module}"
            if root in SANDBOX_BANNED_MODULES:
                return f"Banned module import-from: {module}"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in SANDBOX_BANNED_BUILTINS:
                return f"Forbidden builtin: {node.func.id}()"
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id in SANDBOX_BANNED_MODULES:
                    return f"Forbidden module call: {node.func.value.id}.{node.func.attr}()"

        if isinstance(node, ast.While):
            return "while-loops are forbidden; use bounded for-loops."

    return None


class _Timeout:
    """Context manager for SIGALRM-based timeout (Unix only)."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._old_handler = None

    def _handler(self, signum, frame):
        raise TimeoutError(f"Code execution timed out after {self.seconds}s")

    def __enter__(self):
        try:
            self._old_handler = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        except (AttributeError, ValueError):
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        except (AttributeError, ValueError):
            pass
        return False


def execute_sandbox_code(
    code: str,
    oracle_fn: Optional[Callable] = None,
    timeout: int = 30,
) -> ExecutionResult:
    """Execute sandbox code (env_code, or env_code + solver_code) in a safe namespace.

    The _oracle() function is injected into the namespace for Neural Oracle calls.
    The FINAL_ANSWER variable is extracted from the namespace on success.

    Args:
        code: Python code string to execute.
        oracle_fn: The _oracle(prompt, return_type) callable for Neural Oracles.
        timeout: Maximum execution time in seconds.
    """

    safety_error = validate_sandbox_code(code)
    if safety_error:
        return ExecutionResult(success=False, error=f"Safety check failed: {safety_error}")

    module_name = f"__sandbox_exec_{uuid.uuid4().hex}"
    sandbox_module = types.ModuleType(module_name)
    namespace = sandbox_module.__dict__
    namespace["__builtins__"] = _safe_builtins()
    namespace["__name__"] = module_name
    namespace["__package__"] = None

    # Register sandbox module so decorators like @dataclass can resolve module globals.
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = sandbox_module

    # Inject Neural Oracle into namespace
    if oracle_fn is not None:
        namespace["_oracle"] = oracle_fn

    try:
        with _Timeout(timeout):
            exec(code, namespace)  # noqa: S102
    except TimeoutError as exc:
        return ExecutionResult(success=False, error=str(exc))
    except AssertionError as exc:
        # AssertionError = Parametric Gravity blocked by sandbox constraint
        return ExecutionResult(
            success=False,
            error=f"AssertionError (constraint violation): {exc}",
        )
    except Exception as exc:
        return ExecutionResult(success=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    namespace.pop("__builtins__", None)
    namespace.pop("_oracle", None)

    # Check for FINAL_ANSWER
    final_answer = namespace.get("FINAL_ANSWER")
    output = str(final_answer) if final_answer is not None else ""

    return ExecutionResult(success=True, output=output, namespace=namespace)


# Backward-compatible aliases
def execute_compilation_code(code: str, timeout: int = 30) -> ExecutionResult:
    """Execute context-compilation code and return the resulting namespace."""
    return execute_sandbox_code(code, oracle_fn=None, timeout=timeout)


def execute_query_code(query_code: str, sandbox_namespace: dict, timeout: int = 15) -> ExecutionResult:
    """Execute query code within an existing sandbox namespace (legacy interface)."""

    safety_error = validate_sandbox_code(query_code)
    if safety_error:
        return ExecutionResult(success=False, error=f"Safety check failed: {safety_error}")

    merged = dict(sandbox_namespace)
    merged["__builtins__"] = _safe_builtins()

    try:
        with _Timeout(timeout):
            exec(query_code, merged)  # noqa: S102
    except TimeoutError as exc:
        return ExecutionResult(success=False, error=str(exc))
    except Exception as exc:
        return ExecutionResult(success=False, error=f"{type(exc).__name__}: {exc}")

    answer = merged.get("_answer") or merged.get("FINAL_ANSWER")
    if answer is None:
        return ExecutionResult(
            success=False,
            error="Query code did not set '_answer' or 'FINAL_ANSWER'.",
        )

    return ExecutionResult(success=True, output=str(answer))


def _safe_builtins() -> dict:
    """Return a restricted builtins dict for sandboxed execution."""

    import builtins

    safe = {}
    for name in dir(builtins):
        if name.startswith("_"):
            continue
        if name in SANDBOX_BANNED_BUILTINS:
            continue
        safe[name] = getattr(builtins, name)

    safe["__build_class__"] = builtins.__build_class__
    safe["__name__"] = "__sandbox__"
    safe["__import__"] = _restricted_import
    return safe


def _restricted_import(name, *args, **kwargs):
    """Import guard that only allows sandbox-approved modules."""

    root = name.split(".")[0]
    if root not in SANDBOX_ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed in sandbox.")
    if root in SANDBOX_BANNED_MODULES:
        raise ImportError(f"Module '{name}' is banned in sandbox.")
    return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) else __import__(name, *args, **kwargs)
