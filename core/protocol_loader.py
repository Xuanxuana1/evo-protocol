"""Protocol validation, dynamic loading, and guarded execution helpers."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import multiprocessing as mp
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Any

from core.base_protocol import BaseProtocol, ProtocolResult
from core.base_sandbox_protocol import BaseCaSCompiler, BaseSandboxProtocol, SandboxResult
from core.base_tdg_protocol import BaseTDGCompiler

ALLOWED_IMPORTS = {
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
    "core.base_protocol",
}

BANNED_BUILTINS = {
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

BANNED_MODULE_NAMES = {"os", "sys", "subprocess", "pathlib", "socket", "requests", "urllib"}


def compute_sha(code: str) -> str:
    """Compute short SHA for protocol versioning."""

    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]


def _serialize_result(result: ProtocolResult) -> dict[str, Any]:
    """Serialize ProtocolResult to a plain dict for IPC transport."""

    return {
        "answer": result.answer,
        "confidence": float(result.confidence),
        "reasoning_trace": list(result.reasoning_trace),
        "verification_passed": bool(result.verification_passed),
        "tokens_used": int(result.tokens_used),
        "prompt_tokens": int(result.prompt_tokens),
        "completion_tokens": int(result.completion_tokens),
        "metadata": dict(result.metadata),
    }


def _run_protocol_in_subprocess(pipe_conn, protocol: BaseProtocol, context: str, query: str) -> None:
    """Subprocess entrypoint for hard-timeout execution."""

    payload: dict[str, Any]
    try:
        result = protocol.run(context, query)
        payload = {"ok": True, "result": _serialize_result(result)}
    except BaseException as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        pipe_conn.send(payload)
    except Exception:
        pass
    finally:
        pipe_conn.close()


@dataclass
class ValidationError:
    """Structured error information for self-repair prompts."""

    stage: str
    message: str
    fixable: bool = False

    def to_feedback(self) -> str:
        """Render a compact feedback string for the architect model."""

        tail = "This is likely fixable via targeted edits." if self.fixable else "Please rewrite the problematic section."
        return f"Stage: {self.stage}\nError: {self.message}\n{tail}"


class ProtocolLoader:
    """Validation pipeline + dynamic loader for generated protocol code."""

    def __init__(
        self,
        llm_client,
        model_name: str,
        timeout_seconds: int = 300,
        max_llm_calls_per_task: int = 20,
        use_process_timeout: bool = True,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_llm_calls_per_task = max(1, int(max_llm_calls_per_task))
        self._process_ctx = None
        if use_process_timeout:
            try:
                # Hard-timeout mode is only reliable with fork-based workers.
                self._process_ctx = mp.get_context("fork")
            except ValueError:
                self._process_ctx = None

    def _check_syntax(self, code: str) -> Optional[ValidationError]:
        try:
            ast.parse(code)
            return None
        except SyntaxError as exc:
            return ValidationError(
                stage="syntax",
                message=f"Line {exc.lineno}: {exc.msg}",
                fixable=True,
            )

    def _check_lint(self, code: str) -> Optional[ValidationError]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as file_obj:
            file_obj.write(code)
            tmp_path = Path(file_obj.name)

        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    str(tmp_path),
                    "--select",
                    "F,E,B",
                    "--output-format",
                    "text",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                message = (result.stdout or result.stderr).strip()
                if message:
                    return ValidationError(stage="lint", message=message, fixable=True)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Ruff is optional; skip this stage if unavailable.
            return None
        finally:
            tmp_path.unlink(missing_ok=True)
        return None

    def _check_security(self, code: str) -> Optional[ValidationError]:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if alias.name not in ALLOWED_IMPORTS and root not in ALLOWED_IMPORTS:
                        return ValidationError(
                            stage="security",
                            message=f"Forbidden import: {alias.name}",
                            fixable=True,
                        )
                    if root in BANNED_MODULE_NAMES:
                        return ValidationError(
                            stage="security",
                            message=f"Banned module import: {alias.name}",
                            fixable=True,
                        )

            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if module not in ALLOWED_IMPORTS and root not in ALLOWED_IMPORTS:
                    return ValidationError(
                        stage="security",
                        message=f"Forbidden import-from: {module}",
                        fixable=True,
                    )
                if root in BANNED_MODULE_NAMES:
                    return ValidationError(
                        stage="security",
                        message=f"Banned module import-from: {module}",
                        fixable=True,
                    )

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
                    return ValidationError(
                        stage="security",
                        message=f"Forbidden builtin call: {node.func.id}()",
                        fixable=True,
                    )

                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id in BANNED_MODULE_NAMES:
                        return ValidationError(
                            stage="security",
                            message=f"Forbidden call through module {node.func.value.id}.",
                            fixable=True,
                        )

            # Unbounded while loops are a common source of hanging protocols.
            # Require bounded for-loops so timeout failures are less likely.
            if isinstance(node, ast.While):
                return ValidationError(
                    stage="security",
                    message="Forbidden while-loop: use bounded for-loops with explicit max iterations.",
                    fixable=True,
                )

        return None

    def validate(self, code: str) -> Optional[ValidationError]:
        """Run all validation stages; return first error if found."""

        for check in (self._check_syntax, self._check_lint, self._check_security):
            error = check(code)
            if error is not None:
                return error
        return None

    def load_from_file(self, file_path: str):
        """Load a protocol subclass from a Python source file."""

        module_name = f"evolved_protocol_{Path(file_path).stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            module.BaseProtocol = BaseProtocol
            module.ProtocolResult = ProtocolResult
            module.Any = Any
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseProtocol) and obj is not BaseProtocol:
                    protocol = obj(self.llm_client, self.model_name)
                    setattr(protocol, "max_llm_calls_per_task", self.max_llm_calls_per_task)
                    return protocol

        except Exception:
            return None
        finally:
            sys.modules.pop(module_name, None)

        return None

    def load_from_code(self, code: str) -> tuple[Optional[BaseProtocol], Optional[ValidationError]]:
        """Validate and load protocol from source code string."""

        error = self.validate(code)
        if error is not None:
            return None, error

        sha = compute_sha(code)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{sha}.py",
            encoding="utf-8",
            delete=False,
        ) as file_obj:
            file_obj.write(code)
            temp_path = Path(file_obj.name)

        try:
            protocol = self.load_from_file(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

        if protocol is None:
            return None, ValidationError(
                stage="load",
                message="No valid BaseProtocol subclass could be loaded.",
                fixable=True,
            )

        return protocol, None

    def run_with_timeout(self, protocol: BaseProtocol, context: str, query: str) -> Optional[ProtocolResult]:
        """Run protocol with hard subprocess timeout when available."""

        if self._process_ctx is not None:
            result = self._run_with_process_timeout(protocol, context, query)
            if result is not None:
                return result
            return None

        return self._run_with_thread_timeout(protocol, context, query)

    def _run_with_thread_timeout(
        self, protocol: BaseProtocol, context: str, query: str
    ) -> Optional[ProtocolResult]:
        """Fallback timeout mode using daemon thread join."""

        result_box: dict[str, ProtocolResult] = {}
        error_box: dict[str, BaseException] = {}

        def _target() -> None:
            try:
                result_box["result"] = protocol.run(context, query)
            except BaseException as exc:
                error_box["error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            return None

        if "error" in error_box:
            error = error_box["error"]
            traceback.print_exception(type(error), error, error.__traceback__)
            return None

        return result_box.get("result")

    def _run_with_process_timeout(
        self, protocol: BaseProtocol, context: str, query: str
    ) -> Optional[ProtocolResult]:
        """Execute task in child process and force-kill on timeout."""

        assert self._process_ctx is not None
        parent_conn, child_conn = self._process_ctx.Pipe(duplex=False)
        process = self._process_ctx.Process(
            target=_run_protocol_in_subprocess,
            args=(child_conn, protocol, context, query),
            daemon=True,
        )

        try:
            process.start()
        except Exception:
            parent_conn.close()
            child_conn.close()
            return self._run_with_thread_timeout(protocol, context, query)
        finally:
            child_conn.close()

        process.join(timeout=self.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            parent_conn.close()
            return None

        payload: dict[str, Any] | None = None
        try:
            if parent_conn.poll(0.2):
                payload = parent_conn.recv()
        except (EOFError, OSError):
            payload = None
        finally:
            parent_conn.close()

        if not payload:
            return None
        if not bool(payload.get("ok")):
            error_tb = str(payload.get("traceback") or payload.get("error") or "").strip()
            if error_tb:
                print(error_tb, file=sys.stderr)
            return None

        result_payload = payload.get("result", {})
        if not isinstance(result_payload, dict):
            return None
        try:
            return ProtocolResult(
                answer=str(result_payload.get("answer", "")),
                confidence=float(result_payload.get("confidence", 0.0)),
                reasoning_trace=list(result_payload.get("reasoning_trace", [])),
                verification_passed=bool(result_payload.get("verification_passed", False)),
                tokens_used=int(result_payload.get("tokens_used", 0)),
                prompt_tokens=int(result_payload.get("prompt_tokens", 0)),
                completion_tokens=int(result_payload.get("completion_tokens", 0)),
                metadata=dict(result_payload.get("metadata", {})),
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Sandbox protocol loader (CaS mode)
# ---------------------------------------------------------------------------

SANDBOX_ALLOWED_IMPORTS = ALLOWED_IMPORTS | {
    "pydantic",
    "networkx",
    "enum",
    "datetime",
    "string",
    "operator",
    "abc",
    "core.base_sandbox_protocol",
}


def _serialize_sandbox_result(result: SandboxResult) -> dict[str, Any]:
    """Serialize SandboxResult to a plain dict for IPC transport."""

    return {
        "answer": result.answer,
        "confidence": float(result.confidence),
        "reasoning_trace": list(result.reasoning_trace),
        "verification_passed": bool(result.verification_passed),
        "tokens_used": int(result.tokens_used),
        "prompt_tokens": int(result.prompt_tokens),
        "completion_tokens": int(result.completion_tokens),
        "metadata": dict(result.metadata),
        "sandbox_code": result.sandbox_code,
        "solver_code": result.solver_code,
        "execution_output": result.execution_output,
        "execution_success": bool(result.execution_success),
        "compilation_success": bool(result.compilation_success),
    }


def _run_sandbox_protocol_in_subprocess(
    pipe_conn,
    protocol: BaseCaSCompiler,
    context: str,
    query: str,
) -> None:
    """Subprocess entrypoint for CaS compiler hard-timeout execution."""

    payload: dict[str, Any]
    try:
        result = protocol.run(context, query)
        payload = {"ok": True, "result": _serialize_sandbox_result(result)}
    except BaseException as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        pipe_conn.send(payload)
    except Exception:
        pass
    finally:
        pipe_conn.close()


class SandboxProtocolLoader:
    """Validation pipeline + dynamic loader for CaS sandbox protocol code."""

    def __init__(
        self,
        llm_client,
        model_name: str,
        timeout_seconds: int = 300,
        sandbox_timeout_seconds: int = 30,
        max_llm_calls_per_task: int = 20,
        use_process_timeout: bool = True,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.sandbox_timeout_seconds = max(1, int(sandbox_timeout_seconds))
        self.max_llm_calls_per_task = max(1, int(max_llm_calls_per_task))
        self._process_ctx = None
        if use_process_timeout:
            try:
                self._process_ctx = mp.get_context("fork")
            except ValueError:
                self._process_ctx = None

    def _check_syntax(self, code: str) -> Optional[ValidationError]:
        try:
            ast.parse(code)
            return None
        except SyntaxError as exc:
            return ValidationError(
                stage="syntax",
                message=f"Line {exc.lineno}: {exc.msg}",
                fixable=True,
            )

    def _check_lint(self, code: str) -> Optional[ValidationError]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as file_obj:
            file_obj.write(code)
            tmp_path = Path(file_obj.name)

        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    str(tmp_path),
                    "--select",
                    "F,E,B",
                    "--output-format",
                    "text",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                message = (result.stdout or result.stderr).strip()
                if message:
                    return ValidationError(stage="lint", message=message, fixable=True)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        finally:
            tmp_path.unlink(missing_ok=True)
        return None

    def _check_security(self, code: str) -> Optional[ValidationError]:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if alias.name not in SANDBOX_ALLOWED_IMPORTS and root not in SANDBOX_ALLOWED_IMPORTS:
                        return ValidationError(
                            stage="security",
                            message=f"Forbidden import: {alias.name}",
                            fixable=True,
                        )
                    if root in BANNED_MODULE_NAMES:
                        return ValidationError(
                            stage="security",
                            message=f"Banned module import: {alias.name}",
                            fixable=True,
                        )

            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if module not in SANDBOX_ALLOWED_IMPORTS and root not in SANDBOX_ALLOWED_IMPORTS:
                    return ValidationError(
                        stage="security",
                        message=f"Forbidden import-from: {module}",
                        fixable=True,
                    )
                if root in BANNED_MODULE_NAMES:
                    return ValidationError(
                        stage="security",
                        message=f"Banned module import-from: {module}",
                        fixable=True,
                    )

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
                    return ValidationError(
                        stage="security",
                        message=f"Forbidden builtin call: {node.func.id}()",
                        fixable=True,
                    )
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id in BANNED_MODULE_NAMES:
                        return ValidationError(
                            stage="security",
                            message=f"Forbidden call through module {node.func.value.id}.",
                            fixable=True,
                        )

            if isinstance(node, ast.While):
                return ValidationError(
                    stage="security",
                    message="Forbidden while-loop: use bounded for-loops with explicit max iterations.",
                    fixable=True,
                )

        return None

    def _check_cas_contract(self, code: str) -> Optional[ValidationError]:
        """Enforce that evolved CaS classes only customize compile/solve behavior."""

        tree = ast.parse(code)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            base_names: set[str] = set()
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.add(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.add(base.attr)

            if not ({"BaseCaSCompiler", "BaseSandboxProtocol"} & base_names):
                continue

            methods = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

            required = {"compile_sandbox", "generate_solver"}
            missing = sorted(required - methods)
            if missing:
                return ValidationError(
                    stage="contract",
                    message=f"CaS subclass {node.name} is missing required method(s): {', '.join(missing)}",
                    fixable=True,
                )

            forbidden = {"run", "_call_llm", "_make_oracle_fn"}
            overridden = sorted(forbidden & methods)
            if overridden:
                return ValidationError(
                    stage="contract",
                    message=(
                        f"CaS subclass {node.name} overrides immutable runtime method(s): "
                        f"{', '.join(overridden)}"
                    ),
                    fixable=True,
                )

        return None

    def validate(self, code: str) -> Optional[ValidationError]:
        """Run all validation stages; return first error if found."""

        for check in (self._check_syntax, self._check_lint, self._check_security, self._check_cas_contract):
            error = check(code)
            if error is not None:
                return error
        return None

    def load_from_file(self, file_path: str):
        """Load a CaS compiler subclass from a Python source file."""

        module_name = f"evolved_sandbox_{Path(file_path).stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            module.BaseCaSCompiler = BaseCaSCompiler
            module.BaseSandboxProtocol = BaseSandboxProtocol
            module.SandboxResult = SandboxResult
            module.Any = Any
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseCaSCompiler) and obj is not BaseCaSCompiler:
                    protocol = obj(self.llm_client, self.model_name)
                    setattr(protocol, "sandbox_timeout_seconds", self.sandbox_timeout_seconds)
                    setattr(protocol, "max_llm_calls_per_task", self.max_llm_calls_per_task)
                    return protocol

        except Exception:
            return None
        finally:
            sys.modules.pop(module_name, None)

        return None

    def load_from_code(self, code: str) -> tuple[Optional[BaseCaSCompiler], Optional[ValidationError]]:
        """Validate and load CaS compiler from source code string."""

        error = self.validate(code)
        if error is not None:
            return None, error

        sha = compute_sha(code)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{sha}.py",
            encoding="utf-8",
            delete=False,
        ) as file_obj:
            file_obj.write(code)
            temp_path = Path(file_obj.name)

        try:
            protocol = self.load_from_file(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

        if protocol is None:
            return None, ValidationError(
                stage="load",
                message="No valid BaseCaSCompiler subclass could be loaded.",
                fixable=True,
            )

        return protocol, None

    def run_with_timeout(
        self,
        protocol: BaseCaSCompiler,
        context: str,
        query: str,
    ) -> Optional[SandboxResult]:
        """Run sandbox protocol with hard subprocess timeout when available."""

        if self._process_ctx is not None:
            result = self._run_with_process_timeout(protocol, context, query)
            if result is not None:
                return result
            return None

        return self._run_with_thread_timeout(protocol, context, query)

    def _run_with_thread_timeout(
        self,
        protocol: BaseCaSCompiler,
        context: str,
        query: str,
    ) -> Optional[SandboxResult]:
        """Fallback timeout mode using daemon thread join."""

        result_box: dict[str, SandboxResult] = {}
        error_box: dict[str, BaseException] = {}

        def _target() -> None:
            try:
                result_box["result"] = protocol.run(context, query)
            except BaseException as exc:
                error_box["error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            return None

        if "error" in error_box:
            error = error_box["error"]
            traceback.print_exception(type(error), error, error.__traceback__)
            return None

        return result_box.get("result")

    def _run_with_process_timeout(
        self,
        protocol: BaseCaSCompiler,
        context: str,
        query: str,
    ) -> Optional[SandboxResult]:
        """Execute task in child process and force-kill on timeout."""

        assert self._process_ctx is not None
        parent_conn, child_conn = self._process_ctx.Pipe(duplex=False)
        process = self._process_ctx.Process(
            target=_run_sandbox_protocol_in_subprocess,
            args=(child_conn, protocol, context, query),
            daemon=True,
        )

        try:
            process.start()
        except Exception:
            parent_conn.close()
            child_conn.close()
            return self._run_with_thread_timeout(protocol, context, query)
        finally:
            child_conn.close()

        process.join(timeout=self.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            parent_conn.close()
            return None

        payload: dict[str, Any] | None = None
        try:
            if parent_conn.poll(0.2):
                payload = parent_conn.recv()
        except (EOFError, OSError):
            payload = None
        finally:
            parent_conn.close()

        if not payload:
            return None
        if not bool(payload.get("ok")):
            error_tb = str(payload.get("traceback") or payload.get("error") or "").strip()
            if error_tb:
                print(error_tb, file=sys.stderr)
            return None

        result_payload = payload.get("result", {})
        if not isinstance(result_payload, dict):
            return None
        try:
            return SandboxResult(
                answer=str(result_payload.get("answer", "")),
                confidence=float(result_payload.get("confidence", 0.0)),
                reasoning_trace=list(result_payload.get("reasoning_trace", [])),
                verification_passed=bool(result_payload.get("verification_passed", False)),
                tokens_used=int(result_payload.get("tokens_used", 0)),
                prompt_tokens=int(result_payload.get("prompt_tokens", 0)),
                completion_tokens=int(result_payload.get("completion_tokens", 0)),
                metadata=dict(result_payload.get("metadata", {})),
                sandbox_code=str(result_payload.get("sandbox_code", "")),
                solver_code=str(result_payload.get("solver_code", "")),
                execution_output=str(result_payload.get("execution_output", "")),
                execution_success=bool(result_payload.get("execution_success", False)),
                compilation_success=bool(result_payload.get("compilation_success", False)),
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# TDG protocol loader (Test-Driven Generation mode)
# ---------------------------------------------------------------------------

TDG_ALLOWED_IMPORTS = SANDBOX_ALLOWED_IMPORTS | {
    "core.base_tdg_protocol",
}


def _run_tdg_protocol_in_subprocess(
    pipe_conn,
    protocol: BaseTDGCompiler,
    context: str,
    query: str,
    messages_raw: list | None = None,
) -> None:
    """Subprocess entrypoint for TDG compiler hard-timeout execution."""

    payload: dict[str, Any]
    try:
        result = protocol.run(context, query, messages_raw=messages_raw)
        payload = {"ok": True, "result": _serialize_sandbox_result(result)}
    except BaseException as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    try:
        pipe_conn.send(payload)
    except Exception:
        pass
    finally:
        pipe_conn.close()


class TDGProtocolLoader:
    """Validation pipeline + dynamic loader for TDG protocol code."""

    def __init__(
        self,
        llm_client,
        model_name: str,
        timeout_seconds: int = 300,
        sandbox_timeout_seconds: int = 30,
        max_llm_calls_per_task: int = 20,
        use_process_timeout: bool = True,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.sandbox_timeout_seconds = max(1, int(sandbox_timeout_seconds))
        self.max_llm_calls_per_task = max(1, int(max_llm_calls_per_task))
        self._process_ctx = None
        if use_process_timeout:
            try:
                self._process_ctx = mp.get_context("fork")
            except ValueError:
                self._process_ctx = None

    def _check_syntax(self, code: str) -> Optional[ValidationError]:
        try:
            ast.parse(code)
            return None
        except SyntaxError as exc:
            return ValidationError(
                stage="syntax",
                message=f"Line {exc.lineno}: {exc.msg}",
                fixable=True,
            )

    def _check_lint(self, code: str) -> Optional[ValidationError]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as file_obj:
            file_obj.write(code)
            tmp_path = Path(file_obj.name)

        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    str(tmp_path),
                    "--select",
                    "F,E,B",
                    "--output-format",
                    "text",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                message = (result.stdout or result.stderr).strip()
                if message:
                    return ValidationError(stage="lint", message=message, fixable=True)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        finally:
            tmp_path.unlink(missing_ok=True)
        return None

    def _check_security(self, code: str) -> Optional[ValidationError]:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if alias.name not in TDG_ALLOWED_IMPORTS and root not in TDG_ALLOWED_IMPORTS:
                        return ValidationError(
                            stage="security",
                            message=f"Forbidden import: {alias.name}",
                            fixable=True,
                        )
                    if root in BANNED_MODULE_NAMES:
                        return ValidationError(
                            stage="security",
                            message=f"Banned module import: {alias.name}",
                            fixable=True,
                        )

            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if module not in TDG_ALLOWED_IMPORTS and root not in TDG_ALLOWED_IMPORTS:
                    return ValidationError(
                        stage="security",
                        message=f"Forbidden import-from: {module}",
                        fixable=True,
                    )
                if root in BANNED_MODULE_NAMES:
                    return ValidationError(
                        stage="security",
                        message=f"Banned module import-from: {module}",
                        fixable=True,
                    )

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
                    return ValidationError(
                        stage="security",
                        message=f"Forbidden builtin call: {node.func.id}()",
                        fixable=True,
                    )
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id in BANNED_MODULE_NAMES:
                        return ValidationError(
                            stage="security",
                            message=f"Forbidden call through module {node.func.value.id}.",
                            fixable=True,
                        )

            if isinstance(node, ast.While):
                return ValidationError(
                    stage="security",
                    message="Forbidden while-loop: use bounded for-loops with explicit max iterations.",
                    fixable=True,
                )

        return None

    def _check_tdg_contract(self, code: str) -> Optional[ValidationError]:
        """Enforce that evolved TDG classes only customize compile_tests/generate_answer."""

        tree = ast.parse(code)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            base_names: set[str] = set()
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.add(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.add(base.attr)

            if "BaseTDGCompiler" not in base_names:
                continue

            methods = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            method_nodes = {
                item.name: item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

            required = {"compile_tests", "generate_answer"}
            missing = sorted(required - methods)
            if missing:
                return ValidationError(
                    stage="contract",
                    message=f"TDG subclass {node.name} is missing required method(s): {', '.join(missing)}",
                    fixable=True,
                )

            forbidden = {"run", "_call_llm", "_make_oracle_fn"}
            overridden = sorted(forbidden & methods)
            if overridden:
                return ValidationError(
                    stage="contract",
                    message=(
                        f"TDG subclass {node.name} overrides immutable runtime method(s): "
                        f"{', '.join(overridden)}"
                    ),
                    fixable=True,
                )

            def _calls_self_method(func_node: ast.AST, method_name: str) -> bool:
                for child in ast.walk(func_node):
                    if not isinstance(child, ast.Call):
                        continue
                    func = child.func
                    if (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "self"
                        and func.attr == method_name
                    ):
                        return True
                return False

            compile_node = method_nodes.get("compile_tests")
            if compile_node is not None and not _calls_self_method(compile_node, "_call_llm"):
                return ValidationError(
                    stage="contract",
                    message=(
                        f"TDG subclass {node.name} compile_tests() must call self._call_llm "
                        "to synthesize task-specific tests."
                    ),
                    fixable=True,
                )

            answer_node = method_nodes.get("generate_answer")
            if answer_node is not None and not _calls_self_method(answer_node, "_call_llm"):
                return ValidationError(
                    stage="contract",
                    message=(
                        f"TDG subclass {node.name} generate_answer() must call self._call_llm "
                        "to synthesize grounded answers."
                    ),
                    fixable=True,
                )

            # Reject calls to self.* helper methods that are not defined in the
            # class and not part of the BaseTDGCompiler public interface.
            # Example failure: self._extract_code() called but never defined →
            # passes contract check but raises AttributeError at runtime.
            _ALLOWED_SELF_ATTRS = {
                # BaseTDGCompiler callable methods
                "_call_llm", "_prepare_prompt_text", "_call_budget",
                "_build_test_runner", "_run_tests", "_extract_test_names",
                "_sanitize_generated_code", "_is_syntax_error",
                "_attempt_syntax_repair", "_answer_with_oracle_fallback",
                "_make_oracle_fn", "_derive_dynamic_call_budget",
                "_repair_answer", "_is_transient_llm_error",
                "_extract_max_tokens_cap",
                # Instance attributes (non-callable, but accessed via self.attr)
                "llm", "model", "context_char_limit", "query_char_limit",
                "max_completion_tokens", "oracle_max_completion_tokens",
                "api_timeout_seconds", "sandbox_timeout_seconds",
                "_call_count", "_task_tokens_used", "_task_prompt_tokens",
                "_task_completion_tokens", "_oracle_call_count",
                "_max_llm_calls_current",
            }
            class_defined = set(method_nodes.keys())
            allowed_attrs = _ALLOWED_SELF_ATTRS | class_defined

            def _collect_self_attr_calls(func_node: ast.AST) -> set[str]:
                """Collect names of all self.<name>(...) call sites."""
                called: set[str] = set()
                for child in ast.walk(func_node):
                    if not isinstance(child, ast.Call):
                        continue
                    func_ref = child.func
                    if (
                        isinstance(func_ref, ast.Attribute)
                        and isinstance(func_ref.value, ast.Name)
                        and func_ref.value.id == "self"
                    ):
                        called.add(func_ref.attr)
                return called

            for evolvable_method in ("compile_tests", "generate_answer"):
                fn = method_nodes.get(evolvable_method)
                if fn is None:
                    continue
                for attr in _collect_self_attr_calls(fn):
                    if attr not in allowed_attrs:
                        return ValidationError(
                            stage="contract",
                            message=(
                                f"TDG subclass {node.name}.{evolvable_method}() calls "
                                f"self.{attr}() which is not defined in the class or "
                                "in the BaseTDGCompiler interface. Define the helper "
                                "inside the class or use an allowed base method."
                            ),
                            fixable=True,
                        )

        return None

    def validate(self, code: str) -> Optional[ValidationError]:
        """Run all validation stages; return first error if found."""

        for check in (self._check_syntax, self._check_lint, self._check_security, self._check_tdg_contract):
            error = check(code)
            if error is not None:
                return error
        return None

    def load_from_file(self, file_path: str):
        """Load a TDG compiler subclass from a Python source file."""

        module_name = f"evolved_tdg_{Path(file_path).stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            module.BaseTDGCompiler = BaseTDGCompiler
            module.SandboxResult = SandboxResult
            module.Any = Any
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseTDGCompiler) and obj is not BaseTDGCompiler:
                    protocol = obj(self.llm_client, self.model_name)
                    setattr(protocol, "sandbox_timeout_seconds", self.sandbox_timeout_seconds)
                    setattr(protocol, "max_llm_calls_per_task", self.max_llm_calls_per_task)
                    return protocol

        except Exception:
            return None
        finally:
            sys.modules.pop(module_name, None)

        return None

    def load_from_code(self, code: str) -> tuple[Optional[BaseTDGCompiler], Optional[ValidationError]]:
        """Validate and load TDG compiler from source code string."""

        error = self.validate(code)
        if error is not None:
            return None, error

        sha = compute_sha(code)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{sha}.py",
            encoding="utf-8",
            delete=False,
        ) as file_obj:
            file_obj.write(code)
            temp_path = Path(file_obj.name)

        try:
            protocol = self.load_from_file(str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

        if protocol is None:
            return None, ValidationError(
                stage="load",
                message="No valid BaseTDGCompiler subclass could be loaded.",
                fixable=True,
            )

        return protocol, None

    def run_with_timeout(
        self,
        protocol: BaseTDGCompiler,
        context: str,
        query: str,
        messages_raw: list | None = None,
    ) -> Optional[SandboxResult]:
        """Run TDG protocol with hard subprocess timeout when available."""

        if self._process_ctx is not None:
            result = self._run_with_process_timeout(protocol, context, query, messages_raw)
            if result is not None:
                return result
            return None

        return self._run_with_thread_timeout(protocol, context, query, messages_raw)

    def _run_with_thread_timeout(
        self,
        protocol: BaseTDGCompiler,
        context: str,
        query: str,
        messages_raw: list | None = None,
    ) -> Optional[SandboxResult]:
        """Fallback timeout mode using daemon thread join."""

        result_box: dict[str, SandboxResult] = {}
        error_box: dict[str, BaseException] = {}

        def _target() -> None:
            try:
                result_box["result"] = protocol.run(context, query, messages_raw=messages_raw)
            except BaseException as exc:
                error_box["error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            return None

        if "error" in error_box:
            error = error_box["error"]
            traceback.print_exception(type(error), error, error.__traceback__)
            return None

        return result_box.get("result")

    def _run_with_process_timeout(
        self,
        protocol: BaseTDGCompiler,
        context: str,
        query: str,
        messages_raw: list | None = None,
    ) -> Optional[SandboxResult]:
        """Execute task in child process and force-kill on timeout."""

        assert self._process_ctx is not None
        parent_conn, child_conn = self._process_ctx.Pipe(duplex=False)
        process = self._process_ctx.Process(
            target=_run_tdg_protocol_in_subprocess,
            args=(child_conn, protocol, context, query, messages_raw),
            daemon=True,
        )

        try:
            process.start()
        except Exception:
            parent_conn.close()
            child_conn.close()
            return self._run_with_thread_timeout(protocol, context, query, messages_raw)
        finally:
            child_conn.close()

        process.join(timeout=self.timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            parent_conn.close()
            return None

        payload: dict[str, Any] | None = None
        try:
            if parent_conn.poll(0.2):
                payload = parent_conn.recv()
        except (EOFError, OSError):
            payload = None
        finally:
            parent_conn.close()

        if not payload:
            return None
        if not bool(payload.get("ok")):
            error_tb = str(payload.get("traceback") or payload.get("error") or "").strip()
            if error_tb:
                print(error_tb, file=sys.stderr)
            return None

        result_payload = payload.get("result", {})
        if not isinstance(result_payload, dict):
            return None
        try:
            return SandboxResult(
                answer=str(result_payload.get("answer", "")),
                confidence=float(result_payload.get("confidence", 0.0)),
                reasoning_trace=list(result_payload.get("reasoning_trace", [])),
                verification_passed=bool(result_payload.get("verification_passed", False)),
                tokens_used=int(result_payload.get("tokens_used", 0)),
                prompt_tokens=int(result_payload.get("prompt_tokens", 0)),
                completion_tokens=int(result_payload.get("completion_tokens", 0)),
                metadata=dict(result_payload.get("metadata", {})),
                sandbox_code=str(result_payload.get("sandbox_code", "")),
                solver_code=str(result_payload.get("solver_code", "")),
                execution_output=str(result_payload.get("execution_output", "")),
                execution_success=bool(result_payload.get("execution_success", False)),
                compilation_success=bool(result_payload.get("compilation_success", False)),
            )
        except Exception:
            return None
