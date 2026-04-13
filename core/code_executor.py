"""
Programmatic Tool Calling (PTC) Code Executor — ATLAS Section 2.3 & 2.4.

From the paper:
  "ATLAS replaces [JSON-based tool calling] with a unified programmatic execution
   model, where all tool interactions are mediated by a persistent Python interpreter.
   Tool calls are expressed as function invocations, control flow is encoded
   explicitly using programming constructs, and intermediate results are stored in
   program state rather than surfaced to the model."

Key design choices from Section 2.4 (Scaffolding):
  - MCPServer is pre-injected into the interpreter namespace
  - Errors are enhanced with closest-match suggestions (Section A.6)
  - State persists across code executions within a single task
"""

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from core.mcp_server import MCPServer


class CodeExecutor:
    """
    Persistent Python interpreter for PTC.
    Agent-generated code runs here with MCPServer pre-available.
    State persists across multiple execute() calls within one task episode.
    """

    def __init__(self):
        self._globals: dict[str, Any] = {
            "MCPServer": MCPServer,
            "__builtins__": __builtins__,
        }
        self._execution_log: list[dict] = []

    def execute(self, code: str) -> dict:
        """
        Run agent-generated Python code.

        Returns dict with:
          - success: bool
          - stdout: captured print output
          - result: last expression value (if code ends with an expression)
          - error: error message if failed (with enhanced hints per Section A.6)
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Wrap last expression for capture (like Jupyter)
        exec_code, eval_expr = self._split_last_expr(code)

        result = None
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                if exec_code:
                    exec(exec_code, self._globals)
                if eval_expr:
                    result = eval(eval_expr, self._globals)
        except Exception as e:
            error_msg = self._enhance_error(e, code)
            entry = {
                "success": False,
                "stdout": stdout_buf.getvalue(),
                "result": None,
                "error": error_msg,
                "code": code,
            }
            self._execution_log.append(entry)
            return entry

        entry = {
            "success": True,
            "stdout": stdout_buf.getvalue(),
            "result": result,
            "error": None,
            "code": code,
        }
        self._execution_log.append(entry)
        return entry

    def get_execution_log(self) -> list[dict]:
        return self._execution_log

    def reset(self):
        """Reset interpreter state between tasks."""
        self._globals = {
            "MCPServer": MCPServer,
            "__builtins__": __builtins__,
        }
        self._execution_log = []

    @staticmethod
    def _split_last_expr(code: str) -> tuple[str, str]:
        """Split code into statements + final expression (for result capture)."""
        lines = code.strip().split("\n")
        if not lines:
            return "", ""

        # Try to compile last line as expression
        last_line = lines[-1].strip()
        try:
            compile(last_line, "<expr>", "eval")
            # Last line is an expression — separate it
            exec_part = "\n".join(lines[:-1])
            return exec_part, last_line
        except SyntaxError:
            return code, ""

    @staticmethod
    def _enhance_error(e: Exception, code: str) -> str:
        """
        Informative error messages per ATLAS Section A.6:
        - Incorrect function names: suggest closest tool
        - Incorrect output access: show type hints
        - Argument errors: surface server feedback
        """
        msg = str(e)

        if isinstance(e, AttributeError) and "MCPServer" in type(e).__name__ or "MCPServer" in msg:
            # Already enhanced by MCPServer.__getattr__
            return msg

        if isinstance(e, TypeError) and "unexpected keyword argument" in msg:
            return f"Argument error: {msg}. Check parameter names with get_tools_info()."

        if isinstance(e, (KeyError, IndexError)):
            return (
                f"Output access error: {msg}. "
                "The tool output format may differ from expected. "
                "Use get_tools_info() to check the output schema."
            )

        if isinstance(e, TypeError) and "indices must be integers" in msg:
            return (
                f"{msg}. You may have tried to access a string as a dict. "
                "Check the output type with get_tools_info()."
            )

        # Fallback: include traceback context
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        return f"{msg}\n{''.join(tb[-3:])}"
