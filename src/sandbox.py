"""Sandbox environment for verifying dynamically generated EDA tools."""

from __future__ import annotations

import ast
import copy
from typing import Any, Dict, List, Tuple

FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "__import__", "globals", "locals"}
FORBIDDEN_MODULES = {"os", "sys", "subprocess", "socket", "requests", "shutil", "builtins"}


class SecurityASTValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden from-import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.violations.append(f"Forbidden call: {node.func.id}")
        self.generic_visit(node)


class SandboxRunner:
    @staticmethod
    def audit_ast(code_str: str) -> List[str]:
        """Gate 1: Parse code into AST and check for security and syntax violations."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError as exc:
            return [f"Syntax error on line {exc.lineno}: {exc.msg}"]
        validator = SecurityASTValidator()
        validator.visit(tree)
        return validator.violations

    @staticmethod
    def execute_dry_run(
        code_str: str,
        func_name: str,
        engine_instance: Any,
        test_args: Dict[str, Any],
    ) -> Tuple[bool, Any, str]:
        """Gate 2 & 3: Execute function against an isolated deep-copy of the active netlist."""
        # 1. Compile function into sandbox namespace
        namespace: Dict[str, Any] = {}
        try:
            compiled = compile(code_str, f"<dynamic_{func_name}>", "exec")
            exec(compiled, namespace)
            func = namespace.get(func_name)
            if not func or not callable(func):
                return False, None, f"Function '{func_name}' was not defined in the generated code."
        except Exception as exc:
            return False, None, f"Compilation failed: {exc}"

        # 2. Deep-clone engine netlist
        if engine_instance._netlist is None:
            # If no design is loaded yet, dry-run passes static check
            return True, {}, ""

        mock_engine = copy.copy(engine_instance)
        mock_engine._netlist = copy.deepcopy(engine_instance._netlist)

        # 3. Dry-run invocation
        try:
            result = func(mock_engine, **test_args)
            if not isinstance(result, dict):
                return False, None, f"Function must return a dict, got {type(result).__name__}"

            # Gate 3: Check netlist integrity after execution
            integrity_errors = mock_engine.validate_netlist_integrity()
            if integrity_errors:
                return False, None, f"Netlist graph integrity broken: {'; '.join(integrity_errors[:2])}"

            return True, result, ""
        except Exception as exc:
            return False, None, f"Runtime exception during dry-run: {exc}"

    @staticmethod
    def verify_formal_equivalence(
        code_str: str,
        func_name: str,
        engine_instance: Any,
        test_args: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Gate 4: Formal LEC check via ABC CEC between original and transformed clone."""
        if engine_instance._netlist is None or not engine_instance.original_netlist_path:
            return True, ""

        mock_engine = copy.copy(engine_instance)
        mock_engine._netlist = copy.deepcopy(engine_instance._netlist)

        namespace: Dict[str, Any] = {}
        try:
            exec(compile(code_str, "<dynamic_lec>", "exec"), namespace)
            func = namespace[func_name]
            func(mock_engine, **test_args)
        except Exception as exc:
            return False, f"Transformation crashed during execution: {exc}"

        # Run ABC CEC using built-in check_design_equivalence
        try:
            equiv_res = mock_engine.check_design_equivalence()
            if not equiv_res.get("equivalent", False):
                return False, f"ABC CEC proof failed: Networks are not equivalent ({equiv_res.get('reason', 'CEC mismatch')})"
            return True, ""
        except Exception as exc:
            return False, f"Failed to run equivalence checker: {exc}"
