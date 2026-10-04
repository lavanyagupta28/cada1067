"""Automated test suite for Dynamic Code Generation Extension."""

import os
import sys
import copy
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from src.sandbox import SandboxRunner, SecurityASTValidator
from src.eda_engine import EDAEngine
from src.code_generator import build_synthesis_prompt
from src.dynamic_tools.registry import save_dynamic_tool


def test_ast_security_validator():
    print("Running test_ast_security_validator...")
    
    # 1. Dangerous code with import os
    bad_code_1 = """
def malicious_tool(self):
    import os
    os.system("echo hacked")
    return {}
"""
    violations = SandboxRunner.audit_ast(bad_code_1)
    assert any("os" in v for v in violations), f"Failed to catch forbidden import: {violations}"
    print("  [PASS] Successfully blocked 'import os'")

    # 2. Dangerous code with eval
    bad_code_2 = """
def eval_tool(self):
    eval("1 + 1")
    return {}
"""
    violations = SandboxRunner.audit_ast(bad_code_2)
    assert any("eval" in v for v in violations), f"Failed to catch forbidden call: {violations}"
    print("  [PASS] Successfully blocked 'eval()'")

    # 3. Safe code
    safe_code = """
def count_odd_input_gates(self) -> dict:
    self._require_netlist()
    count = 0
    for name, node in self._netlist.nodes.items():
        if len(node.inputs) % 2 != 0:
            count += 1
    return {"count": count}
"""
    violations = SandboxRunner.audit_ast(safe_code)
    assert len(violations) == 0, f"False positive on safe code: {violations}"
    print("  [PASS] Successfully approved safe EDA code")


def test_dry_run_and_integrity():
    print("\nRunning test_dry_run_and_integrity...")
    engine = EDAEngine()
    test_verilog = "testcase/test01/test01.v"
    assert os.path.exists(test_verilog), f"Missing test netlist: {test_verilog}"
    engine.load(test_verilog)

    # 1. Verify netlist integrity on clean design
    errors = engine.validate_netlist_integrity()
    assert len(errors) == 0, f"Clean design reported integrity errors: {errors}"
    print("  [PASS] Clean design passed integrity audit")

    # 2. Execute dynamic dry-run
    dynamic_code = """
def count_odd_input_gates(self) -> dict:
    self._require_netlist()
    odd_gates = [name for name, node in self._netlist.nodes.items() if len(node.inputs) % 2 != 0]
    return {"odd_gate_count": len(odd_gates), "samples": odd_gates[:3]}
"""
    original_gate_count = len(engine.netlist.nodes)
    ok, res, err = SandboxRunner.execute_dry_run(
        dynamic_code, "count_odd_input_gates", engine, {}
    )
    assert ok, f"Dry-run failed: {err}"
    assert "odd_gate_count" in res, f"Expected odd_gate_count in result, got: {res}"
    assert len(engine.netlist.nodes) == original_gate_count, "Dry run mutated the active netlist!"
    print(f"  [PASS] Dry-run succeeded. Counted {res['odd_gate_count']} odd-input gates without mutating active design")

    # 3. Test integrity audit catching dangling pin
    corrupted_nl = copy.deepcopy(engine.netlist)
    # Add a gate with a completely bogus wire
    from src.netlist_parser import GateNode
    corrupted_nl.nodes["bogus_gate"] = GateNode(
        name="bogus_gate",
        gate_type="and",
        inputs=["non_existent_wire_12345"],
        output="out0"
    )
    corrupted_errors = engine.validate_netlist_integrity(corrupted_nl)
    assert len(corrupted_errors) > 0, "Failed to catch dangling wire in corrupted netlist"
    print(f"  [PASS] Graph integrity audit caught corrupted pin: {corrupted_errors[0]}")


def test_prompt_builder():
    print("\nRunning test_prompt_builder...")
    user_prompt = "Count how many gates have an odd number of inputs in cone out0"
    missing_spec = {
        "tool_name": "count_odd_input_gates",
        "description": "Examine fanin cone of out0 and count gates where len(node.inputs) is odd",
        "parameters_needed": {"output_signal": "string"},
        "is_transformation": False
    }
    prompt = build_synthesis_prompt(user_prompt, missing_spec)
    assert "count_odd_input_gates" in prompt
    assert "self._netlist.nodes" in prompt
    assert "GOLDEN REFERENCE" in prompt
    print("  [PASS] Prompt builder produced rich context packet with API cheat-sheet and golden examples")


def test_persistence_and_loading():
    print("\nRunning test_persistence_and_loading...")
    tool_name = "test_synthetic_tool"
    code = """
def test_synthetic_tool(self) -> dict:
    self._require_netlist()
    return {"status": "synthetic_ok"}
"""
    schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Test synthetic tool",
            "parameters": {"type": "object", "properties": {}}
        }
    }

    # Save to dynamic_tools
    saved_path = save_dynamic_tool(tool_name, code, schema)
    assert saved_path.exists(), f"Failed to save dynamic tool to {saved_path}"
    print(f"  [PASS] Dynamic tool successfully saved to {saved_path}")

    # Clean up test file so it doesn't pollute repository
    saved_path.unlink()
    print("  [PASS] Cleaned up synthetic test file")


if __name__ == "__main__":
    test_ast_security_validator()
    test_dry_run_and_integrity()
    test_prompt_builder()
    test_persistence_and_loading()
    print("\nSUCCESS: ALL TESTS PASSED SUCCESSFULLY!")
