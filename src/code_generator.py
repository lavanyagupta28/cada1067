"""Autonomous Code Generator & Self-Repair Engine for Dynamic EDA Tools."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

from .sandbox import SandboxRunner

logger = logging.getLogger(__name__)

QUERY_EXAMPLE = """
def count_gate_types_in_cone(self, output_signal: str) -> dict:
    self._require_netlist()
    nl = self._netlist
    assert nl is not None

    if output_signal not in nl.wires and output_signal not in nl.primary_outputs:
        raise ValueError(f"Signal {output_signal!r} not found in design.")

    cone_nodes = self._get_fanin_cone_nodes(output_signal)
    breakdown = {}
    for node in cone_nodes:
        gt = node.gate_type.lower()
        breakdown[gt] = breakdown.get(gt, 0) + 1

    return {
        "output_signal": output_signal,
        "total": len(cone_nodes),
        "by_type": breakdown
    }
"""

TRANSFORMATION_EXAMPLE = """
def remove_dangling_gates(self) -> dict:
    self._require_netlist()
    nl = self._netlist
    assert nl is not None

    removed = 0
    active_loads = set(nl.primary_outputs)
    for node in nl.nodes.values():
        active_loads.update(node.inputs)

    dangling = [name for name, node in nl.nodes.items() if node.output not in active_loads]
    for name in dangling:
        del nl.nodes[name]
        removed += 1

    return {"gates_removed": removed}
"""


def build_synthesis_prompt(user_prompt: str, missing_spec: dict) -> str:
    tool_name = missing_spec.get("tool_name", "custom_tool")
    desc = missing_spec.get("description", "")
    params = missing_spec.get("parameters_needed", {})
    is_trans = missing_spec.get("is_transformation", False)
    reference_code = TRANSFORMATION_EXAMPLE if is_trans else QUERY_EXAMPLE

    return f"""
================================================================================
ROLE & TASK SPECIFICATION
================================================================================
You are an expert EDA core developer for the `cada1067` framework.
Write a production-quality Python method for the `EDAEngine` class.

USER PROMPT:
"{user_prompt}"

TOOL NAME TO CREATE:
`def {tool_name}(self, ...) -> dict:`

DETAILED FUNCTIONAL REQUIREMENTS:
{desc}
Required Arguments to accept: {json.dumps(params)}
Is this modifying the circuit?: {is_trans}

================================================================================
CIRCUIT DATA STRUCTURE (self._netlist)
================================================================================
- `self._netlist.nodes`: dict of gate_name -> GateNode
    Each GateNode has:
      - .name (str)
      - .gate_type (str, lower-case: 'and', 'nand', 'or', 'nor', 'xor', 'xnor', 'not', 'buf')
      - .inputs (list[str]: driving wire names)
      - .output (str: driven wire name)
- `self._netlist.dffs`: dict of dff_name -> DFFNode (.name, .ck, .rn, .sn, .d, .q, .qn)
- `self._netlist.wires`: set of all wire names (plus "1'b0" and "1'b1")
- `self._netlist.primary_inputs`: list of primary input names (list[str])
- `self._netlist.primary_outputs`: list of primary output names (list[str])

Engine Helpers Available:
- `self._require_netlist()`: ALWAYS call this on line 1.
- `self._get_fanin_cone_nodes(output_signal)`: returns Set[GateNode] for that cone.
- `self._topological_sort()`: returns topological list of gate names.

================================================================================
GOLDEN REFERENCE CODE EXAMPLE (COPY THIS STYLE)
================================================================================
{reference_code}

================================================================================
OUTPUT FORMAT INSTRUCTIONS
================================================================================
Respond ONLY with a valid JSON object:
{{
  "code": "def {tool_name}(self, ...):\\n    ...",
  "schema": {{
    "type": "function",
    "function": {{
      "name": "{tool_name}",
      "description": "Clear description of what this tool does.",
      "parameters": {{
        "type": "object",
        "properties": {{ ... }}
      }}
    }}
  }},
  "test_arguments": {{ ... }}
}}
""".strip()


class DynamicToolGenerator:
    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._max_retries = 3

    def generate_and_verify(
        self, missing_spec: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], str]:
        """Synthesize, sandbox, verify, and return (success, code, schema, status_message)."""
        tool_name = missing_spec["tool_name"]
        user_prompt = getattr(self._agent, "_active_user_message", "")
        engine = self._agent._engine

        base_prompt = build_synthesis_prompt(user_prompt, missing_spec)
        error_feedback = ""

        for attempt in range(1, self._max_retries + 1):
            prompt = (
                base_prompt
                if not error_feedback
                else f"{base_prompt}\n\n[PREVIOUS ATTEMPT FAILED]:\n{error_feedback}\nPlease fix this bug and regenerate."
            )

            try:
                response_json = self._agent._call_llm_json(prompt)
            except Exception as exc:
                error_feedback = f"Failed to get valid JSON from model: {exc}"
                continue

            code = response_json.get("code", "")
            schema = response_json.get("schema", {})
            test_args = response_json.get("test_arguments", {})

            if not code or not schema:
                error_feedback = "Model output missing 'code' or 'schema' key."
                continue

            # Gate 1: AST Security & Syntax Check
            violations = SandboxRunner.audit_ast(code)
            if violations:
                error_feedback = f"Gate 1 Failed (AST Security/Syntax): {'; '.join(violations)}"
                continue

            # Gate 2 & 3: Dry-Run & Graph Integrity Audit
            ok, res, dry_err = SandboxRunner.execute_dry_run(code, tool_name, engine, test_args)
            if not ok:
                error_feedback = f"Gate 2/3 Failed (Dry-run/Integrity): {dry_err}"
                continue

            # Gate 4: Formal LEC Equivalence (if transformation)
            if missing_spec.get("is_transformation", False) and missing_spec.get("preserves_equivalence", True):
                lec_ok, lec_err = SandboxRunner.verify_formal_equivalence(code, tool_name, engine, test_args)
                if not lec_ok:
                    error_feedback = f"Gate 4 Failed (ABC CEC Equivalence): {lec_err}"
                    continue

            return True, code, schema, f"Tool '{tool_name}' synthesized and verified successfully on attempt {attempt}."

        return False, None, None, f"Failed after {self._max_retries} attempts. Last error: {error_feedback}"
