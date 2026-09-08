"""LLM agent that drives the EDA engine through OpenAI function-calling."""

from __future__ import annotations

import json
import os
import random
import re
import signal as _signal
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to environment variables

from .eda_engine import EDAEngine
from .tools import TOOLS


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class _Timeout:
    """Context manager that raises TimeoutError after *seconds*."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds

    def _handler(self, signum: int, frame: Any) -> None:  # noqa: ANN001
        raise TimeoutError(f"Operation timed out after {self.seconds} s")

    def __enter__(self) -> "_Timeout":
        if hasattr(_signal, "SIGALRM") and hasattr(_signal, "alarm"):
            _signal.signal(_signal.SIGALRM, self._handler)
            _signal.alarm(self.seconds)
        return self

    def __exit__(self, *args: Any) -> None:
        if hasattr(_signal, "alarm"):
            _signal.alarm(0)


# ---------------------------------------------------------------------------
# EDAAgent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert EDA assistant. You have access to a set of netlist analysis "
    "and transformation tools. When the user gives a natural-language request about "
    "a gate-level Verilog netlist, call the appropriate tool(s) in the correct order. "
    "After all tool calls are complete, return a concise, clear natural-language answer "
    "describing what was found or what was changed. Do not discuss scoring or evaluation. "
    "User prompts arrive sequentially. If the current request asks for a result or fact "
    "already recorded in the ordered context history, answer directly from that history "
    "without calling another tool. Call tools only when the needed fact is absent from "
    "history or the user explicitly requests recomputation or current-state analysis. "
    "For numerical or statistical design queries whose answer is not already explicit in "
    "the ordered context history, use tools instead of estimating. Tool results may be "
    "compacted to save tokens; when a tool result includes a file_path, always report "
    "that exact file path to the user along with the count instead of saying the full "
    "list is too long or can be found in the design. For reports of gates with "
    "constant inputs, use find_gates_with_constant_inputs unless the prompt explicitly "
    "says the input is tied to a literal constant. For constant propagation of "
    "reported gates, use simplify_gates_with_constant_inputs. If the relevant prior summary says zero gates were "
    "found, report that there is nothing to simplify. Do not use replace_gate_type_in_cone "
    "for constant propagation or reported-gate simplification; it is only for intentional "
    "gate-library remapping of all gates of a type in a scope. For questions asking "
    "whether there exists a pair of existing signals a,b such that a binary gate "
    "expression like NAND(a,b), OR(a,b), or XOR(a,b) equals a target signal, use "
    "find_binary_gate_equivalent_pair; do not approximate this by listing existing "
    "gates of that type. For prompts asking to list flip-flops driven or clocked by "
    "a named clock signal, use list_flip_flops_by_clock. For requests to report or "
    "count enable/hold structures in flip-flop D-input "
    "logic, use analyze_dff_d_input_structures; do not use register-to-register "
    "path enumeration for that analysis. For highest/lowest fanout "
    "across a signal class, use rank_signals_by_fanout. For functional symmetry "
    "in two named inputs, use check_function_symmetry. For largest/smallest fanin "
    "cones, use rank_signals_by_fanin_cone with gate_count. For depth optimization "
    "where only one named cone must use a restricted gate set and the cost is whole-design "
    "maximum depth, use optimize_depth_preserving_cone_gate_set; when the cost is the "
    "depth of that cone itself, use optimize_cone_depth_preserving_gate_set; "
    "reduce_critical_path with allowed_gates is only for restricting the whole design."
)

# Timeouts (seconds) per category
_FAST_OPS = {"read_design", "write_design", "set_testcase_name"}
_FAST_TIMEOUT = 60
_SLOW_TIMEOUT = 300


class EDAAgent:
    """LLM agent that translates natural-language EDA requests into tool calls.

    Args:
        config:     Parsed YAML config dict.
        eda_engine: Initialised EDAEngine instance.
    """

    def __init__(self, config: Dict[str, Any], eda_engine: EDAEngine) -> None:
        self._config = config
        self._engine = eda_engine
        self._io_handler: Optional[Any] = None  # set later by IOHandler
        self._current_case_name: str = ""
        self._context_history: List[str] = []

        # Developer mode — when enabled, loads the conversation logger
        dev_cfg = config.get("developer", {})
        self._developer_mode: bool = bool(dev_cfg.get("mode", False) or config.get("developer_mode", False))
        if self._developer_mode:
            from .developer import ConversationLogger, NoopLogger, get_logs_dir, extract_case_name  # noqa: E501
            log_base = dev_cfg.get("log_dir", None)
            self._logs_dir = get_logs_dir(log_base if log_base else None)
            self._LoggerCls = ConversationLogger
            self._extract_case_name = extract_case_name
        else:
            from .developer import NoopLogger  # noqa: E501
            self._logs_dir = Path(".")  # unused dummy
            self._LoggerCls = NoopLogger
            self._extract_case_name = lambda _: ""

        provider = config.get("provider", "openai").lower()
        if provider == "openai":
            self._client = self._build_openai_client(config)
            self._model: str = config["openai"]["model"]
        else:
            raise RuntimeError(
                f"Unsupported provider: {provider!r}. Only 'openai' is currently supported."
            )

        gen = config.get("generation", {})
        self._temperature: float = float(gen.get("temperature", 0.2))
        self._max_tokens: int = int(gen.get("max_output_tokens", 4096))

        safety = config.get("safety", {})
        self._max_tool_rounds: int = int(safety.get("max_tool_rounds", 4))
        self._max_inline_items: int = int(safety.get("max_inline_items", 10))
        self._max_tool_result_chars: int = int(
            safety.get("max_tool_result_chars", 2000)
        )
        self._context_history_limit: int = int(
            safety.get("context_history_limit", 20)
        )
        self._llm_min_interval: float = float(
            safety.get("llm_min_interval_seconds", 0.75)
        )
        self._rate_limit_max_retries: int = int(
            safety.get("rate_limit_max_retries", 6)
        )
        self._rate_limit_initial_delay: float = float(
            safety.get("rate_limit_initial_delay_seconds", 1.0)
        )
        self._rate_limit_max_delay: float = float(
            safety.get("rate_limit_max_delay_seconds", 30.0)
        )
        self._rate_limit_jitter: float = float(
            safety.get("rate_limit_jitter_seconds", 0.25)
        )
        self._last_llm_call_started: float = 0.0
        self._active_user_message: str = ""

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_openai_client(config: Dict[str, Any]) -> Any:
        """Build and return an OpenAI client from config."""
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "openai package is not installed. Run: pip install openai>=1.0.0"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            api_key = str(config.get("openai", {}).get("api_key", "")).strip()
        if not api_key or api_key.startswith("YOUR_"):
            raise RuntimeError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY in the environment/.env file or set openai.api_key in config.yaml."
            )
        # Rate-limit retries are handled in _call_llm, where Retry-After and
        # token-reset headers can be honored. Avoid stacking SDK retries with
        # the framework's retry loop.
        return OpenAI(api_key=api_key, max_retries=0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_io_handler(self, io_handler: Any) -> None:
        """Register the IOHandler so the agent can call set_log_file."""
        self._io_handler = io_handler

    def build_state_summary(self) -> str:
        """Return a short string describing the current engine state and all
        previous operations for this testcase.

        The IOHandler calls this after each request to feed context into
        the next request. Includes testcase name, original netlist path, and
        the full history of operations performed so far.
        """
        parts: List[str] = []
        if self._current_case_name:
            parts.append(f"testcase='{self._current_case_name}'")

        if self._engine.original_netlist_path:
            parts.append(f"original_netlist='{self._engine.original_netlist_path}'")

        # Full operation history
        history = self._context_history[-self._context_history_limit:]
        if history:
            parts.append("operations so far in order:")
            if len(self._context_history) > len(history):
                omitted = len(self._context_history) - len(history)
                parts.append(f"  - ... {omitted} earlier operations omitted")
            for item in history:
                parts.append(f"  - {item}")

        return "\n".join(parts) if parts else "(no state)"

    def _trim_context_history(self) -> None:
        """Keep only the most recent compact operation summaries."""
        if self._context_history_limit <= 0:
            self._context_history = []
            return
        overflow = len(self._context_history) - self._context_history_limit
        if overflow > 0:
            del self._context_history[:overflow]

    def _append_context_summary(self, summary: str) -> None:
        """Append an operation summary, merging consecutive replacement counts."""
        if summary.startswith("replace_gate: ") and self._context_history:
            last = self._context_history[-1]
            if last.startswith("replace_gate: "):
                merged = self._merge_replace_gate_summaries(last, summary)
                if merged:
                    self._context_history[-1] = merged
                    self._trim_context_history()
                    return
        self._context_history.append(summary)
        self._trim_context_history()

    def _request_context_summary(
        self, user_message: str, tool_summaries: List[str]
    ) -> Optional[str]:
        """Condense all tool calls from one user prompt into one history item."""
        if not tool_summaries:
            return None

        prompt = " ".join(user_message.split())
        max_prompt_chars = 120
        if len(prompt) > max_prompt_chars:
            prompt = prompt[: max_prompt_chars - 3] + "..."

        counts: Dict[str, int] = {}
        ordered: List[str] = []
        for summary in tool_summaries:
            if summary not in counts:
                ordered.append(summary)
                counts[summary] = 0
            counts[summary] += 1

        details = [
            f"{summary} ({counts[summary]}x)" if counts[summary] > 1 else summary
            for summary in ordered
        ]
        return f"{prompt} => " + "; ".join(details)

    @staticmethod
    def _parse_replace_gate_summary(summary: str) -> Optional[tuple[int, str]]:
        prefix = "replace_gate: "
        if not summary.startswith(prefix):
            return None
        body = summary[len(prefix):]
        marker = " gate(s) -> "
        if marker not in body:
            return None
        count_text, new_type = body.split(marker, 1)
        try:
            return int(count_text), new_type.strip()
        except ValueError:
            return None

    def _merge_replace_gate_summaries(self, first: str, second: str) -> Optional[str]:
        first_parsed = self._parse_replace_gate_summary(first)
        second_parsed = self._parse_replace_gate_summary(second)
        if not first_parsed or not second_parsed:
            return None
        first_count, first_type = first_parsed
        second_count, second_type = second_parsed
        if first_type != second_type:
            return None
        return f"replace_gate: {first_count + second_count} gate(s) -> {first_type}"

    def _specific_gate_type_replacement_from_prompt(self) -> Optional[str]:
        """Infer a source gate type explicitly named in a replacement prompt."""
        prompt = getattr(self, "_active_user_message", "").lower()
        if not any(word in prompt for word in ("replace", "convert", "decompose")):
            return None
        for gate_type in ("nand", "nor", "xnor", "xor", "and", "or", "buf", "not"):
            source_patterns = (
                rf"\b(?:replace|convert|decompose)\s+(?:all\s+)?(?:2-input|two-input)?\s*{gate_type}\s+gates?\b",
                rf"\b(?:replace|convert|decompose)\s+(?:all\s+)?(?:2-input|two-input)?\s*{gate_type}-gates?\b",
                rf"\b(?:replace|convert|decompose)\s+(?:all\s+)?gates?\s+of\s+type\s+{gate_type}\b",
            )
            if any(re.search(pattern, prompt) for pattern in source_patterns):
                return gate_type
        return None

    def _cone_depth_restriction_target_from_prompt(self) -> Optional[str]:
        """Infer the cone target for depth prompts with a cone-only gate restriction."""
        prompt = getattr(self, "_active_user_message", "")
        if "cone" not in prompt.lower():
            return None
        if not any(word in prompt.lower() for word in ("depth", "critical path", "maximum path")):
            return None
        match = re.search(r"\bcone\s+of\s+([A-Za-z_$][\w$]*(?:\[[^\]]+\])?)", prompt, re.IGNORECASE)
        if not match:
            return None
        return match.group(1)

    def _cone_allowed_gates_from_prompt(self) -> Optional[List[str]]:
        """Infer an allowed gate set from phrases like 'only NOR and NOT gates'."""
        prompt = getattr(self, "_active_user_message", "")
        match = re.search(r"\bonly\s+(.+?)\s+gates?\b", prompt, re.IGNORECASE)
        if not match:
            return None
        phrase = match.group(1)
        parts = re.split(r"\s*,\s*|\s+and\s+", phrase, flags=re.IGNORECASE)
        valid = {"nand", "nor", "or", "and", "not", "xor", "xnor", "buf"}
        gates: List[str] = []
        for part in parts:
            gate = part.strip().lower()
            if gate.startswith("and "):
                gate = gate[4:].strip()
            if gate in valid and gate not in gates:
                gates.append(gate)
        return gates or None

    def _prompt_uses_cone_depth_cost(self) -> bool:
        """Return True when the prompt optimizes the named cone's own depth."""
        prompt = getattr(self, "_active_user_message", "").lower()
        return (
            "depth of the cone" in prompt
            or "cone depth" in prompt
            or "cost function is the depth of the cone" in prompt
            or "cost function is depth of the cone" in prompt
        )

    def _prompt_requests_depth_optimization(self) -> bool:
        """Return True only when the request states a depth/path objective."""
        prompt = getattr(self, "_active_user_message", "")
        return bool(
            re.search(
                r"\b(?:depth|logic\s+levels?|critical\s+path|longest\s+path|"
                r"max(?:imum)?\s+path|timing|path\s+delay)\b",
                prompt,
                re.IGNORECASE,
            )
        )

    def _compact_tool_result_for_llm(
        self, tool_name: str, result_str: str
    ) -> str:
        """Shrink a tool result before feeding it back to the LLM."""
        try:
            payload = json.loads(result_str)
        except json.JSONDecodeError:
            return self._truncate_text(result_str)

        # This report can contain thousands of detailed matches.  Preserve its
        # semantics and stable metadata instead of falling through to generic
        # tail truncation, which can hide the count and make arbitrary tail
        # records look like the complete result.
        if tool_name == "analyze_dff_d_input_structures":
            data = payload.get("result")
            if isinstance(data, dict):
                matches = data.get("matches") or data.get("sample_matches") or []
                file_path = data.get("file_path")
                compact_result = {
                    "result_kind": "structural_enable_hold_candidates",
                    "validation": data.get("validation", "structural"),
                    "dffs_examined": data.get("dffs_examined"),
                    "self_feedback_candidates": data.get("self_feedback_candidates"),
                    "count": data.get("count"),
                    "classification_breakdown": data.get(
                        "classification_breakdown", {}
                    ),
                    "sample_matches": matches[:3],
                    "sample_matches_are_complete_list": not bool(file_path),
                    "file_path": file_path,
                    "file_path_type": "local" if file_path else None,
                }
                return json.dumps({"result": compact_result})

        # A singular path is a bounded witness/counterexample, not a collection
        # of independent results.  Keep it intact so the LLM can report a valid
        # source-to-sink path, including its destination.  Bulk results use the
        # plural ``paths`` key and remain subject to normal compaction.
        witness_path_tools = {
            "get_max_depth",
            "get_max_depth_between_endpoint_classes",
            "get_global_max_depth",
            "find_path_avoiding",
            "path_passes_through",
            "derive_boolean_equation",
            "list_primary_ios",
        }
        preserve_path_keys = tool_name in witness_path_tools
        compacted = self._compact_value(
            payload, preserve_path_keys=preserve_path_keys
        )
        compacted_str = json.dumps(compacted)
        if preserve_path_keys:
            # Applying the character-limit fallback here would discard the
            # beginning of a long path and undo the field-level exemption.
            return compacted_str
        if len(compacted_str) <= self._max_tool_result_chars:
            return compacted_str
        tail_limit = max(200, self._max_tool_result_chars - 500)
        return json.dumps(
            {
                "result_summary": "Tool result was compacted because it exceeded the inline character limit.",
                "inline_char_limit": self._max_tool_result_chars,
                "compacted_result_chars": len(compacted_str),
                "tail": compacted_str[-tail_limit:],
            }
        )

    def _compact_value(
        self,
        value: Any,
        *,
        preserve_path_keys: bool = False,
        current_key: Optional[str] = None,
    ) -> Any:
        """Recursively cap large lists and strings in JSON-like data."""
        if isinstance(value, dict):
            return {
                k: self._compact_value(
                    v,
                    preserve_path_keys=preserve_path_keys,
                    current_key=k,
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            if preserve_path_keys and current_key in {
                "path",
                "example_path",
                "counterexample_path",
            }:
                return [
                    self._compact_value(
                        item, preserve_path_keys=preserve_path_keys
                    )
                    for item in value
                ]
            if len(value) <= self._max_inline_items:
                return [
                    self._compact_value(
                        item, preserve_path_keys=preserve_path_keys
                    )
                    for item in value
                ]
            return {
                "inline_items": [
                    self._compact_value(
                        item, preserve_path_keys=preserve_path_keys
                    )
                    for item in value[: self._max_inline_items]
                ],
                "omitted_items": len(value) - self._max_inline_items,
                "total_items": len(value),
            }
        if isinstance(value, str):
            return self._truncate_text(value)
        return value

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self._max_tool_result_chars:
            return text
        return (
            f"[truncated to last {self._max_tool_result_chars} chars]\n"
            + text[-self._max_tool_result_chars :]
        )

    @staticmethod
    def _tool_call_signature(tool_name: str, arguments_json: str) -> str:
        """Return a stable key for detecting repeated identical tool calls."""
        try:
            args = json.loads(arguments_json) if arguments_json else {}
            canonical_args = json.dumps(args, sort_keys=True, separators=(",", ":"))
        except json.JSONDecodeError:
            canonical_args = arguments_json or ""
        return f"{tool_name}:{canonical_args}"

    def _summarize_tool_result(self, tool_name: str, result_str: str) -> str:
        """Create a one-line summary of a tool result for context tracking."""
        try:
            result = json.loads(result_str)
        except json.JSONDecodeError:
            return f"{tool_name} completed"

        if "error" in result:
            return f"{tool_name} FAILED: {self._truncate_text(str(result['error']))}"

        data = result.get("result", result)

        if tool_name == "read_design":
            return f"loaded design via read_design"
        elif tool_name == "write_design":
            return f"wrote design via write_design"
        elif tool_name == "set_testcase_name":
            return f"testcase initialized via set_testcase_name"
        elif tool_name == "count_gates":
            total = data.get("total_gates", "?")
            breakdown = data.get("breakdown", {})
            parts = ", ".join(f"{cnt}x {gt}" for gt, cnt in sorted(breakdown.items()) if cnt > 0)
            return f"count_gates: {total} total gates ({parts})"
        elif tool_name == "count_primary_ios":
            return (
                f"count_primary_ios: "
                f"{data.get('primary_input_ports')} PI ports, "
                f"{data.get('primary_output_ports')} PO ports"
            )
        elif tool_name == "list_primary_ios":
            return (
                f"list_primary_ios: "
                f"{data.get('primary_input_ports')} PI ports/"
                f"{data.get('primary_input_bits')} bits, "
                f"{data.get('primary_output_ports')} PO ports/"
                f"{data.get('primary_output_bits')} bits"
            )
        elif tool_name == "count_gate_types_in_cone":
            by_type = data.get("by_type", {})
            parts = ", ".join(
                f"{count}x {gate_type.upper()}"
                for gate_type, count in sorted(by_type.items())
            )
            sig = data.get("output_signal", "?")
            total = data.get("total", 0)
            return (
                f"count_gate_types_in_cone: cone of {sig} has {total} gates"
                + (f" ({parts})" if parts else "")
            )
        elif tool_name == "find_gates":
            count = data.get("count", 0)
            gate_type = data.get("gate_type")
            input_count = data.get("input_count")
            has_input = data.get("has_input")
            gate_desc = (
                f"{input_count}-input {str(gate_type).upper()} gates"
                if input_count
                else f"{str(gate_type).upper()} gates"
            )
            if gate_type == "any":
                gate_desc = (
                    f"{input_count}-input gates" if input_count else "gates"
                )
            input_desc = ""
            if has_input in {"1'b1", "1", "'1"}:
                input_desc = " with input tied to constant 1"
            elif has_input in {"1'b0", "0", "'0"}:
                input_desc = " with input tied to constant 0"
            elif has_input:
                input_desc = f" with input {has_input}"

            summary = f"find_gates: found {count} {gate_desc}{input_desc}"
            matches = data.get("matches") or data.get("sample_matches") or []
            sample_names = [
                item.get("instance")
                for item in matches[:5]
                if isinstance(item, dict) and item.get("instance")
            ]
            if sample_names:
                summary += f": {', '.join(sample_names)}"
            if data.get("file_path"):
                summary += f"; full list in {data['file_path']}"
            return summary
        elif tool_name == "get_max_depth":
            return f"get_max_depth: depth={data.get('depth')}"
        elif tool_name == "get_max_depth_between_endpoint_classes":
            if data.get("scope") == "global":
                return self._summarize_tool_result(
                    "get_global_max_depth", result_str
                )
            summary = (
                f"get_max_depth_between_endpoint_classes: "
                f"{data.get('source_class')} to {data.get('sink_class')} "
                f"depth={data.get('depth')}"
            )
            if data.get("sink_class") == "DFF_D" and data.get("sink_instance"):
                summary += (
                    f"; {data.get('source_signal')} -> "
                    f"{data.get('sink_instance')}.D "
                    f"(D net {data.get('sink_pin_signal')})"
                )
            if data.get("source_class") == "DFF_Q" and data.get("source_instance"):
                summary += (
                    f"; source {data.get('source_instance')}.Q "
                    f"(Q net {data.get('source_pin_signal')})"
                )
            return summary
        elif tool_name == "get_global_max_depth":
            return (
                f"get_global_max_depth: depth={data.get('depth')} "
                f"({data.get('source_class')} {data.get('source_signal')} to "
                f"{data.get('sink_class')} {data.get('sink_signal')})"
            )
        elif tool_name == "path_passes_through":
            answer = "YES" if data.get("all_paths_pass_through") else "NO"
            path_state = "path exists" if data.get("path_exists") else "no path exists"
            return (
                f"path_passes_through: {answer} for "
                f"{data.get('source')}->{data.get('sink')} via "
                f"{data.get('through')} ({path_state})"
            )
        elif tool_name == "find_articulation_points":
            if "shared_count" in data:
                return self._summarize_tool_result(
                    "find_shared_fanin_gates", result_str
                )
            src = data.get("source", "?")
            snk = data.get("sink", "?")
            count = data.get("count", 0)
            gates = ", ".join(data.get("articulation_points", []))
            return (
                f"find_articulation_points: {count} gate(s) between {src} and {snk}"
                + (f" ({gates})" if gates else "")
            )
        elif tool_name == "find_shared_fanin_gates":
            summary = (
                f"find_shared_fanin_gates: {data.get('shared_count', 0)} shared "
                f"gate(s) between cones of {data.get('signal_a')} "
                f"({data.get('cone_a_count', 0)} gates) and {data.get('signal_b')} "
                f"({data.get('cone_b_count', 0)} gates)"
            )
            if data.get("file_path"):
                summary += f"; full list in {data['file_path']}"
            return summary
        elif tool_name == "is_wire_cut_between_primary_ios":
            answer = "YES" if data.get("is_cut_between_primary_io") else "NO"
            return (
                f"is_wire_cut_between_primary_ios: {answer} for "
                f"{data.get('wire')} "
                f"(on PI-to-PO path: "
                f"{'YES' if data.get('is_on_any_pi_po_path') else 'NO'})"
            )
        elif tool_name == "is_gate_on_any_max_depth_path":
            return (
                f"is_gate_on_any_max_depth_path: {data.get('gate')} "
                f"{'YES' if data.get('on_any_max_depth_path') else 'NO'} "
                f"(global depth {data.get('global_max_depth')})"
            )
        elif tool_name == "get_fanin_cone_depth":
            return (
                f"fanin cone depth of "
                f"{data.get('output')} = "
                f"{data.get('depth')}"
            )
        elif tool_name == "get_deepest_output_fanin_cone":
            # Dispatch may correct a semantically wrong depth-tool selection for
            # an explicit largest/smallest-cone request. Summarize the operation
            # that actually ran so history retains its gate-count metric.
            if data.get("metric") == "gate_count" and "signal_class" in data:
                return self._summarize_tool_result(
                    "rank_signals_by_fanin_cone", result_str
                )
            summary = (
                f"get_deepest_output_fanin_cone: depth={data.get('max_depth')}; "
                f"{data.get('tie_count', 0)} tied output bit(s)"
            )
            outputs = data.get("deepest_outputs", [])
            if outputs:
                summary += f": {', '.join(outputs)}"
            if data.get("file_path"):
                summary += f"; full list in {data['file_path']}"
            return summary
        elif tool_name == "derive_boolean_equation":
            sig = data.get("output_signal", "?")
            dtype = data.get("driver_type", "")
            eq = data.get("equation") or data.get("boundary_state_equation", "")
            if data.get("request_satisfied") is False:
                boundaries = data.get("state_boundaries", [])
                answer = str(data.get("answer", "PI-only expression unavailable")).rstrip(".")
                detail = (
                    f"; DFF-Q/state boundaries: {', '.join(boundaries)}"
                    if boundaries else ""
                )
                return f"derive_boolean_equation: {answer}{detail}."
            if dtype == "dff":
                d_input = data.get("dff_pins", {}).get("d", "?")
                return (
                    "derive_boolean_equation: PI-only expression unavailable; "
                    f"{sig} is registered state; structural equation: {eq}; D={d_input}"
                )
            if data.get("primary_input_only_expression_available") is False:
                boundaries = data.get("state_boundaries", [])
                boundary_text = ", ".join(boundaries) if boundaries else "non-PI boundaries"
                return (
                    "derive_boolean_equation: PI-only expression unavailable; "
                    f"depends on DFF-Q/state boundaries {boundary_text}; "
                    f"boundary-state equation: {eq}"
                )
            return f"derive_boolean_equation: {eq} ({data.get('cone_gate_count', 0)} cone gates)"
        elif tool_name == "get_logic_cone":
            count = data.get("count", 0)
            scope = ""
            if data.get("through_dff"):
                scope = (
                    f"; {data.get('output_signal')} resolved through "
                    f"{data.get('through_dff')}.D to {data.get('resolved_output')}"
                )
            if data.get("file_path"):
                return (
                    f"get_logic_cone: {count} transitive fanin gates{scope}; "
                    f"full list in {data['file_path']}"
                )
            return (
                f"get_logic_cone: {count} transitive fanin gates{scope} "
                f"({', '.join(data.get('gates', []))})"
            )
        elif tool_name == "count_outputs_by_logic_depth":
            return (
                f"count_outputs_by_logic_depth: "
                f"{data.get('count')} outputs "
                f"{data.get('operator')} {data.get('threshold')}"
            )
        elif tool_name == "find_all_paths":
            summary = (
                f"find_all_paths: "
                f"{data.get('count')} paths "
                f"from {data.get('source')} "
                f"to {data.get('sink')}"
            )
            if data.get("file_path"):
                summary += f"; full list in {data['file_path']}"
            return summary
        elif tool_name == "find_gates_with_constant_inputs":
            count = data.get("count", 0)
            gate_type = str(data.get("gate_type", "gate")).upper()
            values = data.get("values", [0, 1])
            complete = "complete" if data.get("complete", True) else "incomplete"
            summary = (
                f"find_gates_with_constant_inputs: found {count} {gate_type} "
                f"gate(s) with constant input(s) {values} ({complete})"
            )
            matches = data.get("matches") or data.get("sample_matches") or []
            sample_names = [
                item.get("instance")
                for item in matches[:5]
                if isinstance(item, dict) and item.get("instance")
            ]
            if sample_names:
                summary += f": {', '.join(sample_names)}"
            if data.get("file_path"):
                summary += f"; full list in {data['file_path']}"
            if data.get("unknown_signal_count"):
                summary += f"; {data['unknown_signal_count']} signal(s) unresolved"
            return summary
        elif tool_name == "simplify_gates_with_constant_inputs":
            return (
                "simplify_gates_with_constant_inputs: simplified "
                f"{data.get('simplified_gates', 0)} "
                f"{str(data.get('gate_type', '?')).upper()} gate(s); "
                f"replacements={data.get('replacement_breakdown', {})}"
            )
        elif tool_name == "find_zero_length_pi_po_paths":
            paths = data.get("paths", [])
            names = [p.get("source") for p in paths[:5] if isinstance(p, dict)]
            detail = f": {', '.join(names)}" if names else ""
            return (
                f"find_zero_length_pi_po_paths: found "
                f"{data.get('count', 0)} direct PI-to-PO wire path(s){detail}"
            )
        elif tool_name == "find_register_to_register_paths":
            summary = f"find_register_to_register_paths: {data.get('count', 0)} paths"
            if data.get("file_path"):
                summary += f"; full list in {data['file_path']}"
            if data.get("truncated"):
                summary += "; result truncated at safety limit"
            return summary
        elif tool_name in {"get_net_fanout", "get_gate_output_fanout"}:
            count = data.get("count", 0)
            if data.get("file_path"):
                return f"{tool_name}: {count} direct gates; full list in {data['file_path']}"
            return f"{tool_name}: {count} direct gates ({', '.join(data.get('fanout', []))})"
        elif tool_name == "get_net_connections":
            drivers = [item.get("instance", "?") for item in data.get("drivers", [])]
            loads = [item.get("instance", "?") for item in data.get("loads", [])]
            summary = (
                f"get_net_connections: {data.get('net_name', '?')} has "
                f"{data.get('driver_count', 0)} driver(s) ({', '.join(drivers)}) and "
                f"{data.get('load_count', 0)} load(s) ({', '.join(loads)})"
            )
            if data.get("file_path"):
                summary += f"; full list in {data['file_path']}"
            return summary
        elif tool_name == "rank_signals_by_fanout":
            rankings = data.get("rankings") or data.get("sample_rankings") or []
            ranking_text = "; ".join(
                f"rank {item.get('rank')}: fanout {item.get('fanout')} "
                f"({', '.join(item.get('signals', []))})"
                for item in rankings
            )
            summary = (
                f"rank_signals_by_fanout: examined "
                f"{data.get('examined_signal_count', 0)} "
                f"{data.get('signal_class', '?')} signal(s); {ranking_text}"
            )
            if data.get("file_path"):
                summary += f"; full ranking in {data['file_path']}"
            return summary
        elif tool_name == "rank_signals_by_fanin_cone":
            rankings = data.get("rankings") or data.get("sample_rankings") or []
            ranking_text = "; ".join(
                f"rank {item.get('rank')}: {data.get('metric')}={item.get('value')} "
                f"({', '.join(item.get('signals', []))})"
                for item in rankings
            )
            summary = (
                f"rank_signals_by_fanin_cone: examined "
                f"{data.get('examined_signal_count', 0)} "
                f"{data.get('signal_class', '?')} signal(s); {ranking_text}"
            )
            if data.get("file_path"):
                summary += f"; full ranking in {data['file_path']}"
            return summary
        elif tool_name == "resolve_name_type":
            return f"resolve_name_type: {data.get('name')} is {data.get('type')}"
        elif tool_name == "replace_gate":
            return (
                f"replace_gate: {data.get('replaced_count', 1)} "
                f"gate(s) -> {data.get('new_type')}"
            )
        elif tool_name == "get_gate_info":
            pins = ", ".join(
                f"{pin}={net}" for pin, net in data.get("pins", {}).items()
            )
            return (
                f"get_gate_info: {data.get('instance')} is {data.get('gate_type')} "
                f"({pins})"
            )
        elif tool_name in {"get_transitive_fanout_cone", "get_reachable_gates_from_gate"}:
            count = data.get("count", 0)
            if data.get("file_path"):
                return f"{tool_name}: {count} reachable gates; full list in {data['file_path']}"
            return f"{tool_name}: {count} reachable gates ({', '.join(data.get('gates', []))})"
        elif tool_name == "find_instances_by_name_pattern":
            count = data.get("count", 0)
            instances = data.get("instances", [])
            return f"find_instances_by_name_pattern: found {count} instances ({', '.join(instances[:5])}{'...' if count > 5 else ''})"
        elif tool_name == "rename_gate":
            return f"rename_gate: '{data.get('old_name')}' → '{data.get('new_name')}'"
        elif tool_name == "rename_wire":
            return f"rename_wire: '{data.get('old_name')}' → '{data.get('new_name')}'"
        elif tool_name == "replace_gate":
            return f"replace_gate: {data.get('replaced')} → {data.get('new_type')}"
        elif tool_name == "replace_pattern":
            return f"replace_pattern: {data.get('replacements', 0)} replacements"
        elif tool_name == "insert_buffers_for_fanout":
            return f"insert_buffers_for_fanout: {data.get('buffers_inserted', 0)} buffers"
        elif tool_name == "insert_dedicated_buffers_for_loads":
            return (
                f"insert_dedicated_buffers_for_loads: "
                f"{data.get('buffers_inserted', 0)} buffers on {data.get('net_name')}"
            )
        elif tool_name == "balance_depth":
            return f"balance_depth: {data.get('buffers_inserted', 0)} buffers"
        elif tool_name == "remove_dangling_gates":
            return f"remove_dangling_gates: {data.get('gates_removed', 0)} gates removed"
        elif tool_name == "collapse_inverter_pairs":
            return (
                f"collapse_inverter_pairs: {data.get('pairs_collapsed', 0)} pairs collapsed, "
                f"{data.get('gates_removed', 0)} gates removed"
            )
        elif tool_name == "replace_gate_type_in_cone":
            ok = data.get("success", False)
            if ok:
                replaced = data.get("replaced", 0)
                skipped  = data.get("skipped", 0)
                src   = data.get("source_type", "?")
                tgt   = data.get("target_types", [])
                scope = data.get("output_signal") or "whole design"
                before_counts = data.get("gate_counts_before", {})
                after_counts = data.get("gate_counts_after", {})
                changed_types = [src]
                changed_types.extend(tgt if isinstance(tgt, list) else [])
                count_parts = []
                for gate_type in sorted(set(changed_types)):
                    key = str(gate_type).upper()
                    before_count = before_counts.get(key, before_counts.get(gate_type))
                    after_count = after_counts.get(key, after_counts.get(gate_type))
                    if before_count is not None and after_count is not None:
                        delta = after_count - before_count
                        count_parts.append(
                            f"{key} {before_count}→{after_count} ({delta:+d})"
                        )
                count_detail = f"; {', '.join(count_parts)}" if count_parts else ""
                return (
                    f"replace_gate_type_in_cone: replaced {replaced}x '{src}' with {tgt} "
                    f"in cone of {scope}" +
                    (f" ({skipped} skipped)" if skipped else "") +
                    count_detail
                )
            else:
                return f"replace_gate_type_in_cone: FAILED — {data.get('reason', 'unknown error')}"
        elif tool_name == "remap_cone_with_gates":
            # The dispatcher may deliberately narrow an over-broad model call
            # to replace_gate_type_in_cone when the prompt names one source
            # gate type.  Format the history using the operation that actually
            # ran, whose result schema has source_type/target_types/replaced,
            # rather than the model-requested tool name.
            if {
                "source_type", "target_types", "replaced"
            }.issubset(data):
                return self._summarize_tool_result(
                    "replace_gate_type_in_cone", result_str
                )
            ok = data.get("success", False)
            if ok:
                before = data.get("gates_before", "?")
                after  = data.get("gates_after", "?")
                sig    = data.get("output_signal", "?")
                gates  = data.get("allowed_gates", [])
                by_type = data.get("gate_types_after", {})
                parts = ", ".join(
                    f"{count}x {gate_type.upper()}"
                    for gate_type, count in sorted(by_type.items())
                )
                detail = f" ({parts})" if parts else ""
                return f"remap_cone_with_gates: cone of {sig} remapped to {gates} — {before} gates → {after} gates{detail}"
            else:
                return f"remap_cone_with_gates: FAILED — {data.get('reason', 'unknown error')}"
        elif tool_name == "remap_design_with_gates":
            if data.get("success", False):
                comb_before = data.get("combinational_gates_before")
                comb_after = data.get("combinational_gates_after")
                dff_count = data.get("dff_count")
                detail = ""
                if comb_before is not None and comb_after is not None:
                    detail = (
                        f" ({comb_before}→{comb_after} combinational"
                        + (f", {dff_count} DFFs preserved" if dff_count is not None else "")
                        + ")"
                    )
                return (
                    f"remap_design_with_gates: whole design remapped to "
                    f"{data.get('allowed_gates', [])} — "
                    f"{data.get('gates_before', '?')} total gates → "
                    f"{data.get('gates_after', '?')} total gates{detail}"
                )
            return f"remap_design_with_gates: FAILED — {data.get('reason', 'unknown error')}"
        elif tool_name == "fraig_merge_equivalent_gates":
            if data.get("success", False):
                comb_before = data.get("combinational_gates_before", data.get("gates_before", "?"))
                comb_after = data.get("combinational_gates_after", data.get("gates_after", "?"))
                total_before = data.get("total_gates_before")
                total_after = data.get("total_gates_after")
                dff_count = data.get("dff_count")
                total_detail = ""
                if total_before is not None and total_after is not None:
                    total_detail = (
                        f"; {total_before}→{total_after} total gates"
                        + (f" including {dff_count} unchanged DFFs" if dff_count is not None else "")
                    )
                return (
                    f"fraig_merge_equivalent_gates: "
                    f"{comb_before} combinational gates → {comb_after} combinational gates "
                    f"(reduction {data.get('gate_reduction', '?')}{total_detail})"
                )
            return f"fraig_merge_equivalent_gates: FAILED"
        elif tool_name == "check_signal_equivalence":
            statement = data.get("statement")
            if statement:
                return f"check_signal_equivalence: {data.get('verdict')} — {statement}"
            return f"check_signal_equivalence: equivalent={data.get('equivalent')}"
        elif tool_name == "check_function_symmetry":
            answer = "YES" if data.get("symmetric") else "NO"
            return (
                f"check_function_symmetry: {answer}; "
                f"{data.get('output_signal')} symmetric in "
                f"{data.get('input_a')} and {data.get('input_b')} "
                f"(proof={data.get('proof')})"
            )
        elif tool_name == "find_binary_gate_equivalent_pair":
            if data.get("exists"):
                return (
                    f"find_binary_gate_equivalent_pair: YES, "
                    f"{data.get('gate_type', '?').upper()}({data.get('a')}, {data.get('b')}) "
                    f"is equivalent to {data.get('target_signal')} "
                    f"({data.get('proof', 'confirmed')})"
                )
            completeness = "complete" if data.get("search_complete") else "incomplete"
            return (
                f"find_binary_gate_equivalent_pair: NO proven pair for "
                f"{data.get('gate_type', '?').upper()}(_, _) equivalent to "
                f"{data.get('target_signal')} ({completeness} search)"
            )
        elif tool_name == "list_flip_flops_by_clock":
            summary = (
                f"list_flip_flops_by_clock: found {data.get('count', 0)} "
                f"DFF(s) clocked by {data.get('clock_signal')}"
            )
            if data.get("file_path"):
                summary += f"; full list in {data['file_path']}"
            return summary
        elif tool_name == "analyze_dff_d_input_structures":
            summary = (
                "analyze_dff_d_input_structures: examined "
                f"{data.get('dffs_examined', 0)} DFF(s), found "
                f"{data.get('count', 0)} structural enable/hold candidate(s); "
                f"self-feedback candidates={data.get('self_feedback_candidates', 0)}; "
                f"breakdown={data.get('classification_breakdown', {})}"
            )
            if data.get("file_path"):
                summary += f"; full report in {data['file_path']}"
            return summary
        elif tool_name == "check_signal_constant":
            answer = "yes" if data.get("always_equal") else "no"
            return (
                f"check_signal_constant: {data.get('signal')} always equals "
                f"{data.get('value')} = {answer}"
            )
        elif tool_name == "check_design_equivalence":
            return (
                f"check_design_equivalence: {data.get('status', 'UNKNOWN')} "
                f"against {data.get('original_netlist', 'original design')}"
            )
        elif tool_name == "optimize_cone_depth":
            return f"optimize_cone_depth: {'OK' if data.get('success') else 'FAILED'}"
        elif tool_name == "reduce_critical_path":
            before = data.get('depth_before', '?')
            after = data.get('depth_after', '?')
            imp = data.get('improvement', '?')
            ok = data.get('success', False)
            restriction = data.get('allowed_gates')
            restricted = f" using only {restriction}" if restriction else ""
            return f"reduce_critical_path: depth {before} → {after} (improved by {imp}){restricted} {'OK' if ok else 'NO IMPROVEMENT'}"
        elif tool_name == "optimize_depth_preserving_cone_gate_set":
            if {
                "gates_before", "gates_after", "gate_types_after"
            }.issubset(data):
                return self._summarize_tool_result(
                    "remap_cone_with_gates", result_str
                )
            before = data.get("depth_before", "?")
            after_unrestricted = data.get("depth_after_unrestricted", "?")
            after = data.get("depth_after", "?")
            imp = data.get("improvement", "?")
            sig = data.get("output_signal", "?")
            gates = data.get("allowed_gates", [])
            status = "OK" if data.get("success", False) else "NO IMPROVEMENT"
            if not data.get("success", False) and data.get("reason"):
                status = f"FAILED: {data.get('reason')}"
            if data.get("restored_input_design"):
                status += "; input design restored"
            return (
                "optimize_depth_preserving_cone_gate_set: "
                f"depth {before} → {after_unrestricted} unrestricted → {after} final "
                f"(improved by {imp}); cone of {sig} restricted to {gates} {status}"
            )
        elif tool_name == "optimize_cone_depth_preserving_gate_set":
            if {
                "gates_before", "gates_after", "gate_types_after"
            }.issubset(data):
                return self._summarize_tool_result(
                    "remap_cone_with_gates", result_str
                )
            before = data.get("depth_before", "?")
            after = data.get("depth_after", "?")
            imp = data.get("improvement", "?")
            sig = data.get("output_signal", "?")
            gates = data.get("allowed_gates", [])
            status = "OK" if data.get("success", False) else "NO IMPROVEMENT"
            if (
                not data.get("success", False)
                and data.get("reason")
                and not data.get("restored_input_design")
            ):
                status = f"FAILED: {data.get('reason')}"
            if data.get("restored_input_design"):
                status = "NO IMPROVEMENT; input design restored"
            return (
                "optimize_cone_depth_preserving_gate_set: "
                f"cone {sig} depth {before} → {after} "
                f"(improved by {imp}); restricted to {gates} {status}"
            )
        else:
            return f"{tool_name} completed"

    # A simple counter for log-file naming (shared across all requests)
    _request_counter: int = 0

    def process_request(self, user_message: str, context_summary: str = "") -> str:
        """Process a natural-language EDA request and return a text response.

        Sends user_message to the LLM with function-calling enabled,
        dispatches tool calls to the EDA engine, and loops until the model
        returns a final text response.  When developer mode is enabled in
        config.yaml, the full conversation is logged to logs/.

        If *context_summary* is non-empty, it is inserted as an additional
        system message so the LLM knows the current engine state (loaded
        design, previous operations, etc.).

        Args:
            user_message:    The user's natural-language request.
            context_summary: Optional summary of previous operations' state.

        Returns:
            Final text response from the LLM.

        Raises:
            RuntimeError: On API errors or unsupported configurations.
        """
        EDAAgent._request_counter += 1

        # Extract case name; if not found in this line, reuse the last known one
        label = self._extract_case_name(user_message)
        if label:
            self._current_case_name = label
        else:
            label = self._current_case_name

        with self._LoggerCls(
            EDAAgent._request_counter, self._logs_dir, self._model, label=label
        ) as conv_log:
            self._active_user_message = user_message
            request_tool_summaries: List[str] = []

            def finish(final_text: str) -> str:
                request_summary = self._request_context_summary(
                    user_message, request_tool_summaries
                )
                if request_summary:
                    self._append_context_summary(request_summary)
                conv_log.log_final_response(final_text)
                return final_text

            conv_log.log_user_input(user_message)
            if context_summary and self._developer_mode:
                conv_log._fh.write(
                    f"  ── CONTEXT ──\n  {context_summary}\n  ─────────────\n\n"
                )

            messages: List[Dict[str, Any]] = []
            messages.append({"role": "system", "content": _SYSTEM_PROMPT})
            if context_summary:
                messages.append(
                    {"role": "system", "content": f"Current state: {context_summary}"}
                )
            messages.append({"role": "user", "content": user_message})

            turn = 0
            tool_rounds = 0
            seen_tool_calls: set[str] = set()
            while True:
                turn += 1
                timeout = _FAST_TIMEOUT  # conservative default

                conv_log.log_llm_prompt(messages, turn)
                t0 = time.monotonic()
                response = self._call_llm(messages, timeout)
                elapsed = time.monotonic() - t0
                conv_log.log_llm_response(response, turn, elapsed)

                choice = response.choices[0]
                msg = choice.message

                # Append assistant message to history
                messages.append(msg.model_dump(exclude_unset=False))

                if choice.finish_reason == "tool_calls" and msg.tool_calls:
                    if tool_rounds >= self._max_tool_rounds:
                        final_text = (
                            "Stopped after reaching the per-request tool-call round "
                            f"limit ({self._max_tool_rounds}) to avoid excessive token use. "
                            "The last tool result is available in the developer log."
                        )
                        return finish(final_text)
                    tool_rounds += 1

                    tool_call_names = [tc.function.name for tc in msg.tool_calls]
                    skip_reduce_for_cone_depth = (
                        "reduce_critical_path" in tool_call_names
                        and "optimize_depth_preserving_cone_gate_set" in tool_call_names
                    )

                    # Dispatch each tool call
                    for tc in msg.tool_calls:
                        tool_name = tc.function.name
                        call_signature = self._tool_call_signature(
                            tool_name, tc.function.arguments
                        )
                        if call_signature in seen_tool_calls:
                            final_text = (
                                "Stopped after the model repeated the same tool call "
                                f"({tool_name}) with the same arguments in one request. "
                                "This guard prevents repeated token spend when a tool "
                                "result is not resolving the request."
                            )
                            return finish(final_text)
                        seen_tool_calls.add(call_signature)

                        if skip_reduce_for_cone_depth and tool_name == "reduce_critical_path":
                            result_str = json.dumps(
                                {
                                    "result": {
                                        "skipped": True,
                                        "reason": (
                                            "Skipped because optimize_depth_preserving_cone_gate_set "
                                            "was also called in this tool batch and supersedes "
                                            "reduce_critical_path for cone-restricted depth optimization."
                                        ),
                                    }
                                }
                            )
                            conv_log.log_tool_result(tool_name, result_str)
                            request_tool_summaries.append(
                                self._summarize_tool_result(tool_name, result_str)
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": result_str,
                                }
                            )
                            continue

                        op_timeout = (
                            _FAST_TIMEOUT if tool_name in _FAST_OPS else _SLOW_TIMEOUT
                        )
                        result_str = self._dispatch_tool(
                            tool_name, tc.function.arguments, op_timeout
                        )
                        conv_log.log_tool_result(tool_name, result_str)

                        # Track operation in context history
                        summary = self._summarize_tool_result(
                            tool_name, result_str
                        )
                        request_tool_summaries.append(summary)

                        # If a design was just loaded, dump the netlist to the log
                        if tool_name == "read_design":
                            conv_log.log_netlist(self._engine.netlist)

                        compact_result_str = self._compact_tool_result_for_llm(
                            tool_name, result_str
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": compact_result_str,
                            }
                        )
                else:
                    # Final text response
                    final_text = msg.content or ""
                    return finish(final_text)

    # ------------------------------------------------------------------
    # Internal: LLM call
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_retry_delay(value: Optional[str]) -> Optional[float]:
        """Parse Retry-After or OpenAI reset-header values into seconds."""
        if not value:
            return None
        text = str(value).strip().lower()
        try:
            return max(0.0, float(text))
        except ValueError:
            pass

        total = 0.0
        pos = 0
        units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        for match in re.finditer(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text):
            if match.start() != pos:
                break
            total += float(match.group(1)) * units[match.group(2)]
            pos = match.end()
        if pos == len(text) and pos > 0:
            return total

        try:
            retry_at = parsedate_to_datetime(str(value))
            now = time.time()
            return max(0.0, retry_at.timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _server_retry_delay(cls, exc: Exception) -> Optional[float]:
        """Read server-directed retry timing from an OpenAI API error."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        for name in ("retry-after", "x-ratelimit-reset-tokens"):
            delay = cls._parse_retry_delay(headers.get(name))
            if delay is not None:
                return delay
        return None

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return whether an SDK exception represents HTTP 429."""
        return (
            getattr(exc, "status_code", None) == 429
            or exc.__class__.__name__ == "RateLimitError"
        )

    def _pace_llm_call(self) -> None:
        """Enforce a minimum interval between all model-call attempts."""
        elapsed = time.monotonic() - self._last_llm_call_started
        remaining = self._llm_min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_llm_call_started = time.monotonic()

    def _call_llm(
        self, messages: List[Dict[str, Any]], timeout: int
    ) -> Any:
        """Call the OpenAI chat completions API.

        Args:
            messages: Conversation history.
            timeout:  Seconds before raising TimeoutError.

        Returns:
            The API response object.

        Raises:
            RuntimeError: On API errors.
        """
        for attempt in range(self._rate_limit_max_retries + 1):
            self._pace_llm_call()
            try:
                with _Timeout(timeout):
                    return self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
            except TimeoutError:
                raise RuntimeError(
                    f"LLM call timed out after {timeout} seconds."
                )
            except Exception as exc:
                if "credit_balance_exhausted" in str(exc) or "insufficient_quota" in str(exc):
                    raise RuntimeError(f"OpenAI API quota exhausted: {exc}") from exc
                if not self._is_rate_limit_error(exc):
                    raise RuntimeError(f"OpenAI API error: {exc}") from exc
                if attempt >= self._rate_limit_max_retries:
                    raise RuntimeError(
                        "OpenAI API rate limit remained active after "
                        f"{self._rate_limit_max_retries} retries: {exc}"
                    ) from exc

                server_delay = self._server_retry_delay(exc)
                exponential_delay = min(
                    self._rate_limit_initial_delay * (2 ** attempt),
                    self._rate_limit_max_delay,
                )
                delay = (
                    server_delay if server_delay is not None
                    else exponential_delay
                )
                delay = min(delay, self._rate_limit_max_delay)
                delay += random.uniform(0.0, self._rate_limit_jitter)
                time.sleep(delay)

        raise RuntimeError("OpenAI API retry loop ended unexpectedly.")

    # ------------------------------------------------------------------
    # Internal: tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(
        self, tool_name: str, arguments_json: str, timeout: int
    ) -> str:
        """Execute a tool call and return a JSON-serialisable result string.

        On failure the error message is returned so the LLM can decide
        how to proceed.

        Args:
            tool_name:       Name of the tool to call.
            arguments_json:  JSON string of tool arguments.
            timeout:         Seconds before aborting.

        Returns:
            JSON string with the tool result or an error description.
        """
        try:
            args: Dict[str, Any] = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON arguments: {exc}"})

        try:
            with _Timeout(timeout):
                result = self._invoke_tool(tool_name, args)
            return json.dumps({"result": result})
        except TimeoutError:
            return json.dumps(
                {"error": f"Tool '{tool_name}' timed out after {timeout} s"}
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Map a tool name to an EDAEngine method and call it.

        Args:
            tool_name: The tool name string.
            args:      Dictionary of parsed arguments.

        Returns:
            The raw Python return value of the EDA operation.

        Raises:
            ValueError: If the tool name is unknown.
        """
        eng = self._engine

        if tool_name == "read_design":
            eng.load(args["filepath"])
            return f"Design loaded from {eng.original_netlist_path!r}"

        if tool_name == "write_design":
            filepath = str(args.get("filepath") or "").strip()
            if not filepath:
                base_name = self._current_case_name or "design"
                filepath = f"{base_name}_out.v"
            eng.save(filepath)
            return f"Design written to {filepath!r}"

        if tool_name == "count_gates":
            return eng.count_gates()

        if tool_name == "count_primary_ios":
            return eng.count_primary_ios()

        if tool_name == "list_primary_ios":
            return eng.list_primary_ios(args.get("io_type"))


        if tool_name == "set_testcase_name":
            case_name: str = args["case_name"]
            log_path: str = str(args.get("log_path") or "").strip()
            if not log_path:
                log_path = f"{case_name}.log"
            if self._io_handler is not None:
                self._io_handler.set_log_file(log_path)
            self._current_case_name = case_name
            return f"Testcase name set to {case_name!r}; log file: {log_path!r}"

        if tool_name == "get_max_depth":
            depth, path = eng.get_max_depth(args["source"], args["sink"])
            return {"depth": depth, "path": path}

        if tool_name == "get_max_depth_between_endpoint_classes":
            prompt = getattr(self, "_active_user_message", "").lower()
            explicit_endpoint_scope = (
                "primary input" in prompt
                or "primary output" in prompt
                or "dff" in prompt
                or bool(re.search(r"\bfrom\b.+\bto\b", prompt))
            )
            whole_design_depth = (
                "depth" in prompt
                and not explicit_endpoint_scope
                and any(
                    phrase in prompt
                    for phrase in (
                        "in the design",
                        "of the design",
                        "entire design",
                        "whole design",
                        "global maximum",
                    )
                )
            )
            if whole_design_depth:
                return eng.get_global_max_depth()
            return eng.get_max_depth_between_endpoint_classes(
                args["source_class"], args["sink_class"]
            )

        if tool_name == "get_global_max_depth":
            return eng.get_global_max_depth()

        if tool_name == "is_gate_on_any_max_depth_path":
            return eng.is_gate_on_any_max_depth_path(args["gate_name"])

        if tool_name == "get_fanin_cone_depth":
            return eng.get_fanin_cone_depth(
                args["output_signal"]
            )
        if tool_name == "get_deepest_output_fanin_cone":
            prompt = getattr(self, "_active_user_message", "").lower()
            size_request = (
                "fanin cone" in prompt
                and any(word in prompt for word in ("largest", "smallest"))
                and not any(word in prompt for word in ("deepest", "shallowest", "depth"))
            )
            if size_request:
                return eng.rank_signals_by_fanin_cone(
                    "PO",
                    "gate_count",
                    "ascending" if "smallest" in prompt else "descending",
                    1,
                    True,
                    100,
                )
            return eng.get_deepest_output_fanin_cone()
        if tool_name == "count_outputs_by_logic_depth":
            return eng.count_outputs_by_logic_depth(
                args["operator"],
                int(args["threshold"])
            )
        if tool_name == "path_passes_through":
            return eng.paths_pass_through_report(
                args["source"], args["sink"], args["node"]
            )

        if tool_name == "is_wire_cut_between_primary_ios":
            return eng.is_wire_cut_between_primary_ios(args["wire_name"])

        if tool_name == "find_path_avoiding":
            path = eng.find_path_avoiding(
                args["source"], args["sink"], args["avoid"]
            )
            return {"path": path}

        if tool_name == "find_articulation_points":
            prompt = getattr(self, "_active_user_message", "").lower()
            if (
                "fanin cone" in prompt
                and any(word in prompt for word in ("shared", "common", "intersection"))
            ):
                return eng.find_shared_fanin_gates(
                    args["source"], args["sink"]
                )
            return eng.find_articulation_points(
                args["source"], args["sink"]
            )

        if tool_name == "get_logic_cone":
            return eng.get_logic_cone_report(args["output_signal"])

        if tool_name == "find_shared_fanin_gates":
            return eng.find_shared_fanin_gates(
                args["signal_a"],
                args["signal_b"],
                int(args.get("inline_limit", 20)),
            )

        if tool_name == "derive_boolean_equation":
            result = eng.derive_boolean_equation(
                args["output_signal"],
                int(args.get("inline_limit", 20)),
            )
            prompt = getattr(self, "_active_user_message", "")
            asks_for_primary_inputs = bool(
                re.search(r"\bprimary[ -]inputs?\b", prompt, re.IGNORECASE)
            )
            if (
                asks_for_primary_inputs
                and result.get("primary_input_only_expression_available") is False
            ):
                signal_name = result.get("output_signal", args["output_signal"])
                state_boundaries = result.get("state_boundaries", [])
                return {
                    "output_signal": signal_name,
                    "driver_type": result.get("driver_type"),
                    "requested_basis": "primary_inputs",
                    "request_satisfied": False,
                    "primary_input_only_expression_available": False,
                    "answer": (
                        f"{signal_name} cannot be expressed solely in terms of current "
                        "primary inputs because it depends on stored DFF-Q state."
                    ),
                    "state_boundaries": state_boundaries,
                    "available_expression_basis": result.get("expression_basis"),
                }
            return result

        if tool_name == "count_cone_gates":
            count = eng.count_cone_gates(args["output_signal"])
            return {"count": count}

        if tool_name == "count_gate_types_in_cone":
            return eng.count_gate_types_in_cone(args["output_signal"])

        if tool_name == "find_all_paths":
            return eng.find_all_paths(
                args["source"],
                args["sink"],
                args.get("inline_limit") or 5,
            )
        if tool_name == "find_zero_length_pi_po_paths":
            return eng.find_zero_length_pi_po_paths()
        if tool_name == "find_register_to_register_paths":
            return eng.find_register_to_register_paths()
        if tool_name == "resolve_name_type":
            return eng.resolve_name_type(args["name"])
        if tool_name == "get_gate_info":
            return eng.get_gate_info(args["gate_name"])
        if tool_name == "find_gates":
            return eng.find_gates(
                args.get("gate_type", ""),
                args.get("input_count"),
                args.get("has_input"),
                min(int(args.get("inline_limit", 10)), 10),
            )
        if tool_name == "find_gates_with_constant_inputs":
            return eng.find_gates_with_constant_inputs(
                args["gate_type"],
                args.get("values", [0, 1]),
                bool(args.get("functional", True)),
                int(args.get("inline_limit", 50)),
            )
        if tool_name == "simplify_gates_with_constant_inputs":
            return eng.simplify_gates_with_constant_inputs(
                args["gate_type"],
                bool(args.get("functional", True)),
                bool(args.get("use_last_report", True)),
            )
        if tool_name == "get_net_fanout":
            return eng.get_fanout_report(args["net_name"])
        if tool_name == "rank_signals_by_fanout":
            return eng.rank_signals_by_fanout(
                args["signal_class"],
                args.get("order", "descending"),
                int(args.get("limit", 1)),
                bool(args.get("include_ties", True)),
                int(args.get("inline_limit", 100)),
            )
        if tool_name == "rank_signals_by_fanin_cone":
            prompt = getattr(self, "_active_user_message", "").lower()
            explicit_size_request = (
                "fanin cone" in prompt
                and any(word in prompt for word in ("largest", "smallest"))
                and not any(word in prompt for word in ("deepest", "shallowest", "depth"))
            )
            return eng.rank_signals_by_fanin_cone(
                "PO" if explicit_size_request and "output" in prompt else args["signal_class"],
                "gate_count" if explicit_size_request else args.get("metric", "gate_count"),
                (
                    "ascending" if "smallest" in prompt else "descending"
                ) if explicit_size_request else args.get("order", "descending"),
                int(args.get("limit", 1)),
                bool(args.get("include_ties", True)),
                int(args.get("inline_limit", 100)),
            )
        if tool_name == "get_net_connections":
            return eng.get_net_connections(
                args["net_name"], int(args.get("inline_limit", 50))
            )
        if tool_name == "get_transitive_fanout_cone":
            return eng.get_reachable_gates_from_net(args["net_name"])
        if tool_name == "get_reachable_gates_from_gate":
            return eng.get_reachable_gates_from_gate(args["gate_name"])
        if tool_name == "list_signals":
            return eng.list_signals()
        if tool_name == "get_gate_output_fanout":
            return eng.get_gate_fanout_report(args["gate_name"])
        if tool_name == "are_same_clock_domain":
            same = eng.are_same_clock_domain(args["dff1"], args["dff2"])
            return {"same_clock_domain": same}

        if tool_name == "list_flip_flops_by_clock":
            return eng.list_flip_flops_by_clock(
                args["clock_signal"],
                int(args.get("inline_limit", 50)),
            )

        if tool_name == "analyze_dff_d_input_structures":
            return eng.analyze_dff_d_input_structures(
                args.get("structure_types"),
                int(args.get("inline_limit", 20)),
            )

        if tool_name == "insert_gate_before":
            new_inst = eng.insert_gate_before(
                args["target_instance"], args["gate_type"], args["extra_input"]
            )
            return {"new_instance": new_inst}

        if tool_name == "rename_gate":
            eng.rename_gate(args["old_name"], args["new_name"])
            return {"old_name": args["old_name"], "new_name": args["new_name"]}

        if tool_name == "rename_wire":
            eng.rename_wire(args["old_name"], args["new_name"])
            return {"old_name": args["old_name"], "new_name": args["new_name"]}

        if tool_name == "replace_gate":
            replacements = args.get("replacements")
            if replacements:
                replaced_instances = []
                new_types = set()
                for replacement in replacements:
                    eng.replace_gate(
                        replacement["instance_name"],
                        replacement["new_gate_type"],
                        replacement.get("extra_input"),
                        replacement.get("new_inputs"),
                    )
                    replaced_instances.append(replacement["instance_name"])
                    new_types.add(replacement["new_gate_type"])
                return {
                    "replaced_count": len(replaced_instances),
                    "replaced": replaced_instances[:10],
                    "new_type": ",".join(sorted(new_types)),
                }

            eng.replace_gate(
                args["instance_name"],
                args["new_gate_type"],
                args.get("extra_input"),
                args.get("new_inputs"),
            )
            return {
                "replaced_count": 1,
                "replaced": args["instance_name"],
                "new_type": args["new_gate_type"],
            }

        if tool_name == "insert_buffers_for_fanout":
            n = eng.insert_buffers_for_fanout(args["net_name"], args["max_fanout"])
            return {"buffers_inserted": n}

        if tool_name == "insert_dedicated_buffers_for_loads":
            n = eng.insert_dedicated_buffers_for_loads(args["net_name"])
            return {"net_name": args["net_name"], "buffers_inserted": n}

        if tool_name == "auto_insert_buffers":
            max_fanout = int(args.get("max_fanout", 4))
            nets = args.get("nets")
            processed = []
            per_net = {}
            total = 0

            if not nets:
                # Work from the complete internal load-pin inventory.  The
                # user-facing list_signals() result is intentionally compact
                # and therefore must not be used for exhaustive processing.
                nl = eng.netlist
                candidates = {
                    input_signal
                    for node in nl.nodes.values()
                    for input_signal in node.inputs
                }
                candidates.update(
                    signal
                    for dff in nl.dffs.values()
                    for signal in (dff.ck, dff.rn, dff.sn, dff.d)
                    if signal
                )
                nets = sorted(candidates - {"1'b0", "1'b1"})

            for net in nets:
                try:
                    fanout = eng.get_fanout(net)
                except Exception:
                    fanout = []
                cnt = len(fanout)
                if cnt > max_fanout:
                    inserted = eng.insert_buffers_for_fanout(net, max_fanout)
                    per_net[net] = {"before": cnt, "buffers_inserted": inserted}
                    total += inserted
                    processed.append(net)

            return {"nets_processed": len(processed), "buffers_inserted": total, "per_net": per_net}

        if tool_name == "balance_depth":
            n = eng.balance_depth(args["source"], args["sinks"])
            return {"buffers_inserted": n}

        if tool_name == "remove_dangling_gates":
            n = eng.remove_dangling_gates()
            return {"gates_removed": n}

        if tool_name == "collapse_inverter_pairs":
            return eng.collapse_inverter_pairs()

        if tool_name == "reduce_critical_path":
            cone_target = self._cone_depth_restriction_target_from_prompt()
            cone_allowed = args.get("allowed_gates") or self._cone_allowed_gates_from_prompt()
            if cone_target and cone_allowed:
                prompt = getattr(self, "_active_user_message", "").lower()
                if self._prompt_uses_cone_depth_cost():
                    return eng.optimize_cone_depth_preserving_gate_set(
                        cone_target,
                        cone_allowed,
                        any(word in prompt for word in ("equivalence", "functionally", "functionality")),
                    )
                return eng.optimize_depth_preserving_cone_gate_set(
                    cone_target,
                    cone_allowed,
                    any(word in prompt for word in ("equivalence", "functionally", "functionality")),
                )
            return eng.reduce_critical_path(args.get("allowed_gates"))

        if tool_name == "optimize_depth_preserving_cone_gate_set":
            if not self._prompt_requests_depth_optimization():
                return eng.remap_cone_with_gates(
                    args["output_signal"],
                    args["allowed_gates"],
                )
            if self._prompt_uses_cone_depth_cost():
                return eng.optimize_cone_depth_preserving_gate_set(
                    args["output_signal"],
                    args["allowed_gates"],
                    bool(args.get("verify_equivalence", False)),
                )
            return eng.optimize_depth_preserving_cone_gate_set(
                args["output_signal"],
                args["allowed_gates"],
                bool(args.get("verify_equivalence", False)),
            )

        if tool_name == "optimize_cone_depth_preserving_gate_set":
            if not self._prompt_requests_depth_optimization():
                return eng.remap_cone_with_gates(
                    args["output_signal"],
                    args["allowed_gates"],
                )
            return eng.optimize_cone_depth_preserving_gate_set(
                args["output_signal"],
                args["allowed_gates"],
                bool(args.get("verify_equivalence", False)),
            )

        if tool_name == "replace_gate_type_in_cone":
            # Guard against a model argument that contradicts an explicitly
            # named source type in the user's request (for example, passing
            # XOR for "replace all XNOR gates").
            source_type = (
                self._specific_gate_type_replacement_from_prompt()
                or args["source_type"]
            )
            return eng.replace_gate_type_in_cone(
                source_type,
                args["target_types"],
                args.get("output_signal"),
            )

        if tool_name == "remap_cone_with_gates":
            source_type = self._specific_gate_type_replacement_from_prompt()
            if source_type:
                return eng.replace_gate_type_in_cone(
                    source_type,
                    args["allowed_gates"],
                    args["output_signal"],
                )
            return eng.remap_cone_with_gates(
                args["output_signal"],
                args["allowed_gates"],
            )

        if tool_name == "remap_design_with_gates":
            return eng.remap_design_with_gates(args["allowed_gates"])

        if tool_name == "fraig_merge_equivalent_gates":
            return eng.fraig_merge_equivalent_gates()

        if tool_name == "optimize_cone_depth":
            success = eng.optimize_cone_depth(
                args["output_signal"], args["max_depth"]
            )
            return {"success": success}

        if tool_name == "replace_pattern":
            n = eng.replace_pattern(args["pattern"], args["replacement"])
            return {"replacements": n}

        if tool_name == "find_instances_by_name_pattern":
            instances = eng.find_instances_by_name_pattern(
                args.get("gate_type", ""), args["name_pattern"]
            )
            return {"instances": instances, "count": len(instances)}

        if tool_name == "check_signal_equivalence":
            equiv = eng.check_signal_equivalence(args["sig1"], args["sig2"])
            sig1 = args["sig1"]
            sig2 = args["sig2"]
            if equiv:
                statement = (
                    f"Signals {sig1} and {sig2} produce identical logic values for all "
                    "input assignments and are functionally equivalent."
                )
            else:
                statement = (
                    f"Signals {sig1} and {sig2} do not produce identical logic values for all "
                    "input assignments and are not functionally equivalent."
                )
            return {
                "signal_1": sig1,
                "signal_2": sig2,
                "equivalent": equiv,
                "identical_for_all_inputs": equiv,
                "verdict": "EQUIVALENT" if equiv else "NOT_EQUIVALENT",
                "statement": statement,
                "response_instruction": "Report the statement without reversing its conclusion.",
            }
        if tool_name == "check_function_symmetry":
            return eng.check_function_symmetry(
                args["output_signal"], args["input_a"], args["input_b"]
            )

        if tool_name == "find_binary_gate_equivalent_pair":
            return eng.find_binary_gate_equivalent_pair(
                args["target_signal"],
                args["gate_type"],
                args.get("candidate_scope", "internal"),
                int(args.get("max_signature_pairs", 50_000_000)),
                int(args.get("max_formal_checks", 3)),
            )

        if tool_name == "check_signal_constant":
            return eng.check_signal_constant(args["signal_name"], args["value"])

        if tool_name == "check_design_equivalence":
            return eng.check_design_equivalence()

        raise ValueError(f"Unknown tool: {tool_name!r}")
