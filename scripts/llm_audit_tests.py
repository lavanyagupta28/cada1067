#!/usr/bin/env python3
"""
LLM-Assisted Quality Audit and Engine Fault Diagnosis Framework.

Evaluates testcase prompt adherence, diagnoses tool selection, validates logic
invariants, checks for hallucinations, and isolates underlying engine defects.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Shared testing utilities
from test_utils import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TESTCASE_DIR,
    discover_testcases,
    ensure_output_directories,
    load_config,
)


@dataclass
class TurnTrace:
    turn_num: int
    user_prompt: str
    tools_called: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    agent_response: str = ""
    elapsed_sec: float = 0.0
    detected_invariants: List[str] = field(default_factory=list)


@dataclass
class TurnEvaluation:
    turn: int
    user_prompt: str
    tool_called: str
    tool_status: str  # "CORRECT", "WRONG_TOOL", "MISSING_TOOL", "UNNECESSARY_TOOL", "N/A"
    accuracy: str     # "ACCURATE", "HALLUCINATED", "INCOMPLETE", "FORMAT_ERROR", "ENGINE_DEFECT"
    comments: str


@dataclass
class TestcaseAuditResult:
    testcase: str
    overall_status: str       # "PASS", "FAIL", "WARNING"
    quality_score: int        # 0 - 10
    issue_category: str       # "NONE", "PROMPT_MISSED", "WRONG_TOOL", "HALLUCINATION", "FORMAT_ERROR", "ENGINE_DEFECT", "TIMEOUT"
    faulty_function: str      # Name of buggy tool or engine function, or "None"
    summary_diagnosis: str
    actionable_fix: str
    invariants_violated: List[str] = field(default_factory=list)
    turn_evaluations: List[TurnEvaluation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trace Extractor & Invariant Checker
# ---------------------------------------------------------------------------

class TraceExtractor:
    """Extracts compact execution traces from prompt.txt, output logs, and tool logs."""

    def __init__(self, testcase_dir: Path, outputs_dir: Path, tool_logs_dir: Path):
        self.testcase_dir = testcase_dir
        self.outputs_dir = outputs_dir
        self.tool_logs_dir = tool_logs_dir

    def get_prompt_lines(self, testcase: str) -> List[str]:
        prompt_file = self.testcase_dir / testcase / "prompt.txt"
        if not prompt_file.is_file():
            return []
        lines: List[str] = []
        for line in prompt_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                lines.append(line)
        return lines

    def get_output_responses(self, testcase: str) -> Dict[int, str]:
        log_file = self.outputs_dir / f"{testcase}.log"
        if not log_file.is_file():
            return {}
        text = log_file.read_text(encoding="utf-8", errors="replace")
        responses: Dict[int, str] = {}
        pattern = re.compile(r"#RESPONSE\s+(\d+)\s*\r?\n(.*?)\r?\n#END\s+\1", re.DOTALL)
        for match in pattern.finditer(text):
            turn_id = int(match.group(1))
            resp_body = match.group(2).strip()
            responses[turn_id] = resp_body
        return responses

    def get_tool_logs_for_test(self, testcase: str) -> List[Path]:
        if not self.tool_logs_dir.is_dir():
            return []
        pattern = f"{testcase}_request_*.log"
        return sorted(self.tool_logs_dir.glob(pattern))

    def parse_single_tool_log(self, log_path: Path) -> Dict[str, Any]:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        data: Dict[str, Any] = {
            "req_num": 0,
            "user_input": "",
            "tool_calls": [],
            "tool_results": [],
            "final_response": "",
            "elapsed": 0.0,
        }

        # Request number
        req_match = re.search(r"Request #(\d+)", text)
        if req_match:
            data["req_num"] = int(req_match.group(1))

        # User input prompt
        input_match = re.search(r"┌─ USER INPUT ─+\s*\r?\n(.*?)\r?\n└─", text, re.DOTALL)
        if input_match:
            data["user_input"] = input_match.group(1).strip()

        # Tool calls from assistant block
        tc_blocks = re.finditer(
            r"tool_call:\s*([A-Za-z0-9_]+)\s*\r?\n\s*arguments:\s*\r?\n\s*(\{.*?\})",
            text,
            re.DOTALL,
        )
        for tc in tc_blocks:
            name = tc.group(1).strip()
            args_raw = tc.group(2).strip()
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {"raw": args_raw}
            data["tool_calls"].append({"name": name, "args": args})

        # Tool calls from arrow summary
        arrow_matches = re.finditer(
            r"(?:→|->)\s*([A-Za-z0-9_]+)\s*(?:\r?\n)\s*args:\s*(\{.*?\})",
            text,
            re.DOTALL,
        )
        for tc in arrow_matches:
            name = tc.group(1).strip()
            if not any(c["name"] == name for c in data["tool_calls"]):
                args_raw = tc.group(2).strip()
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"raw": args_raw}
                data["tool_calls"].append({"name": name, "args": args})

        # Tool results
        res_matches = re.finditer(
            r"── TOOL RESULT:\s*([A-Za-z0-9_]+)\s*──\s*\r?\n(.*?)\r?\n\s*───",
            text,
            re.DOTALL,
        )
        for tr in res_matches:
            tname = tr.group(1).strip()
            body_raw = tr.group(2).strip()
            try:
                body = json.loads(body_raw)
            except Exception:
                body = {"raw": body_raw[:300]}
            data["tool_results"].append({"name": tname, "data": body})

        # Fallback: populate tool_calls from tool_results if empty
        if not data["tool_calls"] and data["tool_results"]:
            for tr in data["tool_results"]:
                data["tool_calls"].append({"name": tr["name"], "args": {}})

        # Final response
        resp_match = re.search(r"┌─ FINAL RESPONSE ─+\s*\r?\n(.*?)\r?\n└─", text, re.DOTALL)
        if resp_match:
            data["final_response"] = resp_match.group(1).strip()

        # Total elapsed time
        time_match = re.search(r"total time:\s*([0-9.]+)s", text)
        if time_match:
            data["elapsed"] = float(time_match.group(1))

        return data

    def extract_testcase_trace(self, testcase: str) -> List[TurnTrace]:
        prompt_lines = self.get_prompt_lines(testcase)
        responses = self.get_output_responses(testcase)
        tool_log_files = self.get_tool_logs_for_test(testcase)

        tool_data_by_req: Dict[int, Dict[str, Any]] = {}
        for f in tool_log_files:
            parsed = self.parse_single_tool_log(f)
            req_num = parsed.get("req_num", 0)
            if req_num > 0:
                tool_data_by_req[req_num] = parsed

        traces: List[TurnTrace] = []
        for i, prompt in enumerate(prompt_lines, start=1):
            t_data = tool_data_by_req.get(i, {})
            resp = responses.get(i, t_data.get("final_response", ""))

            trace = TurnTrace(
                turn_num=i,
                user_prompt=prompt,
                tools_called=t_data.get("tool_calls", []),
                tool_results=t_data.get("tool_results", []),
                agent_response=resp,
                elapsed_sec=t_data.get("elapsed", 0.0),
            )
            trace.detected_invariants = self._check_turn_invariants(trace)
            traces.append(trace)

        return traces

    def _check_turn_invariants(self, turn: TurnTrace) -> List[str]:
        invariants: List[str] = []

        # Check 1: Tool exceptions / error returns
        for res in turn.tool_results:
            r_data = res.get("data", {})
            if isinstance(r_data, dict):
                if "error" in r_data or "exception" in r_data or r_data.get("success") is False:
                    err = r_data.get("error") or r_data.get("exception") or "Tool returned success=False"
                    invariants.append(f"TOOL_EXCEPTION: Tool '{res.get('name')}' reported error: {err}")

        # Check 2: Logic depth path vs length
        for res in turn.tool_results:
            if res.get("name") in ("get_max_depth", "get_longest_path"):
                inner = res.get("data", {}).get("result", {})
                depth = inner.get("depth")
                path = inner.get("path")
                if isinstance(depth, int) and isinstance(path, list):
                    if len(path) == 0 and depth > 0:
                        invariants.append(f"PATH_INVARIANT_VIOLATION: depth is {depth} but path is empty")

        # Check 3: Missing response tags
        if not turn.agent_response:
            invariants.append("MISSING_RESPONSE: No response emitted for this prompt turn")

        return invariants


# ---------------------------------------------------------------------------
# LLM Auditor Prompt Rubric
# ---------------------------------------------------------------------------

SYSTEM_AUDITOR_PROMPT = """You are an expert Electronic Design Automation (EDA) and LLM Systems Auditor.
Your task is to audit an EDA agent's turn-by-turn execution log on a gate-level Verilog netlist testcase.

You will receive:
1. The original testcase prompt turns.
2. The tools called, arguments supplied, and tool return summaries.
3. The final natural-language responses emitted to the user.
4. Any programmatically detected invariant violations (e.g. tool exceptions, empty anomalies).

EVALUATION RUBRIC:
1. PROMPT ADHERENCE: Did the agent answer ALL sub-questions asked in each turn?
2. TOOL SELECTION: Did the agent pick the optimal EDA tool for the job?
   - If user asked for articulation points, did it call find_articulation_points?
   - If user asked for highest fanout net, did it call rank_signals_by_fanout or get_net_fanout on the target?
   - If user asked to collapse inverters, did it call collapse_inverter_pairs?
3. FAITHFULNESS (HALLUCINATION CHECK): Does the final response match the tool's actual return data?
   - If the tool reported 35 gates, did the response say 35, or hallucinate a different count?
   - If the tool gave a file path for long lists, did the agent report that exact path?
4. ENGINE DEFECTS: If the tool itself crashed, returned an error dictionary, or failed, identify the underlying function.

SPECIAL NOTE ON TURN 1:
Turn 1 ("This is the beginning of a new testcase. The case name is testNN.") is framework initialization handled internally by setting the session testcase name. No external tool call is required or expected for Turn 1; mark tool_status as "N/A" and accuracy as "ACCURATE" if acknowledged.

SPECIAL NOTE ON CONTEXT MEMORY:
If the user request asks for a count, status, or metric that was already explicitly computed, reported, or known from a previous turn in the conversation history (e.g. "How many gates were added by the buffer insertion just performed?", "How many NOR gates are now in the design?"), answering directly from context memory WITHOUT invoking a tool is valid, expected, and optimal. Mark tool_status as "N/A" and accuracy as "ACCURATE". Do NOT penalize the agent as MISSING_TOOL or HALLUCINATED when it accurately retrieves a known fact from memory.

OUTPUT FORMAT:
You MUST respond with a single valid JSON object adhering to this structure:
{
  "overall_status": "PASS" | "FAIL" | "WARNING",
  "quality_score": <integer 0 to 10>,
  "issue_category": "NONE" | "PROMPT_MISSED" | "WRONG_TOOL" | "HALLUCINATION" | "FORMAT_ERROR" | "ENGINE_DEFECT" | "TIMEOUT",
  "faulty_function": "<name of faulty tool/function or 'None'>",
  "summary_diagnosis": "<1-3 concise sentences summarizing the main issue or confirming success>",
  "actionable_fix": "<concrete recommendation on what prompt instruction, tool description, or engine code to fix>",
  "turn_evaluations": [
    {
      "turn": <int>,
      "tool_status": "CORRECT" | "WRONG_TOOL" | "MISSING_TOOL" | "UNNECESSARY_TOOL" | "N/A",
      "accuracy": "ACCURATE" | "HALLUCINATED" | "INCOMPLETE" | "FORMAT_ERROR" | "ENGINE_DEFECT",
      "comments": "<brief turn critique>"
    }
  ]
}
"""


class LLMAuditor:
    """Invokes OpenAI LLM to perform semantic evaluation and diagnosis."""

    def __init__(self, config: Dict[str, Any]):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            api_key = str(config.get("openai", {}).get("api_key", "")).strip()
        if not api_key or api_key.startswith("YOUR_"):
            raise RuntimeError("Valid OpenAI API key not found in config.yaml or OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)
        self.model = str(config.get("openai", {}).get("model", "gpt-4o-mini"))

    def audit_testcase(self, testcase: str, traces: List[TurnTrace]) -> TestcaseAuditResult:
        user_content_lines = [f"TESTCASE: {testcase}\n"]

        for t in traces:
            user_content_lines.append(f"--- TURN {t.turn_num} ---")
            user_content_lines.append(f"User Prompt: {t.user_prompt}")
            if t.tools_called:
                calls_str = ", ".join(f"{c['name']}({json.dumps(c.get('args', {}))})" for c in t.tools_called)
                user_content_lines.append(f"Tool(s) Called: {calls_str}")
            else:
                user_content_lines.append("Tool(s) Called: None")

            if t.tool_results:
                res_summaries = []
                for r in t.tool_results:
                    r_str = json.dumps(r.get("data", {}))
                    if len(r_str) > 350:
                        r_str = r_str[:350] + "... [TRUNCATED]"
                    res_summaries.append(f"{r.get('name')}: {r_str}")
                user_content_lines.append(f"Tool Results: {' | '.join(res_summaries)}")

            if t.detected_invariants:
                user_content_lines.append(f"Detected Anomalies: {'; '.join(t.detected_invariants)}")

            user_content_lines.append(f"Agent Final Response: {t.agent_response}\n")

        prompt_payload = "\n".join(user_content_lines)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_AUDITOR_PROMPT},
                    {"role": "user", "content": prompt_payload},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw_json = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_json)

            all_invariants = [inv for t in traces for inv in t.detected_invariants]

            evals: List[TurnEvaluation] = []
            for item in parsed.get("turn_evaluations", []):
                turn_idx = item.get("turn", 0)
                prompt_text = traces[turn_idx - 1].user_prompt if 0 < turn_idx <= len(traces) else ""
                tool_text = ", ".join(c["name"] for c in traces[turn_idx - 1].tools_called) if 0 < turn_idx <= len(traces) else "None"
                evals.append(
                    TurnEvaluation(
                        turn=turn_idx,
                        user_prompt=prompt_text,
                        tool_called=tool_text,
                        tool_status=item.get("tool_status", "CORRECT"),
                        accuracy=item.get("accuracy", "ACCURATE"),
                        comments=item.get("comments", ""),
                    )
                )

            return TestcaseAuditResult(
                testcase=testcase,
                overall_status=parsed.get("overall_status", "PASS"),
                quality_score=int(parsed.get("quality_score", 10)),
                issue_category=parsed.get("issue_category", "NONE"),
                faulty_function=parsed.get("faulty_function", "None"),
                summary_diagnosis=parsed.get("summary_diagnosis", ""),
                actionable_fix=parsed.get("actionable_fix", ""),
                invariants_violated=all_invariants,
                turn_evaluations=evals,
            )

        except Exception as exc:
            return TestcaseAuditResult(
                testcase=testcase,
                overall_status="FAIL",
                quality_score=0,
                issue_category="AUDITOR_ERROR",
                faulty_function="None",
                summary_diagnosis=f"Auditor failed to evaluate testcase: {exc}",
                actionable_fix="Check OpenAI API key and connectivity.",
                invariants_violated=[str(exc)],
                turn_evaluations=[],
            )


# ---------------------------------------------------------------------------
# Report Management
# ---------------------------------------------------------------------------

def save_reports(
    results: List[TestcaseAuditResult],
    json_path: Path,
    csv_path: Path,
) -> None:
    """Save deep JSON reports and summary CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. JSON Report
    json_data = [asdict(r) for r in results]
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(json_data, fh, indent=2)

    # 2. CSV Summary
    fieldnames = [
        "Testcase",
        "Status",
        "Score",
        "Issue_Category",
        "Faulty_Function",
        "Summary_Diagnosis",
        "Actionable_Fix",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Testcase": r.testcase,
                "Status": r.overall_status,
                "Score": r.quality_score,
                "Issue_Category": r.issue_category,
                "Faulty_Function": r.faulty_function,
                "Summary_Diagnosis": r.summary_diagnosis.replace("\n", " "),
                "Actionable_Fix": r.actionable_fix.replace("\n", " "),
            })


# ---------------------------------------------------------------------------
# CLI & Runner
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-Assisted Quality Audit and Engine Fault Diagnosis Framework."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tests",
        nargs="+",
        help="Specific testcases to audit (e.g. --tests test01 test05 test81)",
    )
    group.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Numeric range of tests to audit (e.g. --range 1 91)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Audit all available testcases",
    )

    parser.add_argument(
        "--testcase-dir",
        type=Path,
        default=DEFAULT_TESTCASE_DIR,
        help=f"Directory containing testcases (default: {DEFAULT_TESTCASE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing outputs and logs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pure Python invariant and anomaly checks locally with zero API cost",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    config = load_config(args.config)
    outputs_dir, tool_logs_dir, summary_dir = ensure_output_directories(args.output_dir)

    range_tuple = tuple(args.range) if args.range else None
    targets = discover_testcases(
        testcase_dir=args.testcase_dir,
        tests=args.tests,
        range_spec=range_tuple,
    )

    if not targets:
        print("[ERROR] No matching testcases found.", file=sys.stderr)
        return 1

    extractor = TraceExtractor(args.testcase_dir, outputs_dir, tool_logs_dir)
    auditor = None if args.dry_run else LLMAuditor(config)

    print("=" * 70)
    print("           ICCAD CADA1067 LLM QUALITY AUDITOR")
    print("=" * 70)
    print(f"Mode:         {'ZERO-COST INVARIANT CHECK (Dry Run)' if args.dry_run else 'FULL SEMANTIC LLM AUDIT'}")
    print(f"Target count: {len(targets)} testcase(s)")
    print(f"Outputs dir:  {outputs_dir}")
    print("=" * 70 + "\n")

    results: List[TestcaseAuditResult] = []

    for idx, test_dir in enumerate(targets, start=1):
        tname = test_dir.name
        print(f"[{idx}/{len(targets)}] Auditing {tname:<10} ...", end=" ", flush=True)
        traces = extractor.extract_testcase_trace(tname)

        if not traces:
            print("[EMPTY / MISSING PROMPT]")
            results.append(
                TestcaseAuditResult(
                    testcase=tname,
                    overall_status="FAIL",
                    quality_score=0,
                    issue_category="PROMPT_MISSED",
                    faulty_function="None",
                    summary_diagnosis="No prompt.txt or output log found for testcase.",
                    actionable_fix="Ensure testcase directory exists and was executed.",
                )
            )
            continue

        if args.dry_run:
            invariants = [inv for t in traces for inv in t.detected_invariants]
            status = "FAIL" if invariants else "PASS"
            score = 5 if invariants else 10
            cat = "ENGINE_DEFECT" if invariants else "NONE"
            print(f"[DRY RUN - Invariants: {len(invariants)}]")
            results.append(
                TestcaseAuditResult(
                    testcase=tname,
                    overall_status=status,
                    quality_score=score,
                    issue_category=cat,
                    faulty_function="None",
                    summary_diagnosis=f"Dry run complete. Invariants flagged: {len(invariants)}",
                    actionable_fix="Run without --dry-run for semantic LLM diagnosis.",
                    invariants_violated=invariants,
                )
            )
        else:
            audit_res = auditor.audit_testcase(tname, traces)  # type: ignore[union-attr]
            print(f"[{audit_res.overall_status:<4} - Score: {audit_res.quality_score:>2}/10]")
            results.append(audit_res)

    json_path = summary_dir / "llm_audit_report.json"
    csv_path = summary_dir / "audit_summary.csv"

    save_reports(results, json_path, csv_path)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Audit completed.")
    print(f"  • Deep JSON report: {json_path}")
    print(f"  • Summary CSV:     {csv_path}")
    print("=" * 70)

    print(f"{'Testcase':<12} | {'Status':<8} | {'Score':<5} | {'Issue Category':<16} | {'Faulty Function'}")
    print("-" * 70)
    for r in results:
        print(f"{r.testcase:<12} | {r.overall_status:<8} | {r.quality_score:<5} | {r.issue_category:<16} | {r.faulty_function}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
