#!/usr/bin/env python3
"""
Batch test execution framework for ICCAD Netlist Transformation.

Executes testcases against the main EDA framework, sets up EDA binary paths
(YOSYS_BIN, ABC_BIN), enforces execution timeouts, routes artifacts, and records
summary tables.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Shared testing utilities
from test_utils import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TESTCASE_DIR,
    PROJECT_ROOT,
    discover_testcases,
    ensure_output_directories,
    load_config,
    resolve_python_executable,
    setup_eda_environment,
)


@dataclass
class TestResult:
    name: str
    status: str          # "PASSED", "FAILED", "TIMEOUT", "SKIPPED"
    elapsed_sec: float
    has_log: bool
    has_verilog: bool
    return_code: Optional[int]
    error_message: str = ""


# ---------------------------------------------------------------------------
# Single Testcase Execution
# ---------------------------------------------------------------------------

def run_single_testcase(
    test_dir: Path,
    config_path: Path,
    outputs_dir: Path,
    tool_logs_dir: Path,
    timeout_sec: int,
    python_exe: Path,
    base_env: Dict[str, str],
    verbose: bool = False,
) -> TestResult:
    """Execute main.py for a single testcase and route all output artifacts."""
    test_name = test_dir.name
    prompt_file = test_dir / "prompt.txt"
    out_verilog_name = f"{test_name}_out.v"

    expected_log = outputs_dir / f"{test_name}.log"
    expected_verilog = outputs_dir / out_verilog_name

    if not prompt_file.is_file():
        return TestResult(
            name=test_name,
            status="SKIPPED",
            elapsed_sec=0.0,
            has_log=False,
            has_verilog=False,
            return_code=None,
            error_message="Missing prompt.txt in test directory",
        )

    # Clean old artifacts for this testcase to avoid stale detections
    if expected_log.exists():
        try:
            expected_log.unlink()
        except OSError:
            pass
    if expected_verilog.exists():
        try:
            expected_verilog.unlink()
        except OSError:
            pass

    cmd = [
        str(python_exe),
        "main.py",
        "--config", str(config_path.resolve()),
    ]

    # Child execution environment inherits configured EDA paths
    env = base_env.copy()

    start_time = time.perf_counter()
    status = "FAILED"
    ret_code: Optional[int] = None
    err_msg = ""

    try:
        with prompt_file.open("r", encoding="utf-8", errors="replace") as stdin_fh:
            proc = subprocess.Popen(
                cmd,
                stdin=stdin_fh,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=PROJECT_ROOT,
                env=env,
            )

            try:
                stdout_data, stderr_data = proc.communicate(timeout=timeout_sec)
                ret_code = proc.returncode
                elapsed = time.perf_counter() - start_time

                # Save captured stdout to outputs/testNN.log
                if stdout_data:
                    expected_log.write_text(stdout_data, encoding="utf-8", errors="replace")

                # Check if generated verilog was emitted to root or outputs/
                root_verilog = PROJECT_ROOT / out_verilog_name
                if root_verilog.is_file() and not expected_verilog.is_file():
                    shutil.move(str(root_verilog), str(expected_verilog))

                if ret_code == 0:
                    status = "PASSED"
                else:
                    status = "FAILED"
                    err_msg = (stderr_data or stdout_data or f"Exit code {ret_code}").strip().splitlines()[-1] if (stderr_data or stdout_data) else f"Exit code {ret_code}"

            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_data, stderr_data = proc.communicate()
                elapsed = time.perf_counter() - start_time
                status = "TIMEOUT"
                err_msg = f"Exceeded timeout limit of {timeout_sec}s"
                if stdout_data and not expected_log.exists():
                    expected_log.write_text(stdout_data, encoding="utf-8", errors="replace")

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        status = "FAILED"
        err_msg = str(exc)

    # Route developer logs if written to root logs/
    root_logs_dir = PROJECT_ROOT / "logs"
    if root_logs_dir.is_dir():
        for f in root_logs_dir.glob("*.log"):
            target_name = f"{test_name}_{f.name}" if not f.name.startswith(test_name) else f.name
            target_path = tool_logs_dir / target_name
            try:
                shutil.copy2(str(f), str(target_path))
            except OSError:
                pass

    has_log = expected_log.is_file() and expected_log.stat().st_size > 0
    has_verilog = expected_verilog.is_file() and expected_verilog.stat().st_size > 0

    if verbose and err_msg:
        print(f"\n  [ERROR {test_name}] {err_msg}", file=sys.stderr)

    return TestResult(
        name=test_name,
        status=status,
        elapsed_sec=elapsed,
        has_log=has_log,
        has_verilog=has_verilog,
        return_code=ret_code,
        error_message=err_msg,
    )


# ---------------------------------------------------------------------------
# Summary CSV Management
# ---------------------------------------------------------------------------

def load_existing_summary(csv_path: Path) -> Dict[str, TestResult]:
    """Load existing test entries from summary.csv to allow selective updates."""
    results: Dict[str, TestResult] = {}
    if not csv_path.is_file():
        return results

    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = row.get("Testcase", "").strip()
                if not name:
                    continue
                try:
                    time_sec = float(row.get("Time_sec", 0.0))
                except ValueError:
                    time_sec = 0.0
                ret_code_str = row.get("Return_Code", "").strip()
                ret_code = int(ret_code_str) if ret_code_str.isdigit() else None
                results[name] = TestResult(
                    name=name,
                    status=row.get("Status", "UNKNOWN"),
                    elapsed_sec=time_sec,
                    has_log=row.get("Has_Log", "").upper() == "YES",
                    has_verilog=row.get("Has_Verilog", "").upper() == "YES",
                    return_code=ret_code,
                    error_message=row.get("Details", ""),
                )
    except Exception as exc:
        print(f"[WARN] Could not parse existing summary CSV {csv_path}: {exc}", file=sys.stderr)

    return results


def save_summary_csv(results: List[TestResult], csv_path: Path) -> None:
    """Save full test execution table to CSV."""
    fieldnames = [
        "Testcase",
        "Status",
        "Time_sec",
        "Has_Log",
        "Has_Verilog",
        "Return_Code",
        "Details",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Testcase": r.name,
                "Status": r.status,
                "Time_sec": f"{r.elapsed_sec:.2f}",
                "Has_Log": "YES" if r.has_log else "NO",
                "Has_Verilog": "YES" if r.has_verilog else "NO",
                "Return_Code": r.return_code if r.return_code is not None else "",
                "Details": r.error_message.replace("\n", " "),
            })


# ---------------------------------------------------------------------------
# CLI & Execution Orchestrator
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production batch test execution runner for ICCAD cada1067 framework."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tests",
        nargs="+",
        help="List of specific testcases to run (e.g. --tests test01 test05 test81)",
    )
    group.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Numeric range of testcases to run (e.g. --range 1 91)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all available testcases discovered in the testcase directory",
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
        help=f"Directory to save outputs and summary (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Execution timeout limit in seconds per testcase (default: 300)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Explicit path to Python interpreter with yaml and openai installed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose execution error logs",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    # 1. Environment & Paths Setup
    config = load_config(args.config)
    env = setup_eda_environment(config)
    python_exe = resolve_python_executable(args.python)
    outputs_dir, tool_logs_dir, summary_dir = ensure_output_directories(args.output_dir)
    summary_csv = summary_dir / "summary.csv"

    # 2. Discover testcases
    range_tuple = tuple(args.range) if args.range else None
    targets = discover_testcases(
        testcase_dir=args.testcase_dir,
        tests=args.tests,
        range_spec=range_tuple,
    )

    if not targets:
        print("[ERROR] No matching testcases found.", file=sys.stderr)
        return 1

    print("=" * 70)
    print("           ICCAD CADA1067 BATCH TEST RUNNER")
    print("=" * 70)
    print(f"Python:       {python_exe}")
    print(f"YOSYS_BIN:    {env.get('YOSYS_BIN', 'Not configured')}")
    print(f"ABC_BIN:      {env.get('ABC_BIN', 'Not configured')}")
    print(f"Target count: {len(targets)} testcase(s)")
    print(f"Timeout:      {args.timeout}s per testcase")
    print(f"Outputs dir:  {outputs_dir}")
    print("=" * 70 + "\n")

    # 3. Load previous summary to preserve untouched test records
    cumulative_results = load_existing_summary(summary_csv)
    current_run_results: List[TestResult] = []

    try:
        for idx, test_dir in enumerate(targets, start=1):
            tname = test_dir.name
            print(f"[{idx}/{len(targets)}] Running {tname:<10} ...", end=" ", flush=True)

            res = run_single_testcase(
                test_dir=test_dir,
                config_path=args.config,
                outputs_dir=outputs_dir,
                tool_logs_dir=tool_logs_dir,
                timeout_sec=args.timeout,
                python_exe=python_exe,
                base_env=env,
                verbose=args.verbose,
            )

            current_run_results.append(res)
            cumulative_results[res.name] = res

            color_tag = f"[{res.status}]"
            print(f"{color_tag:<10} in {res.elapsed_sec:.2f}s")

    except KeyboardInterrupt:
        print("\n\n[ABORT] KeyboardInterrupt detected. Saving partial results...")

    # 4. Save summary CSV
    final_list = list(cumulative_results.values())
    final_list.sort(key=lambda r: discover_testcases(testcase_dir=args.testcase_dir, tests=[r.name]))
    save_summary_csv(final_list, summary_csv)

    # 5. Print Execution Table
    passed_count = sum(1 for r in current_run_results if r.status == "PASSED")
    timeout_count = sum(1 for r in current_run_results if r.status == "TIMEOUT")
    failed_count = sum(1 for r in current_run_results if r.status == "FAILED")
    skipped_count = sum(1 for r in current_run_results if r.status == "SKIPPED")
    total_time = sum(r.elapsed_sec for r in current_run_results)

    print("\n" + "=" * 70)
    print("                      EXECUTION SUMMARY")
    print("=" * 70)
    print(f"{'Testcase':<12} | {'Status':<10} | {'Time (s)':<9} | {'Log':<5} | {'Verilog':<8} | Details")
    print("-" * 70)
    for r in current_run_results:
        log_str = "YES" if r.has_log else "NO"
        v_str = "YES" if r.has_verilog else "NO"
        print(f"{r.name:<12} | {r.status:<10} | {r.elapsed_sec:<9.2f} | {log_str:<5} | {v_str:<8} | {r.error_message}")
    print("-" * 70)
    print(f"Total: {len(current_run_results)} | Passed: {passed_count} | Timeout: {timeout_count} | Failed: {failed_count} | Skipped: {skipped_count} | Total Time: {total_time:.2f}s")
    print("=" * 70)
    print(f"Summary saved to: {summary_csv}")

    return 0 if (failed_count == 0 and timeout_count == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
