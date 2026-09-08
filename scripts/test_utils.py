#!/usr/bin/env python3
"""
Shared utilities for test execution, EDA binary resolution, and LLM auditing.

Consolidates environment configuration, binary discovery (YOSYS_BIN, ABC_BIN),
Python interpreter verification, testcase discovery, and config loading.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Standard project directory layout
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTCASE_DIR = PROJECT_ROOT / "testcase"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# Known fallback search paths for OSS CAD Suite binaries on Windows
WINDOWS_FALLBACK_YOSYS = [
    Path(r"C:\Users\lavan\Desktop\Downloads\oss-cad-suite-windows-x64-20260827\oss-cad-suite\bin\yosys.exe"),
    Path(r"C:\oss-cad-suite\bin\yosys.exe"),
]


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load and parse config.yaml safely, returning an empty dict on failure."""
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"[WARN] Failed to read configuration from {config_path}: {exc}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Python Executable Resolution
# ---------------------------------------------------------------------------

def _verify_python_interpreter(py_path: Path) -> bool:
    """Check whether a python interpreter can import yaml and openai."""
    try:
        res = subprocess.run(
            [str(py_path), "-c", "import yaml, openai; print('OK')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.returncode == 0 and "OK" in res.stdout
    except Exception:
        return False


def resolve_python_executable(explicit_path: Optional[str] = None) -> Path:
    """Find a Python binary with yaml and openai installed."""
    if explicit_path:
        explicit = Path(explicit_path)
        if explicit.is_file() and _verify_python_interpreter(explicit):
            return explicit
        raise RuntimeError(f"Specified Python interpreter {explicit_path!r} cannot import yaml and openai.")

    current = Path(sys.executable)
    if _verify_python_interpreter(current):
        return current

    # Common Windows install locations
    candidates = [
        Path(r"C:\Users\lavan\AppData\Local\Programs\Python\Python310\python.exe"),
        Path(r"C:\Users\lavan\AppData\Local\Programs\Python\Python313\python.exe"),
        Path(r"C:\Users\lavan\anaconda3\python.exe"),
    ]
    for c in candidates:
        if c.is_file() and _verify_python_interpreter(c):
            return c

    return current


# ---------------------------------------------------------------------------
# EDA Binary & Environment Setup (YOSYS_BIN, ABC_BIN, PATH)
# ---------------------------------------------------------------------------

def resolve_yosys_binary(config: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Resolve the executable Yosys binary from env, config, PATH, or fallback."""
    # 1. Environment variable
    env_bin = os.environ.get("YOSYS_BIN", "").strip()
    if env_bin:
        p = Path(env_bin)
        if p.is_file():
            return p
        resolved = shutil.which(env_bin)
        if resolved:
            return Path(resolved)

    # 2. Config YAML
    if config:
        cfg_bin = str(config.get("yosys_bin") or "").strip()
        if cfg_bin:
            p = Path(cfg_bin)
            if p.is_file():
                return p

    # 3. System PATH
    resolved = shutil.which("yosys")
    if resolved:
        return Path(resolved)

    # 4. Windows Fallback search
    if sys.platform == "win32":
        for cand in WINDOWS_FALLBACK_YOSYS:
            if cand.is_file():
                return cand

    return None


def resolve_abc_binary(config: Optional[Dict[str, Any]] = None, yosys_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve standalone ABC binary from env, config, PATH, or companion directory."""
    # 1. Environment variable
    env_bin = os.environ.get("ABC_BIN", "").strip()
    if env_bin:
        p = Path(env_bin)
        if p.is_file():
            return p
        resolved = shutil.which(env_bin)
        if resolved:
            return Path(resolved)

    # 2. Config YAML
    if config:
        cfg_bin = str(config.get("abc_bin") or "").strip()
        if cfg_bin:
            p = Path(cfg_bin)
            if p.is_file():
                return p

    # 3. Companion binary next to yosys
    if yosys_path and yosys_path.is_file():
        for name in ("yosys-abc.exe", "yosys-abc", "abc.exe", "abc"):
            candidate = yosys_path.parent / name
            if candidate.is_file():
                return candidate

    # 4. System PATH
    for name in ("yosys-abc", "abc"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)

    return None


def setup_eda_environment(
    config: Optional[Dict[str, Any]] = None,
    base_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Configure and export YOSYS_BIN, ABC_BIN, and runtime library paths.

    Sets environment variables in os.environ and returns a clean child environment
    dict with companion bin and lib directories prepended to PATH.
    """
    env = (base_env or os.environ).copy()

    # Resolve Yosys
    yosys = resolve_yosys_binary(config)
    if yosys and yosys.is_file():
        yosys_str = str(yosys.resolve())
        env["YOSYS_BIN"] = yosys_str
        os.environ["YOSYS_BIN"] = yosys_str

        bin_dir = str(yosys.parent.resolve())
        suite_root = str(yosys.parent.parent.resolve())
        lib_dir = os.path.join(suite_root, "lib")

        extra_dirs = [bin_dir]
        if os.path.isdir(lib_dir):
            extra_dirs.append(lib_dir)

        # Prepend to PATH using os.pathsep (';' on Windows, ':' on Linux)
        current_path = env.get("PATH", "")
        for d in reversed(extra_dirs):
            if d not in current_path:
                current_path = d + os.pathsep + current_path
        env["PATH"] = current_path
        os.environ["PATH"] = current_path

        # Linux/Unix library search path
        if sys.platform != "win32" and os.path.isdir(lib_dir):
            curr_ld = env.get("LD_LIBRARY_PATH", "")
            if lib_dir not in curr_ld:
                env["LD_LIBRARY_PATH"] = (lib_dir + ":" + curr_ld) if curr_ld else lib_dir
                os.environ["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"]

        if os.path.isdir(suite_root):
            env.setdefault("YOSYSHQ_ROOT", suite_root)
            os.environ.setdefault("YOSYSHQ_ROOT", suite_root)

    # Resolve ABC
    abc = resolve_abc_binary(config, yosys)
    if abc and abc.is_file():
        abc_str = str(abc.resolve())
        env["ABC_BIN"] = abc_str
        os.environ["ABC_BIN"] = abc_str

    # Ensure workspace temp root is registered
    workspace_tmp = str((PROJECT_ROOT / "_tmp").resolve())
    os.makedirs(workspace_tmp, exist_ok=True)
    env.update({"TMPDIR": workspace_tmp, "TEMP": workspace_tmp, "TMP": workspace_tmp})

    return env


# ---------------------------------------------------------------------------
# Testcase Discovery & Natural Sorting
# ---------------------------------------------------------------------------

def natural_sort_key(s: str) -> List[object]:
    """Sort strings with embedded numbers naturally (test1 < test2 < test10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", s)]


def discover_testcases(
    testcase_dir: Path = DEFAULT_TESTCASE_DIR,
    tests: Optional[List[str]] = None,
    range_spec: Optional[Tuple[int, int]] = None,
) -> List[Path]:
    """Find and filter testcase directories containing prompt.txt in natural order."""
    if not testcase_dir.is_dir():
        return []

    all_cases = [
        d for d in testcase_dir.iterdir()
        if d.is_dir() and (d / "prompt.txt").is_file()
    ]
    all_cases.sort(key=lambda p: natural_sort_key(p.name))

    if tests:
        target_names = set(tests)
        return [p for p in all_cases if p.name in target_names]

    if range_spec:
        start_idx, end_idx = range_spec
        selected = []
        for p in all_cases:
            match = re.search(r"(\d+)", p.name)
            if match:
                num = int(match.group(1))
                if start_idx <= num <= end_idx:
                    selected.append(p)
        return selected

    return all_cases


# ---------------------------------------------------------------------------
# Directory Management
# ---------------------------------------------------------------------------

def ensure_output_directories(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Tuple[Path, Path, Path]:
    """Create and return (outputs_dir, tool_logs_dir, summary_dir)."""
    outputs_dir = output_dir / "outputs"
    tool_logs_dir = output_dir / "tool_logs"
    summary_dir = output_dir / "summary"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    tool_logs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    return outputs_dir, tool_logs_dir, summary_dir
