"""Live end-to-end integration test for dynamic code generation flow."""

import os
import sys
import yaml
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from src.eda_engine import EDAEngine
from src.agent import EDAAgent


def main():
    print("=================================================================")
    print("STARTING LIVE DYNAMIC TOOL GENERATION TEST")
    print("=================================================================")

    # 1. Load config
    config_path = "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Initialize Engine & load design
    engine = EDAEngine()
    test_design = "testcase/test01/test01.v"
    print(f"\n[1/3] Loading test design: {test_design} ...")
    engine.load(test_design)
    print(f"      Design loaded: {len(engine.netlist.nodes)} gates, {len(engine.netlist.primary_inputs)} PIs, {len(engine.netlist.primary_outputs)} POs.")

    # 3. Initialize Agent
    agent = EDAAgent(config, engine)
    
    # 4. Novel Prompt with no existing tool
    prompt = "List all nets that drive both an AND gate and an OR gate in the design."
    print(f"\n[2/3] Sending novel prompt to agent:")
    print(f"      >> \"{prompt}\"")
    print("      Waiting for LLM planning, declare_missing_tool, code synthesis, sandbox audit, and execution...\n")

    # 5. Process request
    response = agent.process_request(prompt)

    print("\n[3/3] Final Agent Response:")
    print("-----------------------------------------------------------------")
    print(response)
    print("-----------------------------------------------------------------")

    # 6. Check persisted files
    dynamic_dir = Path("src/dynamic_tools")
    generated_files = list(dynamic_dir.glob("tool_*.py"))
    print(f"\nPersisted dynamic tools in {dynamic_dir}:")
    for gf in generated_files:
        print(f"  - {gf.name} ({gf.stat().st_size} bytes)")

    print("\nSUCCESS: End-to-end dynamic code generation flow completed!")


if __name__ == "__main__":
    main()
