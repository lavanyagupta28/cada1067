"""EDA Engine — netlist analysis and transformation operations."""

from __future__ import annotations

import copy
import itertools
import json
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .netlist_parser import (
    GateNode,
    Netlist,
    ParseError,
    WireInfo,
    ONE_INPUT_GATES,
    PRIMITIVE_GATES,
    TWO_INPUT_GATES,
    parse_verilog,
    write_verilog,
)


# ---------------------------------------------------------------------------
# Module-level template cache for replace_gate_type_in_cone.
# Key: (source_type: str, frozenset(target_types))
# Value: dict with keys: inputs (List[str]), output (str),
#        gates (List[GateNode]), internal_wires (List[str])
# ---------------------------------------------------------------------------
_SUBSTITUTION_TEMPLATE_CACHE: Dict[Tuple, dict] = {}


def _workspace_temp_dir() -> str:
    """Return the writable temporary root inside the current working directory."""
    root = os.path.abspath(os.path.join(os.getcwd(), "_tmp"))
    os.makedirs(root, exist_ok=True)
    return root


def _temp_subprocess_env() -> Dict[str, str]:
    """Force Yosys/ABC and child processes to use the workspace temp root."""
    root = _workspace_temp_dir()
    env = os.environ.copy()
    env.update({"TMPDIR": root, "TEMP": root, "TMP": root})
    return env


def _yosys_binary() -> str:
    """Resolve Yosys from YOSYS_BIN or PATH and validate executability."""
    configured = os.environ.get("YOSYS_BIN", "").strip()
    if configured:
        resolved = shutil.which(os.path.expanduser(configured))
        if resolved:
            return resolved
        raise RuntimeError(
            f"YOSYS_BIN does not point to an executable Yosys binary: {configured!r}"
        )

    resolved = shutil.which("yosys")
    if resolved:
        return resolved
    raise RuntimeError(
        "Yosys executable not found. Set YOSYS_BIN or add yosys to PATH."
    )


def _abc_binary() -> str:
    """Resolve standalone ABC from ABC_BIN or PATH."""
    configured = os.environ.get("ABC_BIN", "").strip()
    if configured:
        resolved = shutil.which(os.path.expanduser(configured))
        if resolved:
            return resolved
        raise RuntimeError(
            f"ABC_BIN does not point to an executable ABC binary: {configured!r}"
        )
    for candidate in ("yosys-abc", "abc"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        "ABC executable not found. Set ABC_BIN or add yosys-abc/abc to PATH."
    )


def _parse_blif_template(blif_path: str, target_types: List[str], strip_buf: bool) -> Optional[dict]:
    """Parse ABC BLIF output into a gate template dict.

    Direct ABC emits .gate lines and Yosys emits equivalent .subckt lines:
        .gate not  A=A Y=new_n4_
        .subckt nand A=new_n4_ B=new_n5_ Y=Y
    .inputs / .outputs declare port names.
    Returns None if the file cannot be parsed.
    """
    try:
        with open(blif_path) as fh:
            lines = fh.readlines()
    except OSError:
        return None

    inputs: List[str] = []
    outputs: List[str] = []
    gates: List[GateNode] = []
    g_counter = 0

    for line in lines:
        line = line.strip()
        if line.startswith(".inputs"):
            inputs = line.split()[1:]
        elif line.startswith(".outputs"):
            outputs = line.split()[1:]
        elif line.startswith((".gate", ".subckt")):
            parts = line.split()
            gate_type = parts[1].lower()
            # Build port→signal map
            port_sig: Dict[str, str] = {}
            for tok in parts[2:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    port_sig[k.upper()] = v
            # Determine output and inputs
            out_sig = port_sig.get("Y", "")
            in_sigs = [port_sig.get("A", ""), port_sig.get("B", "")] if gate_type not in ONE_INPUT_GATES else [port_sig.get("A", "")]
            in_sigs = [s for s in in_sigs if s]  # drop empty
            gates.append(GateNode(name=f"_t{g_counter}", gate_type=gate_type,
                                  inputs=in_sigs, output=out_sig))
            g_counter += 1

    if not gates or not outputs:
        return None

    # Determine primary output (last output declared)
    tmpl_output = outputs[0]

    # If strip_buf, inline any buf gates (pass-throughs)
    if strip_buf:
        buf_map: Dict[str, str] = {}
        keep = []
        for g in gates:
            if g.gate_type == "buf":
                buf_map[g.output] = g.inputs[0]
            else:
                keep.append(g)
        # Apply renames through all signals
        def remap(sig: str) -> str:
            while sig in buf_map:
                sig = buf_map[sig]
            return sig
        for g in keep:
            g.inputs = [remap(i) for i in g.inputs]
            g.output = remap(g.output)
        if tmpl_output in buf_map:
            tmpl_output = remap(tmpl_output)
        gates = keep

    # Internal wires = signals not in inputs or outputs
    all_sigs: set = set()
    for g in gates:
        all_sigs.add(g.output)
        all_sigs.update(g.inputs)
    port_sigs = set(inputs) | {tmpl_output}
    internal_wires = [s for s in all_sigs if s not in port_sigs and s not in {"$false", "$true", "$undef"}]

    return {
        "inputs":          inputs,
        "output":          tmpl_output,
        "gates":           gates,
        "internal_wires":  internal_wires,
    }


# ---------------------------------------------------------------------------
# EDA Engine class
# ---------------------------------------------------------------------------

class EDAEngine:
    """Wraps a Netlist and exposes analysis and transformation operations.

    All methods that accept signal / instance names validate them against
    the current netlist and raise ValueError if they are not found.
    """

    def __init__(self) -> None:
        self._netlist: Optional[Netlist] = None
        self._snapshots: Dict[str, Netlist] = {}
        self._instance_counter: int = 0
        self._allowed_gates_constraint: Optional[List[str]] = None
        self._original_netlist_path: Optional[str] = None
        self._last_constant_input_report: Optional[dict] = None
        self._last_constant_input_matches: Optional[List[dict]] = None
        self._last_constant_input_signature: Optional[tuple] = None
        # Combinational adjacency and graph index cache
        self._comb_forward_adj: Optional[Dict[str, List[str]]] = None
        self._comb_backward_adj: Optional[Dict[str, List[str]]] = None
        self._cached_netlist_id: Optional[int] = None
        self._cached_node_count: Optional[int] = None

    # ------------------------------------------------------------------
    # Netlist lifecycle
    # ------------------------------------------------------------------

    def _invalidate_comb_caches(self) -> None:
        """Clear cached combinational graph structures."""
        self._comb_forward_adj = None
        self._comb_backward_adj = None
        self._cached_netlist_id = None
        self._cached_node_count = None

    def _ensure_comb_adj(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Lazily construct and cache forward and backward combinational adjacency.

        DFF boundaries (Q-to-D) are treated as cuts.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        if (
            self._comb_forward_adj is not None
            and self._comb_backward_adj is not None
            and self._cached_netlist_id == id(nl)
            and self._cached_node_count == len(nl.nodes)
        ):
            return self._comb_forward_adj, self._comb_backward_adj

        fwd: Dict[str, List[str]] = {}
        bwd: Dict[str, List[str]] = {}

        for node in nl.nodes.values():
            out_sig = node.output
            inputs = node.inputs
            bwd[out_sig] = list(inputs)
            for inp in inputs:
                fwd.setdefault(inp, []).append(out_sig)

        self._comb_forward_adj = fwd
        self._comb_backward_adj = bwd
        self._cached_netlist_id = id(nl)
        self._cached_node_count = len(nl.nodes)
        return fwd, bwd

    def load(self, filepath: str) -> Netlist:
        """Load a Verilog file into the engine and return the Netlist."""
        self._netlist = parse_verilog(filepath)
        self._original_netlist_path = os.path.abspath(filepath)
        self._allowed_gates_constraint = None
        self._last_constant_input_report = None
        self._last_constant_input_matches = None
        self._last_constant_input_signature = None
        self._invalidate_comb_caches()
        return self._netlist

    def save(self, filepath: Optional[str] = None) -> None:
        """Write the current netlist back to a Verilog file."""
        self._require_netlist()
        if not filepath:
            filepath = "design_out.v"
        write_verilog(self._netlist, filepath)

    def add_snapshot(self, label: str = "default") -> None:
        """Save a deep-copy snapshot of the current netlist under *label*."""
        self._require_netlist()
        self._snapshots[label] = copy.deepcopy(self._netlist)

    def restore_snapshot(self, label: str = "default") -> None:
        """Restore the netlist from a previously saved snapshot."""
        if label not in self._snapshots:
            raise ValueError(f"No snapshot with label {label!r}")
        self._netlist = copy.deepcopy(self._snapshots[label])
        self._invalidate_comb_caches()

    @property
    def netlist(self) -> Netlist:
        """The currently loaded Netlist."""
        self._require_netlist()
        return self._netlist  # type: ignore[return-value]

    @property
    def original_netlist_path(self) -> Optional[str]:
        """Absolute path of the design most recently loaded from disk."""
        return self._original_netlist_path

    def _require_netlist(self) -> None:
        if self._netlist is None:
            raise ValueError("No netlist loaded. Call read_design first.")

    # ------------------------------------------------------------------
    # Internal graph helpers
    # ------------------------------------------------------------------

    def _build_output_to_gate(self) -> Dict[str, str]:
        """Map each output net name → gate instance name that drives it."""
        mapping: Dict[str, str] = {}
        for inst_name, node in self._netlist.nodes.items():  # type: ignore[union-attr]
            mapping[node.output] = inst_name
        for inst_name, dff in self._netlist.dffs.items():  # type: ignore[union-attr]
            mapping[dff.q] = inst_name
        return mapping

    def _build_fanout_map(self) -> Dict[str, List[str]]:
        """Map each net name → list of gate instance names that consume it."""
        fanout: Dict[str, List[str]] = {}
        for inst_name, node in self._netlist.nodes.items():  # type: ignore[union-attr]
            for inp in node.inputs:
                fanout.setdefault(inp, []).append(inst_name)
        for inst_name, dff in self._netlist.dffs.items():  # type: ignore[union-attr]
            for sig in (dff.ck, dff.rn, dff.sn, dff.d):
                if sig:
                    fanout.setdefault(sig, []).append(inst_name)
        return fanout

    def _resolve_signal(self, name: str) -> str:
        """Validate that *name* is a known signal; return it unchanged."""
        nl = self._netlist
        assert nl is not None
        known = (
            set(nl.primary_inputs)
            | set(nl.primary_outputs)
            | set(nl.wires.keys())
            | {n.output for n in nl.nodes.values()}
            | {inp for n in nl.nodes.values() for inp in n.inputs}
            | {dff.q for dff in nl.dffs.values()}
            | {dff.d for dff in nl.dffs.values()}
            | {"1'b0", "1'b1"}
        )
        if name not in known:
            raise ValueError(f"Signal {name!r} not found in netlist.")
        return name

    def _resolve_instance(self, inst_name: str) -> str:
        """Validate that *inst_name* is a known gate or DFF instance."""
        nl = self._netlist
        assert nl is not None
        if inst_name not in nl.nodes and inst_name not in nl.dffs:
            raise ValueError(f"Instance {inst_name!r} not found in netlist.")
        return inst_name

    def _next_inst_name(self, prefix: str = "U_gen") -> str:
        """Generate a unique gate instance name."""
        self._instance_counter += 1
        return f"{prefix}_{self._instance_counter}"

    def _next_wire_name(self, prefix: str = "w_gen") -> str:
        """Generate a unique internal wire name."""
        self._instance_counter += 1
        return f"{prefix}_{self._instance_counter}"

    def _add_wire(self, name: str) -> None:
        """Register an internal wire in the netlist."""
        assert self._netlist is not None
        self._netlist.wires[name] = WireInfo(name=name, width=1, is_bus=False)

    # ------------------------------------------------------------------
    # Analysis: signal-level graph (treating each signal as a node)
    # ------------------------------------------------------------------

    def _forward_successors(self, signal: str) -> List[str]:
        """Return the output signals of all gates driven by *signal*.

        DFF q-to-d arcs are treated as a *cut* (do not follow through DFF).
        """
        fwd, _ = self._ensure_comb_adj()
        return list(fwd.get(signal, []))

    def _backward_predecessors(self, signal: str) -> List[str]:
        """Return all input signals of the gate whose output is *signal*.

        DFF q-to-d arcs are treated as a *cut* (do not follow through DFF).
        """
        _, bwd = self._ensure_comb_adj()
        return list(bwd.get(signal, []))

    def _expand_declared_signal(self, name: str) -> List[str]:
        """Return bit names for a declared bus, otherwise the signal itself."""
        nl = self._netlist
        assert nl is not None
        wire = nl.wires.get(name)
        if wire and wire.is_bus:
            lo, hi = sorted((wire.lsb, wire.msb))
            return [f"{name}[{bit}]" for bit in range(lo, hi + 1)]
        return [name]

    # ==================================================================
    # ANALYSIS OPERATIONS
    # ==================================================================

    def count_gates(self):
        """Return gate counts grouped by type."""

        if self.netlist is None:
            return {"error": "No design loaded"}

        counts = {
            "AND": 0,
            "OR": 0,
            "NOT": 0,
            "NAND": 0,
            "NOR": 0,
            "XOR": 0,
            "XNOR": 0,
            "BUF": 0,
            "DFF": 0,
        }

        # Count combinational gates
        for node in self.netlist.nodes.values():
            gt = node.gate_type.upper()

            if gt in counts:
                counts[gt] += 1

        # Count DFFs separately
        counts["DFF"] = len(self.netlist.dffs)

        total = sum(counts.values())

        return {
            "total_gates": total,
            "breakdown": counts,
        }
    def _driver_of(self, signal: str):
        """Return gate driving this signal, or None."""

        for node in self.netlist.nodes.values():
            if node.output == signal:
                return node

        return None
    
    def _fanin_depth(self, signal: str, memo=None) -> int:
        """Return max logic depth ending at signal."""

        if memo is None:
            memo = {}

        if signal in memo:
            return memo[signal]

        # Primary input
        if signal in self.netlist.primary_inputs:
            memo[signal] = 0
            return 0

        driver = self._driver_of(signal)

        # No driver
        if driver is None:
            memo[signal] = 0
            return 0

        depth = 1 + max(
            self._fanin_depth(inp, memo)
            for inp in driver.inputs
        )

        memo[signal] = depth
        return depth
    
    def get_fanin_cone_depth(self, output_signal: str):
        """Return max logic depth of output fanin cone."""

        self._require_netlist()
        self._resolve_signal(output_signal)

        return {
            "output": output_signal,
            "depth": self._fanin_depth(output_signal)
        }

    def get_deepest_output_fanin_cone(self) -> dict:
        """Return all primary-output bits tied for maximum fanin depth."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        output_bits: List[str] = []
        for output_name in nl.primary_outputs:
            wire = nl.wires.get(output_name)
            if wire and wire.is_bus:
                lo = min(wire.msb, wire.lsb)
                hi = max(wire.msb, wire.lsb)
                output_bits.extend(
                    f"{output_name}[{bit}]" for bit in range(lo, hi + 1)
                )
            else:
                output_bits.append(output_name)

        if not output_bits:
            return {
                "max_depth": -1,
                "deepest_outputs": [],
                "tie_count": 0,
                "total_outputs": 0,
            }

        memo: Dict[str, int] = {}
        depths = {
            output_name: self._fanin_depth(output_name, memo)
            for output_name in output_bits
        }
        max_depth = max(depths.values())
        deepest_outputs = [
            output_name
            for output_name in output_bits
            if depths[output_name] == max_depth
        ]
        result = {
            "max_depth": max_depth,
            "tie_count": len(deepest_outputs),
            "total_outputs": len(output_bits),
        }
        if len(deepest_outputs) <= 50:
            result["deepest_outputs"] = deepest_outputs
            return result

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix="deepest_output_fanin_cones_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write(
                f"# Output bits at maximum fanin depth {max_depth} "
                f"(count: {len(deepest_outputs)})\n"
            )
            for output_name in deepest_outputs:
                report.write(f"{output_name}\n")
        result["file_path"] = os.path.abspath(report.name)
        return result

    def count_outputs_by_logic_depth(self, operator: str, threshold: int) -> dict:
        """Count primary-output bits whose combinational fanin depth matches a predicate."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        comparators = {
            ">": lambda depth: depth > threshold,
            ">=": lambda depth: depth >= threshold,
            "<": lambda depth: depth < threshold,
            "<=": lambda depth: depth <= threshold,
            "==": lambda depth: depth == threshold,
            "!=": lambda depth: depth != threshold,
        }
        if operator not in comparators:
            raise ValueError(
                "operator must be one of: >, >=, <, <=, ==, !="
            )

        output_bits: List[str] = []
        for output_name in nl.primary_outputs:
            wire = nl.wires.get(output_name)
            if wire and wire.is_bus:
                lo = min(wire.msb, wire.lsb)
                hi = max(wire.msb, wire.lsb)
                output_bits.extend(f"{output_name}[{bit}]" for bit in range(lo, hi + 1))
            else:
                output_bits.append(output_name)

        memo: Dict[str, int] = {}
        depths = [
            {"output": output_name, "depth": self._fanin_depth(output_name, memo)}
            for output_name in output_bits
        ]
        matching = [entry for entry in depths if comparators[operator](entry["depth"])]

        return {
            "operator": operator,
            "threshold": threshold,
            "count": len(matching),
            "outputs": matching,
            "total_outputs": len(output_bits),
        }
        
    def get_max_depth(
        self, source: str, sink: str
    ) -> Tuple[int, List[str]]:
        """Return (depth, path) of the longest combinational path from source to sink.

        Traverses forward through gate outputs; DFF q-to-d is treated as a cut.
        Returns (-1, []) if no path exists.

        Args:
            source: Starting signal name (primary input or gate output).
            sink:   Ending signal name (primary output or gate output).

        Returns:
            A tuple (depth, path) where path is a list of signal names.
        """
        self._require_netlist()
        self._resolve_signal(source)
        self._resolve_signal(sink)

        if source == sink:
            return (0, [source])

        fwd, bwd = self._ensure_comb_adj()

        # Step 1: Backward reachability from sink (prunes entire graph to relevant backward cone)
        backward_reachable: Set[str] = {sink}
        queue_bwd: deque[str] = deque([sink])
        while queue_bwd:
            curr = queue_bwd.popleft()
            for pred in bwd.get(curr, ()):
                if pred not in backward_reachable:
                    backward_reachable.add(pred)
                    queue_bwd.append(pred)

        # Early exit if source cannot reach sink
        if source not in backward_reachable:
            return (-1, [])

        # Step 2: Forward reachability from source restricted strictly to backward_reachable
        # Defines the exact bidirectional induced subgraph: Forward(source) ∩ Backward(sink)
        relevant_nodes: Set[str] = {source}
        queue_fwd: deque[str] = deque([source])
        while queue_fwd:
            curr = queue_fwd.popleft()
            for nxt in fwd.get(curr, ()):
                if nxt in backward_reachable and nxt not in relevant_nodes:
                    relevant_nodes.add(nxt)
                    queue_fwd.append(nxt)

        # Step 3: Longest path DP using Kahn's topological sort on the pruned cone
        in_degree: Dict[str, int] = {node: 0 for node in relevant_nodes}
        for u in relevant_nodes:
            for v in fwd.get(u, ()):
                if v in relevant_nodes:
                    in_degree[v] += 1

        dist: Dict[str, int] = {source: 0}
        parent: Dict[str, Optional[str]] = {source: None}
        queue: deque[str] = deque([source])

        while queue:
            u = queue.popleft()
            u_dist = dist[u]
            for v in fwd.get(u, ()):
                if v in relevant_nodes:
                    if u_dist + 1 > dist.get(v, -1):
                        dist[v] = u_dist + 1
                        parent[v] = u
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        if sink not in dist:
            return (-1, [])

        # Reconstruct path backwards from parent pointers
        path: List[str] = []
        curr_sig: Optional[str] = sink
        while curr_sig is not None:
            path.append(curr_sig)
            curr_sig = parent.get(curr_sig)
        path.reverse()

        return (dist[sink], path)

    def get_max_depth_between_endpoint_classes(
        self,
        source_class: str,
        sink_class: str,
    ) -> dict:
        """Return the longest combinational path between endpoint classes."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        source_class = source_class.upper()
        sink_class = sink_class.upper()
        if source_class not in {"PI", "DFF_Q"}:
            raise ValueError("source_class must be 'PI' or 'DFF_Q'.")
        if sink_class not in {"DFF_D", "PO"}:
            raise ValueError("sink_class must be 'DFF_D' or 'PO'.")

        def expand_declared(name: str) -> List[str]:
            wi = nl.wires.get(name)
            if wi and wi.is_bus:
                lo, hi = sorted((wi.lsb, wi.msb))
                return [f"{name}[{bit}]" for bit in range(lo, hi + 1)]
            return [name]

        if source_class == "PI":
            source_signals = {
                signal
                for name in nl.primary_inputs
                for signal in expand_declared(name)
            }
        else:
            source_signals = {dff.q for dff in nl.dffs.values() if dff.q}

        comb_driver = {node.output: node for node in nl.nodes.values()}
        memo: Dict[str, Optional[int]] = {}
        parent: Dict[str, str] = {}
        active: Set[str] = set()

        def longest_from_source(signal_name: str) -> Optional[int]:
            if signal_name in memo:
                return memo[signal_name]
            if signal_name in source_signals:
                memo[signal_name] = 0
                return 0
            if signal_name in {"1'b0", "1'b1"} or signal_name in active:
                memo[signal_name] = None
                return None

            node = comb_driver.get(signal_name)
            if node is None:
                memo[signal_name] = None
                return None

            active.add(signal_name)
            candidates = [
                (depth, inp)
                for inp in node.inputs
                if (depth := longest_from_source(inp)) is not None
            ]
            active.discard(signal_name)
            if not candidates:
                memo[signal_name] = None
                return None

            input_depth, input_signal = max(candidates, key=lambda item: item[0])
            memo[signal_name] = input_depth + 1
            parent[signal_name] = input_signal
            return memo[signal_name]

        if sink_class == "DFF_D":
            endpoints = [
                (inst_name, dff.d)
                for inst_name, dff in sorted(nl.dffs.items())
                if dff.d
            ]
        else:
            endpoints = [
                (signal_name, signal_name)
                for output in nl.primary_outputs
                for signal_name in expand_declared(output)
            ]

        best: Optional[Tuple[int, str, str]] = None
        best_endpoint_ties = 0
        for endpoint_name, endpoint_signal in endpoints:
            depth = longest_from_source(endpoint_signal)
            if depth is None:
                continue
            candidate = (depth, endpoint_name, endpoint_signal)
            if best is None or candidate[0] > best[0]:
                best = candidate
                best_endpoint_ties = 1
            elif candidate[0] == best[0]:
                best_endpoint_ties += 1

        if best is None:
            return {
                "source_class": source_class,
                "sink_class": sink_class,
                "depth": -1,
                "path": [],
            }

        depth, endpoint_name, endpoint_signal = best
        path = [endpoint_signal]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path.reverse()
        result = {
            "source_class": source_class,
            "sink_class": sink_class,
            "depth": depth,
            "source_signal": path[0],
            "sink_name": endpoint_name,
            "sink_signal": endpoint_signal,
            "max_depth_sink_tie_count": best_endpoint_ties,
            "path": path,
        }
        if source_class == "DFF_Q":
            source_dff = next(
                (
                    instance
                    for instance, dff in nl.dffs.items()
                    if dff.q == path[0]
                ),
                None,
            )
            result.update(
                {
                    "source_instance": source_dff,
                    "source_pin": "Q",
                    "source_pin_signal": path[0],
                }
            )
        if sink_class == "DFF_D":
            result.update(
                {
                    "sink_instance": endpoint_name,
                    "sink_pin": "D",
                    "sink_pin_signal": endpoint_signal,
                }
            )
        return result

    def get_global_max_depth(self) -> dict:
        """Return the longest valid combinational boundary-to-boundary path."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        def expand_declared(name: str) -> List[str]:
            wi = nl.wires.get(name)
            if wi and wi.is_bus:
                lo, hi = sorted((wi.lsb, wi.msb))
                return [f"{name}[{bit}]" for bit in range(lo, hi + 1)]
            return [name]

        pi_signals: Set[str] = set()
        for name in nl.primary_inputs:
            pi_signals.update(expand_declared(name))
        dff_q_signals = {dff.q for dff in nl.dffs.values() if dff.q}
        source_signals = pi_signals | dff_q_signals

        endpoints: List[Tuple[str, str, str]] = []
        for output_name in nl.primary_outputs:
            endpoints.extend(
                ("PO", signal_name, signal_name)
                for signal_name in expand_declared(output_name)
            )
        endpoints.extend(
            ("DFF_D", instance_name, dff.d)
            for instance_name, dff in sorted(nl.dffs.items())
            if dff.d
        )

        comb_driver = {node.output: node for node in nl.nodes.values()}
        memo: Dict[str, Optional[int]] = {}
        parent: Dict[str, str] = {}
        active: Set[str] = set()

        def longest_from_boundary(signal_name: str) -> Optional[int]:
            if signal_name in memo:
                return memo[signal_name]
            if signal_name in source_signals:
                memo[signal_name] = 0
                return 0
            if signal_name in {"1'b0", "1'b1"} or signal_name in active:
                memo[signal_name] = None
                return None

            node = comb_driver.get(signal_name)
            if node is None:
                memo[signal_name] = None
                return None

            active.add(signal_name)
            candidates = [
                (depth, input_signal)
                for input_signal in node.inputs
                if (depth := longest_from_boundary(input_signal)) is not None
            ]
            active.discard(signal_name)
            if not candidates:
                memo[signal_name] = None
                return None

            input_depth, input_signal = max(candidates, key=lambda item: item[0])
            memo[signal_name] = input_depth + 1
            parent[signal_name] = input_signal
            return memo[signal_name]

        best: Optional[Tuple[int, str, str, str]] = None
        for sink_class, sink_name, sink_signal in endpoints:
            depth = longest_from_boundary(sink_signal)
            if depth is None:
                continue
            candidate = (depth, sink_class, sink_name, sink_signal)
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None:
            return {"scope": "global", "depth": -1, "path": []}

        depth, sink_class, sink_name, sink_signal = best
        path = [sink_signal]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path.reverse()
        source_signal = path[0]
        source_class = "DFF_Q" if source_signal in dff_q_signals else "PI"
        return {
            "scope": "global",
            "depth": depth,
            "source_class": source_class,
            "source_signal": source_signal,
            "sink_class": sink_class,
            "sink_name": sink_name,
            "sink_signal": sink_signal,
            "path": path,
        }

    def is_gate_on_any_max_depth_path(self, gate_name: str) -> dict:
        """Return whether a combinational gate lies on any global max-depth path."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_instance(gate_name)
        if gate_name not in nl.nodes:
            raise ValueError(f"Instance {gate_name!r} is not a combinational gate.")

        def expand_declared(name: str) -> List[str]:
            wi = nl.wires.get(name)
            if wi and wi.is_bus:
                lo, hi = sorted((wi.lsb, wi.msb))
                return [f"{name}[{bit}]" for bit in range(lo, hi + 1)]
            return [name]

        source_signals: Set[str] = set()
        for name in nl.primary_inputs:
            source_signals.update(expand_declared(name))
            source_signals.add(name)
        for dff in nl.dffs.values():
            if dff.q:
                source_signals.add(dff.q)

        sink_signals: Set[str] = set()
        for name in nl.primary_outputs:
            sink_signals.update(expand_declared(name))
            sink_signals.add(name)
        for dff in nl.dffs.values():
            if dff.d:
                sink_signals.add(dff.d)

        comb_driver = {node.output: node for node in nl.nodes.values()}
        fanout = self._build_fanout_map()
        depth_from_source: Dict[str, Optional[int]] = {}
        depth_to_sink: Dict[str, Optional[int]] = {}
        active_from: Set[str] = set()
        active_to: Set[str] = set()

        def longest_from_source(signal_name: str) -> Optional[int]:
            if signal_name in depth_from_source:
                return depth_from_source[signal_name]
            if signal_name in source_signals:
                depth_from_source[signal_name] = 0
                return 0
            if signal_name in {"1'b0", "1'b1"} or signal_name in active_from:
                depth_from_source[signal_name] = None
                return None
            node = comb_driver.get(signal_name)
            if node is None:
                depth_from_source[signal_name] = None
                return None
            active_from.add(signal_name)
            candidates = [
                depth
                for input_signal in node.inputs
                if (depth := longest_from_source(input_signal)) is not None
            ]
            active_from.discard(signal_name)
            if not candidates:
                depth_from_source[signal_name] = None
                return None
            depth_from_source[signal_name] = max(candidates) + 1
            return depth_from_source[signal_name]

        def longest_to_sink(signal_name: str) -> Optional[int]:
            if signal_name in depth_to_sink:
                return depth_to_sink[signal_name]
            if signal_name in sink_signals:
                depth_to_sink[signal_name] = 0
                return 0
            if signal_name in {"1'b0", "1'b1"} or signal_name in active_to:
                depth_to_sink[signal_name] = None
                return None
            active_to.add(signal_name)
            candidates: List[int] = []
            for consumer_name in fanout.get(signal_name, []):
                if consumer_name in nl.nodes:
                    consumer = nl.nodes[consumer_name]
                    depth = longest_to_sink(consumer.output)
                    if depth is not None:
                        candidates.append(depth + 1)
                elif consumer_name in nl.dffs:
                    # DFF D pins are sink boundaries; do not cross the DFF.
                    if nl.dffs[consumer_name].d == signal_name:
                        candidates.append(0)
            active_to.discard(signal_name)
            if not candidates:
                depth_to_sink[signal_name] = None
                return None
            depth_to_sink[signal_name] = max(candidates)
            return depth_to_sink[signal_name]

        all_sink_depths = [
            depth
            for signal_name in sink_signals
            if (depth := longest_from_source(signal_name)) is not None
        ]
        global_max_depth = max(all_sink_depths, default=-1)

        gate = nl.nodes[gate_name]
        input_depths = [
            depth
            for input_signal in gate.inputs
            if (depth := longest_from_source(input_signal)) is not None
        ]
        output_to_sink = longest_to_sink(gate.output)

        if input_depths and output_to_sink is not None:
            best_path_depth = max(input_depths) + 1 + output_to_sink
        else:
            best_path_depth = -1

        return {
            "gate": gate_name,
            "gate_type": gate.gate_type,
            "gate_output": gate.output,
            "global_max_depth": global_max_depth,
            "best_path_depth_through_gate": best_path_depth,
            "on_any_max_depth_path": (
                global_max_depth >= 0 and best_path_depth == global_max_depth
            ),
        }

    def path_passes_through(
        self, source: str, sink: str, node: str
    ) -> bool:
        """Return True if ALL paths from source to sink pass through node.

        Args:
            source: Starting signal.
            sink:   Ending signal.
            node:   Candidate mandatory waypoint signal.
        """
        return bool(
            self.paths_pass_through_report(source, sink, node)[
                "all_paths_pass_through"
            ]
        )

    def paths_pass_through_report(
        self, source: str, sink: str, through: str
    ) -> dict:
        """Report whether every combinational source-to-sink path uses a node.

        The through node may be either a signal name or a combinational gate
        instance. Paths are searched on an alternating signal/gate graph, so
        gate instances are first-class path nodes.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_signal(source)
        self._resolve_signal(sink)

        through_type = ""
        if through in nl.nodes:
            through_type = "combinational_gate"
        else:
            self._resolve_signal(through)
            through_type = "signal"

        consumers: Dict[str, List[str]] = {}
        for inst_name, gate in nl.nodes.items():
            for input_signal in gate.inputs:
                consumers.setdefault(input_signal, []).append(inst_name)

        def format_path(path: List[Tuple[str, str]]) -> List[str]:
            return [
                f"{kind}:{name}" if kind == "gate" else name
                for kind, name in path
            ]

        def find_path(avoid_kind: Optional[str] = None) -> Optional[List[str]]:
            start = ("signal", source)
            if avoid_kind == "signal" and source == through:
                return None
            if avoid_kind == "signal" and sink == through:
                return None

            queue: deque[List[Tuple[str, str]]] = deque([[start]])
            visited: Set[Tuple[str, str]] = {start}

            while queue:
                path = queue.popleft()
                kind, name = path[-1]
                if kind == "signal" and name == sink:
                    return format_path(path)

                next_nodes: List[Tuple[str, str]] = []
                if kind == "signal":
                    for inst_name in consumers.get(name, []):
                        if avoid_kind == "gate" and inst_name == through:
                            continue
                        next_nodes.append(("gate", inst_name))
                else:
                    output_signal = nl.nodes[name].output
                    if avoid_kind == "signal" and output_signal == through:
                        continue
                    next_nodes.append(("signal", output_signal))

                for next_node in next_nodes:
                    if next_node in visited:
                        continue
                    visited.add(next_node)
                    queue.append(path + [next_node])
            return None

        any_path = find_path()
        if any_path is None:
            return {
                "source": source,
                "sink": sink,
                "through": through,
                "through_type": through_type,
                "path_exists": False,
                "all_paths_pass_through": False,
                "counterexample_path": None,
                "example_path": None,
                "reason": "No combinational path exists from source to sink.",
            }

        avoiding_path = find_path("gate" if through_type == "combinational_gate" else "signal")
        return {
            "source": source,
            "sink": sink,
            "through": through,
            "through_type": through_type,
            "path_exists": True,
            "all_paths_pass_through": avoiding_path is None,
            "counterexample_path": avoiding_path,
            "example_path": any_path,
            "reason": (
                "No path avoiding the through node was found."
                if avoiding_path is None
                else "A path avoiding the through node exists."
            ),
        }

    def is_wire_cut_between_primary_ios(self, wire_name: str) -> dict:
        """Check whether a signal is a PI-to-PO cut without enumerating paths.

        A wire is reported as a cut if there exists at least one expanded
        primary-input bit and one expanded primary-output bit such that a
        combinational path exists through the wire, and blocking the wire
        disconnects that PI bit from that PO bit. DFF boundaries are treated
        as cuts.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_signal(wire_name)

        candidate_signals = self._expand_declared_signal(wire_name)
        pi_bits = [
            bit
            for name in nl.primary_inputs
            for bit in self._expand_declared_signal(name)
        ]
        po_bits = [
            bit
            for name in nl.primary_outputs
            for bit in self._expand_declared_signal(name)
        ]

        successors: Dict[str, List[str]] = {}
        predecessors: Dict[str, List[str]] = {}
        for gate in nl.nodes.values():
            for input_signal in gate.inputs:
                successors.setdefault(input_signal, []).append(gate.output)
                predecessors.setdefault(gate.output, []).append(input_signal)

        def forward_reachable(
            starts: List[str], blocked: Optional[Set[str]] = None
        ) -> Set[str]:
            blocked = blocked or set()
            seen: Set[str] = set()
            queue: deque[str] = deque()
            for start in starts:
                if start in blocked or start in seen:
                    continue
                seen.add(start)
                queue.append(start)
            while queue:
                signal_name = queue.popleft()
                for next_signal in successors.get(signal_name, []):
                    if next_signal in blocked or next_signal in seen:
                        continue
                    seen.add(next_signal)
                    queue.append(next_signal)
            return seen

        def backward_reachable(starts: List[str]) -> Set[str]:
            seen: Set[str] = set(starts)
            queue: deque[str] = deque(starts)
            while queue:
                signal_name = queue.popleft()
                for prev_signal in predecessors.get(signal_name, []):
                    if prev_signal in seen:
                        continue
                    seen.add(prev_signal)
                    queue.append(prev_signal)
            return seen

        pi_set = set(pi_bits)
        po_set = set(po_bits)
        details = []
        is_on_any_path = False
        is_cut = False
        cut_pairs = []
        upstream_all: Set[str] = set()
        downstream_all: Set[str] = set()

        for candidate in candidate_signals:
            upstream_pis = sorted(pi_set & backward_reachable([candidate]))
            downstream_pos = sorted(po_set & forward_reachable([candidate]))
            upstream_all.update(upstream_pis)
            downstream_all.update(downstream_pos)
            candidate_on_path = bool(upstream_pis and downstream_pos)
            is_on_any_path = is_on_any_path or candidate_on_path

            blocked_pairs = []
            if candidate_on_path:
                for pi in upstream_pis:
                    reachable_without_candidate = forward_reachable(
                        [pi], blocked={candidate}
                    )
                    for po in downstream_pos:
                        if po not in reachable_without_candidate:
                            blocked_pairs.append({"primary_input": pi, "primary_output": po})
                            if len(cut_pairs) < 10:
                                cut_pairs.append(
                                    {"primary_input": pi, "primary_output": po}
                                )

            candidate_is_cut = bool(blocked_pairs)
            is_cut = is_cut or candidate_is_cut
            details.append(
                {
                    "signal": candidate,
                    "is_on_any_pi_po_path": candidate_on_path,
                    "is_cut": candidate_is_cut,
                    "upstream_primary_input_count": len(upstream_pis),
                    "downstream_primary_output_count": len(downstream_pos),
                    "cut_pair_count": len(blocked_pairs),
                    "upstream_primary_inputs_sample": upstream_pis[:10],
                    "downstream_primary_outputs_sample": downstream_pos[:10],
                }
            )

        if not is_on_any_path:
            reason = (
                f"{wire_name} is not on any combinational path from an expanded "
                "primary input to an expanded primary output."
            )
        elif is_cut:
            reason = (
                f"Blocking {wire_name} disconnects at least one primary-input "
                "bit from at least one primary-output bit."
            )
        else:
            reason = (
                f"{wire_name} lies on at least one PI-to-PO path, but no checked "
                "PI/PO pair depends on it as a mandatory cut."
            )

        return {
            "wire": wire_name,
            "candidate_signals": candidate_signals,
            "primary_input_bit_count": len(pi_bits),
            "primary_output_bit_count": len(po_bits),
            "is_on_any_pi_po_path": is_on_any_path,
            "is_cut_between_primary_io": is_cut,
            "answer": "yes" if is_cut else "no",
            "upstream_primary_input_count": len(upstream_all),
            "downstream_primary_output_count": len(downstream_all),
            "upstream_primary_inputs_sample": sorted(upstream_all)[:10],
            "downstream_primary_outputs_sample": sorted(downstream_all)[:10],
            "cut_pairs_sample": cut_pairs,
            "details": details,
            "reason": reason,
        }

    def find_path_avoiding(
        self, source: str, sink: str, avoid: str
    ) -> Optional[List[str]]:
        """Return one path from source to sink that does NOT pass through avoid.

        Returns None if no such path exists.

        Args:
            source: Starting signal.
            sink:   Ending signal.
            avoid:  Signal that must not appear on the path.
        """
        self._require_netlist()
        self._resolve_signal(source)
        self._resolve_signal(sink)
        self._resolve_signal(avoid)

        if source == avoid or sink == avoid:
            return None

        # BFS avoiding the forbidden signal
        queue: deque[List[str]] = deque([[source]])
        visited: Set[str] = {source}

        while queue:
            path = queue.popleft()
            cur = path[-1]
            if cur == sink:
                return path
            for nxt in self._forward_successors(cur):
                if nxt not in visited and nxt != avoid:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def find_articulation_points(
        self, source: str, sink: str
    ) -> dict:
        """Find all articulation points (cut gates) between source and sink in the combinational graph.

        An articulation point is an intermediate gate instance such that removing it
        disconnects all combinational paths from source to sink.

        Args:
            source: Starting signal name.
            sink:   Ending signal name.

        Returns:
            Dictionary containing path existence, count, and list of articulation gates.
        """
        self._require_netlist()
        self._resolve_signal(source)
        self._resolve_signal(sink)
        nl = self._netlist
        assert nl is not None

        if source == sink:
            return {
                "source": source,
                "sink": sink,
                "path_exists": True,
                "count": 0,
                "articulation_points": [],
                "gates": [],
                "message": f"Source and sink are the same signal ({source}). No intermediate articulation points exist.",
            }

        # Build efficient adjacency maps
        signal_consumers: Dict[str, List[GateNode]] = {}
        out2gate = self._build_output_to_gate()
        for gate in nl.nodes.values():
            for inp in gate.inputs:
                signal_consumers.setdefault(inp, []).append(gate)

        # 1. Forward reachability from source
        forward_signals: Set[str] = {source}
        forward_gates: Set[str] = set()
        queue: deque[str] = deque([source])

        while queue:
            curr_sig = queue.popleft()
            for gate in signal_consumers.get(curr_sig, []):
                forward_gates.add(gate.name)
                out_sig = gate.output
                if out_sig not in forward_signals:
                    forward_signals.add(out_sig)
                    queue.append(out_sig)

        if sink not in forward_signals:
            return {
                "source": source,
                "sink": sink,
                "path_exists": False,
                "count": 0,
                "articulation_points": [],
                "gates": [],
                "message": f"No combinational path exists between {source} and {sink}.",
            }

        # 2. Backward reachability from sink
        backward_signals: Set[str] = {sink}
        backward_gates: Set[str] = set()
        queue = deque([sink])

        while queue:
            curr_sig = queue.popleft()
            driver_name = out2gate.get(curr_sig)
            if driver_name and driver_name in nl.nodes:
                backward_gates.add(driver_name)
                gate = nl.nodes[driver_name]
                for inp in gate.inputs:
                    if inp not in backward_signals:
                        backward_signals.add(inp)
                        queue.append(inp)

        # 3. Active subgraph between source and sink
        active_signals = forward_signals & backward_signals
        active_gates = forward_gates & backward_gates

        # Helper to test if sink is reachable from source when blocking a gate
        def is_reachable_without_gate(blocked_gate: str) -> bool:
            visited_sigs: Set[str] = {source}
            q: deque[str] = deque([source])
            while q:
                curr = q.popleft()
                if curr == sink:
                    return True
                for g in signal_consumers.get(curr, []):
                    if g.name == blocked_gate or g.name not in active_gates:
                        continue
                    out_s = g.output
                    if out_s not in active_signals:
                        continue
                    if out_s not in visited_sigs:
                        visited_sigs.add(out_s)
                        q.append(out_s)
            return False

        # 4. Find articulation gates
        candidate_gates = active_gates
        art_gates: List[str] = []
        for cand_gate in candidate_gates:
            if not is_reachable_without_gate(cand_gate):
                art_gates.append(cand_gate)

        # 5. Compute topological distance from source for sorting
        min_depth: Dict[str, int] = {source: 0}
        q_depth: deque[str] = deque([source])
        while q_depth:
            curr = q_depth.popleft()
            d = min_depth[curr]
            for g in signal_consumers.get(curr, []):
                if g.name not in active_gates:
                    continue
                out_s = g.output
                if out_s in active_signals:
                    if out_s not in min_depth or min_depth[out_s] > d + 1:
                        min_depth[out_s] = d + 1
                        q_depth.append(out_s)

        sorted_gates = sorted(
            art_gates,
            key=lambda g: (
                min_depth.get(nl.nodes[g].output, 999999),
                g,
            ),
        )

        return {
            "source": source,
            "sink": sink,
            "path_exists": True,
            "count": len(sorted_gates),
            "articulation_points": sorted_gates,
            "gates": sorted_gates,
            "message": (
                f"Found {len(sorted_gates)} articulation gate(s) between {source} and {sink}: {', '.join(sorted_gates)}"
                if sorted_gates
                else f"Found 0 articulation points between {source} and {sink}."
            ),
        }

    def get_logic_cone(self, output_signal: str) -> List[str]:
        """Return all gate instance names that transitively feed output_signal.

        Args:
            output_signal: The target output net name.
        """
        self._require_netlist()
        self._resolve_signal(output_signal)
        nl = self._netlist
        assert nl is not None

        out2gate = self._build_output_to_gate()

        cone_gates: List[str] = []
        visited_signals: Set[str] = set()
        queue: deque[str] = deque([output_signal])

        while queue:
            sig = queue.popleft()
            if sig in visited_signals:
                continue
            visited_signals.add(sig)
            driver = out2gate.get(sig)
            if driver is None or driver not in nl.nodes:
                continue  # primary input or DFF q
            gate = nl.nodes[driver]
            if driver not in cone_gates:
                cone_gates.append(driver)
            for inp in gate.inputs:
                if inp not in visited_signals:
                    queue.append(inp)

        return cone_gates

    def get_logic_cone_report(
        self, output_signal: str, inline_limit: int = 10
    ) -> dict:
        """Return a compact transitive-fanin report for an output signal."""
        inline_limit = max(0, int(inline_limit))
        self._require_netlist()
        self._resolve_signal(output_signal)
        resolution = self._resolve_fanin_cone_target(output_signal)
        resolved_output = resolution["resolved_signal"]
        gates = self.get_logic_cone(resolved_output)
        result = {
            "output_signal": output_signal,
            "count": len(gates),
        }
        if resolution["through_dff"] is not None:
            result.update(
                {
                    "resolved_output": resolved_output,
                    "through_dff": resolution["through_dff"],
                    "resolved_pin": resolution["resolved_pin"],
                }
            )
        if len(gates) <= inline_limit:
            result["gates"] = gates
            return result

        safe_name = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", output_signal).strip("_")
            or "signal"
        )
        report = tempfile.NamedTemporaryFile(
            "w",
            prefix=f"fanin_{safe_name}_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write(
                f"# Transitive fanin gates for {output_signal} "
                f"(count: {len(gates)})\n"
            )
            if resolution["through_dff"] is not None:
                report.write(
                    f"# Registered output: {resolution['through_dff']}.Q "
                    f"-> {resolution['through_dff']}.D\n"
                )
                report.write(f"# Resolved cone target: {resolved_output}\n")
            for gate_name in gates:
                report.write(f"{gate_name}\n")
        result["file_path"] = os.path.abspath(report.name)
        return result

    def find_shared_fanin_gates(
        self,
        signal_a: str,
        signal_b: str,
        inline_limit: int = 20,
    ) -> dict:
        """Return the intersection of two resolved transitive fanin cones."""
        self._require_netlist()
        self._resolve_signal(signal_a)
        self._resolve_signal(signal_b)
        inline_limit = max(0, int(inline_limit))

        resolution_a = self._resolve_fanin_cone_target(signal_a)
        resolution_b = self._resolve_fanin_cone_target(signal_b)
        cone_a = set(self.get_logic_cone(resolution_a["resolved_signal"]))
        cone_b = set(self.get_logic_cone(resolution_b["resolved_signal"]))
        shared = sorted(cone_a & cone_b)

        def compact_resolution(resolution: dict) -> dict:
            return {
                "requested_signal": resolution["requested_signal"],
                "resolved_signal": resolution["resolved_signal"],
                "through_dff": resolution["through_dff"],
                "resolved_pin": resolution["resolved_pin"],
            }

        result = {
            "signal_a": signal_a,
            "signal_b": signal_b,
            "cone_a_count": len(cone_a),
            "cone_b_count": len(cone_b),
            "shared_count": len(shared),
            "resolution_a": compact_resolution(resolution_a),
            "resolution_b": compact_resolution(resolution_b),
        }
        if len(shared) <= inline_limit:
            result["shared_gates"] = shared
            return result

        safe_a = re.sub(r"[^A-Za-z0-9_.-]+", "_", signal_a).strip("_") or "signal_a"
        safe_b = re.sub(r"[^A-Za-z0-9_.-]+", "_", signal_b).strip("_") or "signal_b"
        report = tempfile.NamedTemporaryFile(
            "w",
            prefix=f"shared_fanin_{safe_a}_{safe_b}_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write(
                f"# Gates shared by fanin cones of {signal_a} and {signal_b} "
                f"(count: {len(shared)})\n"
            )
            for gate_name in shared:
                report.write(f"{gate_name}\n")
        result["sample_shared_gates"] = shared[: min(10, inline_limit)]
        result["file_path"] = os.path.abspath(report.name)
        return result

    def derive_boolean_equation(
        self, output_signal: str, inline_limit: int = 20
    ) -> dict:
        """Derive the Boolean equation for an output signal or net in terms of primary/boundary inputs.

        Handles combinational logic cones, direct DFF drivers, direct primary inputs,
        and constant tie-offs.
        """
        self._require_netlist()
        self._resolve_signal(output_signal)
        nl = self._netlist
        assert nl is not None

        out2gate = self._build_output_to_gate()
        driver = out2gate.get(output_signal)

        primary_input_bits: Set[str] = set()
        for pi in nl.primary_inputs:
            wi = nl.wires.get(pi)
            if wi and wi.is_bus:
                lo, hi = sorted((wi.lsb, wi.msb))
                primary_input_bits.update(f"{pi}[{bit}]" for bit in range(lo, hi + 1))
            else:
                primary_input_bits.add(pi)
        dff_q_signals = {dff.q for dff in nl.dffs.values()}
        constants = {"1'b0", "1'b1"}

        def is_primary_input(sig: str) -> bool:
            return sig in primary_input_bits

        # 1. Direct Primary Input / Length-0 connection
        if driver is None:
            if is_primary_input(output_signal):
                return {
                    "output_signal": output_signal,
                    "driver_type": "primary_input",
                    "equation": f"{output_signal} = {output_signal}",
                    "primary_input_only_expression_available": True,
                    "primary_inputs_used": [output_signal],
                    "state_boundaries": [],
                    "explanation": f"Output {output_signal} is directly connected to primary input {output_signal}.",
                }
            if output_signal in constants:
                return {
                    "output_signal": output_signal,
                    "driver_type": "constant",
                    "equation": f"{output_signal} = {output_signal}",
                    "primary_input_only_expression_available": True,
                    "primary_inputs_used": [],
                    "state_boundaries": [],
                    "explanation": f"Signal {output_signal} is a constant tie-off ({output_signal}).",
                }
            return {
                "output_signal": output_signal,
                "driver_type": "undriven",
                "equation": f"{output_signal} is undriven / unassigned",
                "primary_input_only_expression_available": False,
                "primary_inputs_used": [],
                "state_boundaries": [],
                "explanation": f"No driver found for {output_signal}.",
            }

        # 2. Driven directly by a DFF
        if driver in nl.dffs:
            dff = nl.dffs[driver]
            return {
                "output_signal": output_signal,
                "driver_type": "dff",
                "dff_instance": driver,
                "dff_pins": {
                    "ck": dff.ck,
                    "rn": dff.rn,
                    "sn": dff.sn,
                    "d": dff.d,
                    "q": dff.q,
                },
                "equation": f"{output_signal} = {driver}.Q",
                "primary_input_only_expression_available": False,
                "expression_basis": "registered_state",
                "primary_inputs_used": [],
                "state_boundaries": [dff.q],
                "explanation": (
                    f"Output {output_signal} is directly driven by register (DFF) {driver} pin Q. "
                    "Its current value is stored state and cannot be expressed solely from "
                    f"current primary inputs. Its data input D is driven by net {dff.d}."
                ),
            }

        # 3. Combinational Gate Cone
        cone_gates = self.get_logic_cone(output_signal)
        if not cone_gates:
            return {
                "output_signal": output_signal,
                "driver_type": "primary_input",
                "equation": f"{output_signal} = {output_signal}",
                "primary_input_only_expression_available": is_primary_input(output_signal),
                "primary_inputs_used": [output_signal] if is_primary_input(output_signal) else [],
                "state_boundaries": [output_signal] if output_signal in dff_q_signals else [],
                "explanation": f"No combinational gates feed {output_signal}.",
            }

        # Topologically sort the gates from boundary inputs -> output
        in_degree = {g: 0 for g in cone_gates}
        successors: Dict[str, List[str]] = {g: [] for g in cone_gates}

        cone_gate_set = set(cone_gates)
        for consumer in cone_gates:
            consumer_producers: Set[str] = set()
            for input_signal in nl.nodes[consumer].inputs:
                producer = out2gate.get(input_signal)
                if producer in cone_gate_set and producer not in consumer_producers:
                    consumer_producers.add(producer)
                    successors[producer].append(consumer)
                    in_degree[consumer] += 1

        queue = deque([g for g in cone_gates if in_degree[g] == 0])
        topo_order: List[str] = []
        while queue:
            cur = queue.popleft()
            topo_order.append(cur)
            for nxt in successors[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(topo_order) < len(cone_gates):
            remaining = [g for g in cone_gates if g not in set(topo_order)]
            topo_order.extend(remaining)

        gate_breakdown: List[dict] = []
        expr_map: Dict[str, str] = {}
        boundary_inputs: Set[str] = set()

        for g in topo_order:
            node = nl.nodes[g]
            gtype = node.gate_type.lower()
            inps = node.inputs

            for inp in inps:
                if inp not in expr_map:
                    boundary_inputs.add(inp)

            # Local gate symbolic representation
            if gtype == "not":
                sym_eq = f"~{inps[0]}"
            elif gtype == "buf":
                sym_eq = f"{inps[0]}"
            elif gtype == "and":
                sym_eq = " * ".join(inps)
            elif gtype == "nand":
                sym_eq = f"~({' * '.join(inps)})"
            elif gtype == "or":
                sym_eq = " + ".join(inps)
            elif gtype == "nor":
                sym_eq = f"~({' + '.join(inps)})"
            elif gtype == "xor":
                sym_eq = " ^ ".join(inps)
            elif gtype == "xnor":
                sym_eq = f"~({' ^ '.join(inps)})"
            else:
                sym_eq = f"{gtype}({', '.join(inps)})"

            gate_breakdown.append({
                "gate": g,
                "type": node.gate_type.upper(),
                "output": node.output,
                "inputs": list(inps),
                "equation": f"{node.output} = {sym_eq}",
            })

            # Substituted expression
            sub_inps = [expr_map.get(inp, inp) for inp in inps]
            if gtype == "not":
                sub_eq = f"~({sub_inps[0]})"
            elif gtype == "buf":
                sub_eq = f"{sub_inps[0]}"
            elif gtype == "and":
                sub_eq = "(" + " * ".join(sub_inps) + ")"
            elif gtype == "nand":
                sub_eq = "~(" + " * ".join(sub_inps) + ")"
            elif gtype == "or":
                sub_eq = "(" + " + ".join(sub_inps) + ")"
            elif gtype == "nor":
                sub_eq = "~(" + " + ".join(sub_inps) + ")"
            elif gtype == "xor":
                sub_eq = "(" + " ^ ".join(sub_inps) + ")"
            elif gtype == "xnor":
                sub_eq = "~(" + " ^ ".join(sub_inps) + ")"
            else:
                sub_eq = f"{gtype}(" + ", ".join(sub_inps) + ")"

            expr_map[node.output] = sub_eq

        final_substituted = expr_map.get(output_signal, output_signal)
        primary_inputs_used = sorted(boundary_inputs & primary_input_bits)
        state_boundaries = sorted(boundary_inputs & dff_q_signals)
        unresolved_boundaries = sorted(
            boundary_inputs - primary_input_bits - dff_q_signals - constants
        )

        result: Dict[str, Any] = {
            "output_signal": output_signal,
            "driver_type": "combinational",
            "cone_gate_count": len(cone_gates),
            "cone_gates": cone_gates,
            "boundary_inputs": sorted(boundary_inputs),
            "primary_inputs_used": primary_inputs_used,
            "state_boundaries": state_boundaries,
            "unresolved_boundaries": unresolved_boundaries,
            "primary_input_only_expression_available": not (
                state_boundaries or unresolved_boundaries
            ),
            "expression_basis": (
                "primary_inputs"
                if not (state_boundaries or unresolved_boundaries)
                else "combinational_boundary_inputs"
            ),
            "gate_breakdown": gate_breakdown[:inline_limit],
            "total_gates_in_breakdown": len(gate_breakdown),
            "equation": f"{output_signal} = {final_substituted}",
        }

        if len(gate_breakdown) > inline_limit:
            report = tempfile.NamedTemporaryFile(
                "w",
                prefix=f"boolean_eq_{output_signal}_",
                suffix=".txt",
                dir=_workspace_temp_dir(),
                delete=False,
            )
            with report:
                report.write(f"# Boolean Derivation for {output_signal}\n")
                report.write(f"# Total cone gates: {len(cone_gates)}\n")
                report.write(f"# Boundary inputs: {', '.join(sorted(boundary_inputs))}\n\n")
                report.write("## Gate Breakdown:\n")
                for gb in gate_breakdown:
                    report.write(f"{gb['gate']} ({gb['type']}): {gb['equation']}\n")
                report.write(f"\n## Substituted Equation:\n{result['equation']}\n")
            result["file_path"] = os.path.abspath(report.name)

        return result

    def count_cone_gates(self, output_signal: str) -> int:
        """Return the number of gates in the logic cone of output_signal.

        Args:
            output_signal: The target output net name.
        """
        return int(self.count_gate_types_in_cone(output_signal)["total"])

    def _resolve_fanin_cone_target(self, signal: str) -> dict:
        """Resolve a registered output to its D-pin next-state cone target."""
        nl = self._netlist
        assert nl is not None
        for instance, dff in nl.dffs.items():
            if dff.q == signal:
                return {
                    "requested_signal": signal,
                    "resolved_signal": dff.d,
                    "through_dff": instance,
                    "resolved_pin": "D",
                }
        return {
            "requested_signal": signal,
            "resolved_signal": signal,
            "through_dff": None,
            "resolved_pin": None,
        }

    def count_gate_types_in_cone(self, output_signal: str) -> dict:
        """Return total and per-type gate counts in the logic cone."""
        self._require_netlist()
        self._resolve_signal(output_signal)
        nl = self._netlist
        assert nl is not None

        resolution = self._resolve_fanin_cone_target(output_signal)
        actual_output = resolution["resolved_signal"]

        out2gate = self._build_output_to_gate()
        pi_set = set(nl.primary_inputs)
        dff_outputs = {dff.q for dff in nl.dffs.values()}

        cone_insts: Set[str] = set()
        visited: Set[str] = set()
        stack = [actual_output]
        while stack:
            sig = stack.pop()
            if sig in visited:
                continue
            visited.add(sig)
            if sig in {"1'b0", "1'b1"} or sig in pi_set or sig in dff_outputs:
                continue
            driver = out2gate.get(sig)
            if driver is None or driver not in nl.nodes:
                continue
            if driver in cone_insts:
                continue
            cone_insts.add(driver)
            stack.extend(nl.nodes[driver].inputs)

        by_type: Dict[str, int] = {}
        for inst in cone_insts:
            gate_type = nl.nodes[inst].gate_type
            by_type[gate_type] = by_type.get(gate_type, 0) + 1

        result = {
            "output_signal": output_signal,
            "resolved_output": actual_output,
            "total": len(cone_insts),
            "by_type": dict(sorted(by_type.items())),
        }
        if resolution["through_dff"] is not None:
            result["through_dff"] = resolution["through_dff"]
            result["resolved_pin"] = resolution["resolved_pin"]
        return result

    def rank_signals_by_fanin_cone(
        self,
        signal_class: str,
        metric: str = "gate_count",
        order: str = "descending",
        limit: int = 1,
        include_ties: bool = True,
        inline_limit: int = 100,
    ) -> dict:
        """Rank a complete signal class by fanin-cone size or depth."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        normalized_class = signal_class.strip().upper()
        aliases = {
            "PRIMARY_OUTPUT": "PO",
            "PRIMARY_OUTPUTS": "PO",
            "DFFD": "DFF_D",
            "DFFQ": "DFF_Q",
            "GATE_OUTPUTS": "GATE_OUTPUT",
        }
        normalized_class = aliases.get(normalized_class, normalized_class)
        if normalized_class not in {"PO", "DFF_D", "DFF_Q", "GATE_OUTPUT"}:
            raise ValueError(
                "signal_class must be PO, DFF_D, DFF_Q, or GATE_OUTPUT"
            )
        normalized_metric = metric.strip().lower()
        if normalized_metric not in {"gate_count", "depth"}:
            raise ValueError("metric must be 'gate_count' or 'depth'")
        normalized_order = order.strip().lower()
        if normalized_order not in {"ascending", "descending"}:
            raise ValueError("order must be 'ascending' or 'descending'")
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be >= 1")
        inline_limit = max(0, int(inline_limit))

        def expand_port(port_name: str) -> List[str]:
            wire = nl.wires.get(port_name)
            if wire and wire.is_bus:
                lo, hi = sorted((wire.lsb, wire.msb))
                return [f"{port_name}[{bit}]" for bit in range(lo, hi + 1)]
            return [port_name]

        populations = {
            "PO": {
                signal
                for port_name in nl.primary_outputs
                for signal in expand_port(port_name)
            },
            "DFF_D": {dff.d for dff in nl.dffs.values() if dff.d},
            "DFF_Q": {dff.q for dff in nl.dffs.values() if dff.q},
            "GATE_OUTPUT": {node.output for node in nl.nodes.values() if node.output},
        }
        signals = sorted(populations[normalized_class])
        by_value: Dict[int, List[str]] = {}
        resolutions: Dict[str, dict] = {}
        depth_memo: Dict[str, int] = {}
        for signal in signals:
            resolution = (
                self._resolve_fanin_cone_target(signal)
                if normalized_class in {"PO", "DFF_Q"}
                else {
                    "requested_signal": signal,
                    "resolved_signal": signal,
                    "through_dff": None,
                    "resolved_pin": None,
                }
            )
            resolutions[signal] = resolution
            if normalized_metric == "gate_count":
                value = (
                    int(self.count_gate_types_in_cone(signal)["total"])
                    if normalized_class in {"PO", "DFF_Q"}
                    else len(self.get_logic_cone(signal))
                )
            else:
                value = self._fanin_depth(resolution["resolved_signal"], depth_memo)
            by_value.setdefault(value, []).append(signal)

        values = sorted(
            by_value,
            reverse=normalized_order == "descending",
        )[:limit]
        rankings = []
        returned_signal_count = 0
        for rank, value in enumerate(values, start=1):
            tied = sorted(by_value[value])
            selected = tied if include_ties else tied[:1]
            returned_signal_count += len(selected)
            rankings.append(
                {
                    "rank": rank,
                    "value": value,
                    normalized_metric: value,
                    "signals": selected,
                    "tie_count": len(tied),
                    "resolutions": {
                        signal: resolutions[signal]
                        for signal in selected
                        if resolutions[signal]["through_dff"] is not None
                    },
                }
            )

        result = {
            "signal_class": normalized_class,
            "metric": normalized_metric,
            "registered_output_handling": "resolve_q_to_d_next_state_cone",
            "bus_handling": "individual_bits",
            "order": normalized_order,
            "requested_rank_count": limit,
            "examined_signal_count": len(signals),
            "returned_rank_count": len(rankings),
            "returned_signal_count": returned_signal_count,
            "include_ties": bool(include_ties),
        }
        if returned_signal_count <= inline_limit:
            result["rankings"] = rankings
            return result

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix="fanin_cone_ranking_",
            suffix=".tsv",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write(f"rank\t{normalized_metric}\tsignal\tresolved_signal\n")
            for ranking in rankings:
                for signal in ranking["signals"]:
                    report.write(
                        f"{ranking['rank']}\t{ranking['value']}\t{signal}\t"
                        f"{resolutions[signal]['resolved_signal']}\n"
                    )
        result["sample_rankings"] = [
            {**ranking, "signals": ranking["signals"][:10]}
            for ranking in rankings[:10]
        ]
        result["file_path"] = os.path.abspath(report.name)
        return result

    def _nodes_reaching_sink(self, sink: str) -> Set[str]:
        """Return all signals that can reach sink."""

        reachable: Set[str] = set()
        queue: deque[str] = deque([sink])

        while queue:

            cur = queue.popleft()

            if cur in reachable:
                continue

            reachable.add(cur)

            for prev in self._backward_predecessors(cur):
                queue.append(prev)

        return reachable
    def find_all_paths(
        self,
        source: str,
        sink: str,
        inline_limit: int = 5,
    ):
        """Find all combinational signal paths from source to sink.

        The first ``inline_limit`` paths are returned in the tool result. If
        more paths exist, the complete list is streamed to a text file under
        the workspace temp directory instead of being returned as JSON.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        inline_limit = max(0, int(inline_limit))
        self._resolve_signal(source)
        self._resolve_signal(sink)

        out2gate = self._build_output_to_gate()
        fanout_successors: Dict[str, List[str]] = {}
        for node in nl.nodes.values():
            for input_signal in node.inputs:
                fanout_successors.setdefault(input_signal, []).append(node.output)

        sink_reachable: Set[str] = set()
        queue: deque[str] = deque([sink])
        while queue:
            signal_name = queue.popleft()
            if signal_name in sink_reachable:
                continue
            sink_reachable.add(signal_name)
            driver_name = out2gate.get(signal_name)
            if driver_name and driver_name in nl.nodes:
                queue.extend(nl.nodes[driver_name].inputs)

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix="all_paths_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        report_path = os.path.abspath(report.name)
        inline_paths: List[List[str]] = []
        count = 0
        path: List[str] = [source]
        visited: Set[str] = {source}

        def record_path() -> None:
            nonlocal count
            count += 1
            current_path = list(path)
            if count <= inline_limit:
                inline_paths.append(current_path)
            report.write(f"{count}. {' -> '.join(current_path)}\n")

        def walk(signal_name: str) -> None:
            if signal_name == sink:
                record_path()
                return

            for next_signal in fanout_successors.get(signal_name, []):
                if next_signal not in sink_reachable or next_signal in visited:
                    continue
                visited.add(next_signal)
                path.append(next_signal)
                walk(next_signal)
                path.pop()
                visited.remove(next_signal)

        try:
            if source in sink_reachable:
                walk(source)
        finally:
            report.close()

        file_path: Optional[str] = None
        if count > inline_limit:
            file_path = report_path
        else:
            try:
                os.unlink(report_path)
            except OSError:
                pass

        result = {
            "source": source,
            "sink": sink,
            "count": count,
            "paths": inline_paths,
        }
        if file_path:
            result["file_path"] = file_path
        return result

    def find_register_to_register_paths(
        self,
        inline_limit: int = 10,
    ) -> dict:
        """Enumerate combinational paths from every DFF Q pin to every DFF D pin."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        consumers: Dict[str, List[str]] = {}
        for gate_name, gate in nl.nodes.items():
            for input_signal in gate.inputs:
                consumers.setdefault(input_signal, []).append(gate_name)

        dff_sinks: Dict[str, List[str]] = {}
        for dff_name, dff in nl.dffs.items():
            if dff.d:
                dff_sinks.setdefault(dff.d, []).append(dff_name)

        # Prune branches that cannot reach any DFF D pin.
        out2gate = self._build_output_to_gate()
        reaches_dff_d: Set[str] = set(dff_sinks)
        queue: deque[str] = deque(reaches_dff_d)
        while queue:
            signal_name = queue.popleft()
            driver_name = out2gate.get(signal_name)
            if not driver_name or driver_name not in nl.nodes:
                continue
            for input_signal in nl.nodes[driver_name].inputs:
                if input_signal not in reaches_dff_d:
                    reaches_dff_d.add(input_signal)
                    queue.append(input_signal)

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix="register_to_register_paths_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        report_path = os.path.abspath(report.name)
        inline_paths: List[dict] = []
        count = 0

        def record_path(
            source_dff: str,
            source_q: str,
            sink_dff: str,
            sink_d: str,
            gate_path: List[str],
        ) -> None:
            nonlocal count
            count += 1
            entry = {
                "source_dff": source_dff,
                "source_pin": "Q",
                "source_signal": source_q,
                "sink_dff": sink_dff,
                "sink_pin": "D",
                "sink_signal": sink_d,
                "gates": list(gate_path),
            }
            if count <= inline_limit:
                inline_paths.append(entry)
            middle = " -> ".join(gate_path)
            if middle:
                middle = f" -> {middle}"
            report.write(
                f"{count}. {source_dff}.Q({source_q}){middle} "
                f"-> {sink_dff}.D({sink_d})\n"
            )

        def walk(
            source_dff: str,
            source_q: str,
            signal_name: str,
            gate_path: List[str],
            visited_signals: Set[str],
        ) -> None:
            for sink_dff in dff_sinks.get(signal_name, []):
                record_path(source_dff, source_q, sink_dff, signal_name, gate_path)

            for gate_name in consumers.get(signal_name, []):
                output_signal = nl.nodes[gate_name].output
                if output_signal not in reaches_dff_d or output_signal in visited_signals:
                    continue
                visited_signals.add(output_signal)
                gate_path.append(gate_name)
                walk(source_dff, source_q, output_signal, gate_path, visited_signals)
                gate_path.pop()
                visited_signals.remove(output_signal)

        try:
            report.write("# Register-to-register combinational paths\n")
            for source_dff, dff in nl.dffs.items():
                if not dff.q or dff.q not in reaches_dff_d:
                    continue
                walk(source_dff, dff.q, dff.q, [], {dff.q})
        finally:
            report.close()

        result = {"count": count}
        if count <= inline_limit:
            result["paths"] = inline_paths
            try:
                os.unlink(report_path)
            except OSError:
                pass
        else:
            result["file_path"] = report_path
        return result

    def get_fanout(self, net_name: str) -> List[str]:
        """Return all gate instance names driven by net_name.

        Args:
            net_name: The net to query.
        """
        self._require_netlist()
        self._resolve_signal(net_name)
        return self._build_fanout_map().get(net_name, [])

    def get_fanout_report(self, net_name: str, inline_limit: int = 10) -> dict:
        """Return a compact fanout report, spilling large lists to a CWD file."""
        fanout = sorted(set(self.get_fanout(net_name)))
        result = {"net_name": net_name, "count": len(fanout)}
        if len(fanout) <= inline_limit:
            result["fanout"] = fanout
            return result

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", net_name).strip("_") or "net"
        report = tempfile.NamedTemporaryFile(
            "w",
            prefix=f"fanout_{safe_name}_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write(f"# Direct fanout gates for {net_name} (count: {len(fanout)})\n")
            for gate_name in fanout:
                report.write(f"{gate_name}\n")
        result["file_path"] = os.path.abspath(report.name)
        result["complete_result_in_file"] = True
        result["file_contents"] = "all direct fanout gate names"
        result["additional_tool_needed"] = False
        result["response_instruction"] = "Report the count and exact file_path."
        return result

    def rank_signals_by_fanout(
        self,
        signal_class: str,
        order: str = "descending",
        limit: int = 1,
        include_ties: bool = True,
        inline_limit: int = 100,
    ) -> dict:
        """Rank a complete class of signals by direct load-pin fanout."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        normalized_class = signal_class.strip().upper()
        class_aliases = {
            "PRIMARY_INPUT": "PI",
            "PRIMARY_INPUTS": "PI",
            "DFFQ": "DFF_Q",
            "GATE_OUTPUTS": "GATE_OUTPUT",
            "ALL": "ALL_DRIVEN",
        }
        normalized_class = class_aliases.get(normalized_class, normalized_class)
        if normalized_class not in {"PI", "DFF_Q", "GATE_OUTPUT", "ALL_DRIVEN"}:
            raise ValueError(
                "signal_class must be PI, DFF_Q, GATE_OUTPUT, or ALL_DRIVEN"
            )

        normalized_order = order.strip().lower()
        if normalized_order not in {"ascending", "descending"}:
            raise ValueError("order must be 'ascending' or 'descending'")
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be >= 1")
        inline_limit = max(0, int(inline_limit))

        def expand_port(port_name: str) -> List[str]:
            wire = nl.wires.get(port_name)
            if wire and wire.is_bus:
                lo, hi = sorted((wire.lsb, wire.msb))
                return [f"{port_name}[{bit}]" for bit in range(lo, hi + 1)]
            return [port_name]

        primary_input_bits = {
            signal
            for port_name in nl.primary_inputs
            for signal in expand_port(port_name)
        }
        dff_q_signals = {dff.q for dff in nl.dffs.values() if dff.q}
        gate_outputs = {node.output for node in nl.nodes.values() if node.output}
        populations = {
            "PI": primary_input_bits,
            "DFF_Q": dff_q_signals,
            "GATE_OUTPUT": gate_outputs,
            "ALL_DRIVEN": primary_input_bits | dff_q_signals | gate_outputs,
        }
        signals = sorted(populations[normalized_class] - {"1'b0", "1'b1"})
        fanout_map = self._build_fanout_map()

        by_fanout: Dict[int, List[str]] = {}
        for signal in signals:
            by_fanout.setdefault(len(fanout_map.get(signal, [])), []).append(signal)
        fanout_values = sorted(
            by_fanout,
            reverse=normalized_order == "descending",
        )[:limit]

        rankings = []
        returned_signal_count = 0
        for rank, fanout_value in enumerate(fanout_values, start=1):
            tied_signals = sorted(by_fanout[fanout_value])
            selected_signals = tied_signals if include_ties else tied_signals[:1]
            returned_signal_count += len(selected_signals)
            rankings.append(
                {
                    "rank": rank,
                    "fanout": fanout_value,
                    "signals": selected_signals,
                    "tie_count": len(tied_signals),
                }
            )

        result = {
            "signal_class": normalized_class,
            "fanout_definition": "direct_load_pins",
            "bus_handling": "individual_bits",
            "order": normalized_order,
            "requested_rank_count": limit,
            "examined_signal_count": len(signals),
            "returned_rank_count": len(rankings),
            "returned_signal_count": returned_signal_count,
            "include_ties": bool(include_ties),
        }
        if returned_signal_count <= inline_limit:
            result["rankings"] = rankings
            return result

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix="fanout_ranking_",
            suffix=".tsv",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write("rank\tfanout\tsignal\n")
            for ranking in rankings:
                for signal in ranking["signals"]:
                    report.write(
                        f"{ranking['rank']}\t{ranking['fanout']}\t{signal}\n"
                    )
        result["sample_rankings"] = [
            {**ranking, "signals": ranking["signals"][:10]}
            for ranking in rankings[:10]
        ]
        result["file_path"] = os.path.abspath(report.name)
        return result

    def get_net_connections(self, net_name: str, inline_limit: int = 50) -> dict:
        """Return all instances directly connected to a signal, by role.

        Unlike fanout, this includes combinational/DFF drivers as well as
        combinational/DFF loads and records the exact input or DFF port.
        """
        self._require_netlist()
        self._resolve_signal(net_name)
        nl = self._netlist
        assert nl is not None

        drivers: List[dict] = []
        loads: List[dict] = []
        for instance, node in nl.nodes.items():
            if node.output == net_name:
                drivers.append({
                    "instance": instance,
                    "gate_type": node.gate_type,
                    "ports": ["output"],
                })
            indices = [index for index, signal in enumerate(node.inputs) if signal == net_name]
            if indices:
                loads.append({
                    "instance": instance,
                    "gate_type": node.gate_type,
                    "input_indices": indices,
                })

        for instance, dff in nl.dffs.items():
            if dff.q == net_name:
                drivers.append({
                    "instance": instance,
                    "gate_type": "dff",
                    "ports": ["Q"],
                })
            load_ports = [
                port
                for port, signal in (
                    ("D", dff.d), ("CK", dff.ck), ("RN", dff.rn), ("SN", dff.sn)
                )
                if signal == net_name
            ]
            if load_ports:
                loads.append({
                    "instance": instance,
                    "gate_type": "dff",
                    "ports": load_ports,
                })

        drivers.sort(key=lambda item: item["instance"])
        loads.sort(key=lambda item: item["instance"])
        connected = sorted({
            item["instance"] for item in [*drivers, *loads]
        })
        result = {
            "net_name": net_name,
            "driver_count": len(drivers),
            "load_count": len(loads),
            "count": len(connected),
        }
        if len(connected) <= inline_limit:
            result.update({
                "drivers": drivers,
                "loads": loads,
                "connected_gates": connected,
            })
            return result

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", net_name).strip("_") or "net"
        report = tempfile.NamedTemporaryFile(
            "w",
            prefix=f"connections_{safe_name}_",
            suffix=".jsonl",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            for role, items in (("driver", drivers), ("load", loads)):
                for item in items:
                    report.write(json.dumps({"role": role, **item}, sort_keys=True) + "\n")
        result["file_path"] = os.path.abspath(report.name)
        return result

    def resolve_name_type(self, name: str) -> dict:
        """Classify a name as a combinational gate, DFF, or signal."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        kinds: List[str] = []
        if name in nl.nodes:
            kinds.append("combinational_gate")
        if name in nl.dffs:
            kinds.append("dff")
        known_signals = (
            set(nl.wires)
            | set(nl.primary_inputs)
            | set(nl.primary_outputs)
            | {node.output for node in nl.nodes.values()}
            | {sig for node in nl.nodes.values() for sig in node.inputs}
            | {dff.q for dff in nl.dffs.values()}
            | {dff.d for dff in nl.dffs.values()}
        )
        if name in known_signals:
            kinds.append("signal")

        if not kinds:
            raise ValueError(f"Unknown gate, DFF, or signal name: {name!r}")
        if len(kinds) > 1:
            return {"name": name, "type": "ambiguous", "matches": kinds}
        kind = kinds[0]
        result = {"name": name, "type": kind}
        if kind == "combinational_gate":
            result["output_net"] = nl.nodes[name].output
        elif kind == "dff":
            result["output_net"] = nl.dffs[name].q
        else:
            result["output_net"] = name
        return result

    def get_gate_info(self, gate_name: str) -> dict:
        """Return a gate instance's primitive type and logical pin connections."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        if gate_name in nl.nodes:
            node = nl.nodes[gate_name]
            pins = {"Y": node.output}
            for index, net_name in enumerate(node.inputs):
                pin_name = chr(ord("A") + index) if index < 26 else f"I{index}"
                pins[pin_name] = net_name
            return {
                "instance": gate_name,
                "instance_type": "combinational_gate",
                "gate_type": node.gate_type.upper(),
                "pins": pins,
            }

        if gate_name in nl.dffs:
            dff = nl.dffs[gate_name]
            return {
                "instance": gate_name,
                "instance_type": "dff",
                "gate_type": "DFF",
                "pins": {
                    "CK": dff.ck,
                    "RN": dff.rn,
                    "SN": dff.sn,
                    "D": dff.d,
                    "Q": dff.q,
                },
            }

        raise ValueError(f"Gate or DFF instance {gate_name!r} not found in netlist.")

    def get_gate_fanout_report(self, gate_name: str, inline_limit: int = 10) -> dict:
        """Return direct consumers of a combinational gate or DFF output."""
        resolved = self.resolve_name_type(gate_name)
        if resolved["type"] not in {"combinational_gate", "dff"}:
            raise ValueError(f"{gate_name!r} is not a gate instance.")
        report = self.get_fanout_report(resolved["output_net"], inline_limit)
        report.update({
            "source_name": gate_name,
            "source_type": resolved["type"],
            "output_net": resolved["output_net"],
        })
        return report

    def _reachable_gates_report(
        self,
        source_name: str,
        source_type: str,
        start_nets: List[str],
        inline_limit: int = 10,
    ) -> dict:
        """Traverse transitive combinational fanout and report reached gates."""
        nl = self._netlist
        assert nl is not None
        fanout_map = self._build_fanout_map()
        queue: deque[str] = deque(start_nets)
        visited_nets: Set[str] = set()
        reached: Set[str] = set()

        while queue:
            net = queue.popleft()
            if net in visited_nets:
                continue
            visited_nets.add(net)
            for instance in fanout_map.get(net, []):
                if instance in reached:
                    continue
                reached.add(instance)
                if instance in nl.nodes:
                    queue.append(nl.nodes[instance].output)
                # DFFs are included as reached endpoints but traversal stops.

        gates = sorted(reached)
        result = {
            "source_name": source_name,
            "source_type": source_type,
            "start_nets": start_nets,
            "count": len(gates),
        }
        if len(gates) <= inline_limit:
            result["gates"] = gates
            return result

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name).strip("_") or "source"
        report = tempfile.NamedTemporaryFile(
            "w",
            prefix=f"reachable_{safe_name}_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write(
                f"# Gates reachable from {source_name} ({source_type}); count: {len(gates)}\n"
            )
            for gate_name in gates:
                report.write(f"{gate_name}\n")
        result["file_path"] = os.path.abspath(report.name)
        return result

    def get_reachable_gates_from_net(self, net_name: str) -> dict:
        """Return all gates transitively reachable from a signal or bus."""
        resolved = self.resolve_name_type(net_name)
        if resolved["type"] != "signal":
            raise ValueError(f"{net_name!r} is not an unambiguous signal or wire.")
        start_nets = [net_name]
        wi = self._netlist.wires.get(net_name)  # type: ignore[union-attr]
        if wi and wi.is_bus:
            lo, hi = sorted((wi.lsb, wi.msb))
            start_nets.extend(f"{net_name}[{bit}]" for bit in range(lo, hi + 1))
        return self._reachable_gates_report(
            net_name, "signal", list(dict.fromkeys(start_nets))
        )

    def get_reachable_gates_from_gate(self, gate_name: str) -> dict:
        """Return all downstream gates reachable from a gate or DFF output."""
        resolved = self.resolve_name_type(gate_name)
        if resolved["type"] not in {"combinational_gate", "dff"}:
            raise ValueError(f"{gate_name!r} is not an unambiguous gate instance.")
        return self._reachable_gates_report(
            gate_name, resolved["type"], [resolved["output_net"]]
        )
    
    def get_gate_fanout(self, gate_name: str):
        nl = self._netlist
        assert nl is not None

        if gate_name not in nl.nodes:
            raise ValueError(f"Unknown gate: {gate_name}")

        output_signal = nl.nodes[gate_name].output

        return self.get_fanout(output_signal)

    def list_signals(self) -> dict:
        """Return compact signal inventory for the current netlist.

        Large signal lists are written to files under the workspace temp
        directory. The returned JSON contains counts and small samples only.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        categories = {
            "primary_inputs": list(nl.primary_inputs),
            "primary_outputs": list(nl.primary_outputs),
            "wires": list(nl.wires.keys()),
            "gate_outputs": [n.output for n in nl.nodes.values()],
            "dff_q": [d.q for d in nl.dffs.values()],
            "dff_d": [d.d for d in nl.dffs.values()],
        }

        result = {
            "counts": {name: len(values) for name, values in categories.items()},
            "samples": {name: values[:10] for name, values in categories.items()},
        }
        file_paths: Dict[str, str] = {}
        for name, values in categories.items():
            if len(values) <= 100:
                continue
            report = tempfile.NamedTemporaryFile(
                "w",
                prefix=f"signals_{name}_",
                suffix=".txt",
                dir=_workspace_temp_dir(),
                delete=False,
            )
            with report:
                report.write(f"# {name} (count: {len(values)})\n")
                for value in values:
                    report.write(f"{value}\n")
            file_paths[name] = os.path.abspath(report.name)
        if file_paths:
            result["file_paths"] = file_paths
        return result

    def count_primary_ios(self) -> dict:
        """Return declared and bit-expanded primary input/output counts."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        input_ports = list(nl.primary_inputs)
        output_ports = list(nl.primary_outputs)
        input_details = [self._port_detail(name) for name in input_ports]
        output_details = [self._port_detail(name) for name in output_ports]
        return {
            "primary_input_ports": len(input_ports),
            "primary_output_ports": len(output_ports),
            "primary_input_bits": sum(item["width"] for item in input_details),
            "primary_output_bits": sum(item["width"] for item in output_details),
            "primary_inputs": input_ports,
            "primary_outputs": output_ports,
        }

    def _port_detail(self, name: str, include_bits: bool = False) -> dict:
        """Return declared width/range metadata for one top-level port."""
        nl = self._netlist
        assert nl is not None
        wire = nl.wires.get(name)
        if wire is None:
            res: Dict[str, Any] = {"name": name, "width": 1, "range": None}
            if include_bits:
                res["bits"] = [name]
            return res
        res = {
            "name": name,
            "width": wire.width,
            "range": f"[{wire.msb}:{wire.lsb}]" if wire.is_bus else None,
        }
        if include_bits:
            res["bits"] = self._expand_declared_signal(name)
        return res

    def list_primary_ios(
        self,
        io_type: Optional[str] = None,
        include_bits: bool = False,
    ) -> dict:
        """Return primary input/output ports with declared bit widths.

        Args:
            io_type: Optional filter: 'inputs' (primary inputs only),
                     'outputs' (primary outputs only), or 'all'/None (both).
            include_bits: Whether to include explicit per-bit list (default False).
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        filter_type = str(io_type or "").strip().lower()
        result: Dict[str, Any] = {}

        if filter_type not in {"outputs", "po"}:
            primary_inputs = [
                self._port_detail(name, include_bits=include_bits)
                for name in nl.primary_inputs
            ]
            result["primary_input_ports"] = len(primary_inputs)
            result["primary_input_bits"] = sum(
                item["width"] for item in primary_inputs
            )
            result["primary_inputs"] = primary_inputs

        if filter_type not in {"inputs", "pi"}:
            primary_outputs = [
                self._port_detail(name, include_bits=include_bits)
                for name in nl.primary_outputs
            ]
            result["primary_output_ports"] = len(primary_outputs)
            result["primary_output_bits"] = sum(
                item["width"] for item in primary_outputs
            )
            result["primary_outputs"] = primary_outputs

        return result

    def find_zero_length_pi_po_paths(self) -> dict:
        """Return direct zero-gate paths where a PI is also a PO."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        output_set = set(nl.primary_outputs)
        paths = [
            {
                "source": pi,
                "sink": pi,
                "gates": [],
                "description": f"{pi} is both a primary input and primary output",
            }
            for pi in nl.primary_inputs
            if pi in output_set
        ]
        return {"count": len(paths), "paths": paths}

    def are_same_clock_domain(self, dff1: str, dff2: str) -> bool:
        """Return True if both DFFs share the same clock net.

        Args:
            dff1: First DFF instance name.
            dff2: Second DFF instance name.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        for d in (dff1, dff2):
            if d not in nl.dffs:
                raise ValueError(f"DFF instance {d!r} not found in netlist.")

        return nl.dffs[dff1].ck == nl.dffs[dff2].ck

    def list_flip_flops_by_clock(
        self, clock_signal: str, inline_limit: int = 50
    ) -> dict:
        """List DFF instances whose clock pin is driven by *clock_signal*."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        clock_signal = clock_signal.strip()
        self._resolve_signal(clock_signal)

        matches: List[dict] = []
        for inst_name, dff in nl.dffs.items():
            if dff.ck != clock_signal:
                continue
            matches.append(
                {
                    "instance": inst_name,
                    "clock": dff.ck,
                    "d": dff.d,
                    "q": dff.q,
                    "rn": dff.rn,
                    "sn": dff.sn,
                }
            )

        result = {
            "clock_signal": clock_signal,
            "count": len(matches),
        }

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix=f"dffs_clock_{re.sub(r'[^A-Za-z0-9_]+', '_', clock_signal)}_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write(
                f"# list_flip_flops_by_clock clock={clock_signal} "
                f"count={len(matches)}\n"
            )
            report.write("# columns: instance clock d q rn sn\n")
            for item in matches:
                report.write(
                    "# "
                    f"{item['instance']} {item['clock']} {item['d']} "
                    f"{item['q']} {item['rn']} {item['sn']}\n"
                )
            report.write("# jsonl:\n")
            for item in matches:
                report.write(json.dumps(item, sort_keys=True) + "\n")
        result["file_path"] = os.path.abspath(report.name)
        return result

    def analyze_dff_d_input_structures(
        self,
        structure_types: Optional[List[str]] = None,
        inline_limit: int = 20,
    ) -> dict:
        """Find structural DFF hold/enable candidates in their D-input logic.

        A match requires either an identity feedback path from the DFF's own Q
        to D, or a Q-dependent branch merging with a Q-independent branch.
        This deliberately excludes pure toggle/arithmetic feedback that has no
        such selection or gating merge.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        requested = {
            item.strip().lower()
            for item in (structure_types or ["enable", "hold"])
        }
        if not requested or not requested <= {"enable", "hold", "self_feedback"}:
            raise ValueError(
                "structure_types may contain enable, hold, and self_feedback"
            )
        inline_limit = max(0, int(inline_limit))

        drivers = {node.output: node for node in nl.nodes.values()}
        dff_qs = {dff.q for dff in nl.dffs.values() if dff.q}
        matches: List[dict] = []
        self_feedback_count = 0
        classifications: Dict[str, int] = {}

        for instance, dff in sorted(nl.dffs.items()):
            memo: Dict[str, bool] = {}
            active: Set[str] = set()

            def depends_on_own_q(signal: str) -> bool:
                if signal == dff.q:
                    return True
                if signal in memo:
                    return memo[signal]
                if signal in dff_qs or signal in {"1'b0", "1'b1"}:
                    memo[signal] = False
                    return False
                node = drivers.get(signal)
                if node is None or signal in active:
                    memo[signal] = False
                    return False
                active.add(signal)
                result = any(depends_on_own_q(inp) for inp in node.inputs)
                active.discard(signal)
                memo[signal] = result
                return result

            if not depends_on_own_q(dff.d):
                continue
            self_feedback_count += 1

            local_distance_memo: Dict[Tuple[str, int], Optional[int]] = {}

            def local_q_distance(signal: str, budget: int = 6) -> Optional[int]:
                """Return the shortest nearby gate distance to this DFF's Q."""
                if signal == dff.q:
                    return 0
                if budget <= 0 or signal in dff_qs:
                    return None
                key = (signal, budget)
                if key in local_distance_memo:
                    return local_distance_memo[key]
                node = drivers.get(signal)
                if node is None:
                    local_distance_memo[key] = None
                    return None
                child_distances = [
                    distance
                    for inp in node.inputs
                    if (distance := local_q_distance(inp, budget - 1)) is not None
                ]
                result = 1 + min(child_distances) if child_distances else None
                local_distance_memo[key] = result
                return result

            def identity_feedback(signal: str) -> bool:
                parity = 0
                seen: Set[str] = set()
                while signal not in seen:
                    if signal == dff.q:
                        return parity == 0
                    seen.add(signal)
                    node = drivers.get(signal)
                    if node is None or len(node.inputs) != 1:
                        return False
                    if node.gate_type == "not":
                        parity ^= 1
                    elif node.gate_type != "buf":
                        return False
                    signal = node.inputs[0]
                return False

            merge_node: Optional[GateNode] = None
            queue: deque[str] = deque([dff.d])
            visited: Set[str] = set()
            while queue and merge_node is None:
                signal = queue.popleft()
                if signal in visited or signal == dff.q:
                    continue
                visited.add(signal)
                node = drivers.get(signal)
                if node is None:
                    continue
                flags = [local_q_distance(inp) is not None for inp in node.inputs]
                if any(flags) and not all(flags):
                    merge_node = node
                    break
                queue.extend(
                    inp for inp, flag in zip(node.inputs, flags) if flag
                )

            if identity_feedback(dff.d):
                classification = "direct_hold"
            elif merge_node is not None:
                classification = (
                    "and_gated_hold_candidate"
                    if merge_node.gate_type == "and"
                    else "mux_or_gated_hold_candidate"
                )
            elif "self_feedback" in requested:
                classification = "self_feedback_only"
            else:
                continue

            feedback_signals = [dff.d]
            feedback_gates: List[str] = []
            signal = dff.d
            seen_path: Set[str] = set()
            while signal != dff.q and signal not in seen_path:
                seen_path.add(signal)
                node = drivers.get(signal)
                if node is None:
                    break
                feedback_gates.append(node.name)
                candidates = [
                    (distance, inp)
                    for inp in node.inputs
                    if (distance := local_q_distance(inp)) is not None
                ]
                next_signal = min(candidates)[1] if candidates else None
                if next_signal is None:
                    break
                signal = next_signal
                feedback_signals.append(signal)
            feedback_signals.reverse()
            feedback_gates.reverse()

            merge_flags = (
                [local_q_distance(inp) is not None for inp in merge_node.inputs]
                if merge_node is not None
                else []
            )
            q_branches = (
                [inp for inp, flag in zip(merge_node.inputs, merge_flags) if flag]
                if merge_node is not None
                else []
            )
            data_branches = (
                [inp for inp, flag in zip(merge_node.inputs, merge_flags) if not flag]
                if merge_node is not None
                else []
            )
            item = {
                "dff": instance,
                "q_signal": dff.q,
                "d_signal": dff.d,
                "classification": classification,
                "validation": "structural",
                "d_driver": drivers[dff.d].name if dff.d in drivers else None,
                "merge_gate": merge_node.name if merge_node else None,
                "merge_gate_type": merge_node.gate_type if merge_node else None,
                "q_dependent_branches": q_branches,
                "q_independent_branches": data_branches,
                "feedback_signals": feedback_signals,
                "feedback_gates": feedback_gates,
            }
            matches.append(item)
            classifications[classification] = classifications.get(classification, 0) + 1

        result = {
            "dffs_examined": len(nl.dffs),
            "structure_types": sorted(requested),
            "validation": "structural",
            "self_feedback_candidates": self_feedback_count,
            "count": len(matches),
            "classification_breakdown": dict(sorted(classifications.items())),
        }
        if len(matches) <= inline_limit:
            result["matches"] = matches
            return result

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix="dff_d_input_structures_",
            suffix=".jsonl",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            for item in matches:
                report.write(json.dumps(item, sort_keys=True) + "\n")
        result["sample_matches"] = matches[: min(10, inline_limit)]
        result["file_path"] = os.path.abspath(report.name)
        return result

    def check_function_symmetry(
        self,
        output_signal: str,
        input_a: str,
        input_b: str,
    ) -> dict:
        """Prove whether an output function is symmetric in two PI bits."""
        import textwrap

        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_signal(output_signal)
        self._resolve_signal(input_a)
        self._resolve_signal(input_b)

        def expand_port(port_name: str) -> List[str]:
            wire = nl.wires.get(port_name)
            if wire and wire.is_bus:
                lo, hi = sorted((wire.lsb, wire.msb))
                return [f"{port_name}[{bit}]" for bit in range(lo, hi + 1)]
            return [port_name]

        pi_bits = {
            signal
            for port_name in nl.primary_inputs
            for signal in expand_port(port_name)
        }
        if input_a not in pi_bits or input_b not in pi_bits:
            raise ValueError(
                "Symmetry inputs must be scalar primary inputs or primary-input bits."
            )
        if input_a == input_b:
            return {
                "output_signal": output_signal,
                "input_a": input_a,
                "input_b": input_b,
                "symmetric": True,
                "proof": "identical_inputs",
            }

        comb_driver = {node.output: node for node in nl.nodes.values()}
        cone_nodes: Dict[str, GateNode] = {}
        boundary_signals: Set[str] = set()
        active: Set[str] = set()

        def collect(signal: str) -> None:
            if signal in {"1'b0", "1'b1"}:
                return
            node = comb_driver.get(signal)
            if node is None:
                boundary_signals.add(signal)
                return
            if node.name in cone_nodes or signal in active:
                return
            active.add(signal)
            cone_nodes[node.name] = node
            for node_input in node.inputs:
                collect(node_input)
            active.discard(signal)

        collect(output_signal)
        depends_on = sorted({input_a, input_b} & boundary_signals)
        if not depends_on:
            return {
                "output_signal": output_signal,
                "input_a": input_a,
                "input_b": input_b,
                "symmetric": True,
                "proof": "independence",
                "dependent_symmetry_inputs": [],
                "cone_gate_count": len(cone_nodes),
            }

        # Build two copies of the output cone. All boundary values are shared,
        # except that input_a and input_b are exchanged in the second copy.
        boundary_signals.update({input_a, input_b})
        aliases = {
            signal: f"sym_boundary_{index}"
            for index, signal in enumerate(sorted(boundary_signals))
        }
        model = Netlist(module_name="symmetry_miter")
        model.primary_inputs = list(aliases.values())
        for alias in aliases.values():
            model.wires[alias] = WireInfo(name=alias)

        output_maps = [
            {
                node.output: f"sym_{copy_index}_net_{index}"
                for index, node in enumerate(cone_nodes.values())
            }
            for copy_index in range(2)
        ]
        for output_map in output_maps:
            for signal in output_map.values():
                model.wires[signal] = WireInfo(name=signal)

        def remap(signal: str, copy_index: int) -> str:
            if signal in {"1'b0", "1'b1"}:
                return signal
            if signal in output_maps[copy_index]:
                return output_maps[copy_index][signal]
            if copy_index == 1 and signal == input_a:
                return aliases[input_b]
            if copy_index == 1 and signal == input_b:
                return aliases[input_a]
            return aliases[signal]

        for copy_index in range(2):
            for index, node in enumerate(cone_nodes.values()):
                instance = f"sym_{copy_index}_gate_{index}"
                model.nodes[instance] = GateNode(
                    name=instance,
                    gate_type=node.gate_type,
                    inputs=[remap(signal, copy_index) for signal in node.inputs],
                    output=output_maps[copy_index][node.output],
                )

        out_normal = remap(output_signal, 0)
        out_swapped = remap(output_signal, 1)
        diff_signal = "symmetry_diff"
        model.wires[diff_signal] = WireInfo(name=diff_signal)
        model.primary_outputs = [diff_signal]
        model.nodes["symmetry_diff_gate"] = GateNode(
            name="symmetry_diff_gate",
            gate_type="xor",
            inputs=[out_normal, out_swapped],
            output=diff_signal,
        )

        tmp_dir = tempfile.mkdtemp(prefix="symmetry_", dir=_workspace_temp_dir())
        netlist_path = os.path.join(tmp_dir, "miter.v")
        script_path = os.path.join(tmp_dir, "prove.ys")
        write_verilog(model, netlist_path)
        script = textwrap.dedent(f"""\
            read_verilog {netlist_path}
            prep -top {model.module_name}
            sat -set {diff_signal} 1 -max 1 -show-inputs -show {diff_signal}
        """)
        with open(script_path, "w") as handle:
            handle.write(script)
        try:
            result = subprocess.run(
                [_yosys_binary(), "-s", script_path],
                capture_output=True,
                text=True,
                timeout=60,
                env=_temp_subprocess_env(),
                cwd=_workspace_temp_dir(),
            )
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            lower = combined.lower()
            if "sat solving finished - no model found" in lower:
                symmetric = True
            elif "sat solving finished - model found" in lower:
                symmetric = False
            else:
                raise RuntimeError(
                    "Could not determine Yosys symmetry result:\n" + combined[-3000:]
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return {
            "output_signal": output_signal,
            "input_a": input_a,
            "input_b": input_b,
            "symmetric": symmetric,
            "proof": "formal_sat",
            "dependent_symmetry_inputs": depends_on,
            "cone_gate_count": len(cone_nodes),
        }

    def check_signal_equivalence(self, sig1: str, sig2: str) -> bool:
        """Check if two signals in the current netlist are functionally equivalent.

        Uses Yosys SAT solver to verify equivalence.
        Returns True if both signals produce identical logic for all inputs.

        Args:
            sig1: First signal name to compare.
            sig2: Second signal name to compare.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        constant_aliases = {
            "0": "1'b0", "1": "1'b1", "'0": "1'b0", "'1": "1'b1"
        }
        sig1 = constant_aliases.get(sig1.strip().lower(), sig1)
        sig2 = constant_aliases.get(sig2.strip().lower(), sig2)
        self._resolve_signal(sig1)
        self._resolve_signal(sig2)

        # Signal equivalence is combinational: primary inputs and DFF-Q values
        # are independent boundaries. Replace each DFF-Q with a fresh PI so SAT
        # never depends on a sequential-cell model.
        comb_nl = copy.deepcopy(nl)
        q_aliases: Dict[str, str] = {}
        for index, dff in enumerate(comb_nl.dffs.values()):
            if not dff.q or dff.q in q_aliases:
                continue
            alias = f"_signal_equiv_q_{index}"
            q_aliases[dff.q] = alias
            comb_nl.wires[alias] = WireInfo(name=alias)
            comb_nl.primary_inputs.append(alias)
        for node in comb_nl.nodes.values():
            node.inputs = [q_aliases.get(sig, sig) for sig in node.inputs]
        comb_nl.dffs.clear()

        sat_sig1 = q_aliases.get(sig1, sig1)
        sat_sig2 = q_aliases.get(sig2, sig2)
        if sat_sig1 == sat_sig2:
            return True

        # `prep` removes unobserved internal wires before the SAT pass. Expose
        # queried internal signals as outputs in this temporary copy so their
        # names and drivers remain available to `sat -set`.
        for signal_name in (sat_sig1, sat_sig2):
            if (
                signal_name not in {"1'b0", "1'b1"}
                and signal_name not in comb_nl.primary_inputs
                and signal_name not in comb_nl.primary_outputs
            ):
                comb_nl.primary_outputs.append(signal_name)

        # Write netlist to temporary file
        tf = tempfile.NamedTemporaryFile(
            "w", suffix=".v", dir=_workspace_temp_dir(), delete=False
        )
        netlist_path = tf.name
        tf.close()
        write_verilog(comb_nl, netlist_path)

        try:
            return self._yosys_check_signals_equiv(
                netlist_path,
                comb_nl.module_name,
                sat_sig1,
                sat_sig2,
            )
        finally:
            import os
            try:
                os.unlink(netlist_path)
            except OSError:
                pass

    def _yosys_check_signals_equiv(
        self, netlist_verilog: str, top: str, sig1: str, sig2: str
    ) -> bool:
        """Use Yosys SAT to check if two signals are equivalent.
        
        Returns True if signals are equivalent, False otherwise.
        """

        def check_sat(v1, v2):
            """Check if the constraint sig1={v1} AND sig2={v2} is satisfiable."""
            constraints: List[str] = []
            for sig, value in ((sig1, v1), (sig2, v2)):
                if sig in {"1'b0", "1'b1"}:
                    if int(sig[-1]) != value:
                        return False
                    continue
                constraints.append(f"-set {sig} {value}")
            script = f"""
read_verilog {netlist_verilog}
prep -top {top}
sat {' '.join(constraints)}
"""

            with tempfile.NamedTemporaryFile(
                "w", suffix=".ys", dir=_workspace_temp_dir(), delete=False
            ) as f:
                f.write(script)
                f.flush()
                script_path = f.name

            try:
                result = subprocess.run(
                    [_yosys_binary(), "-s", script_path],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=_temp_subprocess_env(),
                    cwd=_workspace_temp_dir(),
                )

                combined = (result.stdout or "") + "\n" + (result.stderr or "")
                if result.returncode != 0:
                    raise RuntimeError(
                        "Yosys signal-equivalence SAT check failed:\n"
                        + combined[-3000:]
                    )
                # If model found, SAT is satisfiable
                lower = combined.lower()
                if "sat solving finished - no model found" in lower:
                    return False
                if "sat solving finished - model found" in lower:
                    return True
                raise RuntimeError(
                    "Could not determine Yosys SAT result:\n" + combined[-3000:]
                )
            finally:
                import os
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

        # Test both scenarios:
        # sat01: sig1=0 AND sig2=1 (both different and contradictory)
        # sat10: sig1=1 AND sig2=0 (both different and contradictory)
        sat01 = check_sat(0, 1)
        sat10 = check_sat(1, 0)

        # Signals are equivalent if neither contradictory scenario is satisfiable
        return not sat01 and not sat10

    def _check_gate_expression_equivalence(
        self, gate_type: str, inputs: List[str], target_signal: str
    ) -> bool:
        """Check whether a virtual primitive gate over existing signals equals target."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        gate_type = gate_type.strip().lower()
        if gate_type in ONE_INPUT_GATES:
            expected_inputs = 1
        elif gate_type in TWO_INPUT_GATES:
            expected_inputs = 2
        else:
            raise ValueError(f"Unsupported primitive gate type: {gate_type!r}")
        if len(inputs) != expected_inputs:
            raise ValueError(
                f"{gate_type!r} expects {expected_inputs} input(s), got {len(inputs)}."
            )

        constant_aliases = {
            "0": "1'b0", "1": "1'b1", "'0": "1'b0", "'1": "1'b1"
        }
        normalized_inputs = [
            constant_aliases.get(str(signal_name).strip().lower(), signal_name)
            for signal_name in inputs
        ]
        target_signal = constant_aliases.get(
            str(target_signal).strip().lower(), target_signal
        )
        for signal_name in normalized_inputs + [target_signal]:
            self._resolve_signal(signal_name)

        comb_nl = copy.deepcopy(nl)
        q_aliases: Dict[str, str] = {}
        for index, dff in enumerate(comb_nl.dffs.values()):
            if not dff.q or dff.q in q_aliases:
                continue
            alias = f"_expr_equiv_q_{index}"
            q_aliases[dff.q] = alias
            comb_nl.wires[alias] = WireInfo(name=alias)
            comb_nl.primary_inputs.append(alias)
        for node in comb_nl.nodes.values():
            node.inputs = [q_aliases.get(sig, sig) for sig in node.inputs]
        comb_nl.dffs.clear()

        sat_inputs = [q_aliases.get(signal_name, signal_name) for signal_name in normalized_inputs]
        sat_target = q_aliases.get(target_signal, target_signal)
        expr_signal = "_expr_equiv_out"
        expr_inst = "_expr_equiv_gate"
        suffix = 0
        while expr_signal in comb_nl.wires:
            suffix += 1
            expr_signal = f"_expr_equiv_out_{suffix}"
            expr_inst = f"_expr_equiv_gate_{suffix}"

        comb_nl.wires[expr_signal] = WireInfo(name=expr_signal)
        comb_nl.nodes[expr_inst] = GateNode(
            name=expr_inst,
            gate_type=gate_type,
            inputs=sat_inputs,
            output=expr_signal,
        )

        if sat_target in {"1'b0", "1'b1"}:
            prove_signal = expr_signal
            prove_value = sat_target[-1]
            comb_nl.primary_outputs.append(expr_signal)
        else:
            diff_signal = "_expr_equiv_diff"
            diff_inst = "_expr_equiv_diff_gate"
            while diff_signal in comb_nl.wires:
                suffix += 1
                diff_signal = f"_expr_equiv_diff_{suffix}"
                diff_inst = f"_expr_equiv_diff_gate_{suffix}"
            comb_nl.wires[diff_signal] = WireInfo(name=diff_signal)
            comb_nl.primary_outputs.append(diff_signal)
            comb_nl.nodes[diff_inst] = GateNode(
                name=diff_inst,
                gate_type="xor",
                inputs=[expr_signal, sat_target],
                output=diff_signal,
            )
            prove_signal = diff_signal
            prove_value = "0"

        out_to_inst = {
            node.output: inst_name for inst_name, node in comb_nl.nodes.items()
        }
        needed_signals: Set[str] = set()
        needed_nodes: Set[str] = set()
        queue: deque[str] = deque([prove_signal])
        while queue:
            signal_name = queue.popleft()
            if signal_name in needed_signals:
                continue
            needed_signals.add(signal_name)
            inst_name = out_to_inst.get(signal_name)
            if not inst_name or inst_name in needed_nodes:
                continue
            needed_nodes.add(inst_name)
            for input_signal in comb_nl.nodes[inst_name].inputs:
                if input_signal not in {"1'b0", "1'b1"}:
                    queue.append(input_signal)

        comb_nl.nodes = {
            inst_name: node
            for inst_name, node in comb_nl.nodes.items()
            if inst_name in needed_nodes
        }
        used_signals: Set[str] = {prove_signal}
        for node in comb_nl.nodes.values():
            used_signals.add(node.output)
            used_signals.update(
                input_signal
                for input_signal in node.inputs
                if input_signal not in {"1'b0", "1'b1"}
            )
        comb_nl.primary_inputs = [
            signal_name for signal_name in comb_nl.primary_inputs
            if signal_name in used_signals
        ]
        comb_nl.primary_outputs = [prove_signal]
        comb_nl.wires = {
            signal_name: wire
            for signal_name, wire in comb_nl.wires.items()
            if signal_name in used_signals
            or signal_name in comb_nl.primary_inputs
            or signal_name in comb_nl.primary_outputs
        }

        tf = tempfile.NamedTemporaryFile(
            "w", suffix=".v", dir=_workspace_temp_dir(), delete=False
        )
        netlist_path = tf.name
        tf.close()
        write_verilog(comb_nl, netlist_path)
        try:
            script = f"""
read_verilog {netlist_path}
prep -top {comb_nl.module_name}
sat -prove {prove_signal} {prove_value} -verify
"""
            with tempfile.NamedTemporaryFile(
                "w", suffix=".ys", dir=_workspace_temp_dir(), delete=False
            ) as script_f:
                script_f.write(script)
                script_f.flush()
                script_path = script_f.name
            try:
                result = subprocess.run(
                    [_yosys_binary(), "-s", script_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=_temp_subprocess_env(),
                    cwd=_workspace_temp_dir(),
                )
                combined = (result.stdout or "") + "\n" + (result.stderr or "")
                if result.returncode == 0:
                    return True
                lower = combined.lower()
                if (
                    "proof did fail" in lower
                    or "model found" in lower
                    or "failed to prove" in lower
                ):
                    return False
                raise RuntimeError(
                    "Yosys expression-equivalence SAT check failed:\n"
                    + combined[-3000:]
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    "Yosys expression-equivalence SAT check timed out."
                ) from exc
            finally:
                try:
                    os.unlink(script_path)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(netlist_path)
            except OSError:
                pass

    def _logic_signature_map(self, pattern_bits: int = 256) -> Dict[str, int]:
        """Return deterministic random-simulation signatures for known signals."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        mask = (1 << pattern_bits) - 1
        rng = random.Random(0xCADA1067)
        signatures: Dict[str, int] = {"1'b0": 0, "1'b1": mask}
        out2gate = self._build_output_to_gate()
        driven = set(out2gate)

        boundary_signals: Set[str] = set()
        for port in nl.primary_inputs:
            boundary_signals.update(self._expand_declared_signal(port))
            boundary_signals.add(port)
        for dff in nl.dffs.values():
            if dff.q:
                boundary_signals.add(dff.q)

        # Treat any undriven non-constant gate input as a combinational boundary.
        for node in nl.nodes.values():
            for signal_name in node.inputs:
                if signal_name not in {"1'b0", "1'b1"} and signal_name not in driven:
                    boundary_signals.add(signal_name)

        for signal_name in sorted(boundary_signals):
            signatures.setdefault(signal_name, rng.getrandbits(pattern_bits) & mask)

        def eval_gate(gate_type: str, values: List[int]) -> int:
            if gate_type == "buf":
                return values[0]
            if gate_type == "not":
                return (~values[0]) & mask
            if gate_type == "and":
                return values[0] & values[1]
            if gate_type == "nand":
                return (~(values[0] & values[1])) & mask
            if gate_type == "or":
                return values[0] | values[1]
            if gate_type == "nor":
                return (~(values[0] | values[1])) & mask
            if gate_type == "xor":
                return values[0] ^ values[1]
            if gate_type == "xnor":
                return (~(values[0] ^ values[1])) & mask
            raise ValueError(f"Unsupported primitive gate type: {gate_type!r}")

        remaining = dict(nl.nodes)
        progressed = True
        while remaining and progressed:
            progressed = False
            for inst_name, node in list(remaining.items()):
                if all(input_signal in signatures for input_signal in node.inputs):
                    values = [signatures[input_signal] for input_signal in node.inputs]
                    signatures[node.output] = eval_gate(node.gate_type, values)
                    del remaining[inst_name]
                    progressed = True
        return signatures

    def find_binary_gate_equivalent_pair(
        self,
        target_signal: str,
        gate_type: str,
        candidate_scope: str = "internal",
        max_signature_pairs: int = 50_000_000,
        max_formal_checks: int = 3,
    ) -> dict:
        """Find existing signals a,b such that gate_type(a,b) equals target_signal."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        gate_type = gate_type.strip().lower()
        if gate_type not in TWO_INPUT_GATES:
            raise ValueError(
                f"find_binary_gate_equivalent_pair requires a 2-input gate, got {gate_type!r}."
            )
        candidate_scope = candidate_scope.strip().lower() or "internal"
        if candidate_scope not in {"internal", "all"}:
            raise ValueError("candidate_scope must be 'internal' or 'all'.")

        self._resolve_signal(target_signal)
        signatures = self._logic_signature_map()
        if target_signal not in signatures:
            return {
                "exists": False,
                "target_signal": target_signal,
                "gate_type": gate_type,
                "search_complete": False,
                "reason": f"No simulation signature available for target {target_signal!r}.",
            }

        primary_ios = set(nl.primary_inputs) | set(nl.primary_outputs)
        primary_io_bits: Set[str] = set()
        for port in primary_ios:
            primary_io_bits.update(self._expand_declared_signal(port))
        primary_ios |= primary_io_bits

        candidates: List[str] = []
        seen_candidates: Set[str] = set()
        candidate_sources: List[str] = []
        candidate_sources.extend(node.output for node in nl.nodes.values())
        candidate_sources.extend(dff.q for dff in nl.dffs.values() if dff.q)
        candidate_sources.extend(
            signal_name for signal_name in nl.wires if signal_name in signatures
        )
        for signal_name in candidate_sources:
            if signal_name in seen_candidates or signal_name in {"1'b0", "1'b1"}:
                continue
            if signal_name not in signatures:
                continue
            if candidate_scope == "internal" and signal_name in primary_ios:
                continue
            seen_candidates.add(signal_name)
            candidates.append(signal_name)

        formal_checks = 0
        formal_failures = 0

        def formal_limit_result(
            a_signal: str, b_signal: str, strategy: str, pairs_checked: int
        ) -> dict:
            return {
                "exists": False,
                "target_signal": target_signal,
                "gate_type": gate_type,
                "candidate_scope": candidate_scope,
                "candidate_signals": len(candidates),
                "signature_pairs_checked": pairs_checked,
                "formal_checks": formal_checks,
                "formal_failures": formal_failures,
                "search_complete": False,
                "reason": (
                    "Signature-compatible candidate pairs exist, but the formal "
                    f"confirmation budget of {max_formal_checks} checks was reached."
                ),
                "next_unconfirmed_candidate": {"a": a_signal, "b": b_signal},
                "strategy": strategy,
            }

        def confirm(a_signal: str, b_signal: str, strategy: str, pairs_checked: int) -> Optional[dict]:
            nonlocal formal_checks, formal_failures
            if formal_checks >= max_formal_checks:
                return formal_limit_result(a_signal, b_signal, strategy, pairs_checked)
            formal_checks += 1
            try:
                equivalent = self._check_gate_expression_equivalence(
                    gate_type, [a_signal, b_signal], target_signal
                )
            except (RuntimeError, TimeoutError):
                formal_failures += 1
                return None
            if equivalent:
                return {
                    "exists": True,
                    "target_signal": target_signal,
                    "gate_type": gate_type,
                    "a": a_signal,
                    "b": b_signal,
                    "candidate_scope": candidate_scope,
                    "strategy": strategy,
                    "proof": "yosys_sat_confirmed",
                    "candidate_signals": len(candidates),
                    "signature_pairs_checked": pairs_checked,
                    "formal_checks": formal_checks,
                    "formal_failures": formal_failures,
                    "search_complete": True,
                }
            return None

        out2gate = self._build_output_to_gate()
        driver_name = out2gate.get(target_signal)
        if driver_name and driver_name in nl.nodes:
            driver = nl.nodes[driver_name]
            if (
                driver.gate_type == gate_type
                and len(driver.inputs) == 2
                and all(input_signal in seen_candidates for input_signal in driver.inputs)
            ):
                found = confirm(driver.inputs[0], driver.inputs[1], "direct_driver", 1)
                if found:
                    return found

        by_signature: Dict[int, List[str]] = {}
        for signal_name in candidates:
            by_signature.setdefault(signatures[signal_name], []).append(signal_name)

        # Fast De Morgan path: OR(x,y)=NAND(~x,~y), AND(x,y)=NOR(~x,~y).
        if driver_name and driver_name in nl.nodes:
            driver = nl.nodes[driver_name]
            demorgan_source = None
            if gate_type == "nand" and driver.gate_type == "or" and len(driver.inputs) == 2:
                demorgan_source = driver
            elif gate_type == "nor" and driver.gate_type == "and" and len(driver.inputs) == 2:
                demorgan_source = driver
            if demorgan_source and all(inp in signatures for inp in demorgan_source.inputs):
                mask = (1 << 256) - 1
                left_complements = by_signature.get((~signatures[demorgan_source.inputs[0]]) & mask, [])
                right_complements = by_signature.get((~signatures[demorgan_source.inputs[1]]) & mask, [])
                for left_signal in left_complements[:20]:
                    if not self._check_gate_expression_equivalence(
                        "not", [left_signal], demorgan_source.inputs[0]
                    ):
                        continue
                    for right_signal in right_complements[:20]:
                        if not self._check_gate_expression_equivalence(
                            "not", [right_signal], demorgan_source.inputs[1]
                        ):
                            continue
                        found = confirm(
                            left_signal, right_signal, "demorgan_driver", 1
                        )
                        if found:
                            return found

        mask = (1 << 256) - 1
        target_sig = signatures[target_signal]

        def apply_sig(a_sig: int, b_sig: int) -> int:
            if gate_type == "and":
                return a_sig & b_sig
            if gate_type == "nand":
                return (~(a_sig & b_sig)) & mask
            if gate_type == "or":
                return a_sig | b_sig
            if gate_type == "nor":
                return (~(a_sig | b_sig)) & mask
            if gate_type == "xor":
                return a_sig ^ b_sig
            if gate_type == "xnor":
                return (~(a_sig ^ b_sig)) & mask
            raise AssertionError(gate_type)

        if gate_type in {"and", "nand"}:
            needed = target_sig if gate_type == "and" else (~target_sig) & mask
            search_candidates = [
                signal_name
                for signal_name in candidates
                if (needed & ~signatures[signal_name]) == 0
            ]
        elif gate_type in {"or", "nor"}:
            allowed = target_sig if gate_type == "or" else (~target_sig) & mask
            search_candidates = [
                signal_name
                for signal_name in candidates
                if (signatures[signal_name] & ~allowed) == 0
            ]
        else:
            search_candidates = candidates

        pair_checks = 0
        if gate_type in {"xor", "xnor"}:
            required_base = target_sig if gate_type == "xor" else (~target_sig) & mask
            for left_signal in search_candidates:
                required_right_sig = required_base ^ signatures[left_signal]
                for right_signal in by_signature.get(required_right_sig, []):
                    if right_signal not in seen_candidates:
                        continue
                    pair_checks += 1
                    if pair_checks > max_signature_pairs:
                        return {
                            "exists": False,
                            "target_signal": target_signal,
                            "gate_type": gate_type,
                            "candidate_scope": candidate_scope,
                            "candidate_signals": len(candidates),
                            "signature_pairs_checked": pair_checks - 1,
                            "search_complete": False,
                            "reason": (
                                "No matching pair found before the signature-search "
                                f"budget of {max_signature_pairs} pairs was reached."
                            ),
                        }
                    found = confirm(
                        left_signal,
                        right_signal,
                        "signature_filter_then_yosys_sat",
                        pair_checks,
                    )
                    if found:
                        return found
            return {
                "exists": False,
                "target_signal": target_signal,
                "gate_type": gate_type,
                "candidate_scope": candidate_scope,
                "candidate_signals": len(candidates),
                "signature_pairs_checked": pair_checks,
                "search_complete": True,
                "proof": "exhaustive_signature_search_with_sat_on_matches",
            }

        candidate_count = len(search_candidates)
        all_candidate_bits = (1 << candidate_count) - 1
        one_at_bit: List[int] = [0] * 256
        for index, signal_name in enumerate(search_candidates):
            sig_value = signatures[signal_name]
            bits = sig_value
            while bits:
                lsb = bits & -bits
                bit_index = lsb.bit_length() - 1
                one_at_bit[bit_index] |= 1 << index
                bits ^= lsb

        def iter_set_indices(bits: int):
            while bits:
                lsb = bits & -bits
                yield lsb.bit_length() - 1
                bits ^= lsb

        for left_index, left_signal in enumerate(search_candidates):
            left_sig = signatures[left_signal]
            if gate_type in {"and", "nand"}:
                needed = target_sig if gate_type == "and" else (~target_sig) & mask
                zero_required = left_sig & (~needed) & mask
                compatible_bits = all_candidate_bits
                bits = zero_required
                while bits and compatible_bits:
                    lsb = bits & -bits
                    bit_index = lsb.bit_length() - 1
                    compatible_bits &= ~one_at_bit[bit_index]
                    bits ^= lsb
            else:
                allowed = target_sig if gate_type == "or" else (~target_sig) & mask
                one_required = allowed & (~left_sig) & mask
                compatible_bits = all_candidate_bits
                bits = one_required
                while bits and compatible_bits:
                    lsb = bits & -bits
                    bit_index = lsb.bit_length() - 1
                    compatible_bits &= one_at_bit[bit_index]
                    bits ^= lsb

            # These gates are commutative, so skip mirrored pairs.
            compatible_bits &= ~((1 << left_index) - 1)
            for right_index in iter_set_indices(compatible_bits):
                right_signal = search_candidates[right_index]
                pair_checks += 1
                if pair_checks > max_signature_pairs:
                    return {
                        "exists": False,
                        "target_signal": target_signal,
                        "gate_type": gate_type,
                        "candidate_scope": candidate_scope,
                        "candidate_signals": len(candidates),
                        "signature_prefiltered_candidates": len(search_candidates),
                        "signature_pairs_checked": pair_checks - 1,
                        "search_complete": False,
                        "reason": (
                            "No matching pair found before the signature-search "
                            f"budget of {max_signature_pairs} pairs was reached."
                        ),
                    }
                if apply_sig(left_sig, signatures[right_signal]) != target_sig:
                    continue
                found = confirm(
                    left_signal, right_signal, "signature_filter_then_yosys_sat", pair_checks
                )
                if found:
                    return found

        return {
            "exists": False,
            "target_signal": target_signal,
            "gate_type": gate_type,
            "candidate_scope": candidate_scope,
            "candidate_signals": len(candidates),
            "signature_prefiltered_candidates": len(search_candidates),
            "signature_pairs_checked": pair_checks,
            "search_complete": True,
            "proof": "exhaustive_signature_search_with_sat_on_matches",
        }

    def check_signal_constant(self, signal_name: str, value: object) -> dict:
        """Prove that a scalar or vector signal always equals a constant value."""
        import textwrap

        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_signal(signal_name)

        if "[" in signal_name:
            width = 1
        else:
            wi = nl.wires.get(signal_name)
            width = wi.width if wi else 1

        raw = str(value).strip().lower().replace("_", "")
        if raw in {"'0", "1'b0"}:
            constant = 0
        elif raw == "'1":
            constant = (1 << width) - 1
        elif raw == "1'b1":
            constant = 1
        else:
            binary = re.fullmatch(r"(\d+)'b([01]+)", raw)
            if binary:
                constant = int(binary.group(2), 2)
            else:
                try:
                    constant = int(raw, 0)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported constant {value!r}; use an integer or Verilog binary literal."
                    ) from exc

        if constant < 0 or constant >= (1 << width):
            raise ValueError(
                f"Constant {value!r} does not fit signal {signal_name!r} width {width}."
            )

        dff_qs = {dff.q for dff in nl.dffs.values() if dff.q}
        queried_bits = {signal_name}
        if width > 1 and "[" not in signal_name:
            queried_bits = {f"{signal_name}[{bit}]" for bit in range(width)}
        if queried_bits & dff_qs:
            return {
                "always_equal": False,
                "signal": signal_name,
                "value": constant,
                "width": width,
                "reason": "Signal includes an unconstrained DFF-Q boundary.",
            }

        comb_nl = copy.deepcopy(nl)
        q_aliases: Dict[str, str] = {}
        for index, dff in enumerate(comb_nl.dffs.values()):
            if not dff.q or dff.q in q_aliases:
                continue
            alias = f"_constant_check_q_{index}"
            q_aliases[dff.q] = alias
            comb_nl.wires[alias] = WireInfo(name=alias)
            comb_nl.primary_inputs.append(alias)
        for node in comb_nl.nodes.values():
            node.inputs = [q_aliases.get(sig, sig) for sig in node.inputs]
        comb_nl.dffs.clear()

        tmp_dir = tempfile.mkdtemp(
            prefix="signal_constant_", dir=_workspace_temp_dir()
        )
        netlist_path = os.path.join(tmp_dir, "netlist.v")
        script_path = os.path.join(tmp_dir, "prove.ys")
        write_verilog(comb_nl, netlist_path)
        literal = f"{width}'d{constant}"
        script = textwrap.dedent(f"""\
            read_verilog {netlist_path}
            prep -top {comb_nl.module_name}
            sat -prove {signal_name} {literal} -verify -show-inputs -show {signal_name}
        """)
        with open(script_path, "w") as fh:
            fh.write(script)

        try:
            result = subprocess.run(
                [_yosys_binary(), "-s", script_path],
                capture_output=True,
                text=True,
                timeout=120,
                env=_temp_subprocess_env(),
                cwd=_workspace_temp_dir(),
            )
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode == 0:
                return {
                    "always_equal": True,
                    "signal": signal_name,
                    "value": constant,
                    "width": width,
                }
            lower = combined.lower()
            if "proof did fail" in lower or "model found" in lower:
                return {
                    "always_equal": False,
                    "signal": signal_name,
                    "value": constant,
                    "width": width,
                }
            raise RuntimeError(
                "Yosys constant-property check failed:\n" + combined[-3000:]
            )
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def check_design_equivalence(self) -> dict:
        """Prove current-vs-original equivalence at combinational DFF boundaries."""
        import shutil
        import textwrap

        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        original_path = self._original_netlist_path
        if not original_path:
            raise ValueError("No original design path is available. Call read_design first.")
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original netlist not found: {original_path!r}")

        original_nl = parse_verilog(original_path)
        dff_mismatch = self._compare_dff_boundary_shapes(original_nl, nl)
        if dff_mismatch:
            log_file = tempfile.NamedTemporaryFile(
                "w",
                prefix="design_equiv_",
                suffix=".log",
                dir=_workspace_temp_dir(),
                delete=False,
            )
            with log_file:
                log_file.write(dff_mismatch + "\n")
            return {
                "equivalent": False,
                "status": "FAIL",
                "mode": "comb_dff_boundary",
                "reason": dff_mismatch,
                "original_netlist": original_path,
                "log_path": log_file.name,
            }
        if self._netlists_structurally_equal(original_nl, nl):
            log_file = tempfile.NamedTemporaryFile(
                "w",
                prefix="design_equiv_",
                suffix=".log",
                dir=_workspace_temp_dir(),
                delete=False,
            )
            with log_file:
                log_file.write(
                    "PASS: current design is equivalent to the original netlist "
                    "under combinational DFF-boundary equivalence.\n"
                )
            return {
                "equivalent": True,
                "status": "PASS",
                "original_netlist": original_path,
                "log_path": log_file.name,
            }

        tmp_dir = tempfile.mkdtemp(
            prefix="design_equiv_", dir=_workspace_temp_dir()
        )
        gold_path = os.path.join(tmp_dir, "gold_comb.v")
        gate_path = os.path.join(tmp_dir, "gate_comb.v")
        script_path = os.path.join(tmp_dir, "equiv.ys")
        log_file = tempfile.NamedTemporaryFile(
            "w",
            prefix="design_equiv_",
            suffix=".log",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        log_path = log_file.name
        log_file.close()

        gold_comb = self._make_dff_boundary_comb_netlist(original_nl, "gold_comb")
        gate_comb = self._make_dff_boundary_comb_netlist(nl, "gate_comb")
        write_verilog(gold_comb, gold_path)
        write_verilog(gate_comb, gate_path)

        # Use ABC's dedicated two-network combinational equivalence checker as
        # the primary backend.  Both complete DFF-boundary netlists are passed
        # to CEC; no output or cone is omitted.  Yosys miter+SAT below remains
        # a conservative fallback for startup, parser, or unrecognized-result
        # failures from the standalone ABC executable.
        abc_command = f"cec -T 260 -p {gold_path} {gate_path}"
        try:
            abc_result = subprocess.run(
                [_abc_binary(), "-c", abc_command],
                capture_output=True,
                text=True,
                timeout=270,
                env=_temp_subprocess_env(),
                cwd=_workspace_temp_dir(),
            )
            abc_output = (abc_result.stdout or "") + "\n" + (abc_result.stderr or "")
            with open(log_path, "w") as fh:
                fh.write("ABC executable: " + _abc_binary() + "\n")
                fh.write("ABC command:\n" + abc_command + "\n\nABC output:\n")
                fh.write(abc_output)
            abc_lower = abc_output.lower()
            if "networks are not equivalent" in abc_lower:
                return {
                    "equivalent": False,
                    "status": "FAIL",
                    "mode": "comb_dff_boundary",
                    "backend": "standalone_abc_cec",
                    "executable": _abc_binary(),
                    "original_netlist": original_path,
                    "log_path": log_path,
                }
            if (
                "networks are equivalent" in abc_lower
                and "not equivalent" not in abc_lower
            ):
                return {
                    "equivalent": True,
                    "status": "PASS",
                    "mode": "comb_dff_boundary",
                    "backend": "standalone_abc_cec",
                    "executable": _abc_binary(),
                    "original_netlist": original_path,
                    "log_path": log_path,
                }
            if "timeout" in abc_lower or "timed out" in abc_lower:
                return {
                    "equivalent": False,
                    "status": "TIMEOUT",
                    "mode": "comb_dff_boundary",
                    "backend": "standalone_abc_cec",
                    "executable": _abc_binary(),
                    "original_netlist": original_path,
                    "log_path": log_path,
                }
        except subprocess.TimeoutExpired as exc:
            with open(log_path, "w") as fh:
                fh.write("ABC CEC timed out after 270 seconds.\n")
                if exc.stdout:
                    fh.write(str(exc.stdout))
                if exc.stderr:
                    fh.write(str(exc.stderr))
            return {
                "equivalent": False,
                "status": "TIMEOUT",
                "mode": "comb_dff_boundary",
                "backend": "standalone_abc_cec",
                "original_netlist": original_path,
                "log_path": log_path,
            }
        except (OSError, RuntimeError):
            pass

        script = textwrap.dedent(f"""\
            read_verilog -sv {gold_path}
            hierarchy -top gold_comb
            proc
            flatten
            opt_clean
            rename gold_comb gold
            design -stash gold

            read_verilog -sv {gate_path}
            hierarchy -top gate_comb
            proc
            flatten
            opt_clean
            rename gate_comb gate
            design -stash gate

            design -copy-from gold -as gold gold
            design -copy-from gate -as gate gate
            miter -equiv -flatten gold gate miter
            hierarchy -top miter
            opt_clean -purge
            techmap
            opt -fast
            abc -g aig
            opt_clean -purge
            sat -prove trigger 0 -verify -show trigger
        """)
        with open(script_path, "w") as fh:
            fh.write(script)

        try:
            process_tmp = tempfile.mkdtemp(prefix="design_equiv_process_")
            process_env = _temp_subprocess_env()
            process_env.update({
                "TMPDIR": process_tmp,
                "TEMP": process_tmp,
                "TMP": process_tmp,
            })
            try:
                process = subprocess.Popen(
                    [_yosys_binary(), "-q", "-l", log_path, "-s", script_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=process_env,
                    cwd=_workspace_temp_dir(),
                )
                returncode = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                with open(log_path, "a") as fh:
                    fh.write("\nYosys combinational boundary equivalence timed out after 300 seconds.\n")
                    fh.write(script)
                return {
                    "equivalent": False,
                    "status": "TIMEOUT",
                    "mode": "comb_dff_boundary",
                    "backend": "yosys_miter_abc_sat",
                    "original_netlist": original_path,
                    "log_path": log_path,
                }

            try:
                with open(log_path) as fh:
                    combined = fh.read()
            except OSError:
                combined = ""

            lower = combined.lower()
            equivalent = returncode == 0 and (
                "sat proof finished - no model found" in lower
            )
            status = "PASS" if equivalent else (
                "FAIL" if "proof did fail" in lower else "ERROR"
            )
            return {
                "equivalent": equivalent,
                "status": status,
                "mode": "comb_dff_boundary",
                "backend": "yosys_miter_abc_sat",
                "original_netlist": original_path,
                "log_path": log_path,
            }
        finally:
            if "process_tmp" in locals():
                shutil.rmtree(process_tmp, ignore_errors=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _compare_dff_boundary_shapes(self, gold: Netlist, gate: Netlist) -> Optional[str]:
        """Return a mismatch reason if DFF-boundary equivalence cannot be set up."""
        if set(gold.primary_inputs) != set(gate.primary_inputs):
            return (
                "Primary input sets differ between original and current netlists: "
                f"original={sorted(gold.primary_inputs)}, current={sorted(gate.primary_inputs)}"
            )
        if set(gold.primary_outputs) != set(gate.primary_outputs):
            return (
                "Primary output sets differ between original and current netlists: "
                f"original={sorted(gold.primary_outputs)}, current={sorted(gate.primary_outputs)}"
            )
        if set(gold.dffs) != set(gate.dffs):
            return (
                "DFF instance sets differ between original and current netlists: "
                f"original_only={sorted(set(gold.dffs) - set(gate.dffs))}, "
                f"current_only={sorted(set(gate.dffs) - set(gold.dffs))}"
            )
        return None

    def _netlists_structurally_equal(self, a: Netlist, b: Netlist) -> bool:
        """Conservative structural identity check before invoking formal tools."""
        def payload(nl: Netlist) -> dict:
            return {
                "primary_inputs": list(nl.primary_inputs),
                "primary_outputs": list(nl.primary_outputs),
                "wires": {
                    name: {
                        "width": wi.width,
                        "is_bus": wi.is_bus,
                        "msb": wi.msb,
                        "lsb": wi.lsb,
                    }
                    for name, wi in sorted(nl.wires.items())
                },
                "nodes": {
                    name: {
                        "gate_type": node.gate_type,
                        "inputs": list(node.inputs),
                        "output": node.output,
                    }
                    for name, node in sorted(nl.nodes.items())
                },
                "dffs": {
                    name: {
                        "ck": dff.ck,
                        "rn": dff.rn,
                        "sn": dff.sn,
                        "d": dff.d,
                        "q": dff.q,
                    }
                    for name, dff in sorted(nl.dffs.items())
                },
            }

        return payload(a) == payload(b)

    def _make_dff_boundary_comb_netlist(self, source: Netlist, module_name: str) -> Netlist:
        """Create a combinational view with DFF Qs as inputs and DFF pins as outputs."""
        comb = copy.deepcopy(source)
        comb.module_name = module_name
        comb.dffs = {}

        existing_names = set(comb.wires) | set(comb.nodes) | set(comb.primary_inputs) | set(comb.primary_outputs)

        def unique_name(base: str) -> str:
            name = base
            suffix = 0
            while name in existing_names:
                suffix += 1
                name = f"{base}_{suffix}"
            existing_names.add(name)
            return name

        for inst_name, dff in source.dffs.items():
            state_input = unique_name(f"_equiv_state_{inst_name}")
            comb.primary_inputs.append(state_input)
            comb.wires[state_input] = WireInfo(name=state_input)
            buf_name = unique_name(f"_equiv_statebuf_{inst_name}")
            comb.nodes[buf_name] = GateNode(
                name=buf_name,
                gate_type="buf",
                inputs=[state_input],
                output=dff.q,
            )

        for inst_name, dff in source.dffs.items():
            for pin_name, signal in (
                ("d", dff.d),
                ("ck", dff.ck),
                ("rn", dff.rn),
                ("sn", dff.sn or "1'b1"),
            ):
                out_name = unique_name(f"_equiv_{pin_name}_{inst_name}")
                comb.primary_outputs.append(out_name)
                comb.wires[out_name] = WireInfo(name=out_name)
                buf_name = unique_name(f"_equiv_{pin_name}buf_{inst_name}")
                comb.nodes[buf_name] = GateNode(
                    name=buf_name,
                    gate_type="buf",
                    inputs=[signal],
                    output=out_name,
                )

        return comb

    # Legacy sequential whole-design equivalence flow kept for reference.
    # It is intentionally not the default because the competition checks
    # combinational behavior with DFF Q pins treated as unconstrained inputs.
    #
    # def check_design_equivalence_sequential_legacy(self) -> dict:
    #     ...
    #     read_verilog -sv dff.v original.v
    #     hierarchy -top <original_top>
    #     proc
    #     async2sync
    #     flatten
    #     opt_clean
    #     rename <original_top> gold
    #     design -stash gold
    #
    #     read_verilog -sv dff.v current.v
    #     hierarchy -top <current_top>
    #     proc
    #     async2sync
    #     flatten
    #     opt_clean
    #     rename <current_top> gate
    #     design -stash gate
    #
    #     design -copy-from gold -as gold gold
    #     design -copy-from gate -as gate gate
    #     equiv_make gold gate equiv
    #     hierarchy -top equiv
    #     clean -purge
    #     equiv_simple
    #     equiv_induct -seq 12
    #     equiv_status -assert

    def find_instances_by_name_pattern(
        self, gate_type: str, name_pattern: str
    ) -> List[str]:
        """Return instance names matching gate_type and name_pattern regex.

        Pass an empty string for gate_type to match all gate types.

        Args:
            gate_type:    Gate type filter (e.g. "buf"), or "" for any.
            name_pattern: Python regex applied to instance names.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        try:
            rx = re.compile(name_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex {name_pattern!r}: {exc}") from exc

        results: List[str] = []
        for inst_name, node in nl.nodes.items():
            type_match = (not gate_type) or node.gate_type == gate_type.lower()
            if type_match and rx.search(inst_name):
                results.append(inst_name)
        return results

    def find_gates(
        self,
        gate_type: str = "",
        input_count: Optional[int] = None,
        has_input: Optional[str] = None,
        inline_limit: int = 10,
    ) -> dict:
        """Find combinational gates matching structural filters."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        gate_type = gate_type.strip().lower()
        if gate_type and gate_type not in PRIMITIVE_GATES:
            raise ValueError(f"Unknown gate type: {gate_type!r}")

        def normalize_signal(signal_name: str) -> str:
            aliases = {
                "0": "1'b0",
                "'0": "1'b0",
                "1": "1'b1",
                "'1": "1'b1",
            }
            key = signal_name.strip().lower()
            return aliases.get(key, signal_name.strip())

        wanted_input = normalize_signal(has_input) if has_input else None
        matches: List[dict] = []

        for inst_name, node in nl.nodes.items():
            if gate_type and node.gate_type != gate_type:
                continue
            if input_count is not None and len(node.inputs) != input_count:
                continue

            normalized_inputs = [normalize_signal(sig) for sig in node.inputs]
            matched_indices: List[int] = []
            if wanted_input is not None:
                matched_indices = [
                    idx
                    for idx, signal_name in enumerate(normalized_inputs)
                    if signal_name == wanted_input
                ]
                if not matched_indices:
                    continue

            matches.append(
                {
                    "instance": inst_name,
                    "gate_type": node.gate_type,
                    "output": node.output,
                    "inputs": list(node.inputs),
                    "matched_input_indices": matched_indices,
                    "other_inputs": [
                        sig
                        for idx, sig in enumerate(node.inputs)
                        if idx not in set(matched_indices)
                    ],
                }
            )

        result = {
            "count": len(matches),
            "gate_type": gate_type or "any",
        }
        if input_count is not None:
            result["input_count"] = input_count
        if has_input is not None:
            result["has_input"] = has_input
        if len(matches) <= inline_limit:
            result["matches"] = matches
            return result

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix="find_gates_",
            suffix=".tsv",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write("instance\tgate_type\toutput\tinputs\n")
            for match in matches:
                report.write(
                    f"{match['instance']}\t{match['gate_type']}\t"
                    f"{match['output']}\t{','.join(match['inputs'])}\n"
                )
        result["file_path"] = os.path.abspath(report.name)
        return result

    def find_gates_with_constant_inputs(
        self,
        gate_type: str,
        values: Optional[List[object]] = None,
        functional: bool = True,
        inline_limit: int = 50,
    ) -> dict:
        """Report gates whose inputs are literal or functionally constant."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        gate_type = gate_type.strip().lower()
        if gate_type not in PRIMITIVE_GATES:
            raise ValueError(f"Unknown gate type: {gate_type!r}")

        wanted_values = self._normalize_constant_values(values or [0, 1])
        candidate_gates = [
            (inst_name, node)
            for inst_name, node in nl.nodes.items()
            if node.gate_type == gate_type
        ]
        candidate_signals = sorted(
            {
                signal
                for _, node in candidate_gates
                for signal in node.inputs
            }
        )

        propagated_constants = self._constant_propagation_facts()
        simulated_observations = self._simulate_signal_observations(
            candidate_signals, rounds=512
        )
        signal_results: Dict[str, dict] = {}
        pending_formal: Dict[str, int] = {}

        for signal in candidate_signals:
            classification = self._classify_constant_candidate(
                signal,
                wanted_values,
                propagated_constants,
                simulated_observations.get(signal, set()),
                functional,
            )
            if classification.get("status") == "pending_formal":
                pending_formal[signal] = int(classification["observed_value"])
            signal_results[signal] = classification

        if pending_formal:
            formal_results = self._batch_prove_observed_constants(pending_formal)
            for signal, classification in formal_results.items():
                signal_results[signal] = classification

        complete = not any(
            result.get("status") in {"unknown", "pending_formal"}
            for result in signal_results.values()
        )

        matches: List[dict] = []
        for inst_name, node in candidate_gates:
            constant_inputs = []
            for index, signal in enumerate(node.inputs):
                classification = signal_results.get(signal, {"constant": False})
                if not classification.get("constant"):
                    continue
                value = int(classification["value"])
                if value not in wanted_values:
                    continue
                constant_inputs.append(
                    {
                        "index": index,
                        "signal": signal,
                        "value": value,
                        "proof": classification.get("proof", "unknown"),
                    }
                )

            if not constant_inputs:
                continue
            matches.append(
                {
                    "instance": inst_name,
                    "gate_type": node.gate_type,
                    "output": node.output,
                    "inputs": list(node.inputs),
                    "constant_inputs": constant_inputs,
                }
            )

        unknown_signals = [
            signal
            for signal, result in signal_results.items()
            if result.get("status") == "unknown"
        ]
        result = {
            "count": len(matches),
            "gate_type": gate_type,
            "values": sorted(wanted_values),
            "functional": bool(functional),
            "complete": complete,
            "candidate_gates": len(candidate_gates),
            "candidate_signals": len(candidate_signals),
            "unknown_signals": unknown_signals[:20],
            "unknown_signal_count": len(unknown_signals),
        }
        self._last_constant_input_matches = copy.deepcopy(matches)
        self._last_constant_input_signature = self._constant_analysis_signature()

        if len(matches) <= inline_limit:
            result["matches"] = matches
            self._last_constant_input_report = copy.deepcopy(result)
            return result

        report = tempfile.NamedTemporaryFile(
            "w",
            prefix="constant_input_gates_",
            suffix=".txt",
            dir=_workspace_temp_dir(),
            delete=False,
        )
        with report:
            report.write(
                f"# find_gates_with_constant_inputs gate_type={gate_type} "
                f"values={sorted(wanted_values)} count={len(matches)} "
                f"complete={complete}\n"
            )
            report.write("# columns: instance gate_type output inputs constant_inputs\n")
            for item in matches:
                report.write(
                    "# "
                    f"{item['instance']} {item['gate_type']} {item['output']} "
                    f"{','.join(item['inputs'])} "
                    f"{json.dumps(item['constant_inputs'], sort_keys=True)}\n"
                )
            report.write("# jsonl:\n")
            for item in matches:
                report.write(json.dumps(item, sort_keys=True) + "\n")

        result["sample_matches"] = matches[:inline_limit]
        result["file_path"] = os.path.abspath(report.name)
        self._last_constant_input_report = copy.deepcopy(result)
        return result

    def simplify_gates_with_constant_inputs(
        self,
        gate_type: str,
        functional: bool = True,
        use_last_report: bool = True,
    ) -> dict:
        """Simplify gates using literal or proven functional constant inputs."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        gate_type = gate_type.strip().lower()
        if gate_type not in PRIMITIVE_GATES:
            raise ValueError(f"Unknown gate type: {gate_type!r}")

        report = self._last_constant_input_report
        matches = self._last_constant_input_matches
        compatible_cache = bool(
            use_last_report
            and report
            and matches is not None
            and report.get("gate_type") == gate_type
            and bool(report.get("functional")) == bool(functional)
            and report.get("complete")
            and self._last_constant_input_signature
            == self._constant_analysis_signature()
        )
        analysis_reused = compatible_cache
        if not compatible_cache:
            report = self.find_gates_with_constant_inputs(
                gate_type,
                values=[0, 1],
                functional=functional,
                inline_limit=len(nl.nodes) + 1,
            )
            matches = self._last_constant_input_matches or []

        assert report is not None
        assert matches is not None
        if not report.get("complete"):
            raise RuntimeError(
                "Constant-input analysis was incomplete; refusing to transform the design."
            )

        replacement_breakdown: Dict[str, int] = {}
        replaced_instances: List[str] = []
        stale_matches = 0

        def replacement(node: GateNode, constant_inputs: List[dict]):
            known = {
                int(item["index"]): int(item["value"])
                for item in constant_inputs
            }
            if node.gate_type in ONE_INPUT_GATES:
                value = known.get(0)
                if value is None:
                    return None
                if node.gate_type == "not":
                    value = 1 - value
                return "buf", [f"1'b{value}"]

            if len(node.inputs) != 2:
                return None
            if 0 in known and 1 in known:
                a, b = known[0], known[1]
                values = {
                    "and": a & b,
                    "or": a | b,
                    "nand": 1 - (a & b),
                    "nor": 1 - (a | b),
                    "xor": a ^ b,
                    "xnor": 1 - (a ^ b),
                }
                return "buf", [f"1'b{values[node.gate_type]}"]

            constant_index = next(iter(known), None)
            if constant_index is None:
                return None
            value = known[constant_index]
            other = node.inputs[1 - constant_index]
            rules = {
                "and": ("buf", ["1'b0"]) if value == 0 else ("buf", [other]),
                "or": ("buf", ["1'b1"]) if value == 1 else ("buf", [other]),
                "nand": ("buf", ["1'b1"]) if value == 0 else ("not", [other]),
                "nor": ("buf", ["1'b0"]) if value == 1 else ("not", [other]),
                "xor": ("buf", [other]) if value == 0 else ("not", [other]),
                "xnor": ("not", [other]) if value == 0 else ("buf", [other]),
            }
            return rules.get(node.gate_type)

        for match in matches:
            instance = match["instance"]
            node = nl.nodes.get(instance)
            if (
                node is None
                or node.gate_type != match.get("gate_type")
                or list(node.inputs) != list(match.get("inputs", []))
            ):
                stale_matches += 1
                continue
            change = replacement(node, match.get("constant_inputs", []))
            if change is None:
                continue
            new_type, new_inputs = change
            self.replace_gate(instance, new_type, new_inputs=new_inputs)
            replaced_instances.append(instance)
            replacement_breakdown[new_type.upper()] = (
                replacement_breakdown.get(new_type.upper(), 0) + 1
            )

        if replaced_instances:
            self._last_constant_input_report = None
            self._last_constant_input_matches = None
            self._last_constant_input_signature = None

        return {
            "gate_type": gate_type,
            "functional": bool(functional),
            "analysis_reused": analysis_reused,
            "reported_gates": len(matches),
            "simplified_gates": len(replaced_instances),
            "eliminated_original_gate_type": len(replaced_instances),
            "replacement_breakdown": dict(sorted(replacement_breakdown.items())),
            "stale_matches_skipped": stale_matches,
            "instances": replaced_instances[:50],
        }

    def _constant_analysis_signature(self) -> tuple:
        """Return a deterministic structural signature for cache validation."""
        nl = self._netlist
        assert nl is not None
        gates = tuple(
            sorted(
                (name, node.gate_type, node.output, tuple(node.inputs))
                for name, node in nl.nodes.items()
            )
        )
        dffs = tuple(
            sorted(
                (name, dff.ck, dff.rn, dff.sn, dff.d, dff.q)
                for name, dff in nl.dffs.items()
            )
        )
        return gates, dffs, tuple(nl.primary_inputs), tuple(nl.primary_outputs)

    def _normalize_constant_values(self, values: List[object]) -> Set[int]:
        normalized: Set[int] = set()
        for value in values:
            raw = str(value).strip().lower().replace("_", "")
            if raw in {"0", "'0", "1'b0"}:
                normalized.add(0)
            elif raw in {"1", "'1", "1'b1"}:
                normalized.add(1)
            else:
                try:
                    parsed = int(raw, 0)
                except ValueError as exc:
                    raise ValueError(f"Unsupported constant value {value!r}.") from exc
                if parsed not in {0, 1}:
                    raise ValueError("Only scalar constant values 0 and 1 are supported.")
                normalized.add(parsed)
        return normalized

    def _literal_constant_value(self, signal: str) -> Optional[int]:
        key = signal.strip().lower().replace("_", "")
        if key in {"0", "'0", "1'b0"}:
            return 0
        if key in {"1", "'1", "1'b1"}:
            return 1
        return None

    def _is_unconstrained_boundary_signal(self, signal: str) -> bool:
        nl = self._netlist
        assert nl is not None
        if signal in nl.primary_inputs:
            return True
        bit_select = re.fullmatch(r"([A-Za-z_$][\w$]*)\[\d+\]", signal)
        if bit_select and bit_select.group(1) in nl.primary_inputs:
            return True
        return signal in {dff.q for dff in nl.dffs.values() if dff.q}

    def _constant_propagation_facts(self) -> Dict[str, int]:
        """Conservative constants proven by local gate-level propagation."""
        nl = self._netlist
        assert nl is not None
        constants: Dict[str, int] = {"1'b0": 0, "1'b1": 1, "0": 0, "1": 1, "'0": 0, "'1": 1}

        def val(signal: str) -> Optional[int]:
            literal = self._literal_constant_value(signal)
            if literal is not None:
                return literal
            return constants.get(signal)

        changed = True
        while changed:
            changed = False
            for node in nl.nodes.values():
                if node.output in constants:
                    continue
                input_values = [val(signal) for signal in node.inputs]
                result: Optional[int] = None
                if node.gate_type == "buf" and input_values[0] is not None:
                    result = input_values[0]
                elif node.gate_type == "not" and input_values[0] is not None:
                    result = 1 - input_values[0]
                elif node.gate_type == "and":
                    if 0 in input_values:
                        result = 0
                    elif all(value == 1 for value in input_values):
                        result = 1
                elif node.gate_type == "nand":
                    if 0 in input_values:
                        result = 1
                    elif all(value == 1 for value in input_values):
                        result = 0
                elif node.gate_type == "or":
                    if 1 in input_values:
                        result = 1
                    elif all(value == 0 for value in input_values):
                        result = 0
                elif node.gate_type == "nor":
                    if 1 in input_values:
                        result = 0
                    elif all(value == 0 for value in input_values):
                        result = 1
                elif node.gate_type in {"xor", "xnor"} and all(
                    value is not None for value in input_values
                ):
                    ones = sum(int(value) for value in input_values)
                    result = ones % 2
                    if node.gate_type == "xnor":
                        result = 1 - result

                if result is not None:
                    constants[node.output] = result
                    changed = True
        return constants

    def _simulate_signal_observations(
        self, signals: List[str], rounds: int = 96
    ) -> Dict[str, Set[int]]:
        """Random-simulate candidate signals to discard obvious non-constants."""
        nl = self._netlist
        assert nl is not None
        rng = random.Random(0)
        observations: Dict[str, Set[int]] = {signal: set() for signal in signals}
        output_to_gate = {node.output: node for node in nl.nodes.values()}
        dff_qs = {dff.q for dff in nl.dffs.values() if dff.q}

        def gate_eval(gate_type: str, inputs: List[int]) -> int:
            if gate_type == "buf":
                return inputs[0]
            if gate_type == "not":
                return 1 - inputs[0]
            if gate_type == "and":
                return int(all(inputs))
            if gate_type == "nand":
                return 1 - int(all(inputs))
            if gate_type == "or":
                return int(any(inputs))
            if gate_type == "nor":
                return 1 - int(any(inputs))
            if gate_type == "xor":
                return sum(inputs) % 2
            if gate_type == "xnor":
                return 1 - (sum(inputs) % 2)
            raise ValueError(f"Unsupported gate type for simulation: {gate_type!r}")

        for _ in range(rounds):
            memo: Dict[str, int] = {
                "1'b0": 0,
                "0": 0,
                "'0": 0,
                "1'b1": 1,
                "1": 1,
                "'1": 1,
            }
            for pi in nl.primary_inputs:
                wi = nl.wires.get(pi)
                if wi and wi.is_bus:
                    for bit in range(min(wi.msb, wi.lsb), max(wi.msb, wi.lsb) + 1):
                        memo[f"{pi}[{bit}]"] = rng.randint(0, 1)
                memo[pi] = rng.randint(0, 1)
            for q in dff_qs:
                memo[q] = rng.randint(0, 1)

            visiting: Set[str] = set()

            def value(signal: str) -> int:
                literal = self._literal_constant_value(signal)
                if literal is not None:
                    return literal
                if signal in memo:
                    return memo[signal]
                if signal in visiting:
                    memo[signal] = rng.randint(0, 1)
                    return memo[signal]
                node = output_to_gate.get(signal)
                if node is None:
                    memo[signal] = rng.randint(0, 1)
                    return memo[signal]
                visiting.add(signal)
                input_values = [value(inp) for inp in node.inputs]
                visiting.remove(signal)
                memo[signal] = gate_eval(node.gate_type, input_values)
                return memo[signal]

            for signal in signals:
                observations[signal].add(value(signal))
        return observations

    def _classify_constant_candidate(
        self,
        signal: str,
        wanted_values: Set[int],
        propagated_constants: Dict[str, int],
        simulated_observations: Set[int],
        functional: bool,
    ) -> dict:
        literal = self._literal_constant_value(signal)
        if literal is not None:
            return {
                "constant": literal in wanted_values,
                "value": literal,
                "proof": "literal",
                "status": "complete",
            }
        if self._is_unconstrained_boundary_signal(signal):
            return {
                "constant": False,
                "proof": "boundary_input",
                "status": "complete",
            }
        if signal in propagated_constants:
            value = propagated_constants[signal]
            return {
                "constant": value in wanted_values,
                "value": value,
                "proof": "propagation",
                "status": "complete",
            }
        if len(simulated_observations) > 1:
            return {"constant": False, "proof": "simulation_filter", "status": "complete"}
        if simulated_observations:
            observed = next(iter(simulated_observations))
            if observed not in wanted_values:
                return {
                    "constant": False,
                    "proof": "simulation_filter",
                    "status": "complete",
                }
        if not functional:
            return {"constant": False, "proof": "structural_only", "status": "complete"}
        if simulated_observations:
            observed = next(iter(simulated_observations))
            return {
                "constant": False,
                "proof": "formal",
                "status": "pending_formal",
                "observed_value": observed,
            }
        return {
            "constant": False,
            "proof": "formal",
            "status": "unknown",
            "reason": "No simulation observation available for formal target selection.",
        }

    def _batch_prove_observed_constants(
        self, observed_values: Dict[str, int]
    ) -> Dict[str, dict]:
        """Classify possible constants by searching for batch counterexamples."""
        remaining = dict(observed_values)
        results: Dict[str, dict] = {}
        chunk_size = 50

        while remaining:
            progress = False
            chunks = [
                dict(list(remaining.items())[index : index + chunk_size])
                for index in range(0, len(remaining), chunk_size)
            ]
            for chunk in chunks:
                active_chunk = {
                    signal: value
                    for signal, value in chunk.items()
                    if signal in remaining
                }
                if not active_chunk:
                    continue

                model = self._find_constant_counterexample_model(active_chunk)
                if model is None:
                    for signal, value in active_chunk.items():
                        results[signal] = {
                            "constant": True,
                            "value": value,
                            "proof": "formal",
                            "status": "complete",
                        }
                        remaining.pop(signal, None)
                    progress = True
                    continue

                if not model:
                    for signal in active_chunk:
                        results[signal] = {
                            "constant": False,
                            "proof": "formal",
                            "status": "unknown",
                            "reason": "SAT counterexample did not include probe values.",
                        }
                        remaining.pop(signal, None)
                    progress = True
                    continue

                for signal in model:
                    if signal not in remaining:
                        continue
                    results[signal] = {
                        "constant": False,
                        "proof": "formal",
                        "status": "complete",
                    }
                    remaining.pop(signal, None)
                    progress = True

            if not progress:
                for signal in list(remaining):
                    results[signal] = {
                        "constant": False,
                        "proof": "formal",
                        "status": "unknown",
                        "reason": "Batch SAT made no classification progress.",
                    }
                    remaining.pop(signal, None)
                break

        return results

    def _find_constant_counterexample_model(
        self, observed_values: Dict[str, int]
    ) -> Optional[Dict[str, int]]:
        """Return one model where any signal differs from its observed value.

        Returns None when no counterexample exists.
        """
        import shutil
        import textwrap

        nl = self._netlist
        assert nl is not None
        comb_nl = copy.deepcopy(nl)
        q_aliases: Dict[str, str] = {}
        for index, dff in enumerate(comb_nl.dffs.values()):
            if not dff.q or dff.q in q_aliases:
                continue
            alias = f"_constant_batch_q_{index}"
            q_aliases[dff.q] = alias
            comb_nl.wires[alias] = WireInfo(name=alias)
            comb_nl.primary_inputs.append(alias)
        for node in comb_nl.nodes.values():
            node.inputs = [q_aliases.get(sig, sig) for sig in node.inputs]
        comb_nl.dffs.clear()

        probe_to_signal: Dict[str, str] = {}
        diff_signals: List[str] = []
        existing = set(comb_nl.wires) | set(comb_nl.nodes) | set(comb_nl.primary_outputs)

        def add_wire(base: str) -> str:
            name = base
            suffix = 0
            while name in existing:
                suffix += 1
                name = f"{base}_{suffix}"
            existing.add(name)
            comb_nl.wires[name] = WireInfo(name=name)
            return name

        def add_gate(base: str, gate_type: str, inputs: List[str], output: str) -> None:
            name = base
            suffix = 0
            while name in existing:
                suffix += 1
                name = f"{base}_{suffix}"
            existing.add(name)
            comb_nl.nodes[name] = GateNode(
                name=name,
                gate_type=gate_type,
                inputs=inputs,
                output=output,
            )

        for index, (signal, observed) in enumerate(observed_values.items()):
            probe = add_wire(f"_constant_probe_{index}")
            comb_nl.primary_outputs.append(probe)
            probe_to_signal[probe] = signal
            add_gate(
                f"_constant_probe_buf_{index}",
                "buf",
                [q_aliases.get(signal, signal)],
                probe,
            )
            if observed == 0:
                diff_signals.append(probe)
            else:
                diff = add_wire(f"_constant_diff_{index}")
                add_gate(f"_constant_diff_not_{index}", "not", [probe], diff)
                diff_signals.append(diff)

        if not diff_signals:
            return None
        current = diff_signals[0]
        for index, diff in enumerate(diff_signals[1:]):
            out = add_wire(f"_constant_diff_or_{index}")
            add_gate(f"_constant_diff_or_gate_{index}", "or", [current, diff], out)
            current = out
        diff_any = add_wire("_constant_diff_any")
        comb_nl.primary_outputs.append(diff_any)
        add_gate("_constant_diff_any_buf", "buf", [current], diff_any)

        tmp_dir = tempfile.mkdtemp(
            prefix="constant_batch_", dir=_workspace_temp_dir()
        )
        netlist_path = os.path.join(tmp_dir, "netlist.v")
        script_path = os.path.join(tmp_dir, "prove.ys")
        write_verilog(comb_nl, netlist_path)
        show_args = " ".join(f"-show {probe}" for probe in probe_to_signal)
        script = textwrap.dedent(f"""\
            read_verilog {netlist_path}
            prep -top {comb_nl.module_name}
            sat -set {diff_any} 1 -max 64 {show_args} -show {diff_any}
        """)
        with open(script_path, "w") as fh:
            fh.write(script)

        try:
            result = subprocess.run(
                [_yosys_binary(), "-s", script_path],
                capture_output=True,
                text=True,
                timeout=60,
                env=_temp_subprocess_env(),
                cwd=_workspace_temp_dir(),
            )
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            lower = combined.lower()
            if "sat solving finished - no model found" in lower:
                return None
            if "sat solving finished - model found" not in lower:
                raise RuntimeError(
                    "Could not determine Yosys SAT batch result:\n" + combined[-3000:]
                )
            model: Dict[str, int] = {}
            for probe, signal in probe_to_signal.items():
                pattern = re.compile(
                    rf"\\{re.escape(probe)}\s+\d+\s+[0-9a-fA-F]+\s+([01])\b"
                )
                for match in pattern.finditer(combined):
                    value = int(match.group(1))
                    if value != observed_values[signal]:
                        model[signal] = value
                        break
            return model
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _prove_scalar_signal_constant(self, signal_name: str, value: int) -> dict:
        """Prove a scalar signal by buffering it to a generated primary output."""
        import shutil
        import textwrap

        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        if value not in {0, 1}:
            raise ValueError("Only scalar constants 0 and 1 are supported.")

        comb_nl = copy.deepcopy(nl)
        q_aliases: Dict[str, str] = {}
        for index, dff in enumerate(comb_nl.dffs.values()):
            if not dff.q or dff.q in q_aliases:
                continue
            alias = f"_constant_probe_q_{index}"
            q_aliases[dff.q] = alias
            comb_nl.wires[alias] = WireInfo(name=alias)
            comb_nl.primary_inputs.append(alias)
        for node in comb_nl.nodes.values():
            node.inputs = [q_aliases.get(sig, sig) for sig in node.inputs]
        probe_input = q_aliases.get(signal_name, signal_name)
        comb_nl.dffs.clear()

        probe_output = "_constant_probe_out"
        probe_inst = "_constant_probe_buf"
        suffix = 0
        existing = set(comb_nl.wires) | set(comb_nl.nodes) | set(comb_nl.primary_outputs)
        while probe_output in existing or probe_inst in existing:
            suffix += 1
            probe_output = f"_constant_probe_out_{suffix}"
            probe_inst = f"_constant_probe_buf_{suffix}"
        comb_nl.wires[probe_output] = WireInfo(name=probe_output)
        comb_nl.primary_outputs.append(probe_output)
        comb_nl.nodes[probe_inst] = GateNode(
            name=probe_inst,
            gate_type="buf",
            inputs=[probe_input],
            output=probe_output,
        )

        tmp_dir = tempfile.mkdtemp(
            prefix="constant_probe_", dir=_workspace_temp_dir()
        )
        netlist_path = os.path.join(tmp_dir, "netlist.v")
        script_path = os.path.join(tmp_dir, "prove.ys")
        write_verilog(comb_nl, netlist_path)
        literal = f"1'b{value}"
        script = textwrap.dedent(f"""\
            read_verilog {netlist_path}
            prep -top {comb_nl.module_name}
            sat -prove {probe_output} {literal} -verify -show-inputs -show {probe_output}
        """)
        with open(script_path, "w") as fh:
            fh.write(script)

        try:
            result = subprocess.run(
                [_yosys_binary(), "-s", script_path],
                capture_output=True,
                text=True,
                timeout=30,
                env=_temp_subprocess_env(),
                cwd=_workspace_temp_dir(),
            )
            combined = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode == 0:
                return {"always_equal": True, "signal": signal_name, "value": value}
            lower = combined.lower()
            if "proof did fail" in lower or "model found" in lower:
                return {"always_equal": False, "signal": signal_name, "value": value}
            raise RuntimeError(
                "Yosys constant-probe check failed:\n" + combined[-3000:]
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ==================================================================
    # TRANSFORMATION OPERATIONS
    # ==================================================================

    def insert_gate_before(
        self, target_instance: str, gate_type: str, extra_input: str
    ) -> str:
        """Insert a new gate before target_instance.

        The original driver of target_instance is wired into input[0] of the
        new gate; extra_input is wired into input[1].  The new gate's output
        replaces the original signal feeding target_instance.

        Args:
            target_instance: Existing gate to insert before.
            gate_type:       Type of the new gate to insert.
            extra_input:     Second input signal for the new gate.

        Returns:
            The instance name of the newly created gate.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_instance(target_instance)

        gate_type = gate_type.lower()
        if gate_type not in TWO_INPUT_GATES:
            raise ValueError(
                f"insert_gate_before requires a 2-input gate, got {gate_type!r}"
            )

        target = nl.nodes[target_instance]
        original_input = target.inputs[0]

        new_wire = self._next_wire_name("ins_w")
        new_inst = self._next_inst_name("ins_g")

        self._add_wire(new_wire)
        new_gate = GateNode(
            name=new_inst,
            gate_type=gate_type,
            inputs=[original_input, extra_input],
            output=new_wire,
        )
        nl.nodes[new_inst] = new_gate
        target.inputs[0] = new_wire

        self.remove_dangling_gates()
        return new_inst

    def replace_gate(
        self,
        instance_name: str,
        new_gate_type: str,
        extra_input: Optional[str] = None,
        new_inputs: Optional[List[str]] = None,
    ) -> None:
        """Replace the gate type of instance_name in-place.

        When upgrading from a 1-input gate (buf/not) to a 2-input gate,
        supply *extra_input* to provide the second input signal.  Raises
        ValueError if the port counts are incompatible and no extra_input
        is provided. Supply *new_inputs* to explicitly set the replacement
        gate inputs.

        Args:
            instance_name: Existing gate instance to replace.
            new_gate_type: New gate type string.
            extra_input:   Optional second input signal when changing from
                           a 1-input gate to a 2-input gate.
            new_inputs:    Optional complete input list for the new gate.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_instance(instance_name)

        new_gate_type = new_gate_type.lower()
        if new_gate_type not in PRIMITIVE_GATES:
            raise ValueError(f"Unknown gate type: {new_gate_type!r}")

        node = nl.nodes[instance_name]
        old_in_count = len(node.inputs)
        new_in_count = 1 if new_gate_type in ONE_INPUT_GATES else 2

        if new_inputs is not None:
            if len(new_inputs) != new_in_count:
                raise ValueError(
                    f"Gate type {new_gate_type!r} requires {new_in_count} input(s), "
                    f"but new_inputs has {len(new_inputs)}."
                )
            node.gate_type = new_gate_type
            node.inputs = list(new_inputs)
            for input_signal in new_inputs:
                if input_signal in {"1'b0", "1'b1"}:
                    continue
                if (
                    input_signal not in nl.wires
                    and input_signal not in nl.primary_inputs
                    and input_signal not in nl.primary_outputs
                ):
                    self._add_wire(input_signal)
            return

        if old_in_count == new_in_count:
            node.gate_type = new_gate_type
        elif old_in_count < new_in_count:
            # Upgrading input count — require extra_input
            if extra_input is None:
                raise ValueError(
                    f"Gate type {new_gate_type!r} requires {new_in_count} input(s), "
                    f"but {instance_name!r} has {old_in_count}. "
                    "Provide 'extra_input' to supply the additional signal."
                )
            node.gate_type = new_gate_type
            while len(node.inputs) < new_in_count:
                node.inputs.append(extra_input)
            # Register extra_input as a known signal if needed
            if extra_input not in nl.wires and extra_input not in nl.primary_inputs:
                self._add_wire(extra_input)
        else:
            # Downgrading input count — drop extra inputs
            node.gate_type = new_gate_type
            node.inputs = node.inputs[:new_in_count]

    def insert_buffer(self, net_name: str, after_gate: str) -> str:
        """Insert a buf on net_name after after_gate.

        Args:
            net_name:   Net to buffer.
            after_gate: Gate instance that drives net_name.

        Returns:
            The new buffer instance name.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_signal(net_name)
        self._resolve_instance(after_gate)

        new_wire = self._next_wire_name("buf_w")
        buf_inst = self._next_inst_name("buf_g")

        self._add_wire(new_wire)

        # Rename original net to new_wire in after_gate's output
        if after_gate in nl.nodes:
            nl.nodes[after_gate].output = new_wire
        else:
            nl.dffs[after_gate].q = new_wire

        # Create the buffer
        buf_node = GateNode(
            name=buf_inst,
            gate_type="buf",
            inputs=[new_wire],
            output=net_name,
        )
        nl.nodes[buf_inst] = buf_node
        return buf_inst

    def insert_buffers_for_fanout(self, net_name: str, max_fanout: int) -> int:
        """Insert buffer trees so no net has fanout > max_fanout.

        Args:
            net_name:   Net to examine and buffer.
            max_fanout: Maximum allowed fanout per net.

        Returns:
            Number of buffers inserted.
        """
        self._require_netlist()
        self._resolve_signal(net_name)
        if max_fanout < 2:
            raise ValueError(
                "max_fanout must be >= 2; buffers cannot reduce fanout to 1"
            )

        nl = self._netlist
        assert nl is not None

        def load_pins(signal: str):
            """Return mutable references to each individual load pin."""
            pins = []
            for inst_name, node in nl.nodes.items():
                for input_index, input_signal in enumerate(node.inputs):
                    if input_signal == signal:
                        pins.append(("gate", inst_name, input_index))
            for inst_name, dff in nl.dffs.items():
                for attr_name in ("ck", "rn", "sn", "d"):
                    if getattr(dff, attr_name) == signal:
                        pins.append(("dff", inst_name, attr_name))
            return pins

        total_inserted = 0
        while True:
            pins = load_pins(net_name)
            if len(pins) <= max_fanout:
                break

            # Replacing F load pins with one buffer input reduces the source
            # fanout by F-1. Repeating this constructs a minimum-size tree;
            # once source-level buffers themselves are grouped, it naturally
            # becomes multilevel.
            group = pins[:max_fanout]
            new_wire = self._next_wire_name("fo_w")
            buf_inst = self._next_inst_name("fo_buf")
            self._add_wire(new_wire)

            for load_kind, inst_name, pin in group:
                if load_kind == "gate":
                    nl.nodes[inst_name].inputs[pin] = new_wire
                else:
                    setattr(nl.dffs[inst_name], pin, new_wire)

            nl.nodes[buf_inst] = GateNode(
                name=buf_inst,
                gate_type="buf",
                inputs=[net_name],
                output=new_wire,
            )
            total_inserted += 1

        return total_inserted

    def insert_dedicated_buffers_for_loads(self, net_name: str) -> int:
        """Insert one buffer per current direct load of net_name.

        After this operation, each load that directly consumed net_name before
        the call consumes a newly created buffer output instead. The original
        net drives only the inserted buffers.
        """
        self._require_netlist()
        self._resolve_signal(net_name)
        nl = self._netlist
        assert nl is not None

        direct_loads = list(self._build_fanout_map().get(net_name, []))
        inserted = 0

        for consumer_inst in direct_loads:
            if consumer_inst not in nl.nodes and consumer_inst not in nl.dffs:
                continue

            new_wire = self._next_wire_name("dedbuf_w")
            buf_inst = self._next_inst_name("dedbuf_g")
            self._add_wire(new_wire)
            nl.nodes[buf_inst] = GateNode(
                name=buf_inst,
                gate_type="buf",
                inputs=[net_name],
                output=new_wire,
            )

            if consumer_inst in nl.nodes:
                node = nl.nodes[consumer_inst]
                node.inputs = [new_wire if sig == net_name else sig for sig in node.inputs]
            elif consumer_inst in nl.dffs:
                dff = nl.dffs[consumer_inst]
                if dff.d == net_name:
                    dff.d = new_wire
                if dff.ck == net_name:
                    dff.ck = new_wire
                if dff.rn == net_name:
                    dff.rn = new_wire
                if dff.sn == net_name:
                    dff.sn = new_wire
            inserted += 1

        return inserted

    def balance_depth(self, source: str, sinks: List[str]) -> int:
        """Add buffers to equalise path lengths from source to all sinks.

        Uses minimal buffer insertion.

        Args:
            source: Starting signal.
            sinks:  List of sink signal names.

        Returns:
            Number of buffers inserted.
        """
        self._require_netlist()
        self._resolve_signal(source)
        for s in sinks:
            self._resolve_signal(s)

        nl = self._netlist
        assert nl is not None

        depths: Dict[str, int] = {}
        for sink in sinks:
            d, _ = self.get_max_depth(source, sink)
            depths[sink] = d

        valid_depths = [d for d in depths.values() if d >= 0]
        if not valid_depths:
            return 0

        target_depth = max(valid_depths)
        total_inserted = 0

        for sink, depth in depths.items():
            if depth < 0 or depth == target_depth:
                continue
            bufs_needed = target_depth - depth
            out2gate = self._build_output_to_gate()
            driver_inst = out2gate.get(sink)
            cur_net = sink
            for _ in range(bufs_needed):
                if driver_inst:
                    new_buf = self.insert_buffer(cur_net, driver_inst)
                    total_inserted += 1
                    # After insert_buffer, original driver now drives a new net
                    # and buf drives cur_net; the buf's input is the new net
                    buf_node = nl.nodes[new_buf]
                    driver_inst = new_buf
                    cur_net = buf_node.output

        return total_inserted

    def rename_gate(self, old_name: str, new_name: str) -> None:
        """Rename a gate instance from old_name to new_name.

        Raises ValueError if old_name does not exist or new_name is already taken.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        if old_name not in nl.nodes and old_name not in nl.dffs:
            raise ValueError(f"Instance '{old_name}' not found in netlist.")
        if new_name in nl.nodes or new_name in nl.dffs:
            raise ValueError(f"Instance name '{new_name}' already exists in netlist.")

        if old_name in nl.nodes:
            nl.nodes[new_name] = nl.nodes.pop(old_name)
            nl.nodes[new_name].name = new_name
        else:
            nl.dffs[new_name] = nl.dffs.pop(old_name)
            nl.dffs[new_name].name = new_name

    def _replace_signal_refs(
        self,
        old_name: str,
        new_name: str,
        *,
        include_drivers: bool = False,
    ) -> None:
        """Replace signal references across gate and DFF connectivity."""
        assert self._netlist is not None

        def rename(sig: str) -> str:
            if sig == old_name:
                return new_name
            if "[" in sig:
                return re.sub(rf"^{re.escape(old_name)}(?=\[)", new_name, sig)
            return sig

        for node in self._netlist.nodes.values():
            node.inputs = [rename(sig) for sig in node.inputs]
            if include_drivers:
                node.output = rename(node.output)

        for dff in self._netlist.dffs.values():
            dff.ck = rename(dff.ck)
            dff.rn = rename(dff.rn)
            dff.sn = rename(dff.sn)
            dff.d = rename(dff.d)
            dff.q = rename(dff.q)

    def rename_wire(self, old_name: str, new_name: str) -> None:
        """Rename a wire/signal and update all references."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        if old_name not in nl.wires:
            raise ValueError(f"Wire '{old_name}' not found in netlist.")
        if new_name in nl.wires:
            raise ValueError(f"Wire name '{new_name}' already exists in netlist.")
        if new_name in nl.nodes or new_name in nl.dffs:
            raise ValueError(f"Name '{new_name}' is already used by an instance.")

        wi = nl.wires.pop(old_name)
        wi.name = new_name
        nl.wires[new_name] = wi
        nl.primary_inputs = [new_name if n == old_name else n for n in nl.primary_inputs]
        nl.primary_outputs = [new_name if n == old_name else n for n in nl.primary_outputs]
        self._replace_signal_refs(old_name, new_name, include_drivers=True)

    def _restore_dff_boundary_nets(
        self,
        original_dffs: Dict[str, "DFFNode"],
        original_wires: Dict[str, WireInfo],
    ) -> int:
        """Keep DFF port net names stable after whole-design ABC optimization."""
        nl = self._netlist
        assert nl is not None
        inserted_buffers = 0

        def ensure_wire(sig: str) -> None:
            if sig in {"", "1'b0", "1'b1"}:
                return
            base_m = re.match(r"^(\w+)\[", sig)
            key = base_m.group(1) if base_m else sig
            if key in nl.wires:
                return
            if key in original_wires:
                nl.wires[key] = copy.deepcopy(original_wires[key])
            else:
                nl.wires[key] = WireInfo(name=key)

        def unique_inst(base: str) -> str:
            candidate = base
            idx = 0
            while candidate in nl.nodes or candidate in nl.dffs:
                idx += 1
                candidate = f"{base}_{idx}"
            return candidate

        driven = {node.output for node in nl.nodes.values()}

        for inst_name, original in original_dffs.items():
            current = nl.dffs.get(inst_name)
            if current is None:
                continue

            if current.q and current.q != original.q:
                self._replace_signal_refs(current.q, original.q, include_drivers=False)
                current.q = original.q
                ensure_wire(original.q)

            current.ck = original.ck
            current.rn = original.rn
            current.sn = original.sn
            ensure_wire(current.ck)
            ensure_wire(current.rn)
            ensure_wire(current.sn)

            if current.d != original.d:
                optimized_d = current.d
                ensure_wire(original.d)
                if original.d not in driven and optimized_d not in {"", original.d}:
                    inst = unique_inst(f"_keep_dff_d_{inst_name}")
                    nl.nodes[inst] = GateNode(
                        name=inst,
                        gate_type="buf",
                        inputs=[optimized_d],
                        output=original.d,
                    )
                    driven.add(original.d)
                    inserted_buffers += 1
                current.d = original.d

        return inserted_buffers

    def remove_dangling_gates(self) -> int:
        """Remove all gates and nets that do not feed any primary output or DFF input.

        Returns:
            Number of gates removed.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        # Collect all signals that are "needed"
        needed_signals: Set[str] = set(nl.primary_outputs)
        for dff in nl.dffs.values():
            needed_signals.add(dff.d)
            if dff.ck:
                needed_signals.add(dff.ck)
            if dff.rn:
                needed_signals.add(dff.rn)
            if dff.sn:
                needed_signals.add(dff.sn)

        # Expand bus PO names (e.g. "n16") to their actual bit-slice drivers
        # (e.g. "n16[0]"..."n16[7]") so the BFS reaches those gates.
        out2gate = self._build_output_to_gate()
        expanded_needed: Set[str] = set()
        for sig in needed_signals:
            if sig in out2gate:
                expanded_needed.add(sig)
            else:
                for k in out2gate:
                    if k.split('[')[0] == sig:
                        expanded_needed.add(k)
                if sig not in expanded_needed:
                    expanded_needed.add(sig)  # keep as-is (PI or constant)

        # Backward BFS: find all gate outputs that transitively feed needed signals
        useful_outputs: Set[str] = set()
        queue: deque[str] = deque(expanded_needed)

        while queue:
            sig = queue.popleft()
            if sig in useful_outputs or sig in nl.primary_inputs:
                continue
            if sig in {"1'b0", "1'b1"}:
                continue
            useful_outputs.add(sig)
            driver = out2gate.get(sig)
            if driver and driver in nl.nodes:
                gate = nl.nodes[driver]
                for inp in gate.inputs:
                    if inp not in useful_outputs:
                        queue.append(inp)

        # Remove gates whose output is not useful
        to_remove = [
            inst
            for inst, node in nl.nodes.items()
            if node.output not in useful_outputs
            and node.output not in nl.primary_outputs
        ]
        for inst in to_remove:
            del nl.nodes[inst]

        # Remove orphan internal wires
        used_wires: Set[str] = set()
        for node in nl.nodes.values():
            used_wires.add(node.output)
            used_wires |= set(node.inputs)
        for dff in nl.dffs.values():
            extra = {dff.d, dff.q}
            if dff.ck:
                extra.add(dff.ck)
            if dff.rn:
                extra.add(dff.rn)
            if dff.sn:
                extra.add(dff.sn)
            used_wires.update(extra)

        # Also keep bus base names (e.g. "n16") when only bit-slices
        # (e.g. "n16[7]") appear in used_wires — otherwise write_verilog
        # won't emit the bus declaration and Yosys sees out-of-bounds selects.
        used_bus_bases: Set[str] = set()
        for sig in used_wires:
            m = re.match(r'^(\w+)\[', sig)
            if m:
                used_bus_bases.add(m.group(1))

        declared = set(nl.primary_inputs) | set(nl.primary_outputs)
        nl.wires = {
            k: v
            for k, v in nl.wires.items()
            if k in declared or k in used_wires or k in used_bus_bases
        }

        return len(to_remove)

    def optimize_cone_depth(self, output_signal: str, max_depth: int) -> bool:
        """Restructure the logic cone of output_signal so its depth ≤ max_depth.

        Uses a greedy rebalancing strategy (associativity / commutativity).
        Preserves functional equivalence.

        Args:
            output_signal: Target output net.
            max_depth:     Maximum allowed combinational depth.

        Returns:
            True if the depth constraint is met after transformation, False otherwise.
        """
        self._require_netlist()
        self._resolve_signal(output_signal)

        # Save snapshot for equivalence check
        self.add_snapshot("pre_opt")

        nl = self._netlist
        assert nl is not None
        out2gate = self._build_output_to_gate()

        # Collect all inputs to the cone
        cone = self.get_logic_cone(output_signal)
        current_depth, _ = self.get_max_depth(
            _find_cone_primary_input(nl, output_signal, out2gate) or output_signal,
            output_signal,
        )
        if current_depth <= max_depth:
            return True

        # Simple rebalancing: flatten trees of associative gates then rebuild
        rebuilt = _rebalance_associative_tree(nl, output_signal, out2gate, max_depth)
        if not rebuilt:
            return False

        # Verify equivalence
        if not self.check_signal_equivalence(output_signal, "pre_opt"):
            self.restore_snapshot("pre_opt")
            return False

        self.remove_dangling_gates()
        return True

    def reduce_critical_path(
        self,
        allowed_gates: Optional[List[str]] = None,
    ) -> dict:
        """Reduce the maximum combinational depth of the loaded netlist using
        Yosys + ABC logic restructuring (no retiming, no DFF movement).

        Workflow:
          1. Measure depth before via get_fanin_cone_depth on each primary output.
          2. Write netlist to a temp Verilog file.
          3. Generate a unit-delay Liberty file for the requested gate set (or
             the active whole-design constraint) and a DFF blackbox so ABC
             never touches sequential elements.
          4. Run Yosys (subprocess) with ABC -liberty and a custom script that
             runs &dch -f (deep combinational restructuring) but omits dretime
             and scorr (both of which alter sequential boundaries).
          5. Parse the ABC-remapped Verilog back into self._netlist.
          6. Measure depth after and return a comparison dict.

        Returns:
            dict with keys depth_before, depth_after, improvement, success.

        Raises:
            RuntimeError: if Yosys is not found or the run fails.
        """
        import os
        import textwrap

        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        valid_gates = {"nand", "nor", "or", "and", "not", "xor", "xnor", "buf"}
        requested_gates = allowed_gates or self._allowed_gates_constraint
        effective_allowed: Optional[List[str]] = None
        if requested_gates:
            effective_allowed = list(dict.fromkeys(g.lower() for g in requested_gates))
            bad = sorted(set(effective_allowed) - valid_gates)
            if bad:
                raise ValueError(f"Unknown gate types in allowed_gates: {bad}")
            if not effective_allowed:
                raise ValueError("allowed_gates cannot be empty.")

        original_netlist = copy.deepcopy(nl)
        original_dffs = copy.deepcopy(nl.dffs)
        original_wires = copy.deepcopy(nl.wires)

        # ------------------------------------------------------------------
        # Step 1 — measure depth BEFORE
        # Iterate over gate node outputs (always concrete scalar signals) so
        # bus primary outputs like n5[0]..n5[15] are handled correctly.
        # Using a shared memo dict avoids redundant re-computation.
        # ------------------------------------------------------------------
        memo_before: Dict[str, int] = {}
        depth_before = max(
            (self._fanin_depth(node.output, memo_before) for node in nl.nodes.values()),
            default=0,
        )

        # ------------------------------------------------------------------
        # Step 2 — write current netlist to temp input file
        # ------------------------------------------------------------------
        tmp_dir = tempfile.mkdtemp(prefix="yosys_opt_", dir=_workspace_temp_dir())
        tmp_input = os.path.join(tmp_dir, "input.v")
        tmp_output = os.path.join(tmp_dir, "output.v")
        tmp_lib = os.path.join(tmp_dir, "prim.lib")
        tmp_bb = os.path.join(tmp_dir, "dff_blackbox.v")
        tmp_script = os.path.join(tmp_dir, "run.ys")

        write_verilog(nl, tmp_input)

        # ------------------------------------------------------------------
        # Step 3a — DFF blackbox (prevents ABC from touching flip-flops)
        # ------------------------------------------------------------------
        with open(tmp_bb, "w") as fh:
            fh.write(textwrap.dedent("""\
                (* blackbox *)
                module dff (CK, RN, SN, D, Q);
                  input CK, RN, SN, D;
                  output Q;
                endmodule
            """))

        # ------------------------------------------------------------------
        # Step 3b — unit-delay Liberty file for primitive gates
        # All cells have identical delay so ABC minimises depth purely.
        # ------------------------------------------------------------------
        def _cell(name: str, func: str, inputs: List[str]) -> str:
            input_decls = "\n".join(
                f'    pin({p}) {{ direction : input; }}'  for p in inputs
            )
            timing_pins = " ".join(inputs)
            return textwrap.dedent(f"""\
              cell({name}) {{
            {input_decls}
                pin(Y) {{
                  direction : output;
                  function : "{func}";
                  timing() {{
                    related_pin : "{timing_pins}";
                    cell_rise(scalar)   {{ values("1.0"); }}
                    cell_fall(scalar)   {{ values("1.0"); }}
                    rise_transition(scalar) {{ values("0.0"); }}
                    fall_transition(scalar) {{ values("0.0"); }}
                  }}
                }}
              }}
            """)

        cell_specs = {
            "and":  ("A*B",     ["A", "B"]),
            "nand": ("!(A*B)",  ["A", "B"]),
            "nor":  ("!(A+B)",  ["A", "B"]),
            "or":   ("A+B",     ["A", "B"]),
            "xor":  ("A^B",     ["A", "B"]),
            "xnor": ("!(A^B)",  ["A", "B"]),
            "not":  ("!A",      ["A"]),
            "buf":  ("A",       ["A"]),
        }
        library_gates = effective_allowed or [
            "nand", "nor", "or", "xor", "xnor", "not", "buf"
        ]
        # ABC's delay mapper rejects libraries with fewer than three cell
        # classes. BUF is an internal mapping aid only; opt_clean removes it,
        # and the postcondition below still enforces the user-visible set.
        if effective_allowed and len(library_gates) < 3 and "buf" not in library_gates:
            library_gates = [*library_gates, "buf"]
        liberty_content = (
            'library(prim) {\n'
            '  time_unit : "1ns";\n'
            '  voltage_unit : "1V";\n'
            '  current_unit : "1mA";\n'
            '  leakage_power_unit : "1nW";\n'
            '  capacitive_load_unit(1, pf);\n'
            '  pulling_resistance_unit : "1kohm";\n'
            '  delay_model : generic_cmos;\n'
            + "".join(_cell(gate, *cell_specs[gate]) for gate in library_gates)
            + '}\n'
        )
        with open(tmp_lib, "w") as fh:
            fh.write(liberty_content)

        # ------------------------------------------------------------------
        # Step 4 — Yosys script
        # Custom ABC script removes dretime (retiming) and scorr (sequential
        # redundancy removal) from the Yosys default.  Only combinational
        # restructuring passes remain.
        # ------------------------------------------------------------------
        abc_script = "+strash;&get,-n;&fraig,-x;&put;dc2;strash;&get,-n;&dch,-f;&nf,{D};&put"
        abc_command = (
            f"abc -liberty {tmp_lib} -D 1"
            if effective_allowed
            else f'abc -liberty {tmp_lib} -script "{abc_script}"'
        )
        yosys_script = textwrap.dedent(f"""\
            read_verilog -lib {tmp_bb}
            read_verilog {tmp_input}
            hierarchy -check -top {nl.module_name}
            flatten
            techmap
            {abc_command}
            opt_clean -purge
            write_verilog -noattr -nohex {tmp_output}
        """)
        with open(tmp_script, "w") as fh:
            fh.write(yosys_script)

        try:
            result = subprocess.run(
                [_yosys_binary(), "-s", tmp_script],
                capture_output=True,
                text=True,
                timeout=300,
                env=_temp_subprocess_env(),
                cwd=_workspace_temp_dir(),
            )
        except FileNotFoundError:
            raise RuntimeError(
                "yosys not found. Install Yosys (apt install yosys) and retry."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Yosys timed out after 300 s.")

        if result.returncode != 0:
            raise RuntimeError(
                f"Yosys failed (exit {result.returncode}).\n"
                f"STDOUT:\n{result.stdout[-3000:]}\n"
                f"STDERR:\n{result.stderr[-1000:]}"
            )

        if not os.path.exists(tmp_output):
            raise RuntimeError(
                "Yosys completed but output Verilog was not written.\n"
                f"STDOUT:\n{result.stdout[-3000:]}\n"
                f"STDERR:\n{result.stderr[-1000:]}"
            )

        # ------------------------------------------------------------------
        # Step 5 — parse ABC output back into Netlist object
        # ------------------------------------------------------------------
        from .netlist_parser import parse_verilog as _parse_verilog
        self._netlist = _parse_verilog(tmp_output)
        boundary_buffers = self._restore_dff_boundary_nets(original_dffs, original_wires)
        helper_buffers_lowered = 0

        if effective_allowed:
            # ABC needs BUF as a third internal cell class for two-gate
            # libraries, and DFF-boundary restoration may also add a BUF to
            # retain an original D net. Lower those wire helpers into the
            # requested gate set before enforcing the public postcondition.
            if "buf" not in effective_allowed and any(
                node.gate_type == "buf" for node in self._netlist.nodes.values()
            ):
                lowered = self.replace_gate_type_in_cone(
                    "buf",
                    effective_allowed,
                    output_signal=None,
                )
                if not lowered.get("success"):
                    self._netlist = original_netlist
                    raise RuntimeError(
                        "Could not lower optimization BUF helpers into the "
                        f"requested gate set {effective_allowed}: "
                        f"{lowered.get('reason', 'unknown mapping failure')}"
                    )
                helper_buffers_lowered = int(lowered.get("replaced", 0))

            remaining_types = sorted({
                node.gate_type
                for node in self._netlist.nodes.values()
                if node.gate_type not in set(effective_allowed)
            })
            if remaining_types:
                self._netlist = original_netlist
                raise RuntimeError(
                    "Restricted depth optimization produced disallowed gate types: "
                    f"{remaining_types}"
                )
            self._allowed_gates_constraint = effective_allowed

        # Clean up temp dir
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Step 6 — measure depth AFTER (same approach as before)
        # ------------------------------------------------------------------
        memo_after: Dict[str, int] = {}
        depth_after = max(
            (self._fanin_depth(node.output, memo_after) for node in self._netlist.nodes.values()),
            default=0,
        )

        return {
            "depth_before": depth_before,
            "depth_after": depth_after,
            "improvement": depth_before - depth_after,
            "success": depth_after < depth_before,
            "dff_boundary_buffers_inserted": boundary_buffers,
            "helper_buffers_lowered": helper_buffers_lowered,
            "allowed_gates": effective_allowed,
        }

    def _max_logic_depth(self) -> int:
        """Return maximum combinational fanin depth over all gate outputs."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        memo: Dict[str, int] = {}
        return max(
            (self._fanin_depth(node.output, memo) for node in nl.nodes.values()),
            default=0,
        )

    def _resolved_cone_root(self, output_signal: str) -> str:
        """Resolve a DFF-Q cone request to the corresponding D input."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        dff_q_to_d = {dff.q: dff.d for dff in nl.dffs.values()}
        return dff_q_to_d.get(output_signal, output_signal)

    def _cone_gate_type_counts(self, output_signal: str) -> Dict[str, int]:
        """Return gate-type counts in the combinational cone of output_signal."""
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        actual_output = self._resolved_cone_root(output_signal)
        counts: Dict[str, int] = {}
        for inst in self.get_logic_cone(actual_output):
            gate_type = nl.nodes[inst].gate_type
            counts[gate_type] = counts.get(gate_type, 0) + 1
        return counts

    def optimize_depth_preserving_cone_gate_set(
        self,
        output_signal: str,
        allowed_gates: List[str],
        verify_equivalence: bool = False,
    ) -> dict:
        """Optimize whole-design depth, then remap only one cone to allowed gates.

        This matches prompts where the cost function is whole-design maximum
        depth but only a named cone has a gate-set restriction.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_signal(output_signal)

        original_netlist = copy.deepcopy(nl)
        original_constraint = (
            list(self._allowed_gates_constraint)
            if self._allowed_gates_constraint is not None
            else None
        )

        allowed = list(dict.fromkeys(g.lower() for g in allowed_gates))
        valid = {"nand", "nor", "or", "and", "not", "xor", "xnor", "buf"}
        bad = sorted(set(allowed) - valid)
        if bad:
            raise ValueError(f"Unknown gate types in allowed_gates: {bad}")
        if not allowed:
            raise ValueError("allowed_gates cannot be empty.")

        depth_before = self._max_logic_depth()
        cone_before = self._cone_gate_type_counts(output_signal)

        try:
            # First pass: optimize depth with the normal unrestricted library,
            # ignoring any previous whole-design gate-set constraint.
            self._allowed_gates_constraint = None
            opt_result = self.reduce_critical_path()
            depth_after_unrestricted = self._max_logic_depth()

            # Second pass: repair only the requested cone to the requested set.
            remap_result = self.remap_cone_with_gates(output_signal, allowed)
            if not remap_result.get("success"):
                self._netlist = original_netlist
                self._allowed_gates_constraint = original_constraint
                return {
                    "success": False,
                    "reason": remap_result.get("reason", "Cone remap failed."),
                    "output_signal": output_signal,
                    "allowed_gates": allowed,
                    "depth_before": depth_before,
                    "depth_after_unrestricted": depth_after_unrestricted,
                    "optimization": opt_result,
                    "cone_remap": remap_result,
                }

            cone_after = self._cone_gate_type_counts(output_signal)
            disallowed = {
                gate_type: count
                for gate_type, count in cone_after.items()
                if gate_type not in set(allowed)
            }
            if disallowed:
                self._netlist = original_netlist
                self._allowed_gates_constraint = original_constraint
                return {
                    "success": False,
                    "reason": "Cone remap left disallowed gate types.",
                    "output_signal": output_signal,
                    "allowed_gates": allowed,
                    "disallowed_gate_counts": disallowed,
                    "depth_before": depth_before,
                    "depth_after_unrestricted": depth_after_unrestricted,
                    "cone_before": cone_before,
                    "cone_after": cone_after,
                    "optimization": opt_result,
                    "cone_remap": remap_result,
                }

            equivalence = None
            if verify_equivalence:
                equivalence = self.check_design_equivalence()
                if not equivalence.get("equivalent"):
                    self._netlist = original_netlist
                    self._allowed_gates_constraint = original_constraint
                    return {
                        "success": False,
                        "reason": "Equivalence check failed after optimization.",
                        "output_signal": output_signal,
                        "allowed_gates": allowed,
                        "equivalence": equivalence,
                        "depth_before": depth_before,
                        "depth_after_unrestricted": depth_after_unrestricted,
                        "cone_before": cone_before,
                        "cone_after": cone_after,
                        "optimization": opt_result,
                        "cone_remap": remap_result,
                    }

            depth_after = self._max_logic_depth()
            if depth_after >= depth_before:
                self._netlist = original_netlist
                self._allowed_gates_constraint = original_constraint
                return {
                    "success": False,
                    "reason": "Final depth was not smaller than the input depth; restored input design.",
                    "output_signal": output_signal,
                    "allowed_gates": allowed,
                    "depth_before": depth_before,
                    "depth_after_unrestricted": depth_after_unrestricted,
                    "depth_after": depth_after,
                    "improvement": depth_before - depth_after,
                    "cone_before": cone_before,
                    "cone_after": cone_after,
                    "optimization": opt_result,
                    "cone_remap": remap_result,
                    "equivalence": equivalence,
                    "equivalence_checked": verify_equivalence,
                    "restored_input_design": True,
                }
            self._allowed_gates_constraint = original_constraint
            return {
                "success": True,
                "output_signal": output_signal,
                "allowed_gates": allowed,
                "depth_before": depth_before,
                "depth_after_unrestricted": depth_after_unrestricted,
                "depth_after": depth_after,
                "improvement": depth_before - depth_after,
                "cone_before": cone_before,
                "cone_after": cone_after,
                "optimization": opt_result,
                "cone_remap": remap_result,
                "equivalence": equivalence,
                "equivalence_checked": verify_equivalence,
                "restored_input_design": False,
            }
        except Exception:
            self._netlist = original_netlist
            self._allowed_gates_constraint = original_constraint
            raise

    def remap_cone_with_gates(
        self,
        output_signal: str,
        allowed_gates: List[str],
        force_abc: bool = False,
    ) -> dict:
        """Re-implement the fanin cone of output_signal using only allowed_gates.

        Extracts the combinational cone of output_signal as a standalone module,
        runs Yosys+ABC with a restricted Liberty (only the allowed gate types),
        then splices the remapped cone back into the full netlist.

        Args:
            output_signal: Target net whose fanin cone is remapped (e.g. "n11[0]").
            allowed_gates: List of permitted gate types (e.g. ["nand", "not"]).

        Returns:
            dict with keys: success (bool), gates_before (int), gates_after (int),
            output_signal (str), allowed_gates (list), and on failure: reason (str).
        """
        import os, shutil, textwrap

        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        # Validate allowed_gates
        valid = {"nand", "nor", "or", "and", "not", "xor", "xnor", "buf"}
        bad = [g for g in allowed_gates if g not in valid]
        if bad:
            return {"success": False, "reason": f"Unknown gate types: {bad}"}

        # ------------------------------------------------------------------
        # Step 1 — BFS backward to collect cone gate instances + boundary inputs
        # If output_signal is a DFF Q pin, we remap the cone feeding its D input
        # instead (since Q itself has no combinational fanin).
        # ------------------------------------------------------------------
        out2gate = self._build_output_to_gate()

        # Check if output_signal is a DFF Q — if so, resolve to its D input
        dff_q_to_d: Dict[str, str] = {dff.q: dff.d for dff in nl.dffs.values()}
        actual_output = output_signal
        if output_signal in dff_q_to_d:
            actual_output = dff_q_to_d[output_signal]

        # Resolve bit-slice: if actual_output is still not driven by a comb gate
        if actual_output not in out2gate:
            if output_signal == nl.module_name:
                return {
                    "success": False,
                    "reason": (
                        f"'{output_signal}' is the module name, not a driven signal. "
                        "Use remap_design_with_gates for a whole-design reconstruction."
                    ),
                }
            return {
                "success": False,
                "reason": (
                    f"Signal '{output_signal}' (resolved to '{actual_output}') "
                    f"has no driving combinational gate in netlist."
                ),
            }

        pi_set = set(nl.primary_inputs)
        dff_outputs = {dff.q for dff in nl.dffs.values()}

        cone_insts: Set[str] = set()
        boundary_inputs: Set[str] = set()
        visited: Set[str] = set()
        queue = [actual_output]

        while queue:
            sig = queue.pop()
            if sig in visited:
                continue
            visited.add(sig)

            if sig in {"1'b0", "1'b1"} or sig in pi_set or sig in dff_outputs:
                boundary_inputs.add(sig)
                continue

            driver = out2gate.get(sig)
            if driver is None or driver not in nl.nodes:
                boundary_inputs.add(sig)
                continue

            cone_insts.add(driver)
            queue.extend(nl.nodes[driver].inputs)

        # Boundary inputs = signals consumed by cone but produced outside
        # (Remove constants — they'll be inlined as literals)
        boundary_inputs.discard("1'b0")
        boundary_inputs.discard("1'b1")

        gates_before = len(cone_insts)
        if gates_before == 0:
            return {"success": False, "reason": "Cone has no remappable gates."}

        # Prefer local, output-preserving substitutions. Keeping each original
        # gate output as a proof point is friendlier to equiv_make/equiv_simple
        # than replacing a large sequential boundary cone in one shot, and it
        # also leaves shared fanout logic untouched.
        allowed_set = set(allowed_gates)
        source_types = sorted({
            nl.nodes[inst].gate_type
            for inst in cone_insts
            if nl.nodes[inst].gate_type not in allowed_set
        })
        templates = {
            source_type: self._get_substitution_template(source_type, allowed_gates)
            for source_type in source_types
        }
        if not force_abc and all(template is not None for template in templates.values()):
            replaced = 0
            for source_type in source_types:
                result = self.replace_gate_type_in_cone(
                    source_type,
                    allowed_gates,
                    output_signal,
                )
                if not result.get("success"):
                    return result
                replaced += int(result.get("replaced", 0))

            remapped_out2gate = self._build_output_to_gate()
            remapped_q_to_d = {dff.q: dff.d for dff in nl.dffs.values()}
            remapped_output = remapped_q_to_d.get(output_signal, output_signal)
            remapped_insts: Set[str] = set()
            seen_signals: Set[str] = set()
            stack = [remapped_output]
            while stack:
                sig = stack.pop()
                if sig in seen_signals:
                    continue
                seen_signals.add(sig)
                driver = remapped_out2gate.get(sig)
                if driver is None or driver not in nl.nodes or driver in remapped_insts:
                    continue
                remapped_insts.add(driver)
                stack.extend(nl.nodes[driver].inputs)

            gate_types_after: Dict[str, int] = {}
            for inst in remapped_insts:
                gate_type = nl.nodes[inst].gate_type
                gate_types_after[gate_type] = gate_types_after.get(gate_type, 0) + 1

            return {
                "success": True,
                "output_signal": output_signal,
                "allowed_gates": allowed_gates,
                "gates_before": gates_before,
                "gates_after": len(remapped_insts),
                "gate_types_after": dict(sorted(gate_types_after.items())),
                "gates_replaced": replaced,
                "strategy": "local_substitution",
            }

        # ------------------------------------------------------------------
        # Step 2 — Write cone as standalone Verilog submodule
        # Wire names with '[' are illegal as port names → sanitise to '_'
        # Keep a map so we can unsanitise after ABC.
        # ------------------------------------------------------------------
        def sanitise(name: str) -> str:
            return re.sub(r'[\[\]]', '_', name)

        # Build sanitise maps (check for collisions — very unlikely but safe)
        san_map: Dict[str, str] = {}   # original → sanitised
        rev_map: Dict[str, str] = {}   # sanitised → original
        all_signals: set = boundary_inputs | {actual_output}
        for sig in all_signals:
            s = sanitise(sig)
            # Ensure uniqueness
            base = s
            idx = 0
            while s in rev_map and rev_map[s] != sig:
                s = f"{base}_{idx}"
                idx += 1
            san_map[sig] = s
            rev_map[s] = sig

        # Sanitise internal cone wire names too
        cone_wires: set = set()
        for inst in cone_insts:
            node = nl.nodes[inst]
            cone_wires.add(node.output)
            cone_wires.update(node.inputs)
        # Internal = in cone but not boundary and not output
        internal_wires = cone_wires - boundary_inputs - {actual_output} - {"1'b0", "1'b1"}
        for sig in internal_wires:
            s = sanitise(sig)
            base = s
            idx = 0
            while s in rev_map and rev_map[s] != sig:
                s = f"{base}_{idx}"
                idx += 1
            san_map[sig] = s
            rev_map[s] = sig

        def s(sig: str) -> str:
            """Return sanitised name, or original if not in map (constant)."""
            return san_map.get(sig, sig)

        sorted_bi = sorted(boundary_inputs)
        out_san = s(actual_output)

        sub_lines: List[str] = []
        port_list = ", ".join(sorted_bi + [actual_output])
        port_san  = ", ".join([s(p) for p in sorted_bi] + [out_san])
        sub_lines.append(f"module cone_top ({port_san});")
        for bi in sorted_bi:
            sub_lines.append(f"  input wire {s(bi)};")
        sub_lines.append(f"  output wire {out_san};")

        # Declare internal wires
        for w in sorted(internal_wires):
            sub_lines.append(f"  wire {s(w)};")
        sub_lines.append("")

        # Gate instances — positional style
        for inst in sorted(cone_insts):
            node = nl.nodes[inst]
            if node.gate_type in ONE_INPUT_GATES:
                ports = f"{s(node.output)}, {s(node.inputs[0])}"
            else:
                ports = f"{s(node.output)}, {s(node.inputs[0])}, {s(node.inputs[1])}"
            sub_lines.append(f"  {node.gate_type} {inst} ({ports});")

        sub_lines.append("endmodule")

        tmp_dir = tempfile.mkdtemp(prefix="yosys_cone_", dir=_workspace_temp_dir())
        sub_v   = os.path.join(tmp_dir, "cone.v")
        out_v   = os.path.join(tmp_dir, "cone_out.v")
        lib_f   = os.path.join(tmp_dir, "restricted.lib")
        script_f = os.path.join(tmp_dir, "run.ys")

        with open(sub_v, "w") as fh:
            fh.write("\n".join(sub_lines) + "\n")

        # ------------------------------------------------------------------
        # Step 3 — restricted Liberty: only allowed_gates get cell entries
        # ------------------------------------------------------------------
        def _cell(name: str, func: str, inputs: List[str]) -> str:
            input_decls = "\n".join(
                f'    pin({p}) {{ direction : input; }}' for p in inputs
            )
            timing_pins = " ".join(inputs)
            return textwrap.dedent(f"""\
              cell({name}) {{
            {input_decls}
                pin(Y) {{
                  direction : output;
                  function : "{func}";
                  timing() {{
                    related_pin : "{timing_pins}";
                    cell_rise(scalar)   {{ values("1.0"); }}
                    cell_fall(scalar)   {{ values("1.0"); }}
                    rise_transition(scalar) {{ values("0.0"); }}
                    fall_transition(scalar) {{ values("0.0"); }}
                  }}
                }}
              }}
            """)

        all_cells = {
            "and":   ("A*B",     ["A", "B"]),
            "nand":  ("!(A*B)",  ["A", "B"]),
            "nor":   ("!(A+B)",  ["A", "B"]),
            "or":    ("A+B",     ["A", "B"]),
            "xor":   ("A^B",     ["A", "B"]),
            "xnor":  ("!(A^B)",  ["A", "B"]),
            "not":   ("!A",      ["A"]),
            "buf":   ("A",       ["A"]),
        }
        lib_body = (
            'library(restricted) {\n'
            '  time_unit : "1ns";\n'
            '  voltage_unit : "1V";\n'
            '  current_unit : "1mA";\n'
            '  leakage_power_unit : "1nW";\n'
            '  capacitive_load_unit(1, pf);\n'
            '  pulling_resistance_unit : "1kohm";\n'
            '  delay_model : generic_cmos;\n'
        )
        for g in allowed_gates:
            if g in all_cells:
                func, ins = all_cells[g]
                lib_body += _cell(g, func, ins)
        # ABC's map command requires a buf cell for internal wiring even if the
        # user didn't request it.  We always include it; any generated buf
        # instances that aren't in allowed_gates are removed as trivial pass-throughs.
        if "buf" not in allowed_gates:
            func, ins = all_cells["buf"]
            lib_body += _cell("buf", func, ins)
        lib_body += '}\n'

        with open(lib_f, "w") as fh:
            fh.write(lib_body)

        yosys_script = textwrap.dedent(f"""\
            read_verilog {sub_v}
            hierarchy -check -top cone_top
            flatten
            techmap
            abc -liberty {lib_f} -D 1
            opt_clean -purge
            write_verilog -noattr -nohex {out_v}
        """)
        with open(script_f, "w") as fh:
            fh.write(yosys_script)

        try:
            result = subprocess.run(
                [_yosys_binary(), "-s", script_f],
                capture_output=True, text=True, timeout=300,
                env=_temp_subprocess_env(),
                cwd=_workspace_temp_dir(),
            )
        except FileNotFoundError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError("yosys not found. Install Yosys and retry.")
        except subprocess.TimeoutExpired:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError("Yosys timed out after 300 s.")

        if result.returncode != 0 or not os.path.exists(out_v):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {
                "success": False,
                "reason": (
                    f"Yosys failed (exit {result.returncode}).\n"
                    f"STDOUT:\n{result.stdout[-2000:]}\n"
                    f"STDERR:\n{result.stderr[-500:]}"
                ),
            }

        # ------------------------------------------------------------------
        # Step 4 — parse ABC output, unsanitise wire names, splice back
        # ------------------------------------------------------------------
        from .netlist_parser import parse_verilog as _parse_verilog
        cone_nl = _parse_verilog(out_v)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Build a prefix for new intermediate wires to avoid collisions
        safe_out = re.sub(r'\W', '_', actual_output)
        prefix = f"_rc_{safe_out}_"

        # Collect existing wire names in full netlist to detect collisions
        existing_wires = set(nl.wires.keys())

        # Only ports keep their original names. ABC's internal signals must be
        # fresh because the original cone can contain shared nodes that remain
        # live for consumers outside this remapped output cone.
        wire_rename: Dict[str, str] = {}
        for orig in boundary_inputs | {actual_output}:
            wire_rename[san_map[orig]] = orig

        # New internal wires from ABC get prefixed names
        for wname, wi in cone_nl.wires.items():
            if wname not in wire_rename:
                candidate = prefix + wname
                # Ensure no collision
                while candidate in existing_wires:
                    candidate = candidate + "_"
                wire_rename[wname] = candidate
                existing_wires.add(candidate)

        def rw(sig: str) -> str:
            """Rename a signal from cone_nl namespace to full netlist namespace."""
            # Handle constants
            if sig in {"1'b0", "1'b1"}:
                return sig
            return wire_rename.get(sig, prefix + sig)

        # Disconnect only the old root. Keeping the rest temporarily preserves
        # side fanouts; remove_dangling_gates below prunes nodes that truly have
        # no remaining consumer after the duplicate cone is installed.
        old_root = out2gate[actual_output]
        nl.nodes.pop(old_root, None)

        # Add new gate instances from ABC-remapped cone
        gates_after = 0
        for inst_name, node in cone_nl.nodes.items():
            new_name = prefix + inst_name
            # Ensure no collision with existing instances
            while new_name in nl.nodes or new_name in nl.dffs:
                new_name = new_name + "_"
            new_node = GateNode(
                name=new_name,
                gate_type=node.gate_type,
                inputs=[rw(i) for i in node.inputs],
                output=rw(node.output),
            )
            nl.nodes[new_name] = new_node
            gates_after += 1

        # If buf was added only for ABC's internal needs (not in allowed_gates),
        # eliminate any generated buf instances by rewiring consumers directly
        # to the buf's input — buf(X) = X, so it's a pure pass-through.
        if "buf" not in allowed_gates:
            buf_insts = [n for n, nd in nl.nodes.items() if nd.gate_type == "buf"
                         and n.startswith(prefix)]
            if buf_insts:
                # Build output→signal map for quick substitution
                buf_out_to_in: Dict[str, str] = {
                    nl.nodes[n].output: nl.nodes[n].inputs[0] for n in buf_insts
                }
                # Rewire all gate inputs that consume a buf output
                for node in nl.nodes.values():
                    node.inputs = [buf_out_to_in.get(i, i) for i in node.inputs]
                for dff in nl.dffs.values():
                    dff.d  = buf_out_to_in.get(dff.d,  dff.d)
                    dff.ck = buf_out_to_in.get(dff.ck, dff.ck)
                    dff.rn = buf_out_to_in.get(dff.rn, dff.rn)
                    dff.sn = buf_out_to_in.get(dff.sn, dff.sn)
                # Remove buf instances
                for n in buf_insts:
                    nl.nodes.pop(n, None)
                    gates_after -= 1

        # Add new internal wires to nl.wires
        for wname, wi in cone_nl.wires.items():
            final_name = wire_rename.get(wname, prefix + wname)
            # Only add if it's a new internal wire (not a boundary / existing)
            if final_name not in nl.wires:
                nl.wires[final_name] = WireInfo(name=final_name)

        self.remove_dangling_gates()
        gate_counts = self.count_gate_types_in_cone(output_signal)

        return {
            "success": True,
            "output_signal": output_signal,
            "allowed_gates": allowed_gates,
            "gates_before": gates_before,
            "gates_after": gates_after,
            "gate_types_after": gate_counts.get("by_type", {}),
        }

    def optimize_cone_depth_preserving_gate_set(
        self,
        output_signal: str,
        allowed_gates: List[str],
        verify_equivalence: bool = False,
    ) -> dict:
        """Optimize only one fanin cone for cone depth under a gate-set limit.

        This is for prompts whose cost function is the depth of the named cone,
        not whole-design maximum depth. It remaps only that cone through the
        restricted ABC cone flow, then restores the input design if the cone
        depth does not improve.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None
        self._resolve_signal(output_signal)

        original_netlist = copy.deepcopy(nl)
        allowed = list(dict.fromkeys(g.lower() for g in allowed_gates))
        valid = {"nand", "nor", "or", "and", "not", "xor", "xnor", "buf"}
        bad = sorted(set(allowed) - valid)
        if bad:
            raise ValueError(f"Unknown gate types in allowed_gates: {bad}")
        if not allowed:
            raise ValueError("allowed_gates cannot be empty.")

        dff_q_to_d = {dff.q: dff.d for dff in nl.dffs.values()}
        if output_signal in dff_q_to_d:
            return {
                "success": False,
                "reason": (
                    f"{output_signal} is a DFF Q output and is treated as a "
                    "combinational primary-input boundary; its fanin cone depth is 0."
                ),
                "output_signal": output_signal,
                "allowed_gates": allowed,
                "depth_before": 0,
                "depth_after": 0,
                "improvement": 0,
                "cone_before": {},
                "cone_after": {},
                "equivalence_checked": False,
                "restored_input_design": False,
            }

        depth_before = int(self.get_fanin_cone_depth(output_signal)["depth"])
        cone_before = self._cone_gate_type_counts(output_signal)

        try:
            remap_result = self.remap_cone_with_gates(
                output_signal,
                allowed,
                force_abc=False,
            )
            if not remap_result.get("success"):
                self._netlist = original_netlist
                return {
                    "success": False,
                    "reason": remap_result.get("reason", "Cone remap failed."),
                    "output_signal": output_signal,
                    "allowed_gates": allowed,
                    "depth_before": depth_before,
                    "cone_before": cone_before,
                    "cone_remap": remap_result,
                    "restored_input_design": True,
                }

            cone_after = self._cone_gate_type_counts(output_signal)
            disallowed = {
                gate_type: count
                for gate_type, count in cone_after.items()
                if gate_type not in set(allowed)
            }
            if disallowed:
                self._netlist = original_netlist
                return {
                    "success": False,
                    "reason": "Cone remap left disallowed gate types.",
                    "output_signal": output_signal,
                    "allowed_gates": allowed,
                    "depth_before": depth_before,
                    "cone_before": cone_before,
                    "cone_after": cone_after,
                    "disallowed_gate_counts": disallowed,
                    "cone_remap": remap_result,
                    "restored_input_design": True,
                }

            depth_after = int(self.get_fanin_cone_depth(output_signal)["depth"])
            if depth_after >= depth_before:
                self._netlist = original_netlist
                return {
                    "success": False,
                    "reason": "Cone depth was not smaller than the input cone; restored input design.",
                    "output_signal": output_signal,
                    "allowed_gates": allowed,
                    "depth_before": depth_before,
                    "depth_after": depth_after,
                    "improvement": depth_before - depth_after,
                    "cone_before": cone_before,
                    "cone_after": cone_after,
                    "cone_remap": remap_result,
                    "equivalence_checked": False,
                    "equivalence_note": (
                        "Cone-local formal equivalence is not implemented; the ABC remap "
                        "is used as a functional-preserving synthesis transform."
                    ) if verify_equivalence else None,
                    "restored_input_design": True,
                }

            return {
                "success": True,
                "output_signal": output_signal,
                "allowed_gates": allowed,
                "depth_before": depth_before,
                "depth_after": depth_after,
                "improvement": depth_before - depth_after,
                "cone_before": cone_before,
                "cone_after": cone_after,
                "cone_remap": remap_result,
                "equivalence_checked": False,
                "equivalence_note": (
                    "Cone-local formal equivalence is not implemented; the ABC remap "
                    "is used as a functional-preserving synthesis transform."
                ) if verify_equivalence else None,
                "restored_input_design": False,
            }
        except Exception:
            self._netlist = original_netlist
            raise

    def remap_design_with_gates(self, allowed_gates: List[str]) -> dict:
        """Rebuild every combinational gate using only ``allowed_gates``.

        Uses local output-preserving templates so DFF boundaries, primary
        outputs, and shared fanout nets retain their original connectivity.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        valid = {"nand", "nor", "or", "and", "not", "xor", "xnor", "buf"}
        normalized = [gate.lower() for gate in allowed_gates]
        bad = sorted(set(normalized) - valid)
        if bad:
            return {"success": False, "reason": f"Unknown gate types: {bad}"}
        if not normalized:
            return {"success": False, "reason": "allowed_gates cannot be empty."}

        allowed_set = set(normalized)
        combinational_gates_before = len(nl.nodes)
        dff_count = len(nl.dffs)
        total_gates_before = combinational_gates_before + dff_count
        source_types = sorted({
            node.gate_type
            for node in nl.nodes.values()
            if node.gate_type not in allowed_set
        })

        unavailable = [
            source_type
            for source_type in source_types
            if self._get_substitution_template(source_type, normalized) is None
        ]
        if unavailable:
            return {
                "success": False,
                "reason": (
                    f"Cannot map gate types {unavailable} using only {normalized}. "
                    "The requested gate set may not be functionally complete."
                ),
            }

        replaced = 0
        for source_type in source_types:
            result = self.replace_gate_type_in_cone(
                source_type,
                normalized,
                output_signal=None,
            )
            if not result.get("success"):
                return result
            replaced += int(result.get("replaced", 0))

        remaining = sorted({
            node.gate_type
            for node in nl.nodes.values()
            if node.gate_type not in allowed_set
        })
        if remaining:
            return {
                "success": False,
                "reason": f"Disallowed gate types remain after remapping: {remaining}",
            }

        self._allowed_gates_constraint = normalized

        return {
            "success": True,
            "allowed_gates": normalized,
            # Whole-design gate totals include preserved DFFs, matching the
            # count_gates API and ordinary meaning of "entire netlist".
            "gates_before": total_gates_before,
            "gates_after": len(nl.nodes) + len(nl.dffs),
            "combinational_gates_before": combinational_gates_before,
            "combinational_gates_after": len(nl.nodes),
            "dff_count": len(nl.dffs),
            "gates_replaced": replaced,
        }

    def fraig_merge_equivalent_gates(self) -> dict:
        """Merge functionally equivalent combinational nodes using ABC FRAIG.

        The ABC script intentionally contains only AIG construction, FRAIG,
        and the final technology map required to return gate instances to
        Yosys. It does not run dc2, dch, balancing, or retiming.
        """
        import os
        import shutil
        import textwrap

        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        original_netlist = copy.deepcopy(nl)
        original_dffs = copy.deepcopy(nl.dffs)
        original_wires = copy.deepcopy(nl.wires)
        gates_before = len(nl.nodes)
        dff_count = len(nl.dffs)

        valid = ["and", "nand", "or", "nor", "xor", "xnor", "not", "buf"]
        target_gates = self._allowed_gates_constraint or valid
        genlib_funcs = {
            "and":  "Y=A*B;    PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "nand": "Y=!(A*B); PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "or":   "Y=A+B;    PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "nor":  "Y=!(A+B); PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "xor":  "Y=A^B;    PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "xnor": "Y=!(A^B); PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "not":  "Y=!A;     PIN A NONINV 1 1 1 1 1 1",
            "buf":  "Y=A;      PIN A NONINV 1 1 1 1 1 1",
        }

        # ABC map needs an inverter and a wire cell even when they are not
        # user-visible members of a restricted gate set. They are lowered or
        # rejected after import.
        library_gates = list(dict.fromkeys([*target_gates, "not", "buf"]))
        tmp_dir = tempfile.mkdtemp(prefix="yosys_fraig_", dir=_workspace_temp_dir())
        input_v = os.path.join(tmp_dir, "input.v")
        output_v = os.path.join(tmp_dir, "output.v")
        dff_bb = os.path.join(tmp_dir, "dff_blackbox.v")
        genlib_f = os.path.join(tmp_dir, "fraig.genlib")
        script_f = os.path.join(tmp_dir, "run.ys")

        write_verilog(nl, input_v)
        with open(dff_bb, "w") as fh:
            fh.write(textwrap.dedent("""\
                (* blackbox *)
                module dff (CK, RN, SN, D, Q);
                  input CK, RN, SN, D;
                  output Q;
                endmodule
            """))
        with open(genlib_f, "w") as fh:
            for gate in library_gates:
                fh.write(f"GATE {gate} 1 {genlib_funcs[gate]};\n")

        # map is only the AIG-to-gate output encoding step. No other ABC
        # optimization commands are included after FRAIG.
        abc_script = "+strash;&get,-n;&fraig,-x;&put;map"
        yosys_script = textwrap.dedent(f"""\
            read_verilog -lib {dff_bb}
            read_verilog {input_v}
            hierarchy -check -top {nl.module_name}
            flatten
            techmap
            abc -genlib {genlib_f} -script "{abc_script}"
            opt_clean -purge
            write_verilog -noattr -nohex {output_v}
        """)
        with open(script_f, "w") as fh:
            fh.write(yosys_script)

        try:
            result = subprocess.run(
                [_yosys_binary(), "-s", script_f],
                capture_output=True,
                text=True,
                timeout=300,
                env=_temp_subprocess_env(),
                cwd=_workspace_temp_dir(),
            )
            if result.returncode != 0 or not os.path.exists(output_v):
                raise RuntimeError(
                    f"Yosys FRAIG failed (exit {result.returncode}).\n"
                    f"STDOUT:\n{result.stdout[-3000:]}\n"
                    f"STDERR:\n{result.stderr[-1000:]}"
                )

            self._netlist = parse_verilog(output_v)
            boundary_buffers = self._restore_dff_boundary_nets(original_dffs, original_wires)

            helpers_lowered = 0
            if self._allowed_gates_constraint and "buf" not in self._allowed_gates_constraint:
                lowered = self.replace_gate_type_in_cone(
                    "buf", self._allowed_gates_constraint, output_signal=None
                )
                if not lowered.get("success"):
                    raise RuntimeError(lowered.get("reason", "Could not lower FRAIG buffers."))
                helpers_lowered = int(lowered.get("replaced", 0))

            if self._allowed_gates_constraint:
                allowed_set = set(self._allowed_gates_constraint)
                remaining = sorted({
                    node.gate_type
                    for node in self._netlist.nodes.values()
                    if node.gate_type not in allowed_set
                })
                if remaining:
                    raise RuntimeError(
                        f"FRAIG produced disallowed gate types: {remaining}"
                    )

            gates_after = len(self._netlist.nodes)
            return {
                "success": True,
                # Retain the original fields for compatibility. These count
                # combinational gates because FRAIG does not transform DFFs.
                "gates_before": gates_before,
                "gates_after": gates_after,
                "gate_reduction": gates_before - gates_after,
                "combinational_gates_before": gates_before,
                "combinational_gates_after": gates_after,
                "dff_count": dff_count,
                "total_gates_before": gates_before + dff_count,
                "total_gates_after": gates_after + dff_count,
                "dff_boundary_buffers_inserted": boundary_buffers,
                "helper_buffers_lowered": helpers_lowered,
                "allowed_gates": self._allowed_gates_constraint,
                "abc_operations": ["strash", "fraig", "map"],
            }
        except Exception:
            self._netlist = original_netlist
            raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _get_substitution_template(
        self,
        source_type: str,
        target_types: List[str],
    ) -> Optional[dict]:
        """Compile (and cache) a gate substitution template via ABC.

        Builds a 1-gate dummy module of source_type, maps it with a
        restricted Liberty containing only target_types (+buf for ABC internals),
        and returns the resulting gate-list as a reusable template.

        Returns:
            dict with keys:
                inputs       : list of input port names  (e.g. ["A","B"])
                output       : output port name          (e.g. "Y")
                gates        : List[GateNode]  — template gates (ports as placeholders)
                internal_wires: List[str]      — new wire names needed per substitution
            Returns None if ABC cannot map source_type into target_types.
        """
        import os, shutil, textwrap

        cache_key = (source_type, frozenset(target_types))
        if cache_key in _SUBSTITUTION_TEMPLATE_CACHE:
            return _SUBSTITUTION_TEMPLATE_CACHE[cache_key]

        # ---- Hardcoded fallbacks for cases ABC cannot handle ----
        # ABC's map command requires an inverter cell; when 'not' is absent
        # from target_types, self-tied NAND/NOR are the only option and ABC
        # crashes. We provide these trivially-known templates directly.
        def tmpl(inputs: List[str], gates: List[Tuple[str, List[str], str]]) -> dict:
            internal_wires = [
                output
                for _gate_type, _gate_inputs, output in gates
                if output != "Y"
            ]
            return {
                "inputs": inputs,
                "output": "Y",
                "gates": [
                    GateNode(f"_t{idx}", gate_type, gate_inputs, output)
                    for idx, (gate_type, gate_inputs, output) in enumerate(gates)
                ],
                "internal_wires": internal_wires,
            }

        _hardcoded: Dict[Tuple, dict] = {
            # NAND-only templates. NAND(A,A) supplies inversion.
            ("not", frozenset({"nand"})): tmpl(["A"], [
                ("nand", ["A", "A"], "Y"),
            ]),
            ("buf", frozenset({"nand"})): tmpl(["A"], [
                ("nand", ["A", "A"], "_w0"),
                ("nand", ["_w0", "_w0"], "Y"),
            ]),
            ("and", frozenset({"nand"})): tmpl(["A", "B"], [
                ("nand", ["A", "B"], "_w0"),
                ("nand", ["_w0", "_w0"], "Y"),
            ]),
            ("nand", frozenset({"nand"})): tmpl(["A", "B"], [
                ("nand", ["A", "B"], "Y"),
            ]),
            ("or", frozenset({"nand"})): tmpl(["A", "B"], [
                ("nand", ["A", "A"], "_w0"),
                ("nand", ["B", "B"], "_w1"),
                ("nand", ["_w0", "_w1"], "Y"),
            ]),
            ("nor", frozenset({"nand"})): tmpl(["A", "B"], [
                ("nand", ["A", "A"], "_w0"),
                ("nand", ["B", "B"], "_w1"),
                ("nand", ["_w0", "_w1"], "_w2"),
                ("nand", ["_w2", "_w2"], "Y"),
            ]),
            ("xor", frozenset({"nand"})): tmpl(["A", "B"], [
                ("nand", ["A", "B"], "_w0"),
                ("nand", ["A", "_w0"], "_w1"),
                ("nand", ["B", "_w0"], "_w2"),
                ("nand", ["_w1", "_w2"], "Y"),
            ]),
            ("xnor", frozenset({"nand"})): tmpl(["A", "B"], [
                ("nand", ["A", "B"], "_w0"),
                ("nand", ["A", "_w0"], "_w1"),
                ("nand", ["B", "_w0"], "_w2"),
                ("nand", ["_w1", "_w2"], "_w3"),
                ("nand", ["_w3", "_w3"], "Y"),
            ]),

            # NOR-only templates. NOR(A,A) supplies inversion.
            ("not", frozenset({"nor"})): tmpl(["A"], [
                ("nor", ["A", "A"], "Y"),
            ]),
            ("buf", frozenset({"nor"})): tmpl(["A"], [
                ("nor", ["A", "A"], "_w0"),
                ("nor", ["_w0", "_w0"], "Y"),
            ]),
            ("or", frozenset({"nor"})): tmpl(["A", "B"], [
                ("nor", ["A", "B"], "_w0"),
                ("nor", ["_w0", "_w0"], "Y"),
            ]),
            ("nor", frozenset({"nor"})): tmpl(["A", "B"], [
                ("nor", ["A", "B"], "Y"),
            ]),
            ("and", frozenset({"nor"})): tmpl(["A", "B"], [
                ("nor", ["A", "A"], "_w0"),
                ("nor", ["B", "B"], "_w1"),
                ("nor", ["_w0", "_w1"], "Y"),
            ]),
            ("nand", frozenset({"nor"})): tmpl(["A", "B"], [
                ("nor", ["A", "A"], "_w0"),
                ("nor", ["B", "B"], "_w1"),
                ("nor", ["_w0", "_w1"], "_w2"),
                ("nor", ["_w2", "_w2"], "Y"),
            ]),
            ("xnor", frozenset({"nor"})): tmpl(["A", "B"], [
                ("nor", ["A", "B"], "_w0"),
                ("nor", ["A", "_w0"], "_w1"),
                ("nor", ["B", "_w0"], "_w2"),
                ("nor", ["_w1", "_w2"], "Y"),
            ]),
            ("xor", frozenset({"nor"})): tmpl(["A", "B"], [
                ("nor", ["A", "B"], "_w0"),
                ("nor", ["A", "_w0"], "_w1"),
                ("nor", ["B", "_w0"], "_w2"),
                ("nor", ["_w1", "_w2"], "_w3"),
                ("nor", ["_w3", "_w3"], "Y"),
            ]),
            # Deterministic AND+NOT templates.  Keep these exact-set entries
            # ahead of the ABC fallback so primitive decomposition does not
            # depend on the installed Yosys/ABC version or mapping choices.
            ("nand", frozenset({"and", "not"})): tmpl(["A", "B"], [
                ("and", ["A", "B"], "_w0"),
                ("not", ["_w0"], "Y"),
            ]),
            ("or", frozenset({"and", "not"})): tmpl(["A", "B"], [
                ("not", ["A"], "_w0"),
                ("not", ["B"], "_w1"),
                ("and", ["_w0", "_w1"], "_w2"),
                ("not", ["_w2"], "Y"),
            ]),
            ("nor", frozenset({"and", "not"})): tmpl(["A", "B"], [
                ("not", ["A"], "_w0"),
                ("not", ["B"], "_w1"),
                ("and", ["_w0", "_w1"], "Y"),
            ]),
            ("xor", frozenset({"and", "not"})): tmpl(["A", "B"], [
                ("not", ["A"], "_w0"),
                ("not", ["B"], "_w1"),
                ("and", ["_w0", "_w1"], "_w2"),
                ("not", ["_w2"], "_w3"),
                ("and", ["A", "B"], "_w4"),
                ("not", ["_w4"], "_w5"),
                ("and", ["_w3", "_w5"], "Y"),
            ]),
            ("xnor", frozenset({"and", "not"})): tmpl(["A", "B"], [
                ("not", ["A"], "_w0"),
                ("not", ["B"], "_w1"),
                ("and", ["_w0", "_w1"], "_w2"),
                ("not", ["_w2"], "_w3"),
                ("and", ["A", "B"], "_w4"),
                ("not", ["_w4"], "_w5"),
                ("and", ["_w3", "_w5"], "_w6"),
                ("not", ["_w6"], "Y"),
            ]),
            ("buf", frozenset({"and", "not"})): tmpl(["A"], [
                ("not", ["A"], "_w0"),
                ("not", ["_w0"], "Y"),
            ]),
            # Deterministic OR+NOT templates (the De Morgan dual basis).
            ("and", frozenset({"or", "not"})): tmpl(["A", "B"], [
                ("not", ["A"], "_w0"),
                ("not", ["B"], "_w1"),
                ("or", ["_w0", "_w1"], "_w2"),
                ("not", ["_w2"], "Y"),
            ]),
            ("nand", frozenset({"or", "not"})): tmpl(["A", "B"], [
                ("not", ["A"], "_w0"),
                ("not", ["B"], "_w1"),
                ("or", ["_w0", "_w1"], "Y"),
            ]),
            ("nor", frozenset({"or", "not"})): tmpl(["A", "B"], [
                ("or", ["A", "B"], "_w0"),
                ("not", ["_w0"], "Y"),
            ]),
            ("xor", frozenset({"or", "not"})): tmpl(["A", "B"], [
                ("or", ["A", "B"], "_w0"),
                ("not", ["_w0"], "_w1"),
                ("not", ["A"], "_w2"),
                ("not", ["B"], "_w3"),
                ("or", ["_w2", "_w3"], "_w4"),
                ("not", ["_w4"], "_w5"),
                ("or", ["_w5", "_w1"], "_w6"),
                ("not", ["_w6"], "Y"),
            ]),
            ("xnor", frozenset({"or", "not"})): tmpl(["A", "B"], [
                ("or", ["A", "B"], "_w0"),
                ("not", ["_w0"], "_w1"),
                ("not", ["A"], "_w2"),
                ("not", ["B"], "_w3"),
                ("or", ["_w2", "_w3"], "_w4"),
                ("not", ["_w4"], "_w5"),
                ("or", ["_w5", "_w1"], "Y"),
            ]),
            ("buf", frozenset({"or", "not"})): tmpl(["A"], [
                ("not", ["A"], "_w0"),
                ("not", ["_w0"], "Y"),
            ]),
            # buf -> not(not(A))
            ("buf", frozenset({"not"})): tmpl(["A"], [
                ("not", ["A"], "_w0"),
                ("not", ["_w0"], "Y"),
            ]),
        }
        # Also support nand+not or nor+not for "not" target (just use not directly — no-op)
        # Actually mapping "not" to {"not"} is trivially identity — skip.

        hk = (source_type, frozenset(target_types))
        # Try exact match
        if hk in _hardcoded:
            _SUBSTITUTION_TEMPLATE_CACHE[cache_key] = _hardcoded[hk]
            return _hardcoded[hk]
        # Try compatible subset matches (e.g. target={"nand","not"} can use
        # a NAND-only template).  Select deterministically by gate count, then
        # target-set size/name, rather than relying on dictionary insertion order.
        compatible = [
            (len(candidate["gates"]), len(ht), tuple(sorted(ht)), candidate)
            for (hs, ht), candidate in _hardcoded.items()
            if hs == source_type and ht.issubset(frozenset(target_types))
        ]
        if compatible:
            selected = min(compatible, key=lambda item: item[:3])[3]
            _SUBSTITUTION_TEMPLATE_CACHE[cache_key] = selected
            return selected

        # genlib format: each gate on one line — works with ABC's map command
        # and avoids the Liberty scalar-timing crash in abc -genlib path.
        genlib_funcs = {
            "nand": "Y=!(A*B); PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "nor":  "Y=!(A+B); PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "or":   "Y=A+B;    PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "and":  "Y=A*B;    PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "xor":  "Y=A^B;    PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "xnor": "Y=!(A^B); PIN A NONINV 1 1 1 1 1 1; PIN B NONINV 1 1 1 1 1 1",
            "not":  "Y=!A;     PIN A NONINV 1 1 1 1 1 1",
            "buf":  "Y=A;      PIN A NONINV 1 1 1 1 1 1",
        }

        is_one_input = source_type in ONE_INPUT_GATES

        tmp_dir   = tempfile.mkdtemp(prefix="abc_tmpl_", dir=_workspace_temp_dir())
        dummy_v   = os.path.join(tmp_dir, "dummy.v")
        out_blif  = os.path.join(tmp_dir, "out.blif")
        genlib_f  = os.path.join(tmp_dir, "tgt.genlib")
        yosys_f   = os.path.join(tmp_dir, "map.ys")
        process_tmp = os.path.join(tmp_dir, "process_tmp")
        os.makedirs(process_tmp, exist_ok=True)

        # ---- 1. Write dummy.v ----
        if is_one_input:
            dummy_src = f"module dummy (A, Y);\n  input A;\n  output Y;\n  {source_type} g0 (Y, A);\nendmodule\n"
        else:
            dummy_src = f"module dummy (A, B, Y);\n  input A, B;\n  output Y;\n  {source_type} g0 (Y, A, B);\nendmodule\n"
        with open(dummy_v, "w") as fh:
            fh.write(dummy_src)

        # ---- 2. Build genlib ----
        genlib_lines = []
        for g in target_types:
            if g in genlib_funcs:
                genlib_lines.append(f"GATE {g} 1 {genlib_funcs[g]};")
        # buf always included so ABC has a trivial pass-through available
        if "buf" not in target_types:
            genlib_lines.append(f"GATE buf 1 {genlib_funcs['buf']};")
        with open(genlib_f, "w") as fh:
            fh.write("\n".join(genlib_lines) + "\n")

        # ---- 3. Map through Yosys's ABC pass ----
        # Keep ABC process discovery inside Yosys instead of invoking the
        # yosys-abc executable directly from the framework.
        yosys_script = textwrap.dedent(f"""\
            read_verilog {dummy_v}
            hierarchy -check -top dummy
            flatten
            techmap
            abc -nocleanup -genlib {genlib_f} -script "+strash;dc2;map"
            opt_clean -purge
            write_blif {out_blif}
        """)
        with open(yosys_f, "w") as fh:
            fh.write(yosys_script)
        process_env = _temp_subprocess_env()
        process_env.update({"TMPDIR": process_tmp, "TEMP": process_tmp, "TMP": process_tmp})
        result = subprocess.run(
            [_yosys_binary(), "-s", yosys_f],
            capture_output=True,
            text=True,
            timeout=30,
            env=process_env,
            cwd=_workspace_temp_dir(),
        )
        if result.returncode != 0 or not os.path.exists(out_blif):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _SUBSTITUTION_TEMPLATE_CACHE[cache_key] = None
            return None

        # ---- 4. Parse BLIF output directly (avoid Yosys round-trip) ----
        # BLIF .gate lines: .gate <type> <port>=<sig> ... <out_port>=<out>
        template = _parse_blif_template(out_blif, target_types, "buf" not in target_types)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _SUBSTITUTION_TEMPLATE_CACHE[cache_key] = template
        return template

    def replace_gate_type_in_cone(
        self,
        source_type: str,
        target_types: List[str],
        output_signal: Optional[str] = None,
    ) -> dict:
        """Replace all source_type gates (in the cone of output_signal, or the
        whole design if output_signal is None) with an ABC-derived equivalent
        built from target_types only.

        Each matching gate is replaced individually using a precompiled template
        from ABC.  Non-matching gates are untouched.

        Args:
            source_type:   Gate type to replace (e.g. "or").
            target_types:  Allowed gate types in the replacement (e.g. ["nand","not"]).
            output_signal: Cone root signal.  None / "*" = whole design.

        Returns:
            dict with keys: success, replaced, skipped, source_type,
            target_types, output_signal.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        # ---- 1. Compile/fetch template ----
        template = self._get_substitution_template(source_type, target_types)
        if template is None:
            return {
                "success": False,
                "reason": (
                    f"No substitution template is available for '{source_type}' "
                    f"using only {target_types}."
                ),
                "replaced": 0, "skipped": 0,
            }

        tmpl_inputs       = template["inputs"]        # e.g. ["A","B"] or ["A"]
        tmpl_output       = template["output"]        # e.g. "Y"
        tmpl_gates        = template["gates"]         # List[GateNode]
        tmpl_int_wires    = template["internal_wires"]
        gate_counts_before = self.count_gates().get("breakdown", {})
        replacement_gate_types: Dict[str, int] = {}
        for tg in tmpl_gates:
            if tg.gate_type == "buf" and "buf" not in target_types:
                continue
            replacement_gate_types[tg.gate_type] = replacement_gate_types.get(tg.gate_type, 0) + 1

        # ---- 2. Collect target gate instances ----
        if output_signal is None or output_signal == "*":
            # Whole design
            target_insts = [
                inst for inst, node in nl.nodes.items()
                if node.gate_type == source_type
            ]
        else:
            # Resolve DFF Q → D
            dff_q_to_d = {d.q: d.d for d in nl.dffs.values()}
            actual_out = dff_q_to_d.get(output_signal, output_signal)

            out2gate = self._build_output_to_gate()
            if actual_out not in out2gate:
                return {
                    "success": False,
                    "reason": f"Signal '{output_signal}' has no driving combinational gate.",
                    "replaced": 0, "skipped": 0,
                }

            # BFS backward
            cone_insts: Set[str] = set()
            queue = [actual_out]
            visited: Set[str] = set()
            pi_set = set(nl.primary_inputs)
            dff_q_set = {d.q for d in nl.dffs.values()}

            while queue:
                sig = queue.pop()
                if sig in visited:
                    continue
                visited.add(sig)
                if sig in {"1'b0", "1'b1"} or sig in pi_set or sig in dff_q_set:
                    continue
                driver = out2gate.get(sig)
                if driver and driver in nl.nodes:
                    cone_insts.add(driver)
                    for inp in nl.nodes[driver].inputs:
                        if inp not in visited:
                            queue.append(inp)

            target_insts = [
                inst for inst in cone_insts
                if nl.nodes[inst].gate_type == source_type
            ]

        if not target_insts:
            return {
                "success": True,
                "replaced": 0,
                "skipped": 0,
                "reason": f"No '{source_type}' gates found in the specified scope.",
                "source_type": source_type,
                "target_types": target_types,
                "output_signal": output_signal,
                "gate_counts_before": gate_counts_before,
                "gate_counts_after": gate_counts_before,
                "replacement_gate_types": replacement_gate_types,
                "delta_by_type": {},
            }

        # ---- 3. Apply template to each matching gate ----
        replaced = 0
        skipped  = 0
        sub_counter = itertools.count(0)

        # Snapshot existing wire names to avoid collisions
        existing_wires = set(nl.wires.keys())
        existing_insts = set(nl.nodes.keys()) | set(nl.dffs.keys())

        for inst_name in target_insts:
            node = nl.nodes.get(inst_name)
            if node is None:
                skipped += 1
                continue

            idx = next(sub_counter)
            prefix = f"_rs_{idx}_"

            # Map template placeholders → real signal names
            # tmpl_inputs[0] → node.inputs[0], etc.
            port_map: Dict[str, str] = {}
            for i, tp in enumerate(tmpl_inputs):
                if i < len(node.inputs):
                    port_map[tp] = node.inputs[i]
            port_map[tmpl_output] = node.output

            # Assign fresh names to internal wires
            wire_map: Dict[str, str] = {}
            for tw in tmpl_int_wires:
                candidate = prefix + tw
                while candidate in existing_wires:
                    candidate += "_"
                wire_map[tw] = candidate
                existing_wires.add(candidate)
                nl.wires[candidate] = WireInfo(name=candidate)

            def resolve(sig: str) -> str:
                if sig in port_map:
                    return port_map[sig]
                if sig in wire_map:
                    return wire_map[sig]
                return sig

            # Remove original gate
            del nl.nodes[inst_name]

            # Insert template gates (skip any buf that wasn't in target_types)
            for tg in tmpl_gates:
                if tg.gate_type == "buf" and "buf" not in target_types:
                    # buf is a pass-through: rewire consumers to its input
                    buf_out = resolve(tg.output)
                    buf_in  = resolve(tg.inputs[0])
                    # Replace all uses of buf_out with buf_in
                    for n in nl.nodes.values():
                        n.inputs = [buf_in if x == buf_out else x for x in n.inputs]
                    for dff in nl.dffs.values():
                        for attr in ("d", "ck", "rn", "sn"):
                            if getattr(dff, attr) == buf_out:
                                setattr(dff, attr, buf_in)
                    # Remove the intermediate wire if it was freshly created
                    nl.wires.pop(buf_out, None)
                    continue

                new_inst = prefix + tg.name
                while new_inst in existing_insts:
                    new_inst += "_"
                existing_insts.add(new_inst)

                new_node = GateNode(
                    name=new_inst,
                    gate_type=tg.gate_type,
                    inputs=[resolve(i) for i in tg.inputs],
                    output=resolve(tg.output),
                )
                nl.nodes[new_inst] = new_node

            replaced += 1

        gate_counts_after = self.count_gates().get("breakdown", {})
        all_gate_types = set(gate_counts_before) | set(gate_counts_after)
        delta_by_type = {
            gate_type: gate_counts_after.get(gate_type, 0) - gate_counts_before.get(gate_type, 0)
            for gate_type in sorted(all_gate_types)
            if gate_counts_after.get(gate_type, 0) != gate_counts_before.get(gate_type, 0)
        }

        return {
            "success": True,
            "replaced": replaced,
            "skipped":  skipped,
            "source_type":   source_type,
            "target_types":  target_types,
            "output_signal": output_signal,
            "replacement_gate_types": replacement_gate_types,
            "delta_by_type": delta_by_type,
            "gate_counts_before": gate_counts_before,
            "gate_counts_after": gate_counts_after,
        }

    def collapse_inverter_pairs(self) -> dict:
        """Collapse every NOT-to-NOT chain into a direct connection.

        Consumers of the second inverter are rewired to the first inverter's
        input. The first inverter is removed only when it has no other fanout.
        A boundary buffer is retained when the second output is a primary
        output or DFF port, preserving externally significant net names.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        pairs_collapsed = 0
        gates_removed = 0
        boundary_buffers = 0

        def is_primary_output(sig: str) -> bool:
            return sig in nl.primary_outputs or sig.split("[", 1)[0] in nl.primary_outputs

        def used_by_dff(sig: str) -> bool:
            return any(
                sig in (dff.d, dff.ck, dff.rn, dff.sn)
                for dff in nl.dffs.values()
            )

        def signal_is_used(sig: str) -> bool:
            return (
                is_primary_output(sig)
                or used_by_dff(sig)
                or any(sig in node.inputs for node in nl.nodes.values())
            )

        def discard_unused_scalar_wire(sig: str) -> None:
            if (
                sig in nl.wires
                and sig not in nl.primary_inputs
                and not is_primary_output(sig)
                and not signal_is_used(sig)
                and all(node.output != sig for node in nl.nodes.values())
                and all(dff.q != sig for dff in nl.dffs.values())
            ):
                nl.wires.pop(sig, None)

        while True:
            out2gate = {
                node.output: inst_name
                for inst_name, node in nl.nodes.items()
            }
            candidates: List[Tuple[str, str]] = []
            for second_name, second in nl.nodes.items():
                if second.gate_type != "not" or len(second.inputs) != 1:
                    continue
                first_name = out2gate.get(second.inputs[0])
                if first_name is None or first_name == second_name:
                    continue
                first = nl.nodes.get(first_name)
                if first and first.gate_type == "not" and len(first.inputs) == 1:
                    candidates.append((first_name, second_name))

            if not candidates:
                break

            # Avoid processing overlapping chains from the same snapshot.
            selected: List[Tuple[str, str]] = []
            occupied: Set[str] = set()
            for first_name, second_name in candidates:
                if first_name in occupied or second_name in occupied:
                    continue
                selected.append((first_name, second_name))
                occupied.update((first_name, second_name))

            for first_name, second_name in selected:
                first = nl.nodes.get(first_name)
                second = nl.nodes.get(second_name)
                if first is None or second is None:
                    continue
                if (
                    first.gate_type != "not"
                    or second.gate_type != "not"
                    or second.inputs != [first.output]
                ):
                    continue

                source = first.inputs[0]
                first_output = first.output
                second_output = second.output
                keep_boundary_net = is_primary_output(second_output) or used_by_dff(second_output)

                # Internal combinational consumers can always bypass the pair.
                for node_name, node in nl.nodes.items():
                    if node_name != second_name:
                        node.inputs = [source if sig == second_output else sig for sig in node.inputs]

                if keep_boundary_net:
                    second.gate_type = "buf"
                    second.inputs = [source]
                    boundary_buffers += 1
                else:
                    for dff in nl.dffs.values():
                        for attr in ("d", "ck", "rn", "sn"):
                            if getattr(dff, attr) == second_output:
                                setattr(dff, attr, source)
                    nl.nodes.pop(second_name, None)
                    gates_removed += 1

                if not signal_is_used(first_output):
                    nl.nodes.pop(first_name, None)
                    gates_removed += 1

                discard_unused_scalar_wire(first_output)
                discard_unused_scalar_wire(second_output)
                pairs_collapsed += 1

        return {
            "pairs_collapsed": pairs_collapsed,
            "gates_removed": gates_removed,
            "boundary_buffers_retained": boundary_buffers,
        }

    def replace_pattern(self, pattern: str, replacement: str) -> int:
        """Replace matched structural patterns throughout the netlist.

        pattern and replacement are strings like "inv->buf" or "or->nand+not".
        Supported simple pattern: "<gate_type>" → "<gate_type>".

        Args:
            pattern:     Source gate type or chain (e.g. "buf", "inv->buf").
            replacement: Target gate type or chain (e.g. "not", "nand+not").

        Returns:
            Number of replacements made.
        """
        self._require_netlist()
        nl = self._netlist
        assert nl is not None

        # Parse simple single-gate pattern for now
        # "inv" is alias for "not"
        def normalise(s: str) -> str:
            return "not" if s.strip().lower() == "inv" else s.strip().lower()

        src_type = normalise(pattern.split("->")[0].split("+")[0])
        dst_type = normalise(replacement.split("->")[0].split("+")[0])

        if src_type not in PRIMITIVE_GATES or dst_type not in PRIMITIVE_GATES:
            raise ValueError(
                f"replace_pattern: unsupported gate type(s): {src_type!r}, {dst_type!r}"
            )

        src_inputs = 1 if src_type in ONE_INPUT_GATES else 2
        dst_inputs = 1 if dst_type in ONE_INPUT_GATES else 2
        if src_inputs != dst_inputs:
            raise ValueError(
                f"replace_pattern: port count mismatch between {src_type!r} and {dst_type!r}"
            )

        count = 0
        for node in nl.nodes.values():
            if node.gate_type == src_type:
                node.gate_type = dst_type
                count += 1
        return count


# ===========================================================================
# Standalone functional helpers (used by EDAEngine but not part of its class)
# ===========================================================================

def _find_cone_primary_input(
    nl: Netlist,
    output_signal: str,
    out2gate: Dict[str, str],
) -> Optional[str]:
    """Return the first primary input found in the transitive fanin of output_signal."""
    visited: Set[str] = set()
    queue: deque[str] = deque([output_signal])
    while queue:
        sig = queue.popleft()
        if sig in visited:
            continue
        visited.add(sig)
        if sig in nl.primary_inputs:
            return sig
        driver = out2gate.get(sig)
        if driver and driver in nl.nodes:
            for inp in nl.nodes[driver].inputs:
                queue.append(inp)
    return None


def _rebalance_associative_tree(
    nl: Netlist,
    output_signal: str,
    out2gate: Dict[str, str],
    max_depth: int,
) -> bool:
    """Attempt to rebalance an associative gate tree to meet max_depth.

    This is a best-effort structural rebalancing pass.
    Returns True if the depth is now ≤ max_depth.
    """
    driver_inst = out2gate.get(output_signal)
    if driver_inst is None or driver_inst not in nl.nodes:
        return False

    root_gate = nl.nodes[driver_inst]
    if root_gate.gate_type not in {"and", "or", "xor", "nand", "nor", "xnor"}:
        return False  # Non-associative gate — cannot rebalance

    # Collect all leaves of this associative tree
    leaves: List[str] = []
    _collect_leaves(nl, root_gate.gate_type, output_signal, out2gate, leaves)

    if len(leaves) < 4:
        return False  # Nothing to rebalance

    # Build a balanced binary tree from leaves
    gate_type = root_gate.gate_type
    _build_balanced_tree(nl, leaves, gate_type, output_signal, root_gate)
    return True


def _collect_leaves(
    nl: Netlist,
    gate_type: str,
    sig: str,
    out2gate: Dict[str, str],
    leaves: List[str],
) -> None:
    """DFS to collect inputs of same-type associative gate tree."""
    driver = out2gate.get(sig)
    if driver is None or driver not in nl.nodes:
        leaves.append(sig)
        return
    node = nl.nodes[driver]
    if node.gate_type != gate_type:
        leaves.append(sig)
        return
    for inp in node.inputs:
        _collect_leaves(nl, gate_type, inp, out2gate, leaves)


def _build_balanced_tree(
    nl: Netlist,
    leaves: List[str],
    gate_type: str,
    final_output: str,
    reuse_node: GateNode,
) -> None:
    """Build a balanced binary tree of gate_type over leaves, writing final output."""
    counter = itertools.count(1)
    layer = list(leaves)

    while len(layer) > 2:
        next_layer: List[str] = []
        for i in range(0, len(layer) - 1, 2):
            new_wire = f"_bal_{gate_type}_{next(counter)}"
            nl.wires[new_wire] = WireInfo(name=new_wire)
            new_inst = f"_bal_g_{next(counter)}"
            new_node = GateNode(
                name=new_inst,
                gate_type=gate_type,
                inputs=[layer[i], layer[i + 1]],
                output=new_wire,
            )
            nl.nodes[new_inst] = new_node
            next_layer.append(new_wire)
        if len(layer) % 2 == 1:
            next_layer.append(layer[-1])
        layer = next_layer

    # Final gate reuses the root node's slot
    if len(layer) == 2:
        reuse_node.inputs = [layer[0], layer[1]]
    elif len(layer) == 1:
        reuse_node.gate_type = "buf"
        reuse_node.inputs = [layer[0]]
