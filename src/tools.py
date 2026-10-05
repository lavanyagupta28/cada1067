"""OpenAI function-calling tool schemas for the EDA engine."""

from __future__ import annotations

from typing import Any, Dict, List


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_design",
            "description": (
                "Call this to load a gate-level Verilog netlist from disk into the EDA engine. "
                "Must be called before any analysis or transformation operation. "
                "The filepath argument should include the directory and filename."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the Verilog (.v) file to load.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_design",
            "description": (
                "Call this to write the current (possibly modified) netlist back to a "
                "Verilog file on disk. Use after any transformation to persist changes. "
                "If filepath is omitted, write '<case_name>_out.v' in the current working directory"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": (
                            "Optional destination path for the output Verilog file. "
                            "Relative paths are saved relative to the current working directory."
                        ),
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_gates",
            "description": (
                "Count gates across the WHOLE currently loaded design and return the total "
                "plus a breakdown by gate type. Use this for whole-design questions asking "
                "how many gates of a particular type are in the netlist, such as "
                "'How many AND gates are now in the reconstructed netlist?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_testcase_name",
            "description": (
                "Call this when the user specifies a testcase name (e.g., 'test8', 'test35'). "
                "Sets the active case name and opens the corresponding log file for output mirroring. "
                "If log_path is omitted, use '<case_name>.log' in the current working directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_name": {
                        "type": "string",
                        "description": "The short identifier for this testcase (e.g. 'test8').",
                    },
                    "log_path": {
                        "type": "string",
                        "description": (
                            "Optional path to the log file (e.g. 'test8.log'). "
                            "Relative paths are saved relative to the current working directory."
                        ),
                    },
                },
                "required": ["case_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fanin_cone_depth",
            "description": (
                "Return the maximum logic depth "
                "of the fanin cone of an output signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string"
                    }
                },
                "required": ["output_signal"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_outputs_by_logic_depth",
            "description": (
                "Count how many primary-output bits have combinational fanin logic depth "
                "matching a comparison predicate. Use this exact tool for prompts such as "
                "'How many outputs have a logic depth greater than 4?', 'less than 3', "
                "'equal to 0', 'at least 5', or 'at most 2'. Bus outputs are expanded "
                "and counted per output bit. DFF boundaries are treated as cuts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operator": {
                        "type": "string",
                        "enum": [">", ">=", "<", "<=", "==", "!="],
                        "description": (
                            "Comparison operator. Map natural language as: greater than => >, "
                            "at least => >=, less/smaller than => <, at most => <=, "
                            "equal to/exactly => ==, not equal to => !=."
                        ),
                    },
                    "threshold": {
                        "type": "integer",
                        "description": "Depth threshold to compare against.",
                    },
                },
                "required": ["operator", "threshold"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deepest_output_fanin_cone",
            "description": (
                "Find the primary-output bit or tied bits with the deepest combinational "
                "fanin cone, measured by logic depth. DFF outputs are treated as boundaries. "
                "Do not use this for 'largest' or 'smallest fanin cone'; those mean cone "
                "gate count and must use rank_signals_by_fanin_cone with metric=gate_count."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_global_max_depth",
            "description": (
                "Return the maximum combinational depth anywhere in the design, across "
                "PI/DFF-Q sources and PO/DFF-D sinks."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_max_depth_between_endpoint_classes",
            "description": (
                "Compute the maximum combinational logic depth across endpoint CLASSES. "
                "Invoke this exact tool for requests such as 'from any primary input to any DFF D-pin' "
                "or 'from any PI to any primary output'. For maximum register-to-register "
                "depth, use source_class DFF_Q and sink_class DFF_D. This evaluates all "
                "matching endpoints in one "
                "operation. For a DFF_D sink, sink_instance is the DFF, sink_pin is D, and "
                "sink_pin_signal is the net connected to that D pin (not the DFF Q output). "
                "Do not pass literal strings PI, DFF, or PO to get_max_depth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_class": {
                        "type": "string",
                        "enum": ["PI", "DFF_Q"],
                        "description": "Source endpoint class.",
                    },
                    "sink_class": {
                        "type": "string",
                        "enum": ["DFF_D", "PO"],
                        "description": "DFF_D means every DFF D-pin; PO means every primary-output bit.",
                    },
                },
                "required": ["source_class", "sink_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "is_gate_on_any_max_depth_path",
            "description": (
                "Determine whether a combinational gate instance lies on at least one "
                "global maximum-depth combinational path of the design. Use this exact "
                "tool for prompts such as 'Determine whether gate g0 lies on any "
                "maximum-depth path of the design'. This does not require a specific "
                "source/sink and does not enumerate paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_name": {
                        "type": "string",
                        "description": "Combinational gate instance name, such as g0.",
                    },
                },
                "required": ["gate_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_max_depth",
            "description": (
                "Find the longest combinational path between two CONCRETE signal names. "
                "Both source and sink must be actual netlist signals such as n0 or n16[2]. "
                "Never pass endpoint classes such as PI, DFF, DFF_D, or PO here; use "
                "get_max_depth_between_endpoint_classes for those requests. "
                "DFF boundaries are treated as cuts. "
                "Returns the depth (integer) and the list of signals on the longest path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Starting signal name (e.g. a primary input).",
                    },
                    "sink": {
                        "type": "string",
                        "description": "Ending signal name (e.g. a primary output).",
                    },
                },
                "required": ["source", "sink"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "path_passes_through",
            "description": (
                "Call this to determine whether ALL combinational paths from source to sink "
                "pass through a specific intermediate node. The node may be either a signal "
                "name or a combinational gate instance such as g0. Use this exact tool for "
                "prompts like 'Does every path from input n2 to output n12 pass through gate g0?'. "
                "Do not use this for prompts asking whether a wire is a cut between any "
                "primary input and any primary output; use is_wire_cut_between_primary_ios. "
                "The result reports whether any source-to-sink path exists, whether all such "
                "paths pass through the node, and a counterexample path when one avoids it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Starting signal name."},
                    "sink": {"type": "string", "description": "Ending signal name."},
                    "node": {
                        "type": "string",
                        "description": "The intermediate signal or combinational gate instance to test.",
                    },
                },
                "required": ["source", "sink", "node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "is_wire_cut_between_primary_ios",
            "description": (
                "Determine whether a wire/signal is a cut between any primary input and any "
                "primary output. Use this exact tool for prompts like 'Determine whether wire "
                "n55104 is a cut between any primary input and any primary output. Report yes "
                "or no.' This expands bus PI/PO ports to bits, treats DFF boundaries as cuts, "
                "does not enumerate all paths, and never requires literal source='PI' or "
                "sink='PO'. The returned answer is yes iff blocking the wire disconnects at "
                "least one expanded PI bit from at least one expanded PO bit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "wire_name": {
                        "type": "string",
                        "description": "Wire or signal name to test, such as n55104.",
                    },
                },
                "required": ["wire_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_path_avoiding",
            "description": (
                "Call this to find one combinational path from source to sink that does NOT "
                "pass through the specified signal. "
                "Returns the path as a list of signal names, or null if no such path exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Starting signal name."},
                    "sink": {"type": "string", "description": "Ending signal name."},
                    "avoid": {
                        "type": "string",
                        "description": "Signal that must not appear on the returned path.",
                    },
                },
                "required": ["source", "sink", "avoid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_articulation_points",
            "description": (
                "Find all articulation points (cut gates / mandatory intermediate gate instances) "
                "in the combinational graph between a source signal and a sink signal. "
                "Use this exact tool for prompts like 'Find all articulation points in the "
                "combinational graph between n2 and n14' or 'List all cut/bottleneck gates "
                "between source and sink'. Returns the list of gate instances that every "
                "combinational path from source to sink must traverse."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Starting signal name (e.g. 'n2').",
                    },
                    "sink": {
                        "type": "string",
                        "description": "Ending signal name (e.g. 'n14').",
                    },
                },
                "required": ["source", "sink"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_signals_by_fanin_cone",
            "description": (
                "Rank every signal in a complete class by fanin-cone gate count or "
                "logic depth. Use this exact tool with signal_class=PO and metric=gate_count "
                "for questions such as 'Which output has the largest fanin cone?'. "
                "Use gate_count for largest/smallest cone questions and "
                "depth only for deepest/shallowest questions. PO buses are expanded "
                "into bits, and registered PO/DFF-Q signals use the same Q-to-D "
                "next-state cone resolution as count_gate_types_in_cone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_class": {
                        "type": "string",
                        "enum": ["PO", "DFF_D", "DFF_Q", "GATE_OUTPUT"],
                        "description": "Complete class of signals to rank.",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["gate_count", "depth"],
                        "description": "Ranking metric; largest cone means gate_count.",
                    },
                    "order": {
                        "type": "string",
                        "enum": ["descending", "ascending"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of distinct metric ranks to return.",
                    },
                    "include_ties": {
                        "type": "boolean",
                        "description": "Return all signals tied at selected ranks.",
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": "Maximum signals inline before writing a TSV file.",
                    },
                },
                "required": ["signal_class", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logic_cone",
            "description": (
                "Call this to retrieve all gate instance names that directly or transitively "
                "feed (drive) the given output signal. When the requested output is a "
                "DFF-Q signal, the user-facing report resolves it to that DFF's D-pin "
                "next-state combinational cone, consistently with count_gate_types_in_cone. "
                "Returns the gate names inline or in a file for a large cone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string",
                        "description": "The output net whose transitive fanin is requested.",
                    }
                },
                "required": ["output_signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_shared_fanin_gates",
            "description": (
                "Find all gate instances shared by the transitive fanin cones of two "
                "signals. This computes the intersection of two backward logic cones; "
                "use it for gates shared/common between fanin-cone requests. It is not "
                "an articulation-point or source-to-sink path query. Registered outputs "
                "are resolved through their D-input next-state cones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_a": {"type": "string", "description": "First signal or output."},
                    "signal_b": {"type": "string", "description": "Second signal or output."},
                    "inline_limit": {
                        "type": "integer",
                        "description": "Maximum shared gates inline before writing a report file.",
                    },
                },
                "required": ["signal_a", "signal_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "derive_boolean_equation",
            "description": (
                "Derive the Boolean equation for an output signal or net in terms of its primary inputs "
                "or boundary inputs. Use this exact tool for prompts like 'Derive the Boolean equation for "
                "output <signal> in terms of its primary inputs'. It extracts the logic cone, gate equations, "
                "step-by-step substitutions, and handles flip-flops (DFFs) and direct wire connections. "
                "If a PI-only expression is unavailable, report that before any boundary-input equation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string",
                        "description": "Target signal or net name (e.g. 'n7', 'n12', 'n22[0]').",
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": "Maximum number of gate equations to return inline (default: 20).",
                    },
                },
                "required": ["output_signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_cone_gates",
            "description": (
                "Call this to count the number of gates in the logic cone of an output signal. "
                "Returns an integer gate count. Do not use this for counts by gate type "
                "such as 'how many NAND gates in the cone'; use count_gate_types_in_cone instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string",
                        "description": "The output net whose logic cone gate count is requested.",
                    }
                },
                "required": ["output_signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_gate_types_in_cone",
            "description": (
                "Count gates by gate type in the logic cone of an output signal. Use this "
                "for prompts such as 'How many NAND gates are in the restructured cone of n8?' "
                "or any cone question asking for AND/OR/NOT/NAND/NOR/XOR/XNOR/BUF counts. "
                "Returns total and a by_type dictionary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string",
                        "description": "The output net whose logic cone should be counted.",
                    }
                },
                "required": ["output_signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_zero_length_pi_po_paths",
            "description": (
                "Find all paths of length 0 from primary inputs to primary outputs, i.e. "
                "direct wire connections where the same signal is both a PI and a PO and "
                "no gates are traversed. Use this exact tool for prompts like 'Find all "
                "paths of length 0 (direct wire connections from PI to PO)'. Do not call "
                "find_all_paths with literal source='PI' or sink='PO' for this request."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_all_paths",
            "description": (
                "List all combinational paths between a source signal and sink signal. "
                "Use only when both source and sink are concrete signal names. Do not pass "
                "literal strings PI or PO, and do not use this for 'paths of length 0' "
                "from any PI to any PO; use find_zero_length_pi_po_paths for that. "
                "Only the first 5 paths are returned inline by default. If more than "
                "5 paths exist, the complete path list is written to a text file under "
                "./_tmp/ and the result returns the file_path. There is no artificial "
                "maximum path-count cap; enumeration continues until all acyclic "
                "combinational paths are found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string"
                    },
                    "sink": {
                        "type": "string"
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": (
                            "Optional number of paths to include inline in the tool result. "
                            "Defaults to 5. Do not increase this unless explicitly requested."
                        )
                    }
                },
                "required": [
                    "source",
                    "sink"
                ]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_register_to_register_paths",
            "description": (
                "List ALL register-to-register paths through combinational logic. This exact tool must "
                "be used for requests such as 'List all register-to-register paths in this design'. "
                "It traverses from every DFF Q pin to reached DFF D pins and reports the ordered gates. "
                "Do not use list_signals, get_max_depth, or get_max_depth_between_endpoint_classes. "
                "Up to 10 paths are returned inline; larger results are written under ./_tmp/ and the "
                "result returns the path count and file path. The full enumeration is streamed to the "
                "file so large path sets are not returned as giant JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_name_type",
            "description": (
                "Determine whether a provided identifier is a combinational gate instance, DFF instance, "
                "or signal/wire. Call this first when the wording is ambiguous, then select the matching "
                "gate-based or net-based fanout/reachability tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Identifier to classify, such as g0 or n16.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gate_info",
            "description": (
                "Return the exact primitive gate type and every logical pin-to-net connection for a "
                "combinational gate or DFF instance. Use this tool directly for requests such as "
                "'What type of gate is g0? Report its gate type and pin connections.' The result "
                "identifies AND, NAND, NOR, NOT, BUF, OR, XOR, XNOR, or DFF; do not use "
                "resolve_name_type or a fanout tool for this request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_name": {
                        "type": "string",
                        "description": "Combinational gate or DFF instance name, such as g0.",
                    }
                },
                "required": ["gate_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transitive_fanout_cone",
            "description": (
                "Compute the transitive fanout cone of a SIGNAL or WIRE. Return every gate "
                "reachable downstream, not only directly connected gates. Traversal follows "
                "combinational fanout, includes reached DFFs, and stops at DFF boundaries. "
                "For more than 10 gates, the complete list is written under ./_tmp/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "net_name": {
                        "type": "string",
                        "description": "Signal or bus from which downstream reachability starts.",
                    }
                },
                "required": ["net_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reachable_gates_from_gate",
            "description": (
                "Find ALL gates transitively reachable downstream from the OUTPUT of a gate or DFF instance. "
                "Use after resolve_name_type identifies the source as a combinational_gate or dff. "
                "For more than 10 gates, the complete list is written under ./_tmp/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_name": {
                        "type": "string",
                        "description": "Gate or DFF instance whose output starts the traversal.",
                    }
                },
                "required": ["gate_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_net_connections",
            "description": (
                "Return ALL instances directly connected to a SIGNAL or WIRE, including both "
                "the instance(s) that drive it and the gates/DFF ports that consume it. Use this "
                "for wording such as 'all gates connected to signal n1' or 'what drives and uses n1'. "
                "For downstream consumers only, use get_net_fanout instead. Do not call this after "
                "get_net_fanout has returned a complete file-backed result for the same signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "net_name": {
                        "type": "string",
                        "description": "Signal whose direct drivers and loads should be listed.",
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": "Maximum unique connected instances returned inline. Defaults to 50.",
                    },
                },
                "required": ["net_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_net_fanout",
            "description": (
                "Return only the immediate/direct fanout gates of a SIGNAL or WIRE. Do not use "
                "this tool for a transitive fanout cone. Do not pass a gate instance such as g0; "
                "use get_gate_output_fanout. "
                "For 10 or fewer gates, returns the names inline so they must be listed in the response. "
                "For more than 10 gates, writes the complete list to a text file in the current working "
                "directory temp folder and returns only the count and file path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "net_name": {
                        "type": "string",
                        "description": "The net name whose fanout list is requested.",
                    }
                },
                "required": ["net_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_signals_by_fanout",
            "description": (
                "Rank every signal in a complete signal class by direct load-pin fanout. "
                "Use this for highest/lowest or top-N fanout comparisons across primary "
                "inputs, DFF-Q signals, gate outputs, or all driven signals. Primary-input "
                "buses are expanded into individual bits. Do not use list_signals samples "
                "for exhaustive fanout ranking; use get_net_fanout for one named net."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_class": {
                        "type": "string",
                        "enum": ["PI", "DFF_Q", "GATE_OUTPUT", "ALL_DRIVEN"],
                        "description": "Complete class of signals to rank.",
                    },
                    "order": {
                        "type": "string",
                        "enum": ["descending", "ascending"],
                        "description": "Descending for highest fanout; ascending for lowest.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of distinct fanout ranks to return. Defaults to 1.",
                    },
                    "include_ties": {
                        "type": "boolean",
                        "description": "Return every signal tied at each selected rank. Defaults to true.",
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": "Maximum returned signals inline before writing a TSV file.",
                    },
                },
                "required": ["signal_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gate_output_fanout",
            "description": (
                "Find every gate directly connected to the OUTPUT NET of a gate or DFF instance. "
                "Use this exact tool for wording such as 'gates connected to the output of g0'. "
                "Do not treat the gate instance name as a signal. Large results are written under ./_tmp/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_name": {
                        "type": "string"
                    }
                },
                "required": ["gate_name"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "are_same_clock_domain",
            "description": (
                "Call this to check whether two flip-flop (DFF) instances share the same clock net. "
                "Returns true if both DFFs are in the same clock domain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dff1": {
                        "type": "string",
                        "description": "First DFF instance name.",
                    },
                    "dff2": {
                        "type": "string",
                        "description": "Second DFF instance name.",
                    },
                },
                "required": ["dff1", "dff2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_flip_flops_by_clock",
            "description": (
                "List all flip-flop/DFF instances whose clock pin is exactly the given "
                "clock signal. Use this exact tool for prompts like 'List all flip-flops "
                "driven by clock n0' or 'which DFFs are clocked by clk?'. Do not use "
                "get_transitive_fanout_cone for clock-pin queries. Results are "
                "reported as a total count plus a full-list file_path under ./_tmp/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clock_signal": {
                        "type": "string",
                        "description": "Clock net name to match against each DFF CK pin.",
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": "Deprecated; DFF records are written to file_path instead of returned inline.",
                    },
                },
                "required": ["clock_signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_dff_d_input_structures",
            "description": (
                "Analyze every DFF's D-input logic for structural enable or hold "
                "patterns involving feedback from that DFF's own Q output. Use this "
                "for requests to report or count flip-flops with enable/hold logic "
                "implemented through multiplexing or gates. Do not enumerate "
                "register-to-register paths for this purpose. Results are structural "
                "candidates, with an exact count and a file_path for large reports."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "structure_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["enable", "hold"]},
                        "description": "Requested structure classes; defaults to both enable and hold.",
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": "Maximum matches returned inline; larger reports are written to a file.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_gate_before",
            "description": (
                "Call this to insert a new 2-input gate immediately before an existing gate instance. "
                "The original driver signal becomes input[0] of the new gate; "
                "extra_input becomes input[1]. "
                "The new gate's output replaces the original signal feeding the target. "
                "Returns the new gate's instance name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_instance": {
                        "type": "string",
                        "description": "Existing gate instance to insert before.",
                    },
                    "gate_type": {
                        "type": "string",
                        "description": "Type of gate to insert (e.g. 'and', 'or', 'nand').",
                    },
                    "extra_input": {
                        "type": "string",
                        "description": "Second input signal for the new gate.",
                    },
                },
                "required": ["target_instance", "gate_type", "extra_input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_gate",
            "description": (
                "Call this to replace the gate type of an existing instance in-place. "
                "When replacing a 1-input gate (buf/not) with a 2-input gate (and/or/nand/nor/xor/xnor), "
                "supply extra_input to provide the additional input signal. "
                "Use new_inputs to explicitly choose the inputs kept by the replacement. "
                "Use this exact tool after find_gates for constant-input propagation, such as "
                "simplifying nand(Y, A, 1'b1) or nand(Y, 1'b1, A) into not(Y, A). "
                "When several reported instances need the same kind of structural simplification, "
                "use the replacements batch argument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_name": {
                        "type": "string",
                        "description": "Instance name of the gate to replace.",
                    },
                    "new_gate_type": {
                        "type": "string",
                        "description": "New gate type (e.g. 'and', 'not', 'xor').",
                    },
                    "extra_input": {
                        "type": "string",
                        "description": (
                            "Optional second input signal. Required when upgrading from "
                            "a 1-input gate (buf/not) to a 2-input gate."
                        ),
                    },
                    "new_inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional complete input list for the replacement gate. "
                            "Use this when reducing a 2-input gate to a 1-input gate "
                            "and the kept input is not simply the first existing input."
                        ),
                    },
                    "replacements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "instance_name": {
                                    "type": "string",
                                    "description": "Instance name of the gate to replace.",
                                },
                                "new_gate_type": {
                                    "type": "string",
                                    "description": "Replacement gate type.",
                                },
                                "new_inputs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Complete input list for this replacement.",
                                },
                                "extra_input": {
                                    "type": "string",
                                    "description": "Optional extra input for this replacement.",
                                },
                            },
                            "required": ["instance_name", "new_gate_type"],
                        },
                        "description": (
                            "Optional batch of replacements. Use this after a finder returns "
                            "several instances that should all receive structural replacements."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_dedicated_buffers_for_loads",
            "description": (
                "Call this when the request says each direct load of a signal/net must be "
                "driven through its own dedicated BUF gate. This inserts one BUF per "
                "current direct load, reconnects every original load to a unique buffer "
                "output, and leaves the original net driving only the inserted buffers. "
                "Use this for wording like 'each load of n2 is driven through a dedicated "
                "buffer'. Do not use insert_buffers_for_fanout for that wording."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "net_name": {
                        "type": "string",
                        "description": "The signal/net whose current direct loads should each get a dedicated BUF.",
                    },
                },
                "required": ["net_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_buffers_for_fanout",
            "description": (
                "Call this to insert buffer trees on a net so that no downstream gate "
                "has fanout exceeding max_fanout. "
                "This may leave some loads directly connected to the original net. "
                "Do not use this when every load needs its own dedicated buffer; use "
                "insert_dedicated_buffers_for_loads instead. "
                "Returns the number of buffers inserted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "net_name": {
                        "type": "string",
                        "description": "The net to apply fanout buffering to.",
                    },
                    "max_fanout": {
                        "type": "integer",
                        "description": "Maximum allowed fanout per net segment.",
                    },
                },
                "required": ["net_name", "max_fanout"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_dangling_gates",
            "description": (
                "Call this to remove all gates and nets that do not transitively feed "
                "any primary output or DFF input. DFFs are preserved as combinational "
                "boundaries. Returns the number of gates removed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collapse_inverter_pairs",
            "description": (
                "Find every back-to-back inverter pair (NOT followed directly by NOT) "
                "and collapse the pair into a direct wire connection. Use this exact tool "
                "when asked to remove double inversions or collapse consecutive inverters. "
                "Shared fanouts and primary-output/DFF boundary net names are preserved. "
                "Returns the number of pairs collapsed and gates removed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_pattern",
            "description": (
                "Call this to replace structural gate patterns throughout the entire netlist. "
                "The pattern and replacement are gate type strings such as 'buf', 'not', 'and'. "
                "Simple chain notation 'A->B' or compound 'A+B' is also accepted. "
                "Returns the number of replacements made."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Source gate type or pattern (e.g. 'buf', 'inv->buf').",
                    },
                    "replacement": {
                        "type": "string",
                        "description": "Target gate type or pattern (e.g. 'not', 'nand+not').",
                    },
                },
                "required": ["pattern", "replacement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_gates",
            "description": (
                "Find combinational gate instances by structural filters: gate type, input count, "
                "and/or exact input signal. Use this before generic replacements for prompts like "
                "'all 2-input NAND gates that have one input tied to constant 1'. Also use this "
                "for listing gates of a type with their input and output signals, such as 'List "
                "all NAND gates in this design with their input and output signals'. For constant 1, "
                "pass has_input=\"1'b1\"; for constant 0, pass has_input=\"1'b0\". The result includes "
                "the matching instances, their inputs, matched input indices, and other_inputs. "
                "If the result contains file_path, the complete list has been written there and "
                "the final answer must report that exact path. "
                "CRITICAL: This tool ONLY filters by gate_type and input_count. If the user prompt "
                "requires filtering by whether the gate's output drives a primary output pin, another "
                "specific gate type, or any downstream sink topology, do NOT call find_gates; "
                "you MUST call declare_missing_tool instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_type": {
                        "type": "string",
                        "description": "Gate type filter such as 'nand', 'and', or ''. Empty string matches all types.",
                    },
                    "input_count": {
                        "type": "integer",
                        "description": "Optional exact number of inputs to match, such as 2.",
                    },
                    "has_input": {
                        "type": "string",
                        "description": "Optional exact input signal to match, such as \"1'b1\" or \"1'b0\".",
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return inline before writing the full list to ./_tmp/. Defaults to 10.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_gates_with_constant_inputs",
            "description": (
                "Report combinational gates whose inputs are constant 0 or 1 under the "
                "competition model: primary inputs and DFF-Q outputs are unconstrained, "
                "and constants are proven on a temporary analysis model without modifying "
                "the design. Use this exact tool for prompts like 'Report any NAND gates "
                "with constant inputs (0 or 1)' or 'find gates with functionally constant "
                "inputs'. Do not use this for wording that specifically says an input is "
                "'tied to' a literal constant; use find_gates for structural tied constants."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_type": {
                        "type": "string",
                        "description": "Gate type filter such as 'nand' or 'and'.",
                    },
                    "values": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "description": "Constant values to report, usually [0, 1].",
                    },
                    "functional": {
                        "type": "boolean",
                        "description": "When true, prove non-literal signal constants formally.",
                    },
                    "inline_limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return inline before writing the full list to ./_tmp/. Defaults to 50.",
                    },
                },
                "required": ["gate_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simplify_gates_with_constant_inputs",
            "description": (
                "Simplify gates previously reported with literal or proven functional "
                "constant inputs while preserving their output nets. Use this for "
                "requests to propagate constant inputs through reported NAND/AND/OR/"
                "NOR/XOR/XNOR/NOT/BUF gates. It reuses a compatible complete prior "
                "report when available and otherwise performs the functional analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_type": {
                        "type": "string",
                        "description": "Reported gate type to simplify, such as 'nand'.",
                    },
                    "functional": {
                        "type": "boolean",
                        "description": "Use formally proven functional constants as well as literals. Defaults to true.",
                    },
                    "use_last_report": {
                        "type": "boolean",
                        "description": "Reuse the compatible complete preceding report. Defaults to true.",
                    },
                },
                "required": ["gate_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_instances_by_name_pattern",
            "description": (
                "Call this to search for gate instances by gate type and/or instance name pattern. "
                "Pass an empty string for gate_type to match all types. "
                "name_pattern is a Python regex applied to instance names. "
                "Returns a list of matching instance names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_type": {
                        "type": "string",
                        "description": "Gate type filter (e.g. 'buf'). Empty string matches all types.",
                    },
                    "name_pattern": {
                        "type": "string",
                        "description": "Python regex pattern to match against instance names.",
                    },
                },
                "required": ["gate_type", "name_pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_signal_constant",
            "description": (
                "Prove whether a scalar or complete bus is ALWAYS equal to a constant regardless "
                "of all primary inputs and DFF-Q boundary values. Invoke this exact tool for prompts "
                "such as 'is output n16 always 0?', 'is this signal constant?', or 'regardless of "
                "all inputs'. Do not use check_signal_equivalence for constant-property questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_name": {
                        "type": "string",
                        "description": "Scalar signal, bit-select, or complete bus to prove.",
                    },
                    "value": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "description": "Expected constant, such as 0, 1, 8'b0, '0, or '1.",
                    },
                },
                "required": ["signal_name", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_function_symmetry",
            "description": (
                "Prove whether the Boolean function at one output signal is symmetric "
                "with respect to two scalar primary inputs or primary-input bits. This "
                "checks invariance when the two input values are swapped. Use this for "
                "symmetry questions; do not compare the output separately to each input "
                "with check_signal_equivalence. DFF-Q values are independent boundaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string",
                        "description": "Signal whose Boolean function is checked.",
                    },
                    "input_a": {
                        "type": "string",
                        "description": "First scalar PI or PI bit, such as n3.",
                    },
                    "input_b": {
                        "type": "string",
                        "description": "Second scalar PI or PI bit, such as n9[0].",
                    },
                },
                "required": ["output_signal", "input_a", "input_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_signal_equivalence",
            "description": (
                "Compare TWO SIGNALS inside the current netlist for functional equivalence. "
                "Use only when the request explicitly names two signals; this does not compare designs. "
                "Uses Yosys SAT solver for precise equivalence checking on circuits of any size. "
                "Returns true if the signals are logically equivalent, false otherwise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sig1": {
                        "type": "string",
                        "description": "First signal name to compare.",
                    },
                    "sig2": {
                        "type": "string",
                        "description": "Second signal name to compare.",
                    }
                },
                "required": ["sig1", "sig2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_binary_gate_equivalent_pair",
            "description": (
                "Find whether there exists a pair of existing netlist signals (a,b) such "
                "that a virtual 2-input primitive gate gate_type(a,b) is functionally "
                "equivalent to target_signal. Use this exact tool for prompts like "
                "'does there exist internal signals a,b such that NAND(a,b) equals n25?' "
                "Candidates may be outputs of any gate type; do not use find_gates for "
                "this existential expression query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_signal": {
                        "type": "string",
                        "description": "Target signal whose function should equal gate_type(a,b).",
                    },
                    "gate_type": {
                        "type": "string",
                        "enum": ["and", "or", "nand", "nor", "xor", "xnor"],
                        "description": "The virtual binary gate to apply to candidate signals.",
                    },
                    "candidate_scope": {
                        "type": "string",
                        "enum": ["internal", "all"],
                        "description": (
                            "Use 'internal' to exclude primary input/output ports; use 'all' "
                            "only if the prompt explicitly permits primary IO candidates."
                        ),
                    },
                    "max_signature_pairs": {
                        "type": "integer",
                        "description": (
                            "Optional safety budget for broad signature pair search. The "
                            "tool reports search_complete=false if this budget is reached."
                        ),
                    },
                    "max_formal_checks": {
                        "type": "integer",
                        "description": (
                            "Optional safety budget for expensive Yosys SAT confirmations "
                            "after signature filtering. Defaults to a small safe value."
                        ),
                    },
                },
                "required": ["target_signal", "gate_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_design_equivalence",
            "description": (
                "Prove that the CURRENT TRANSFORMED DESIGN is equivalent to the ORIGINAL NETLIST "
                "loaded by read_design. Invoke this exact non-mutating tool for requests such as "
                "'prove the transformed design is equivalent to the pre-transformation netlist' "
                "or 'verify whole-design equivalence'. It serializes the current netlist and runs "
                "standalone ABC CEC on the full networks at DFF boundaries, treating DFF Q pins as "
                "unconstrained inputs and comparing primary outputs plus DFF D/control pins. "
                "A Yosys miter with internal ABC simplification and SAT is retained as a conservative "
                "fallback for ABC startup/parser errors. Never call FRAIG as a standalone substitute for a proof request."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_gate",
            "description": (
                "Rename a gate instance in the currently loaded netlist. "
                "Only the instance name changes — connections and logic are preserved exactly. "
                "Invoke this when asked to rename, relabel, or change the name of a gate instance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_name": {
                        "type": "string",
                        "description": "Current instance name of the gate to rename.",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New instance name for the gate.",
                    },
                },
                "required": ["old_name", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_wire",
            "description": (
                "Rename a wire or signal in the currently loaded netlist and update all references. "
                "Use this when asked to rename, relabel, or change the identifier of a wire/net. "
                "Only the signal name changes; gate and DFF connectivity is preserved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "old_name": {
                        "type": "string",
                        "description": "Current wire/signal name to rename.",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "New wire/signal name.",
                    },
                },
                "required": ["old_name", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_gate_type_in_cone",
            "description": (
                "Replace all instances of a specific gate type (e.g. OR) in the fanin cone "
                "of an output signal with a functionally equivalent circuit built from the target gate types. "
                "Unlike remap_cone_with_gates, this ONLY touches the specified source gate type — "
                "all other gates in the cone remain unchanged. "
                "Use this when asked to 'replace OR gates with NAND+NOT' or 'convert XOR to NAND-only' "
                "within a cone or across the whole design. "
                "Do NOT use this for constant propagation, gates with inputs tied to constants, "
                "or prompts referring to 'reported gates'; use find_gates followed by replace_gate "
                "with explicit new_inputs for those cases. "
                "This is a broad technology-remapping tool, not a local simplification tool. "
                "Set output_signal to null to apply across the entire design. "
                "Returns success, replaced (count), skipped (count), replacement_gate_types, "
                "delta_by_type, and gate_counts_after so follow-up gate-count questions can "
                "use updated counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "description": "Gate type to replace. E.g. 'or', 'xor', 'nor', 'and', 'not', 'buf'.",
                    },
                    "target_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Allowed gate types in the replacement. "
                            "E.g. ['nand', 'not'] or ['nor', 'not']. "
                            "Must form a functionally complete set for the source gate."
                        ),
                    },
                    "output_signal": {
                        "type": "string",
                        "description": (
                            "Cone root signal (e.g. 'n11[0]'). "
                            "Only gates in the fanin cone of this signal are replaced. "
                            "Omit or set to null to apply across the entire design."
                        ),
                    },
                },
                "required": ["source_type", "target_types"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fraig_merge_equivalent_gates",
            "description": (
                "Use Yosys+ABC FRAIG to find, formally prove, and merge functionally equivalent "
                "combinational nodes across the whole design. Invoke this exact tool when asked "
                "to merge functionally equivalent gates or equivalent gate pairs. The ABC flow "
                "runs FRAIG only; it does not run dc2, dch, balancing, depth optimization, or retiming. "
                "DFF boundaries and an active whole-design gate restriction are preserved."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remap_design_with_gates",
            "description": (
                "Reconstruct the ENTIRE netlist/whole design using only the specified gate types. "
                "Invoke this exact tool when the request says 'entire netlist', 'whole design', "
                "or asks to restrict every combinational gate in the design. Do not pass a module "
                "name to remap_cone_with_gates for whole-design requests. DFFs and their boundary "
                "nets are preserved. Example: allowed_gates=['and', 'not']."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "allowed_gates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The only combinational gate types permitted afterward. "
                            "Valid values: 'and', 'nand', 'nor', 'or', 'not', "
                            "'xor', 'xnor', and 'buf'."
                        ),
                    },
                },
                "required": ["allowed_gates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remap_cone_with_gates",
            "description": (
                "Re-implement the fanin cone of a given output signal using ONLY the specified gate types. "
                "This is only for a specific driven signal, never an entire module or whole design. "
                "Uses Yosys+ABC with a restricted Liberty library so the result is guaranteed to use "
                "only the allowed gates. Handles any source gate type (OR, NOR, XOR, etc.) automatically. "
                "Use this only when asked to remap, restrict, rebuild, or convert the entire cone "
                "to a restricted set (e.g. 'replace all gates in cone of n11[0] with only NAND and NOT gates'). "
                "Do not use this when the prompt names one source gate type such as 'replace all OR gates'; "
                "use replace_gate_type_in_cone for that narrower operation. "
                "Returns success, gates_before, gates_after."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string",
                        "description": "The output net whose fanin cone will be remapped (e.g. 'n11[0]').",
                    },
                    "allowed_gates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of gate types the cone may use after remapping. "
                            "Valid values: 'and', 'nand', 'nor', 'or', 'not', 'xor', 'xnor', 'buf'. "
                            "Example: ['nand', 'not']"
                        ),
                    },
                },
                "required": ["output_signal", "allowed_gates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_primary_ios",
            "description": (
                "Count primary inputs and primary outputs in the current design. Use this "
                "exact tool for prompts such as 'Determine the number of primary inputs "
                "and outputs.' Returns declared port counts and bit-expanded counts. "
                "Do not use this when the prompt asks to list port names with bit widths; "
                "use list_primary_ios instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_primary_ios",
            "description": (
                "List primary input and primary output ports with declared bit widths, "
                "Verilog ranges, and expanded bit names. Use this exact tool for prompts "
                "such as 'Please list all the primary inputs of this design with their bit "
                "widths', 'list all primary outputs with widths', or 'show all primary I/O "
                "ports and their bit widths'. Do not use list_signals for primary port "
                "width questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "io_type": {
                        "type": "string",
                        "enum": ["all", "inputs", "outputs"],
                        "description": "Optional filter: 'inputs' to list primary inputs only, 'outputs' to list primary outputs only, or 'all' (default).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_depth_preserving_cone_gate_set",
            "description": (
                "Optimize/minimize the WHOLE DESIGN maximum logic depth while preserving a "
                "gate-set restriction on only one named fanin cone. Use this exact tool for "
                "prompts like 'minimize maximum path depth, ensuring the cone of n15 contains "
                "only AND, OR, and NOT gates' or 'optimize depth while the cone of n11[0] "
                "continues to use only NAND and NOT gates' when the cost function is the "
                "maximum logic depth of the final design. Internally this first runs "
                "unrestricted whole-design depth optimization, then remaps only the requested "
                "cone to allowed_gates. Do not use reduce_critical_path(allowed_gates=...) "
                "for cone-only restrictions, because that restricts the entire design. "
                "Do not use this when the cost function is the depth of the named cone itself; "
                "use optimize_cone_depth_preserving_gate_set instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string",
                        "description": "The output/net whose fanin cone must obey allowed_gates.",
                    },
                    "allowed_gates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Only these combinational gate types may remain in the named cone. "
                            "Valid values: 'and', 'nand', 'nor', 'or', 'not', 'xor', 'xnor', 'buf'."
                        ),
                    },
                    "verify_equivalence": {
                        "type": "boolean",
                        "description": (
                            "Optional expensive equivalence check against the originally loaded design. "
                            "Set true when the prompt explicitly asks to ensure functional equivalence."
                        ),
                    },
                },
                "required": ["output_signal", "allowed_gates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_cone_depth_preserving_gate_set",
            "description": (
                "Optimize/minimize the depth of one named fanin cone while preserving a "
                "gate-set restriction on that same cone. Use this exact tool for prompts like "
                "'Optimize the depth of the cone of n8 while ensuring the cone of n8 maintains "
                "only NAND and NOT gates' or prompts where the cost function is explicitly "
                "'the depth of the cone'. This does not optimize the whole design first; it "
                "extracts/remaps only the named cone and restores the input design if the cone "
                "depth does not improve. Do not use optimize_depth_preserving_cone_gate_set "
                "for cone-depth cost prompts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_signal": {
                        "type": "string",
                        "description": "The output/net whose fanin cone depth should be optimized.",
                    },
                    "allowed_gates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Only these combinational gate types may remain in the named cone. "
                            "Valid values: 'and', 'nand', 'nor', 'or', 'not', 'xor', 'xnor', 'buf'."
                        ),
                    },
                    "verify_equivalence": {
                        "type": "boolean",
                        "description": (
                            "Optional. The tool uses functional-preserving synthesis/remapping; "
                            "full-design equivalence is intentionally not run for this cone-local flow."
                        ),
                    },
                },
                "required": ["output_signal", "allowed_gates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_signals",
            "description": (
                "Return a compact signal inventory for the currently loaded netlist. "
                "Large lists are written to ./_tmp/ and only counts, samples, and file paths "
                "are returned inline. Do not use this for simple primary input/output counts; "
                "use count_primary_ios instead. Do not use this for listing primary input "
                "or output bit widths; use list_primary_ios instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reduce_critical_path",
            "description": (
                "Invoke this without fail when asked to reduce, minimize or optimize the critical path depth "
                "or maximum logic depth of the design. "
                "Uses Yosys+ABC logic restructuring to reduce combinational depth. "
                "When the request says the ENTIRE netlist/design must remain restricted to specific gate types, "
                "pass those exact types in allowed_gates (for example ['and', 'not']). "
                "If only a named cone has a gate-set restriction, use "
                "optimize_depth_preserving_cone_gate_set instead. "
                "If omitted, a restriction established by remap_design_with_gates is inherited. "
                "No retiming or sequential changes are made — only combinational logic is restructured. "
                "Returns depth_before, depth_after, improvement and success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "allowed_gates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional exclusive gate set for the optimized result. "
                            "Use only when the prompt says the entire design must remain those gates only."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "auto_insert_buffers",
            "description": (
                "Automatically find nets with fanout > max_fanout and insert buffers so no gate drives more than max_fanout loads. "
                "If 'nets' is provided (array of net names), only those nets are processed; otherwise all known nets are checked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_fanout": {"type": "integer", "description": "Maximum allowed fanout per net."},
                    "nets": {"type": "array", "items": {"type": "string"}, "description": "Optional list of net names to process (default: all)."}
                },
                "required": ["max_fanout"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_missing_tool",
            "description": (
                "Synthesize and create a new custom tool dynamically. Call this tool whenever "
                "the user's request asks for a query, property, analysis, or transformation that "
                "is not directly supported by an exact existing tool (for example: finding gates whose "
                "output drives a primary output pin, finding nets that drive multiple gate types, "
                "checking gate input pin counts/parity, custom topological queries, or novel logic transformations). "
                "Provide an exhaustive specification so the engine can generate, test, and register the new tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Canonical snake_case name for the new tool (e.g. 'count_odd_input_gates').",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "An exhaustive, precise functional specification of the required tool. "
                            "You MUST include: "
                            "1. The exact mathematical calculation, topological query, or mutation rule. "
                            "2. How input parameters (e.g. signal names, threshold numbers) should be used. "
                            "3. What keys and values must be returned in the output dictionary. "
                            "4. Any special conditions (e.g., what to return if 0 matches are found)."
                        ),
                    },
                    "is_transformation": {
                        "type": "boolean",
                        "description": "True if the tool modifies the netlist; False if it is a read-only query.",
                    },
                    "preserves_equivalence": {
                        "type": "boolean",
                        "description": "True if transformation must preserve combinational logic equivalence. Default True.",
                    },
                    "parameters_needed": {
                        "type": "object",
                        "description": "Key-value dictionary of parameter names, their types, and descriptions.",
                    },
                },
                "required": ["tool_name", "description", "is_transformation", "parameters_needed"],
            },
        },
    },
]

