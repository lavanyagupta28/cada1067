"""I/O handler — reads requests from stdin and writes tagged responses to stdout."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Optional

from .agent import EDAAgent


class IOHandler:
    """Manages the stdin → agent → stdout/log pipeline.

    Args:
        agent: An initialised EDAAgent instance.
    """

    def __init__(self, agent: EDAAgent) -> None:
        self._agent = agent
        self._agent.set_io_handler(self) #bidirectional linkage between IOhandler and agent
        self._log_file: Optional[IO[str]] = None
        self._log_path: Optional[Path] = None
        self._response_counter: int = 0

        # Ensure UTF-8 I/O encoding on Windows without encoding exceptions
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if hasattr(sys.stdin, "reconfigure"):
            try:
                sys.stdin.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_log_file(self, path: str) -> None:
        """Open a fresh protocol log at *path* for this process.

        Re-selecting the active path is a no-op so a repeated tool call cannot
        truncate responses already written by this process.

        Args:
            path: File-system path of the log file.
        """
        log_path = Path(path).resolve()
        if self._log_file is not None and self._log_path == log_path:
            return
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
        self._log_file = log_path.open("w", encoding="utf-8")
        self._log_path = log_path

    def run(self) -> None:
        """Main loop: read lines from stdin, invoke the agent, write responses.

        Reads until EOF.  After printing each ``#END <id>`` block, stdout is
        flushed before blocking on the next line.

        A running context summary is built after each request and passed to
        the next one so the LLM knows the current engine state (loaded design,
        previous operation results, etc.).
        """
        context_summary: str = ""

        try:
            for raw_line in sys.stdin:
                line = raw_line.rstrip("\r\n")

                self._response_counter += 1
                resp_id = self._response_counter

                try:
                    response_text = self._agent.process_request(line, context_summary)
                except Exception as exc:
                    response_text = f"[ERROR] {exc}"

                self._emit_response(resp_id, response_text)

                # Build context for the next request
                context_summary = self._agent.build_state_summary()
        finally:
            if self._log_file is not None:
                self._log_file.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_response(self, resp_id: int, text: str) -> None:
        """Write a response block to stdout (and optionally the log file).

        Format::

            #RESPONSE <id>
            <text>
            #END <id>

        Args:
            resp_id: Monotonically increasing response identifier.
            text:    The response body text.
        """
        block = f"#RESPONSE {resp_id}\n{text}\n#END {resp_id}\n"
        if self._log_file is not None:
            self._log_file.write(block)
            self._log_file.flush()

        sys.stdout.write(block)
        sys.stdout.flush()
