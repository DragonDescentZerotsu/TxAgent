#!/usr/bin/env python3
"""Export one group branch's exact prompt, reasoning trace, and outputs as text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_group(run_dir: Path, group_key: str) -> dict[str, Any]:
    path = run_dir / "group_reasoning_outputs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        output = json.loads(line)
        if group_key.lower() in str(output.get("group_id") or "").lower():
            return output
    raise ValueError(f"No group containing {group_key!r} in {path}")


def _message(messages: list[dict[str, Any]], role: str) -> str:
    return next(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == role
    )


def export_group_branch(run_dir: Path, group_key: str) -> str:
    output = _read_group(run_dir, group_key)
    llm = output.get("llm") or {}
    validation = llm.get("structured_output_validation") or {}
    messages = llm.get("messages") or []
    assistant = next(
        (
            str(message.get("content") or "")
            for message in reversed(messages)
            if message.get("role") == "assistant"
        ),
        "",
    )
    lines = [
        "GROUP BRANCH: PROMPT, THINKING TRACE, AND VERBATIM OUTPUT",
        "",
        f"Run: {run_dir}",
        f"Group: {output.get('group_id')}",
        f"Status: {output.get('status')}",
        "Structured-output validation: "
        f"valid={validation.get('valid')}; attempts={validation.get('attempt_count')}",
        "",
        "===== SYSTEM PROMPT =====",
        _message(messages, "system"),
        "",
        "===== USER PROMPT =====",
        _message(messages, "user"),
        "",
        "===== THINKING TRACE =====",
        str(llm.get("reasoning_content") or "(none captured)"),
        "",
        "===== VERBATIM ASSISTANT OUTPUT =====",
        assistant or "(verbatim assistant message unavailable)",
        "",
        "===== PARSED STRUCTURED OUTPUT =====",
        json.dumps(llm.get("content"), indent=2, ensure_ascii=False),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("group_key")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output_path = args.output or args.run_dir / (
        f"{args.group_key.replace('.', '_')}_branch_prompt_thinking_trace.txt"
    )
    output_path.write_text(
        export_group_branch(args.run_dir, args.group_key),
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
