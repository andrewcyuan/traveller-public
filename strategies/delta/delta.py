"""A query-local tool loop over an immutable trajectory directory."""

import base64
import json
import mimetypes
import re
import subprocess
import sys
from pathlib import Path


ASSETS = Path(__file__).parent


def tool(name, description, properties, required):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOLS = [
    tool(
        "read_file",
        "Read a local text file, optionally matching a regex. Offsets and limits count characters. Use on trajectory summaries first.",
        {
            "path": {"type": "string"},
            "pattern": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 160000},
        },
        ["path"],
    ),
    tool(
        "inspect_trajectory",
        "Run the original scripts/inspect_trajectory.py helper. Supply its command-line arguments as an array, starting with trajectory ID.",
        {
            "args": {"type": "array", "items": {"type": "string"}},
        },
        ["args"],
    ),
    tool(
        "view_image",
        "View a local question image or trajectory screenshot.",
        {
            "path": {"type": "string"},
        },
        ["path"],
    ),
    tool(
        "write_output",
        "Write memory_module_output.json. Follow the exact schema in INSTRUCTION.md.",
        {
            "memory_markdown": {"type": "string"},
            "trajectory_spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "trajectory_id": {"type": "string"},
                        "start_state_index": {"type": "integer", "minimum": 0},
                        "end_state_index": {"type": "integer", "minimum": 0},
                    },
                    "required": [
                        "trajectory_id",
                        "start_state_index",
                        "end_state_index",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        ["memory_markdown", "trajectory_spans"],
    ),
]


def image_part(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}


def local_path(workspace, name):
    """Allow only query-local files and the linked source trajectory tree."""
    path = (workspace / name).resolve()
    roots = [workspace.resolve(), (workspace / "trajectories").resolve()]
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError("Path is outside the query workspace")
    return path


def validate_output(payload, workspace):
    if set(payload) != {"memory_markdown", "trajectory_spans"}:
        raise ValueError(
            "Output must have exactly memory_markdown and trajectory_spans"
        )
    if not isinstance(payload["memory_markdown"], str):
        raise TypeError("memory_markdown must be text")
    if not isinstance(payload["trajectory_spans"], list):
        raise TypeError("trajectory_spans must be a list")
    total = 0
    for span in payload["trajectory_spans"]:
        if set(span) != {"trajectory_id", "start_state_index", "end_state_index"}:
            raise ValueError("Invalid span fields")
        start, end = span["start_state_index"], span["end_state_index"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError("Invalid inclusive state indices")
        tid = span["trajectory_id"]
        if not isinstance(tid, str) or re.fullmatch(r"[A-Za-z0-9_-]+", tid) is None:
            raise ValueError("Invalid trajectory ID")
        source = local_path(workspace, f"trajectories/{tid}/trajectory.json")
        trajectory = json.loads(source.read_text())
        if end >= len(trajectory["states"]):
            raise ValueError("Span exceeds trajectory length")
        total += end - start + 1
    if total > 20:
        raise ValueError("Spans exceed the original 20-state limit")
    return payload


def execute_tool(workspace, name, arguments):
    if name == "read_file":
        text = local_path(workspace, arguments["path"]).read_text()
        if arguments.get("pattern"):
            regex = re.compile(arguments["pattern"], re.IGNORECASE)
            text = "\n".join(
                f"{i}: {line}"
                for i, line in enumerate(text.splitlines(), 1)
                if regex.search(line)
            )
        start = arguments.get("offset", 0)
        limit = arguments.get("limit", 24000)
        if (
            type(start) is not int
            or start < 0
            or type(limit) is not int
            or not 1 <= limit <= 160000
        ):
            raise ValueError("Invalid character offset/limit")
        return (
            f"Characters {start}:{min(start + limit, len(text))} of {len(text)}\n"
            + text[start : start + limit]
        )
    if name == "inspect_trajectory":
        args = arguments["args"]
        if not args or re.fullmatch(r"[A-Za-z0-9_-]+", args[0]) is None:
            raise ValueError("First argument must be a trajectory ID")
        allowed = {
            "--state",
            "--span",
            "--match",
            "--max-text-chars",
            "--max-thought-chars",
            "--max-url-chars",
            "--max-match-lines",
            "--max-match-line-chars",
        }
        if len(args) % 2 != 1 or any(
            args[i] not in allowed for i in range(1, len(args), 2)
        ):
            raise ValueError("Use only the documented inspection options")
        local_path(workspace, f"trajectories/{args[0]}/trajectory.json")
        result = subprocess.run(
            [sys.executable, str(ASSETS / "scripts/inspect_trajectory.py"), *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return (result.stdout + result.stderr)[:160000]
    if name == "view_image":
        return image_part(local_path(workspace, arguments["path"]))
    if name == "write_output":
        payload = validate_output(arguments, workspace)
        (workspace / "memory_module_output.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        return "Output saved."
    raise ValueError("Unknown tool")

