"""State-search retrieval over original trajectory evidence."""

import gzip
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from strategies.delta.delta import ASSETS, TOOLS, execute_tool, tool, validate_output
from strategies.delta.loop import tool_loop

SEARCH_HINT = (
    "You can also use search_states to search the complete original state text "
    "across all trajectories, independently of the job ranking. Choose focused "
    "terms or quoted phrases from the question and observations. Results are "
    "navigation hints: inspect their original states or screenshots before "
    "using them as evidence. No minimum number of searches is required."
)
SEARCH_TOOL = tool(
    "search_states",
    "Search all original state text with SQLite FTS5. Use quoted phrases, AND, "
    "OR, NOT, parentheses or prefix*. Default is AND between terms. Results "
    "include trajectory/state IDs and matched snippets; inspect the source "
    "for full context. This search never selects reader evidence automatically.",
    {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 40},
    },
    ["query"],
)


def build_index(source, output):
    """Index original state text without modifying sources or reading labels."""
    source, output = Path(source).resolve(), Path(output).resolve()
    paths = sorted(source.glob("*/trajectory.json"))
    assert paths, source
    pins = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    identity = {"format": 1, "source": str(source), "source_sha256": pins}
    if output.exists():
        with sqlite3.connect(output.as_uri() + "?mode=ro", uri=True) as db:
            saved = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
            assert saved == identity, "Source index no longer matches its input"
            count = db.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        return {**identity, "states": count, "index": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.sqlite")
    with sqlite3.connect(temporary) as db:
        db.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
        db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(identity),))
        db.execute(
            "CREATE VIRTUAL TABLE states USING fts5("
            "text, trajectory_id UNINDEXED, state_index UNINDEXED, "
            "url UNINDEXED, screenshot UNINDEXED, source_sha256 UNINDEXED)"
        )
        count = 0
        for path in paths:
            trajectory = json.loads(path.read_text())
            assert trajectory["id"] == path.parent.name
            for index, state in enumerate(trajectory["states"]):
                assert state["state_index"] == index
                db.execute(
                    "INSERT INTO states VALUES (?,?,?,?,?,?)",
                    (
                        state["text"], trajectory["id"], index,
                        state.get("url"), state.get("screenshot"),
                        pins[str(path.relative_to(source))],
                    ),
                )
                count += 1
        db.execute("INSERT INTO states(states) VALUES ('optimize')")
    temporary.replace(output)
    return {**identity, "states": count, "index": str(output)}


def search_states(workspace, arguments):
    query = arguments["query"]
    limit = arguments.get("limit", 12)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Supply a nonempty FTS5 query")
    if type(limit) is not int or not 1 <= limit <= 40:
        raise ValueError("Search limit must be between 1 and 40")
    index = (workspace / "state-index.sqlite").resolve()
    try:
        with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            fetched = db.execute(
                "SELECT trajectory_id, state_index, text, url, screenshot, "
                "source_sha256, bm25(states) AS score, "
                "snippet(states, 0, '[', ']', ' … ', 64) AS snippet "
                "FROM states WHERE states MATCH ? "
                "ORDER BY score, trajectory_id, CAST(state_index AS INTEGER) LIMIT ?",
                (query, limit),
            ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"FTS5 search failed: {error}") from error
    rows = [
        {
            **dict(row), "state_index": int(row["state_index"]), "rank": rank,
            "id": f"{row['trajectory_id']}:{row['state_index']}",
        }
        for rank, row in enumerate(fetched, 1)
    ]
    result = {
        "query": query, "limit": limit,
        "matches": [
            {key: value for key, value in row.items() if key != "text"}
            for row in rows
        ],
    }
    with gzip.open(workspace / "state-search-results.jsonl.gz", "at") as stream:
        stream.write(json.dumps({"query": query, "limit": limit, "candidates": rows,
                                 "tool_result": result}, ensure_ascii=False) + "\n")
    return json.dumps(result, ensure_ascii=False)


def execute(workspace, name, arguments):
    if name == "search_states":
        return search_states(workspace, arguments)
    return execute_tool(workspace, name, arguments)


def query(client, workspace):
    workspace = Path(workspace).resolve()
    prompt = (ASSETS / "query_prompt.txt").read_text()
    instructions = (ASSETS / "INSTRUCTION.md").read_text()
    bridge = "\nTool interface: read_file replaces cat/sed/rg; inspect_trajectory runs the supplied Python helper; view_image opens images; write_output writes the required JSON file. INSTRUCTION.md is reproduced below.\n"
    packet = (workspace / "question.json").read_text()
    text = (
        prompt + bridge + instructions
        + "\n\nThe identical local question.json is included here to avoid a separate initial read:\n"
        + packet
    )
    text += "\n\n" + SEARCH_HINT
    history = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
    return tool_loop(
        client, workspace, history,
        tools=TOOLS + [SEARCH_TOOL],
        execute=execute,
        validate=validate_output,
        model=os.environ.get("DELTA_CONTROLLER_MODEL", "openai/gpt-6-luna"),
        reasoning_effort="high",
        max_turns=48,
    )
