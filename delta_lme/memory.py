"""Portable Insert and Query API for state-search retrieval."""
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import sys
import uuid
from importlib import import_module
from pathlib import Path
from strategies.delta.delta import ASSETS
from .budget import BudgetExceeded
from .checkpoint import write_json, fingerprint
from .state_search import query as search_query, build_index
from .ranking import rank
from .native import CodexMemory, harness, memory as native
PROFILE = "state-search"

def ensure_summaries(source_root):
    """Render the original deterministic indexes, never model-generated facts."""
    source_root = Path(source_root).resolve()
    concise = source_root / "TRAJECTORY_SUMMARY_CONCISE.md"
    full = source_root / "TRAJECTORY_SUMMARY_FULL.md"
    if concise.exists() and full.exists():
        return
    with (source_root / ".delta-summary.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if concise.exists() and full.exists():
            return
        subprocess.run(
            [
                sys.executable,
                str(ASSETS / "scripts/render_trajectory_summary.py"),
                str(source_root),
                "--concise-output",
                str(concise),
                "--full-output",
                str(full),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

def source_records(source_root):
    source_root = Path(source_root)
    text = (source_root / "TRAJECTORY_SUMMARY_FULL.md").read_text()
    parts = re.split(r"^## \d+\. ([A-Za-z0-9_-]+)\s*$", text, flags=re.MULTILINE)
    records = [
        {"id": parts[i], "text": parts[i + 1].strip()} for i in range(1, len(parts), 2)
    ]
    assert records and len(records) == len({row["id"] for row in records})
    assert {row["id"] for row in records} == {
        path.parent.name for path in source_root.glob("*/trajectory.json")
    }
    return records

def question_packet(question, ranking, image=False):
    packet = {
        "question": question,
        "ordered_jobs": ranking["ordered_jobs"],
        "search_hint": "Use these fallible relevance hints to shortlist; all original trajectories remain searchable. Scores are not facts or proof. Verify exact scope and preserve contrary observations.",
    }
    if image:
        packet["image"] = "question_image.png"
    return packet


def decode(workspace, question_id="query"):
    decoder = CodexMemory({"workspace_dir": str(workspace), "evidence_mode": "both",
                           "codex_params": {"binary": sys.executable}})
    output = decoder._normalize_output_for_query(workspace / "memory_module_output.json")
    assert not output["trajectory_spans_invalid"]
    assert sum(s["end_state_index"] - s["start_state_index"] + 1
               for s in output["trajectory_spans_valid"]) <= 20
    context, original, final = harness.truncate_memory_context(
        decoder._build_memory_context_from_output(output), max_tokens=200000,
        question_id=question_id)
    return {"output": output, "context": context, "original_context_tokens": original,
            "context_tokens": final}


def query_evidence(client, source_root, workspace, question, query_image=None,
                   *, state_index=None):
    source_root, workspace = Path(source_root).resolve(), Path(workspace).resolve()
    ensure_summaries(source_root)
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".query.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = {"question": question, "profile": PROFILE,
                    "image_sha256": hashlib.sha256(Path(query_image).read_bytes()).hexdigest() if query_image else None,
                    "source": str(source_root),
                    "summary_sha256": hashlib.sha256((source_root / "TRAJECTORY_SUMMARY_FULL.md").read_bytes()).hexdigest()}
        identity_path = workspace / "query-identity.json"
        if identity_path.exists():
            assert json.loads(identity_path.read_text()) == identity
        else:
            assert not (workspace / "memory_module_output.json").exists()
            write_json(identity_path, identity)
        assert not (workspace / "query-error.json").exists(), "Terminal query failure is preserved"
        link = workspace / "trajectories"
        if not link.exists():
            link.symlink_to(source_root, target_is_directory=True)
        assert link.resolve() == source_root
        for name in ("INSTRUCTION.md", "query_prompt.txt"):
            shutil.copyfile(ASSETS / name, workspace / name)
        shutil.copytree(ASSETS / "scripts", workspace / "scripts", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        if query_image:
            shutil.copyfile(query_image, workspace / "question_image.png")
        index = workspace / "state-index.sqlite"
        if state_index:
            if not index.exists(): index.symlink_to(Path(state_index).resolve())
            assert index.resolve() == Path(state_index).resolve()
        else:
            build_index(source_root, index)
        if not (workspace / "memory_module_output.json").exists():
            try:
                ranking = rank(client, question, source_records(source_root))
                write_json(workspace / "ranking.json", ranking)
                write_json(workspace / "question.json", question_packet(question, ranking, bool(query_image)))
                search_query(client, workspace)
            except BudgetExceeded:
                raise
            except Exception as error:
                write_json(workspace / "query-error.json", {"error": type(error).__name__ + ": " + str(error)})
                raise
        return decode(workspace)


@native.register_memory
class DeltaMemory(CodexMemory):
    memory_type = "delta_lme"

    def __init__(self, memory_params):
        super().__init__({**memory_params, "evidence_mode": "both",
                          "codex_params": {**memory_params.get("codex_params", {}), "binary": sys.executable}})

    @property
    def memory_config(self):
        params = {"evidence_mode": "both", "profile": PROFILE,
                  **{k: self.memory_params[k] for k in ("client_factory", "client_configuration") if k in self.memory_params}}
        if self.trajectory_pool_root is not None:
            params["trajectory_pool_root"] = str(self.trajectory_pool_root)
        return {"memory_type": self.memory_type, "memory_params": params}

    def insert(self, trajectory):
        count = len(self.inserted_trajectory_ids)
        super().insert(trajectory)
        if len(self.inserted_trajectory_ids) != count:
            for name in ("TRAJECTORY_SUMMARY_CONCISE.md", "TRAJECTORY_SUMMARY_FULL.md"):
                (self.workspace_dir / "trajectories" / name).unlink(missing_ok=True)
            (self.workspace_dir / "state-index.sqlite").unlink(missing_ok=True)

    def prepare(self):
        source = self.workspace_dir / "trajectories"
        ensure_summaries(source)
        build_index(source, self.workspace_dir / "state-index.sqlite")

    def query(self, query, query_image=None):
        specification = self.memory_params.get("client_factory")
        if not specification or ":" not in specification:
            raise ValueError("Supply a budgeted client_factory module:function")
        module, name = specification.rsplit(":", 1)
        factory = getattr(import_module(module), name)
        invocation = self.get_query_context().get("query_invocation_id") or uuid.uuid4().hex
        workspace = self.workspace_dir / "delta_queries" / hashlib.sha256(invocation.encode()).hexdigest()
        workspace.mkdir(parents=True, exist_ok=True)
        client = factory(workspace, dict(self.memory_params.get("client_configuration", {})))
        self.prepare()
        return query_evidence(client, self.workspace_dir / "trajectories", workspace,
                              query, query_image,
                              state_index=self.workspace_dir / "state-index.sqlite")["context"]
