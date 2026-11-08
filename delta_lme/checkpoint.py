"""Exclusive-writer, atomic JSON checkpoints independent of strategy state."""

import fcntl
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Any, Self, TextIO


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def implementation_hash(paths: Sequence[Path]) -> str:
    root = Path(os.path.commonpath([path.parent for path in paths]))
    return fingerprint(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


def write_json(path: Path, value: Any) -> None:
    """Commit a complete file using a same-directory temporary and atomic replace."""
    encoded = canonical_json(value).encode()
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_manifest(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("signature") != fingerprint(
        manifest["identity"]
    ):
        raise ValueError("invalid run manifest")
    return manifest


class RunStore:
    def __init__(
        self, output: Path, identity: Mapping[str, Any], *, resume: bool = False
    ):
        self.output = output
        self.identity = dict(identity)
        self.signature = fingerprint(identity)
        self.resume = resume
        self.lock: TextIO | None = None

    def __enter__(self) -> Self:
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "run.lock").open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = self.output / "manifest.json"
            if self.resume:
                if read_manifest(self.output)["signature"] != self.signature:
                    raise ValueError(
                        "resume inputs, settings, or implementation differ from this run"
                    )
            else:
                if manifest_path.exists() or (self.output / "checkpoint.json").exists():
                    raise ValueError(
                        "run already exists; use resume or a new output directory"
                    )
                write_json(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "signature": self.signature,
                        "identity": self.identity,
                    },
                )
        except BaseException:
            self.lock.close()
            self.lock = None
            raise
        return self

    def load(self) -> dict[str, Any] | None:
        path = self.output / "checkpoint.json"
        if not path.exists():
            return None
        envelope = json.loads(path.read_text())
        if (
            envelope.get("schema_version") != 1
            or envelope.get("signature") != self.signature
            or envelope.get("sha256") != fingerprint(envelope.get("checkpoint"))
        ):
            raise ValueError("checkpoint is incompatible or corrupt")
        return envelope["checkpoint"]

    def save(self, checkpoint: Mapping[str, Any]) -> None:
        write_json(
            self.output / "checkpoint.json",
            {
                "schema_version": 1,
                "signature": self.signature,
                "sha256": fingerprint(checkpoint),
                "checkpoint": checkpoint,
            },
        )

    def event(self, value: Mapping[str, Any]) -> None:
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.lock is not None:
            self.lock.close()
            self.lock = None
