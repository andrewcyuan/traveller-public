"""Persistent reservations; every paid attempt counts against an explicit cap."""
import fcntl
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from .checkpoint import write_json

class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetLedger:
    path: Path
    limit_usd: float = 75.0

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {
                    "limit_usd": self.limit_usd,
                    "spent_usd": 0.0,
                    "uncertain_usd": 0.0,
                    "reservations": {},
                    "calls": 0,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            if data["limit_usd"] != self.limit_usd:
                raise ValueError("budget limit differs from the existing ledger")
            yield data
            write_json(self.path, data)

    def reserve(self, amount: float, model: str) -> str:
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("reservation must be finite and positive")
        key = uuid4().hex
        with self.locked() as data:
            total = (
                data["spent_usd"]
                + data["uncertain_usd"]
                + sum(item["usd"] for item in data["reservations"].values())
            )
            if total + amount > self.limit_usd:
                raise BudgetExceeded(f"${self.limit_usd:.2f} research budget exhausted")
            data["reservations"][key] = {
                "usd": amount,
                "model": model,
                "at": datetime.now(UTC).isoformat(),
            }
        return key

    def finish(self, key: str, cost: float | None) -> None:
        with self.locked() as data:
            held = data["reservations"].pop(key)["usd"]
            data["calls"] += 1
            if cost is None or not math.isfinite(cost) or cost < 0:
                data["uncertain_usd"] += held
            else:
                data["spent_usd"] += cost
                if cost > held:
                    # Never issue another call after a violated cost bound.
                    data["uncertain_usd"] = self.limit_usd

    def snapshot(self) -> dict[str, Any]:
        with self.locked() as data:
            return json.loads(json.dumps(data))
