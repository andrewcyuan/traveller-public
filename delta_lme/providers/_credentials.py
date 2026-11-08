"""Load provider credentials from the process environment or repo-local .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def api_key(name: str) -> str:
    load_dotenv(REPO_ROOT / ".env", override=False)
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} in the environment or repo .env file")
    return value
