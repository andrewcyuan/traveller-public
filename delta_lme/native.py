"""Load the pinned, unmodified LongMemEval V2 implementation."""
import sys
from importlib import import_module
from pathlib import Path
VENDOR = Path(__file__).resolve().parents[1] / "vendor/LongMemEval-V2"
sys.path.insert(0, str(VENDOR))
harness = import_module("evaluation.harness")
metrics = import_module("evaluation.qa_eval_metrics")
CodexMemory = import_module("memory_modules.codex").CodexMemory
memory = import_module("memory_modules.memory")
