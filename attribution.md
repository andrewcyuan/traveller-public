# Third-party notices

Native benchmark files under `vendor/LongMemEval-V2/` come from https://github.com/xiaowu0162/LongMemEval-V2 at commit `2cc8c540bdb87fe6761629b585e727e1c4704520`. The Apache 2.0 license is retained at `vendor/LongMemEval-V2/LICENSE`.

Modification: `memory_modules/memory.py` omits automatic imports of five unused memory implementations (no_retrieval, agentrunbook_c, agentrunbook_c_v2, agentrunbook_r, rag). The remaining native files, including answer parsing and scoring, are unchanged. Only the runtime dependency subset is distributed.

AgentRunbook-C instructions, query prompt, and helper scripts under `strategies/delta/` derive from that upstream project and remain subject to its license.

Dataset downloads are pinned to Hugging Face dataset `xiaowu0162/longmemeval-v2`, revision `f152293e235517d504809563c833d7190b8c713b`. Dataset and screenshot rights are governed by their source terms. No dataset files are distributed in this repository.

Traveller code was extracted from the working tree of `longmemeval-v2-delta`, based on commit `4bc1e6f`, including its reader image-limit updates. This extraction does not grant a new license to Traveller-authored code. Python dependencies retain their respective licenses.
