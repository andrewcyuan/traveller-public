"""Query-scoped first-reply transport through OpenRouter, with a shared cap."""
import asyncio
import fcntl
import gzip
import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from .budget import BudgetExceeded, BudgetLedger
from .checkpoint import canonical_json
from .http import request_json
from .redaction import strip_images
from .reader import returned_message


def api_key():
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or OPENROUTER_KEY before paid runs")
    return key

GEMINI_MODEL = "gemini-3.8-flash"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
TOGETHER_QWEN = "Qwen/Qwen3.5-9B"
DIRECT_MODELS = {GEMINI_MODEL, DEEPSEEK_MODEL, TOGETHER_QWEN}

def _provider_key(name):
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    key = os.environ.get(name)
    if not key:
        raise ValueError(f"Set {name} before paid runs")
    return key

def _chat_messages(history):
    messages=[]
    for item in history:
        if item.get("type") == "provider_message":
            messages.append(item["message"])
        elif item.get("type") == "function_call":
            if messages and messages[-1].get("role")=="assistant" and any(c["id"]==item["call_id"] for c in messages[-1].get("tool_calls",[])):
                continue
            messages.append({"role":"assistant","tool_calls":[{"id":item["call_id"],"type":"function",
                "function":{"name":item["name"],"arguments":item["arguments"]}}]})
        elif item.get("type") == "function_call_output":
            messages.append({"role":"tool","tool_call_id":item["call_id"],"content":str(item["output"])})
        elif item.get("role") in ("user","assistant","system"):
            content=item.get("content","")
            if isinstance(content,list):
                parts=[]
                for part in content:
                    if part.get("type") in ("input_text","output_text","text"):
                        parts.append({"type":"text","text":part.get("text","")})
                    elif part.get("type") in ("input_image","image_url"):
                        image=part.get("image_url")
                        if isinstance(image,dict): image=image.get("url")
                        parts.append({"type":"image_url","image_url":{"url":image}})
                content=parts
            messages.append({"role":item["role"],"content":content})
    return messages

def _chat_tools(tools):
    return [{"type":"function","function":{"name":t["name"],"description":t.get("description",""),
        "parameters":t.get("parameters",{})}} for t in tools]

def _responses_shape(raw):
    choice=(raw.get("choices") or [{}])[0]
    msg=choice.get("message") or {}
    output=[{"type":"provider_message","message":msg}]
    for call in msg.get("tool_calls") or []:
        fn=call.get("function",{})
        output.append({"type":"function_call","call_id":call.get("id"),"name":fn.get("name"),
                       "arguments":fn.get("arguments","{}")})
    finish=choice.get("finish_reason")
    usage=raw.get("usage") or {}
    normalized={**usage,"input_tokens":usage.get("prompt_tokens",0),"output_tokens":usage.get("completion_tokens",0)}
    return {"status":"incomplete" if finish=="length" else "completed","output":output,"usage":normalized,
            "provider_response":raw}

async def _post_direct(model,payload):
    if model == GEMINI_MODEL:
        assert "input" in payload and "messages" not in payload, "Gemini is retrieval-only"
    import httpx
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if model==GEMINI_MODEL:
        url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        key=_provider_key("GEMINI_API_KEY")
    else:
        url="https://api.together.xyz/v1/chat/completions"
        key=_provider_key("TOGETHER_API_KEY")
    request={"model":model,"messages":_chat_messages(payload["input"]) if "input" in payload else payload["messages"],
             "max_tokens":payload.get("max_output_tokens",payload.get("max_tokens",20000))}
    if model==GEMINI_MODEL:
        request.update(reasoning_effort=payload.get("reasoning",{}).get("effort","high"),
                       tools=_chat_tools(payload.get("tools",[])),tool_choice="auto")
    elif model==DEEPSEEK_MODEL:
        request.update(reasoning_effort="high",tools=_chat_tools(payload.get("tools",[])),tool_choice="auto")
    else:
        request.update(temperature=payload.get("temperature",0.6),top_p=payload.get("top_p",0.95),top_k=20)
        request.pop("max_tokens",None);request["max_tokens"]=20000
    async with asyncio.timeout(900):
        async with httpx.AsyncClient(timeout=900) as session:
            response=await session.post(url,json=request,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
            if response.is_error:
                raise RuntimeError(f"{model} HTTP {response.status_code}: {response.text[:1000]}")
    raw=response.json()
    if "input" not in payload:
        return {**raw,"wire_request":strip_images(request),"provider":"together","usage":{**(raw.get("usage") or {}),"input_tokens":(raw.get("usage") or {}).get("prompt_tokens",0),
                "output_tokens":(raw.get("usage") or {}).get("completion_tokens",0),
                "cost":_direct_cost(model,raw.get("usage") or {})}}
    result=_responses_shape(raw)
    result["wire_request"]=strip_images(request)
    result["usage"]["cost"]=_direct_cost(model,result.get("usage",{}))
    return result

def _direct_cost(model,usage):
    i=float(usage.get("input_tokens",usage.get("prompt_tokens",0)) or 0)
    o=float(usage.get("output_tokens",usage.get("completion_tokens",0)) or 0)
    if model==GEMINI_MODEL: return i*0.75/1_000_000+o*3.75/1_000_000
    if model==DEEPSEEK_MODEL: return i*0.30/1_000_000+o*1.20/1_000_000
    if model==TOGETHER_QWEN: return i*0.17/1_000_000+o*0.25/1_000_000
    return None

READER_MODEL = "Qwen/Qwen3.5-9B" if os.environ.get("DELTA_READER_PROVIDER", "openrouter") == "together" else "qwen/qwen3.5-9b"
READER = {
    "model": READER_MODEL,
    "max_tokens": 20000,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "reasoning": {"enabled": True},
    **({} if READER_MODEL == TOGETHER_QWEN else {"provider": {
        "only": ["deepinfra", "parasail"],
        "order": ["parasail", "deepinfra"],
        "quantizations": ["bf16"],
        "allow_fallbacks": True,
        "require_parameters": True,
    }}),
}

CONTROLLER_MODEL = os.environ.get("DELTA_CONTROLLER_MODEL", "openai/gpt-6-luna")

LIMITS = {
    ("v1/responses", "openai/gpt-6-luna"): 0.4,
    ("v1/responses", GEMINI_MODEL): 0.4,
    ("v1/responses", DEEPSEEK_MODEL): 0.4,
    ("alpha/decisions", "typesafe/jev-1.13"): 0.03,
    ("v1/chat/completions", READER_MODEL): 0.04,
    ("v1/chat/completions", "openai/gpt-5.2"): 1.0,
}

def attempts_used(folder, cache_key, model=None):
    path = folder / "calls.jsonl.gz"
    if not path.exists():
        return 0
    with gzip.open(path, "rt") as stream:
        return sum(
            not row.get("cache_hit")
            and (
                row.get("model") == READER["model"]
                if model == READER["model"]
                else row.get("cache_key") == cache_key
            )
            for row in (json.loads(line) for line in stream)
        )


@contextmanager
def model_slot(root, model, partition=None):
    folder = root / "model-slots" / model.replace("/", "_")
    if partition:
        folder /= partition
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(4 if model == READER["model"] else 6):
            lock = (folder / f"{index}.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                yield folder
            finally:
                lock.close()
            return
        time.sleep(0.05)


def pace(folder, *, cooldown=False):
    with (folder / "rate.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        deadline = float(lock.read() or "0")
        if not cooldown:
            time.sleep(max(0, deadline - time.monotonic()))
        lock.seek(0)
        lock.truncate()
        lock.write(str(max(deadline, time.monotonic() + (30 if cooldown else 1))))
        lock.flush()


def validate(route, payload):
    model = payload.get("model")
    bound = LIMITS.get((route, model))
    if bound is None:
        raise ValueError("Unregistered route/model")
    if (
        model == READER["model"]
        and {key: value for key, value in payload.items() if key != "messages"}
        != READER
    ):
        raise ValueError("Required reader configuration changed")
    if model in {"openai/gpt-6-luna", GEMINI_MODEL, DEEPSEEK_MODEL} and (
        payload.get("max_output_tokens") != 8192
        or payload.get("reasoning", {}).get("effort") != "high"
    ):
        raise ValueError("Unregistered controller configuration")
    if model == "typesafe/jev-1.13" and (
        len(payload.get("questions", {})) > 16
        or len(canonical_json(payload).encode()) > 160000
    ):
        raise ValueError("Unbounded classifier payload")
    if model == "openai/gpt-5.2" and (
        payload.get("max_tokens") != 4096
        or len(canonical_json(payload).encode()) > 200000
    ):
        raise ValueError("Unbounded native judge payload")
    return bound


def complete(route, response):
    if route == "v1/chat/completions":
        return returned_message(response)
    if route == "alpha/decisions":
        return isinstance(response.get("answers"), dict)
    # Cache incomplete algorithmic replies too; never regenerate token-limit failures.
    return response.get("status") in ("completed", "incomplete", "failed")


@dataclass(frozen=True)
class Client:
    root: Path
    folder: Path
    role: str
    budget_usd: float

    @property
    def ledger(self):
        return BudgetLedger(self.root / "budget.json", self.budget_usd)

    def log(self, call):
        self.folder.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.folder / "calls.jsonl.gz", "at") as stream:
            stream.write(
                json.dumps({"at": datetime.now(UTC).isoformat(), **call}) + "\n"
            )

    def post(self, route, payload):
        if payload.get("model") in {"openai/gpt-6-luna", GEMINI_MODEL, DEEPSEEK_MODEL}:
            assert payload["reasoning"]["effort"] == "high"
        bound = validate(route, payload)
        assert self.folder.resolve().is_relative_to(self.root.resolve())
        body = canonical_json(payload).encode()
        key = hashlib.sha256(route.encode() + body).hexdigest()
        # Per-query scope prevents cross-question caching of the generic first
        # controller turn from understating cold online retrieval latency.
        scope = str(self.folder.relative_to(self.root))
        cache_key = hashlib.sha256((scope + key).encode()).hexdigest()
        locks = self.root / "request-locks"
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / (cache_key + ".lock")).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self.dispatch(route, payload, body, key, cache_key, bound)

    def dispatch(self, route, payload, body, key, cache_key, bound):
        cache = self.root / "requests.sqlite"
        with sqlite3.connect(cache, timeout=60) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
            )
            found = db.execute(
                "SELECT response FROM responses WHERE key=?", (cache_key,)
            ).fetchone()
        base = {
            "key": key,
            "cache_key": cache_key,
            "route": route,
            "model": payload["model"],
            "request": strip_images(payload),
        }
        if found:
            response = json.loads(found[0])
            self.log({**base, "response": response, "cache_hit": True})
            return response
        for attempt in range(
            attempts_used(self.folder, cache_key, payload["model"]), 3
        ):
            ledger = self.ledger
            while True:
                try:
                    reservation = ledger.reserve(bound, payload["model"])
                    break
                except BudgetExceeded:
                    snapshot = ledger.snapshot()
                    if (
                        snapshot["spent_usd"] + snapshot["uncertain_usd"] + bound
                        > ledger.limit_usd
                    ):
                        raise
                    time.sleep(1)
            started = time.perf_counter()
            # Separate controller queues for the parallel benchmarks; their
            # real API durations remain measured. Reader capacity stays shared.
            partition = (
                self.role
                if payload["model"] in {"openai/gpt-6-luna", GEMINI_MODEL, DEEPSEEK_MODEL}
                else "bf16-pool"
                if payload["model"] == READER["model"]
                else None
            )
            with model_slot(self.root, payload["model"], partition) as slots:
                reader = payload["model"] == READER["model"]
                if reader:
                    pace(slots)
                try:
                    response = asyncio.run(
                        _post_direct(payload["model"], payload)
                    ) if payload["model"] in DIRECT_MODELS else asyncio.run(
                        request_json(
                            "https://openrouter.ai/api/" + route,
                            body,
                            api_key(),
                            timeout=900 if reader else 240,
                        )
                    )
                    if response.get("error") and not (
                        route == "v1/responses"
                        and response.get("status") in ("incomplete", "failed")
                    ):
                        raise RuntimeError(str(response["error"]))
                except BaseException as error:
                    ledger.finish(reservation, None)
                    self.log(
                        {
                            **base,
                            "attempt": attempt + 1,
                            "error": str(error),
                            "seconds": time.perf_counter() - started,
                            "reservation_usd": bound,
                        }
                    )
                    if (
                        isinstance(error, (KeyboardInterrupt, SystemExit))
                        or attempt == 2
                    ):
                        raise
                    if reader and "429" in str(error):
                        pace(slots, cooldown=True)
                    else:
                        time.sleep(30 if "429" in str(error) else 2)
                    continue
            cost = (response.get("usage") or {}).get("cost")
            try:
                reported_cost = float(cost) if cost is not None else None
            except (TypeError, ValueError, OverflowError):
                reported_cost = None
            ledger.finish(reservation, reported_cost)
            self.log(
                {
                    **base,
                    "attempt": attempt + 1,
                    "response": response,
                    "seconds": time.perf_counter() - started,
                    "reservation_usd": bound,
                }
            )
            if not complete(route, response):
                if attempt == 2:
                    raise ValueError("Three empty or malformed transport replies")
                continue
            if reader:
                if READER_MODEL == TOGETHER_QWEN:
                    assert response["provider"] == "together"
                else:
                    assert response["provider"].lower() in ("deepinfra", "parasail")
                    assert response["model"] in (READER["model"], READER["model"] + "-20260310")
            with sqlite3.connect(cache, timeout=60) as db:
                db.execute(
                    "INSERT OR IGNORE INTO responses VALUES (?,?)",
                    (cache_key, json.dumps(response)),
                )
            return response
        raise AssertionError("Unreachable transport retry exit")
