"""Shared query-local file-agent tool loop for Delta benchmark adapters."""

import json
import re


def tool_loop(
    client,
    workspace,
    history,
    *,
    tools,
    execute,
    validate,
    model,
    reasoning_effort,
    max_turns,
):
    trace = workspace / "agent-trace.jsonl"
    for turn in range(max_turns):
        result = client.post(
            "v1/responses",
            {
                "model": model,
                "input": history,
                "tools": tools,
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": 8192,
                "store": False,
            },
        )
        output = result.get("output", [])
        with trace.open("a") as stream:
            stream.write(json.dumps({"turn": turn, "response": result}) + "\n")
        if result.get("status") != "completed":
            raise RuntimeError(f"Incomplete retrieval response: {result.get('status')}")
        history.extend(output)
        calls = [item for item in output if item["type"] == "function_call"]
        images = []
        for call in calls:
            try:
                arguments = json.loads(call["arguments"])
                response = execute(workspace, call["name"], arguments)
                if isinstance(response, dict):
                    images.append(response)
                    response = "Image attached in the following user message."
            except (ValueError, TypeError, OSError, KeyError, re.error) as error:
                response = f"Tool error: {error}"
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": response,
                }
            )
            with trace.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "tool": call["name"],
                            "arguments": call["arguments"],
                            "result": response,
                        }
                    )
                    + "\n"
                )
        if images:
            history.append({"role": "user", "content": images})
        target = workspace / "memory_module_output.json"
        if target.exists():
            return validate(json.loads(target.read_text()), workspace)
        if not calls:
            history.append(
                {
                    "role": "user",
                    "content": "Write your final result with write_output, following INSTRUCTION.md.",
                }
            )
    raise RuntimeError("Retrieval exceeded its turn limit")
