"""First native Chat Completion capture, independent of answer correctness."""

from types import SimpleNamespace

from .native import harness as upstream


def returned_message(response):
    choices = response.get("choices") if isinstance(response, dict) else None
    return (
        isinstance(choices, list)
        and bool(choices)
        and isinstance(choices[0], dict)
        and isinstance(choices[0].get("message"), dict)
    )


def native_text(response):
    assert returned_message(response), "No returned native message"
    return upstream.extract_text_from_response_message(
        SimpleNamespace(**response["choices"][0]["message"])
    )


def first_native(calls, model):
    """Select first returned message after transport errors; never later content."""
    found = []
    for index, call in enumerate(calls):
        if call.get("cache_hit") or call.get("model") != model:
            continue
        if call.get("route") != "v1/chat/completions" or "error" in call:
            continue
        response = call.get("response")
        if returned_message(response):
            found.append((index, call))
    if not found:
        return None, {"returned_outputs": 0, "unused_returned_outputs": 0}
    index, first = found[0]
    return first, {
        "first_receipt_index": index,
        "returned_outputs": len(found),
        "unused_returned_outputs": len(found) - 1,
        "unused_reader_attempts": sum(
            i > index
            and not c.get("cache_hit")
            and c.get("model") == model
            and c.get("route") == "v1/chat/completions"
            for i, c in enumerate(calls)
        ),
        "first_native_text_empty": not bool(native_text(first["response"])),
    }
