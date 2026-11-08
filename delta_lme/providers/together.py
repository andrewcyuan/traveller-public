"""Direct Together AI client for the Qwen 3.5 9B chat model."""

from ._credentials import api_key

MODEL = "Qwen/Qwen3.5-9B"


class TogetherQwenClient:
    """Small SDK wrapper; returns the Together ChatCompletion unchanged."""

    def __init__(self, key: str | None = None):
        from together import Together

        self._client = Together(api_key=key or api_key("TOGETHER_API_KEY"))

    def complete(self, messages, **options):
        return self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            **options,
        )
