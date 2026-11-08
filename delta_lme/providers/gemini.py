"""Direct Google Gemini API client for Gemini 3.8 Flash."""

from ._credentials import api_key

MODEL = "gemini-3.8-flash"


class GeminiFlashClient:
    """Small SDK wrapper; returns the Google GenAI response unchanged."""

    def __init__(self, key: str | None = None):
        from google import genai

        self._client = genai.Client(api_key=key or api_key("GEMINI_API_KEY"))

    def generate(self, contents, **options):
        return self._client.models.generate_content(
            model=MODEL,
            contents=contents,
            **options,
        )
