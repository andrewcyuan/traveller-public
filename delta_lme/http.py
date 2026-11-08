import asyncio
import httpx

async def request_json(url, body, api_key, timeout=240):
    """Bound total time even when a server sends periodic keepalive data."""
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                content=body,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
            )
            if response.is_error:
                error = response.json().get("error", {})
                metadata = error.get("metadata", {})
                raise RuntimeError(
                    f"OpenRouter HTTP {response.status_code}: {metadata.get('provider_name', '')} {metadata.get('provider_error_code', error.get('message', 'request failed'))}"
                )
            return response.json()
