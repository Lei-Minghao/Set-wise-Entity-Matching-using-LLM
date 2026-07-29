import os
import time
from dataclasses import dataclass, field
from functools import wraps

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


# ---------------------------------------------------------------------------
# Shared token counter
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    prompt_tokens:     int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens     += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        return self

    def __repr__(self) -> str:
        return (f"TokenUsage(prompt={self.prompt_tokens}, "
                f"completion={self.completion_tokens}, "
                f"total={self.total_tokens})")



# ---------------------------------------------------------------------------
# Rate limiters
# ---------------------------------------------------------------------------

class SimpleRateLimiter:
    """Throttles API requests to stay within rate limits."""

    def __init__(self, max_requests_per_minute: int = 20):
        self.max_rpm = max_requests_per_minute
        self.request_times = []

    def wait_if_needed(self):
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        if len(self.request_times) >= self.max_rpm:
            sleep_time = 60 - (current_time - self.request_times[0]) + 0.1
            print(f"[WAIT] Rate limit approaching. Waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        self.request_times.append(time.time())


# Global rate limiters — adjust to match your API tier
_gemini_rate_limiter = SimpleRateLimiter(max_requests_per_minute=15)



def set_gemini_rate_limit(requests_per_minute: int):
    global _gemini_rate_limiter
    _gemini_rate_limiter = SimpleRateLimiter(max_requests_per_minute=requests_per_minute)
    print(f"[OK] Gemini rate limit set to {requests_per_minute} requests/minute")

# ---------------------------------------------------------------------------
# Vertex AI / Gemini
# ---------------------------------------------------------------------------

from google import genai


# Top-level classes required for diskcache pickling
class _GeminiMessage:
    def __init__(self, content: str):
        self.content = content


class _GeminiChoice:
    def __init__(self, content: str):
        self.message = _GeminiMessage(content)


class _GeminiResponse:
    """Thin wrapper that makes a Vertex AI response look like an OpenAI response."""

    def __init__(self, text: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.text = text
        self.usage = None
        self.prompt_tokens     = prompt_tokens
        self.completion_tokens = completion_tokens
        self.choices = [_GeminiChoice(text)]


def get_vertex_client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )


def _messages_to_prompt(messages: list[dict]) -> str:
    return "".join(f"{msg['role']}: {msg['content']}\n" for msg in messages)


@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=10))
def gemini_chat_complete(
    messages: list[dict],
    model: str = "gemini-2.5-flash-lite",
    client: genai.Client | None = None,
    token_usage: TokenUsage | None = None,
    **kwargs,
) -> _GeminiResponse:
    _gemini_rate_limiter.wait_if_needed()
    if client is None:
        client = get_vertex_client()

    prompt = _messages_to_prompt(messages)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": kwargs.get("temperature", 0.0),
            "max_output_tokens": kwargs.get("max_tokens", 1024),
        },
    )

    text = getattr(response, "text", None)
    if not text:
        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason
        except (AttributeError, IndexError):
            pass
        raise ValueError(
            f"Empty Gemini response (finish_reason={finish_reason}). Raw: {response}"
        )

    # Extract token counts from usage_metadata
    prompt_tokens     = 0
    completion_tokens = 0
    try:
        meta = response.usage_metadata
        prompt_tokens     = getattr(meta, "prompt_token_count",     0) or 0
        completion_tokens = getattr(meta, "candidates_token_count", 0) or 0
        # fallback: derive completion from total - prompt
        if completion_tokens == 0:
            total = getattr(meta, "total_token_count", 0) or 0
            completion_tokens = max(0, total - prompt_tokens)
    except AttributeError:
        pass

    if token_usage is not None:
        token_usage.prompt_tokens     += prompt_tokens
        token_usage.completion_tokens += completion_tokens

    return _GeminiResponse(text, prompt_tokens, completion_tokens)