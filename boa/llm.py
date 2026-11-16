"""Gemini wrapper with patient retries. LLaMEA's built-in Gemini_LLM gives up after 5 short retries,
which turns transient 503/429 errors into wasted candidates; here such errors are retried with
exponential backoff + jitter and do not consume LLaMEA's budget.
Subclasses LLaMEA's Gemini_LLM / OpenAI_LLM (https://github.com/XAI-liacs/LLaMEA, van Stein & Baeck 2025); only the retry logic is new."""
import random, time
from llamea.llm import Gemini_LLM


def _retriable(err):
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    s = str(err)
    return any(t in s for t in ("503", "429", "500", "502", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                                "timed out", "Timeout", "Connection", "disconnected"))


class RobustGemini(Gemini_LLM):
    def query(self, session_messages, max_retries=40, base_delay=8, max_delay=300, **kwargs):
        last = None
        for attempt in range(max_retries + 1):
            try:
                # one attempt only; backoff handled here
                return super().query(session_messages, max_retries=0, **kwargs)
            except Exception as err:  # noqa
                last = err
                if not _retriable(err) or attempt == max_retries:
                    raise
                time.sleep(min(max_delay, base_delay * 2 ** min(attempt, 6)) * (0.5 + random.random()))
        raise last


# ---- Cerebras (OpenAI-compatible endpoint, free tier) -------------------------------------------------
import openai
from llamea.llm import OpenAI_LLM


class RobustCerebras(OpenAI_LLM):
    """gpt-oss-120b on Cerebras' OpenAI-compatible API (https://inference-docs.cerebras.ai).
    Same patient retry policy as RobustGemini; 429/5xx/timeouts are retried with exponential backoff."""

    def __init__(self, api_key, model="gpt-oss-120b", temperature=1.0, base_url="https://api.cerebras.ai/v1",
                 reasoning_effort="low", **kw):
        super().__init__(api_key, model=model, temperature=temperature, **kw)
        self.reasoning_effort = reasoning_effort
        self._client_kwargs = dict(api_key=api_key, base_url=base_url)
        self.client = openai.OpenAI(**self._client_kwargs)

    def query(self, session_messages, max_retries=40, base_delay=8, max_delay=300, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=session_messages, temperature=self.temperature,
                    max_completion_tokens=16000, reasoning_effort=self.reasoning_effort)
                # thinking models can exhaust the token budget while reasoning and return no content
                return resp.choices[0].message.content or ""
            except Exception as err:  # noqa
                if not _retriable(err) or attempt == max_retries:
                    raise
                time.sleep(min(max_delay, base_delay * 2 ** min(attempt, 6)) * (0.5 + random.random()))


def _flatten(content):
    """Newer Mistral models may return a list of chunks (text / thinking); keep only the text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
        d = c if isinstance(c, dict) else getattr(c, "model_dump", lambda: {})()
        if d.get("type", "text") == "text" and isinstance(d.get("text"), str):
            parts.append(d["text"])
    return "".join(parts)


class RobustMistral(OpenAI_LLM):
    """Mistral La Plateforme (OpenAI-compatible chat endpoint, https://docs.mistral.ai). Same retry policy."""

    def __init__(self, api_key, model="codestral-latest", temperature=0.7, base_url="https://api.mistral.ai/v1",
                 reasoning_effort=None, max_tokens=12000, **kw):
        super().__init__(api_key, model=model, temperature=temperature, **kw)
        self.reasoning_effort, self.max_tokens = reasoning_effort, max_tokens
        self._client_kwargs = dict(api_key=api_key, base_url=base_url)
        self.client = openai.OpenAI(**self._client_kwargs, max_retries=0)

    def query(self, session_messages, max_retries=40, base_delay=8, max_delay=300, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
                resp = self.client.chat.completions.create(
                    model=self.model, messages=session_messages, temperature=self.temperature,
                    max_tokens=self.max_tokens, **extra)
                return _flatten(resp.choices[0].message.content)
            except Exception as err:  # noqa
                if not _retriable(err) or attempt == max_retries:
                    raise
                time.sleep(min(max_delay, base_delay * 2 ** min(attempt, 6)) * (0.5 + random.random()))
