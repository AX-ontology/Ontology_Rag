from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import signal
import time
from typing import Any
import urllib.error
import urllib.request

from .utils import read_json, stable_id, text, write_json


class LLMDeadlineExceeded(TimeoutError):
    """Raised when an entire HTTP response exceeds the configured deadline."""


class LLMResponseError(RuntimeError):
    """The provider responded, but its JSON was incomplete or malformed."""


class _response_deadline:
    """Bound total response time, including chunked bodies that never finish."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.previous_handler: Any = None
        self.previous_timer: tuple[float, float] = (0.0, 0.0)

    def __enter__(self) -> None:
        if not hasattr(signal, "setitimer"):
            return
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._raise_timeout)
        self.previous_timer = signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *_: Any) -> None:
        if not hasattr(signal, "setitimer"):
            return
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        if self.previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *self.previous_timer)

    def _raise_timeout(self, *_: Any) -> None:
        raise LLMDeadlineExceeded(
            f"LLM response exceeded total deadline ({self.seconds:g}s)"
        )


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    api_base: str
    model: str
    timeout: float
    max_retries: int
    rate_limit_seconds: float
    max_tokens: int = 8192
    thinking: str = ""
    max_tokens_limit: int = 8192

    @classmethod
    def from_env(cls) -> "LLMConfig":
        key = os.environ.get("TASKFRAME_API_KEY", "").strip()
        model = os.environ.get("TASKFRAME_MODEL", "").strip()
        if not key or not model:
            raise ValueError(
                "TASKFRAME_API_KEY와 TASKFRAME_MODEL이 필요합니다. "
                ".env를 작성하거나 --backend heuristic을 사용하세요."
            )
        api_base = os.environ.get(
            "TASKFRAME_API_BASE", "https://api.openai.com/v1"
        ).rstrip("/")
        default_thinking = "disabled" if "deepseek" in api_base.casefold() else ""
        max_tokens = int(os.environ.get("TASKFRAME_MAX_TOKENS", "8192"))
        return cls(
            api_key=key,
            api_base=api_base,
            model=model,
            timeout=float(os.environ.get("TASKFRAME_TIMEOUT", "180")),
            max_retries=int(os.environ.get("TASKFRAME_MAX_RETRIES", "4")),
            rate_limit_seconds=float(
                os.environ.get("TASKFRAME_RATE_LIMIT_SECONDS", "0.2")
            ),
            max_tokens=max_tokens,
            thinking=os.environ.get(
                "TASKFRAME_THINKING", default_thinking
            ).strip().casefold(),
            max_tokens_limit=int(os.environ.get(
                "TASKFRAME_MAX_TOKENS_LIMIT",
                str(max(max_tokens, 65536) if "deepseek" in api_base.casefold() else max_tokens),
            )),
        )


def parse_json(content: str) -> Any:
    value = content.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        starts = sorted(pos for pos in (value.find("{"), value.find("[")) if pos >= 0)
        # Only parse the outermost value. A truncated object may still contain
        # a complete nested array; accepting it silently loses sibling fields.
        for start in starts[:1]:
            try:
                result, _ = decoder.raw_decode(value[start:])
                return result
            except json.JSONDecodeError:
                pass
    raise ValueError("LLM 응답에서 JSON을 찾지 못했습니다.")


class ChatJSONClient:
    def __init__(self, config: LLMConfig, cache_dir: Path, use_cache: bool = True):
        self.config = config
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_response(stage: str, parsed: Any) -> None:
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a complete JSON object")
        required: tuple[str, ...] = ()
        if stage.startswith("llm_1_"):
            required = ("chunks",)
        elif stage == "llm_2_local_taskframe_induction":
            required = ("taskframes", "task_relations", "triple_link_decisions")
        elif stage == "llm_2_global_taskframe_reconciliation":
            required = ("taskframes", "task_relations")
        for field in required:
            if not isinstance(parsed.get(field), list):
                raise ValueError(f"LLM response requires array field: {field}")

    def _cache_path(self, stage: str, key: str, system: str, user: str) -> Path:
        call_id = stable_id(
            "CALL", self.config.api_base, self.config.model, self.config.thinking,
            self.config.max_tokens, stage, key, system, user,
            length=24,
        )
        return self.cache_dir / stage / f"{call_id}.json"

    def has_cached_call(self, stage: str, key: str, system: str, user: str) -> bool:
        return self.use_cache and self._cache_path(stage, key, system, user).exists()

    def call(self, stage: str, key: str, system: str, user: str) -> Any:
        cache_path = self._cache_path(stage, key, system, user)
        if self.use_cache and cache_path.exists():
            cached = read_json(cache_path)
            try:
                self._validate_response(stage, cached.get("parsed"))
                if cached.get("finish_reason") == "length":
                    raise ValueError("Truncated cache")
            except ValueError:
                pass
            else:
                return cached["parsed"]
            # Legacy parsing could salvage a nested array from a truncated
            # object. Preserve those responses for audit, then request the
            # complete object (including relation/link decision fields).
            backup = self.cache_dir / "_legacy_incomplete" / stage / (
                f"{cache_path.stem}.{time.time_ns()}.json"
            )
            backup.parent.mkdir(parents=True, exist_ok=True)
            cache_path.replace(backup)
            print(f"[llm-cache-invalid] stage={stage} saved={backup}", flush=True)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.config.thinking in {"enabled", "disabled"}:
            payload["thinking"] = {"type": self.config.thinking}
        error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            request = urllib.request.Request(
                f"{self.config.api_base}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with _response_deadline(self.config.timeout):
                    with urllib.request.urlopen(
                        request, timeout=self.config.timeout
                    ) as response:
                        envelope = json.loads(response.read().decode())
                choice = envelope["choices"][0]
                content = choice["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        text(part.get("text")) for part in content if isinstance(part, dict)
                    )
                try:
                    if choice.get("finish_reason") == "length":
                        raise ValueError("LLM output truncated (finish_reason=length)")
                    parsed = parse_json(str(content))
                    self._validate_response(stage, parsed)
                except ValueError as exc:
                    diagnostic = {
                        "stage": stage,
                        "model": self.config.model,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": envelope.get("usage", {}),
                        "content_chars": len(str(content)),
                        "requested_max_tokens": payload["max_tokens"],
                        "error": str(exc),
                    }
                    write_json(
                        self.cache_dir / "_errors" / stage
                        / f"{cache_path.stem}.attempt{attempt}.json",
                        diagnostic,
                    )
                    if choice.get("finish_reason") == "length":
                        ceiling = max(self.config.max_tokens, self.config.max_tokens_limit)
                        payload["max_tokens"] = min(ceiling, payload["max_tokens"] * 2)
                    raise ValueError(
                        f"{exc}; finish_reason={choice.get('finish_reason')}; "
                        f"content_chars={len(str(content))}; "
                        f"completion_tokens={envelope.get('usage', {}).get('completion_tokens')}"
                    ) from exc
                write_json(
                    cache_path,
                    {
                        "stage": stage, "model": self.config.model, "parsed": parsed,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": envelope.get("usage", {}),
                    },
                )
                if self.config.rate_limit_seconds:
                    time.sleep(self.config.rate_limit_seconds)
                return parsed
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:1200]
                error = RuntimeError(f"LLM HTTP {exc.code}: {body}")
                if exc.code == 400 and "response_format" in body:
                    payload.pop("response_format", None)
                elif exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise error
            except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
                error = exc
            if attempt < self.config.max_retries:
                print(
                    f"[llm-retry] stage={stage} attempt={attempt}/"
                    f"{self.config.max_retries} error={type(error).__name__}: {error}"
                )
                time.sleep(min(30, 2 ** (attempt - 1) + random.random()))
        failure_type = LLMResponseError if isinstance(error, ValueError) else RuntimeError
        raise failure_type(f"LLM 호출 실패(stage={stage}): {error}")
