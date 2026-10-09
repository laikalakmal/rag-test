"""
src/utils/gemini_client.py
--------------------------
Centralized Google Gemini client and rate-limiter tailored to Google AI Studio
model quotas and rate limits.

Features:
- Accurate rate limits per model (RPM, TPM, RPD) based on Google AI Studio quotas.
- Automatic request pacing with sliding-window rate limiting.
- Robust response parsing for string, list, and multi-part dictionary outputs.
- Exponential backoff retry on 429 (RESOURCE_EXHAUSTED) and 503 (UNAVAILABLE),
  respecting Google's 'retry in Xs' directives.
- Automatic fallback model chains.
"""

import collections
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("GeminiClient")

# Model rate limit specifications directly from user's Google AI Studio quota table:
GEMINI_MODEL_LIMITS: Dict[str, Dict[str, Any]] = {
    # Flash Lite models: 15 RPM, 250K TPM, 500 RPD
    # Safe pacing: max 14 RPM cap, minimum 4.2s inter-request delay
    "gemini-3.5-flash-lite": {
        "rpm": 15,
        "max_rpm": 14,
        "tpm": 250_000,
        "rpd": 500,
        "safe_interval": 4.2,
    },
    "gemini-3.1-flash-lite": {
        "rpm": 15,
        "max_rpm": 14,
        "tpm": 250_000,
        "rpd": 500,
        "safe_interval": 4.2,
    },
    "gemini-3.1-flash-lite-preview": {
        "rpm": 15,
        "max_rpm": 14,
        "tpm": 250_000,
        "rpd": 500,
        "safe_interval": 4.2,
    },
    "gemini-flash-lite-latest": {
        "rpm": 15,
        "max_rpm": 14,
        "tpm": 250_000,
        "rpd": 500,
        "safe_interval": 4.2,
    },

    # 2.5 Flash Lite: 10 RPM, 250K TPM, 20 RPD
    "gemini-2.5-flash-lite": {
        "rpm": 10,
        "max_rpm": 9,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 6.5,
    },

    # Regular Flash models: 5 RPM, 250K TPM, 20 RPD
    # Safe pacing: max 4 RPM cap, minimum 13.0s inter-request delay
    "gemini-3.8-flash": {
        "rpm": 5,
        "max_rpm": 4,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 13.0,
    },
    "gemini-3.7-flash": {
        "rpm": 5,
        "max_rpm": 4,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 13.0,
    },
    "gemini-3.6-flash": {
        "rpm": 5,
        "max_rpm": 4,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 13.0,
    },
    "gemini-3.5-flash": {
        "rpm": 5,
        "max_rpm": 4,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 13.0,
    },
    "gemini-3-flash": {
        "rpm": 5,
        "max_rpm": 4,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 13.0,
    },
    "gemini-3-flash-preview": {
        "rpm": 5,
        "max_rpm": 4,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 13.0,
    },
    "gemini-2.5-flash": {
        "rpm": 5,
        "max_rpm": 4,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 13.0,
    },
    "gemini-flash-latest": {
        "rpm": 5,
        "max_rpm": 4,
        "tpm": 250_000,
        "rpd": 20,
        "safe_interval": 13.0,
    },

    # Pro models: 0 quota
    "gemini-2.5-pro": {
        "rpm": 0,
        "max_rpm": 0,
        "tpm": 0,
        "rpd": 0,
        "safe_interval": 60.0,
    },
    "gemini-pro-latest": {
        "rpm": 0,
        "max_rpm": 0,
        "tpm": 0,
        "rpd": 0,
        "safe_interval": 60.0,
    },
    "gemini-3.1-pro": {
        "rpm": 0,
        "max_rpm": 0,
        "tpm": 0,
        "rpd": 0,
        "safe_interval": 60.0,
    },
    "gemini-3.1-pro-preview": {
        "rpm": 0,
        "max_rpm": 0,
        "tpm": 0,
        "rpd": 0,
        "safe_interval": 60.0,
    },
}


def get_model_specs(model_name: str) -> Dict[str, Any]:
    """Look up rate limit specifications for a model from user's exact quota table."""
    clean_name = model_name.lower().replace("models/", "").strip()
    if clean_name in GEMINI_MODEL_LIMITS:
        return GEMINI_MODEL_LIMITS[clean_name]

    # Heuristic matching if name varies slightly
    if "flash-lite" in clean_name or "flash_lite" in clean_name:
        return {
            "rpm": 15,
            "max_rpm": 14,
            "tpm": 250_000,
            "rpd": 500,
            "safe_interval": 4.2,
        }
    elif "flash" in clean_name:
        return {
            "rpm": 5,
            "max_rpm": 4,
            "tpm": 250_000,
            "rpd": 20,
            "safe_interval": 13.0,
        }
    elif "pro" in clean_name:
        return {
            "rpm": 0,
            "max_rpm": 0,
            "tpm": 0,
            "rpd": 0,
            "safe_interval": 60.0,
        }

    # Safe generic default (treat like Flash Lite)
    return {
        "rpm": 15,
        "max_rpm": 14,
        "tpm": 250_000,
        "rpd": 500,
        "safe_interval": 4.2,
    }


class GeminiRateLimiter:
    """Thread-safe sliding window rate limiter per model name."""
    _instances: Dict[str, "GeminiRateLimiter"] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.specs = get_model_specs(model_name)
        
        self.max_rpm = self.specs.get("max_rpm", 14)
        self.min_interval = self.specs.get("safe_interval", 4.2)
        self.rpd = self.specs.get("rpd", 500)

        if self.max_rpm == 0:
            logger.warning(
                f"Model '{model_name}' has 0 RPM quota in your Google AI Studio account. "
                "Calls will be blocked or fail. Please use 'gemini-3.5-flash-lite' or 'gemini-3.1-flash-lite'."
            )

        self.timestamps: collections.deque = collections.deque()
        self.last_call_time = 0.0
        self.lock = threading.Lock()
        self.daily_calls = 0

    def downgrade_to_free_tier(self) -> None:
        """Dynamically throttles pacing to 14 RPM if Google API reports a Free-Tier quota limit."""
        with self.lock:
            if self.max_rpm > 14:
                logger.info(
                    f"GeminiRateLimiter: Detected Free Tier quota for {self.model_name}. "
                    "Auto-adjusting to 14 RPM and 4.1s inter-call spacing."
                )
                self.max_rpm = 14
                self.min_interval = 4.1

    @classmethod
    def get_limiter(cls, model_name: str) -> "GeminiRateLimiter":
        clean_name = model_name.lower().replace("models/", "").strip()
        with cls._lock:
            if clean_name not in cls._instances:
                cls._instances[clean_name] = cls(clean_name)
            return cls._instances[clean_name]

    def acquire(self) -> None:
        """Blocks until it is safe to issue a request under model quota constraints."""
        if self.max_rpm == 0:
            raise RuntimeError(
                f"Model '{self.model_name}' has 0 RPM quota in Google AI Studio. "
                "Please configure 'gemini-3.5-flash-lite' or 'gemini-3.1-flash-lite' instead."
            )

        with self.lock:
            now = time.time()

            if self.daily_calls >= self.rpd:
                logger.warning(
                    f"Model '{self.model_name}' has reached its daily limit of {self.rpd} requests "
                    f"(current count: {self.daily_calls}). Google API may return 429 quota exhausted."
                )

            # 1. Enforce minimum inter-request interval
            elapsed_since_last = now - self.last_call_time
            if elapsed_since_last < self.min_interval:
                sleep_needed = self.min_interval - elapsed_since_last
                time.sleep(sleep_needed)
                now = time.time()

            # 2. Enforce sliding window (60-second) RPM cap
            window_start = now - 60.0
            while self.timestamps and self.timestamps[0] < window_start:
                self.timestamps.popleft()

            if len(self.timestamps) >= self.max_rpm:
                oldest = self.timestamps[0]
                sleep_needed = (oldest + 60.0) - now + 0.05
                if sleep_needed > 0:
                    logger.debug(
                        f"Rate limiter pacing: model={self.model_name} at {len(self.timestamps)}/{self.max_rpm} RPM. "
                        f"Sleeping {sleep_needed:.2f}s."
                    )
                    time.sleep(sleep_needed)
                    now = time.time()
                    window_start = now - 60.0
                    while self.timestamps and self.timestamps[0] < window_start:
                        self.timestamps.popleft()

            # Record this call
            self.timestamps.append(now)
            self.last_call_time = now
            self.daily_calls += 1


def extract_gemini_text(response: Any) -> str:
    """
    Robustly extracts plain text from Langchain / Google GenAI responses.
    Handles str, AIMessage, list of dicts, and multi-part content blocks.
    """
    if response is None:
        return ""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    text_parts.append(part["text"])
                else:
                    text_parts.append(str(part))
            elif hasattr(part, "text"):
                text_parts.append(getattr(part, "text"))
            else:
                text_parts.append(str(part))
        return "".join(text_parts)

    return str(content)


def invoke_gemini_with_retry(
    llm: Any,
    prompt: Any,
    model_name: str,
    max_retries: int = 4,
    fallback_models: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Invokes Gemini with automatic rate limiting, backoff on 429/503 errors,
    and automatic model fallback if quota or availability errors persist.
    """
    limiter = GeminiRateLimiter.get_limiter(model_name)
    current_model = model_name
    current_llm = llm

    fallback_queue = list(fallback_models or [])

    for attempt in range(max_retries):
        limiter.acquire()
        try:
            response = current_llm.invoke(prompt)
            text = extract_gemini_text(response)
            return text.strip().strip('"').strip("'")

        except Exception as e:
            err_msg = str(e)
            is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
            is_service_unavailable = "503" in err_msg or "UNAVAILABLE" in err_msg
            is_not_found = "404" in err_msg or "NOT_FOUND" in err_msg

            if is_not_found and fallback_queue:
                next_model = fallback_queue.pop(0)
                logger.warning(f"Model {current_model} not found/deprecated. Switching to fallback {next_model}.")
                current_model = next_model
                limiter = GeminiRateLimiter.get_limiter(current_model)
                from langchain_google_genai import ChatGoogleGenerativeAI
                current_llm = ChatGoogleGenerativeAI(
                    model=current_model,
                    temperature=getattr(llm, "temperature", 0.0),
                    google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                )
                continue

            if is_rate_limit or is_service_unavailable:
                if is_rate_limit and ("free_tier" in err_msg.lower() or "limit: 15" in err_msg.lower()):
                    limiter.downgrade_to_free_tier()

                # Check for explicit retry-after directive in error message
                wait_time = 2.0 * (2 ** attempt)
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                if match:
                    wait_time = max(float(match.group(1)) + 1.0, wait_time)

                logger.warning(
                    f"Gemini {current_model} temporary throttle/spillover (attempt {attempt+1}/{max_retries}). "
                    f"Waiting {wait_time:.1f}s before retry... Error: {err_msg[:120]}"
                )
                time.sleep(wait_time)

                # If last retry failed on this model and we have a fallback, switch
                if attempt == max_retries - 2 and fallback_queue:
                    next_model = fallback_queue.pop(0)
                    logger.info(f"Switching from throttled {current_model} to fallback model {next_model}.")
                    current_model = next_model
                    limiter = GeminiRateLimiter.get_limiter(current_model)
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    current_llm = ChatGoogleGenerativeAI(
                        model=current_model,
                        temperature=getattr(llm, "temperature", 0.0),
                        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                    )
            else:
                logger.error(f"Non-retryable error invoking Gemini {current_model}: {e}")
                raise

    raise RuntimeError(f"Exceeded max retries ({max_retries}) calling Gemini {current_model}.")
