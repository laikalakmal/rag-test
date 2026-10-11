import os
import time
from typing import Optional

import requests
from langchain_core.language_models.llms import LLM
import logging

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


class OllamaError(Exception):
    """Base exception for Ollama errors."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the specified model is not found on the Ollama server (HTTP 404)."""
    pass


class OllamaInfrastructureError(OllamaError):
    """Raised when Ollama is unreachable, crashes (HTTP 5xx), or times out after retries."""
    pass


def verify_ollama_model(model_name: str, api_url: Optional[str] = None) -> tuple[bool, str]:
    """
    Pre-flight check: Verifies that Ollama is running and the specified model is installed.
    Returns (True, "OK") or (False, error_message).
    """
    if not api_url:
        api_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate").strip()

    base_url = api_url.rsplit("/api/", 1)[0]
    tags_url = f"{base_url}/api/tags"

    try:
        resp = requests.get(tags_url, timeout=5)
        resp.raise_for_status()
        tags_data = resp.json()
        models = [m.get("name") for m in tags_data.get("models", [])]

        target = model_name.strip()
        target_with_tag = target if ":" in target else f"{target}:latest"

        matched = False
        for m in models:
            if m == target or m == target_with_tag or m.split(":")[0] == target.split(":")[0]:
                matched = True
                break

        if not matched:
            avail_str = ", ".join(models) if models else "None"
            return False, (
                f"Model '{model_name}' is not installed in Ollama. "
                f"Available models: [{avail_str}]. "
                f"Please run 'ollama pull {model_name}' in your terminal before running this evaluation."
            )

        return True, "OK"

    except requests.exceptions.ConnectionError:
        return False, (
            f"Cannot connect to Ollama server at {base_url}. "
            "Is Ollama running? Please start Ollama ('ollama serve') before running this evaluation."
        )
    except Exception as e:
        return False, f"Failed to check Ollama models at {tags_url}: {e}"


class OllamaLLM(LLM):
    model: str = "llama3"
    api_url: str = "http://localhost:11434/api/generate"
    timeout: int = 120
    temperature: float = 0.2

    def _post_with_retry(self, payload: dict) -> dict:
        """Internal helper to POST to Ollama with retry and exponential backoff.
        Retries up to 3 times (5s, 15s, 30s) on connection, timeout, or HTTP 5xx errors.
        Fails fast immediately on HTTP 404 Model Not Found.
        Never uses fallback models or alternative providers.
        """
        max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        retry_delays = [5.0, 15.0, 30.0]
        attempt = 0

        while True:
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 404:
                    err_text = ""
                    try:
                        err_text = response.json().get("error", "")
                    except Exception:
                        pass
                    raise OllamaModelNotFoundError(
                        f"Model '{self.model}' not found in Ollama (HTTP 404). {err_text}. "
                        f"Run 'ollama pull {self.model}' to download it."
                    )
                response.raise_for_status()
                return response.json()

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                attempt += 1
                if attempt > max_retries:
                    logger.critical(f"Ollama connection/timeout error after {max_retries} retries: {e}. Halting.")
                    raise OllamaInfrastructureError(
                        f"Ollama server connection/timeout failed after {max_retries} attempts ({e}). Halting evaluation."
                    )
                delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                logger.warning(
                    f"Ollama request error ({type(e).__name__}: {e}); waiting {delay:.1f}s before retry {attempt}/{max_retries}..."
                )
                time.sleep(delay)

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else 500
                if 500 <= status < 600:
                    attempt += 1
                    if attempt > max_retries:
                        logger.critical(f"Ollama server error {status} after {max_retries} retries: {e}. Halting.")
                        raise OllamaInfrastructureError(
                            f"Ollama server returned HTTP {status} after {max_retries} attempts. Halting evaluation."
                        )
                    delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                    logger.warning(
                        f"Ollama server error {status}; waiting {delay:.1f}s before retry {attempt}/{max_retries}..."
                    )
                    time.sleep(delay)
                else:
                    raise OllamaInfrastructureError(f"Ollama returned HTTP error {status}: {e}")

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                 "num_thread": 8
            }
        }
        data = self._post_with_retry(payload)
        # Optional cooldown after a successful request
        cooldown = float(os.getenv("OLLAMA_COOLDOWN_SECONDS", "0.5"))
        if cooldown > 0:
            time.sleep(cooldown)
        return data.get("response", "").strip()

    @property
    def _llm_type(self) -> str:
        return "ollama"


def build_ollama_llm(config=None) -> OllamaLLM:
    load_dotenv()
    temperature = config.agent.temperature if config else float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

    return OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama3").strip(),
        api_url=os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate").strip(),
        timeout=int(os.getenv("OLLAMA_TIMEOUT", "500")),
        temperature=temperature
    )


def get_llm(provider: Optional[str] = None, config=None) -> LLM:
    load_dotenv()
    selected = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
    if selected in {"ollama", "local_ollama"}:
        return build_ollama_llm(config)
    elif selected == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # We use ChatGoogleGenerativeAI since it supports the newer Gemini models
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            temperature=config.agent.temperature if config else float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{selected}'. Use 'ollama' or 'gemini'."
    )