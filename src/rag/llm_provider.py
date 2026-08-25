import os
import time
from typing import Optional

import requests
from langchain_core.language_models.llms import LLM


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


class OllamaLLM(LLM):
    model: str = "llama3"
    api_url: str = "http://localhost:11434/api/generate"
    timeout: int = 120
    temperature: float = 0.2

    def _post_with_retry(self, payload: dict) -> dict:
        """Internal helper to POST to Ollama with retry and exponential backoff.
        Reads configuration from environment variables:
          OLLAMA_MAX_RETRIES (default 3)
          OLLAMA_BACKOFF_FACTOR (default 2.0)
          OLLAMA_INITIAL_DELAY (default 0.5 seconds)
        """
        max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        backoff_factor = float(os.getenv("OLLAMA_BACKOFF_FACTOR", "2.0"))
        initial_delay = float(os.getenv("OLLAMA_INITIAL_DELAY", "0.5"))
        attempt = 0
        while True:
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                attempt += 1
                if attempt > max_retries:
                    logger.error(f"Ollama request failed after {max_retries} retries: {e}")
                    raise
                delay = initial_delay * (backoff_factor ** (attempt - 1))
                logger.warning(f"Ollama request error ({e}); retry {attempt}/{max_retries} after {delay:.2f}s")
                time.sleep(delay)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if 500 <= status < 600:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(f"Ollama server error after {max_retries} retries: {e}")
                        raise
                    delay = initial_delay * (backoff_factor ** (attempt - 1))
                    logger.warning(f"Ollama server error {status}; retry {attempt}/{max_retries} after {delay:.2f}s")
                    time.sleep(delay)
                else:
                    raise

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
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