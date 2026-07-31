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

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        response = requests.post(
            self.api_url,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

        data = response.json()
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