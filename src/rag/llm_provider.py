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


class OpenRouterLLM(LLM):
    api_key: str
    model: str = "qwen/qwen3-next-80b-a3b-instruct:free"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    timeout: int = 60

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(text_parts).strip()
        return str(content).strip()

    @property
    def _llm_type(self) -> str:
        return "openrouter"


class HuggingFaceInferenceLLM(LLM):
    api_key: str = ""
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    api_url: str
    timeout: int = 60
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False
    wait_for_model: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 10.0

    def _parse_response_text(self, data) -> str:
        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                    return "".join(parts).strip()
                return str(content).strip()
            if "error" in data:
                raise RuntimeError(f"Hugging Face inference error: {data['error']}")
        raise RuntimeError(f"Unexpected Hugging Face response payload: {data}")

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }

        if not self.do_sample:
            payload["temperature"] = 0

        payload["extra_body"] = {
            "options": {
                "wait_for_model": self.wait_for_model,
            }
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                    continue
                raise RuntimeError(f"Hugging Face request failed: {exc}") from exc

            if response.status_code in {429, 503} and attempt < self.max_retries:
                time.sleep(self.retry_delay_seconds)
                continue

            if response.status_code >= 400:
                raise RuntimeError(f"Hugging Face error {response.status_code}: {response.text}")

            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(f"Hugging Face returned non-JSON response: {response.text}") from exc

            return self._parse_response_text(data)

        raise RuntimeError("Hugging Face request failed after retries")

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

class GeminiLLM(LLM):
    api_key: str
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: Optional[int] = None
    max_output_tokens: int = 1024
    _client = None

    class Config:
        arbitrary_types_allowed = True

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai is required for GeminiLLM. Install it with `pip install google-genai`."
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_generation_config(self) -> dict:
        config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.top_k is not None:
            config["top_k"] = self.top_k
        return config

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if text:
            return str(text).strip()

        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        piece = getattr(part, "text", "")
                        if piece:
                            parts.append(piece)
            if parts:
                return "".join(parts).strip()

        raise RuntimeError(f"Gemini returned empty response: {response}")

    @property
    def _llm_type(self) -> str:
        return "gemini"

def build_openrouter_llm() -> OpenRouterLLM:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY. Add it to .env or export it in your shell.")

    return OpenRouterLLM(
        api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-5.2"),
        site_url=os.getenv("OPENROUTER_SITE_URL"),
        site_name=os.getenv("OPENROUTER_SITE_NAME"),
        timeout=int(os.getenv("OPENROUTER_TIMEOUT", "60")),
    )


def build_huggingface_llm() -> HuggingFaceInferenceLLM:
    load_dotenv()
    api_key = os.getenv("HF_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", "")).strip()

    model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2").strip()
    api_url = os.getenv(
        "HF_API_URL",
        "https://router.huggingface.co/v1/chat/completions",
    ).strip()

    return HuggingFaceInferenceLLM(
        api_key=api_key,
        model=model,
        api_url=api_url,
        timeout=int(os.getenv("HF_TIMEOUT", "60")),
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "256")),
        temperature=float(os.getenv("HF_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("HF_TOP_P", "0.9")),
        do_sample=_env_bool("HF_DO_SAMPLE", False),
        wait_for_model=_env_bool("HF_WAIT_FOR_MODEL", True),
        max_retries=int(os.getenv("HF_MAX_RETRIES", "3")),
        retry_delay_seconds=float(os.getenv("HF_RETRY_DELAY_SECONDS", "10")),
    )


def build_gemini_llm() -> GeminiLLM:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Add it to .env or export it in your shell.")

    top_k_raw = os.getenv("GEMINI_TOP_K")
    top_k = int(top_k_raw) if top_k_raw not in {None, ""} else None

    return GeminiLLM(
        api_key=api_key,
        model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip(),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("GEMINI_TOP_P", "0.9")),
        top_k=top_k,
        max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")),
    )



def get_llm(provider: Optional[str] = None) -> LLM:
    load_dotenv()
    selected = (provider or os.getenv("LLM_PROVIDER", "openrouter")).strip().lower()
    if selected == "openrouter":
        return build_openrouter_llm()
    if selected in {"hf", "huggingface", "huggingface_inference"}:
        return build_huggingface_llm()
    if selected in {"gemini", "google", "google_gemini"}:
        return build_gemini_llm()
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{selected}'. Use 'openrouter', 'huggingface', 'gemini', or 'local_t5'."
    )