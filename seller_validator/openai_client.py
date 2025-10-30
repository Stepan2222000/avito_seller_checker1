"""Обёртка над OpenAI SDK для вызова Llama через DeepInfra."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, Sequence

from openai import (
    APITimeoutError,
    APIError,
    APIStatusError,
    AsyncOpenAI,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)
from pydantic import ValidationError

from .config import Config
from .gemini_client import SellerVerdictSchema
from .prompt_builder import build_prompt_sections

logger = logging.getLogger("seller_validator.gemini")


class OpenAIAnalyzer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=self.config.deepinfra_api_key,
            base_url=self.config.deepinfra_base_url,
            timeout=self.config.llm_request_timeout,
        )
        self._semaphore = asyncio.Semaphore(self.config.llm_concurrency)

    async def close(self) -> None:
        await self._client.close()

    async def analyze(
        self,
        *,
        seller_url: str,
        seller_name: str | None,
        items: Sequence[Dict[str, Any]],
        attempt: int,
    ) -> Dict[str, Any]:
        prompt_sections = build_prompt_sections(
            seller_url=seller_url, seller_name=seller_name, items=items
        )
        system_prompt = prompt_sections[0]
        user_prompt = "\n\n".join(prompt_sections[1:])
        attempt_no = 0
        last_error: Exception | None = None
        while attempt_no < self.config.llm_max_retries:
            attempt_no += 1
            try:
                logger.info(
                    "Запрос к Llama (DeepInfra): url=%s, объявлений=%s, попытка=%s",
                    seller_url,
                    len(items),
                    attempt_no,
                )
                async with self._semaphore:
                    request_kwargs: dict[str, object] = {
                        "model": self.config.deepinfra_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": self.config.deepinfra_temperature,
                    }
                    # DeepSeek Terminus поддерживает режим «thinking»,
                    # который включается флагом reasoning_enabled.
                    if self.config.deepinfra_reasoning_enabled is not None:
                        request_kwargs["reasoning_enabled"] = self.config.deepinfra_reasoning_enabled
                    reasoning_effort = (self.config.deepinfra_reasoning_effort or "").strip()
                    if reasoning_effort:
                        request_kwargs["reasoning_effort"] = reasoning_effort

                    response = await self._client.chat.completions.create(**request_kwargs)
                parsed = self._parse_response(response)
                logger.info(
                    "Ответ Llama (DeepInfra): url=%s, попытка=%s, verdict=%s, confidence=%.2f",
                    seller_url,
                    attempt_no,
                    parsed.get("verdict"),
                    float(parsed.get("confidence", 0.0)),
                )
                parsed["llm_attempts"] = attempt_no
                parsed["gemini_attempts"] = attempt_no
                return parsed
            except (APITimeoutError, APIError, APIStatusError, RateLimitError) as exc:
                last_error = exc
                message = str(exc).lower()
                status_code = getattr(exc, "status_code", None)
                logger.warning(
                    "APIError Llama (DeepInfra): url=%s, попытка=%s, ошибка=%s",
                    seller_url,
                    attempt_no,
                    exc,
                )
                await asyncio.sleep(self._backoff(attempt_no))
                continue
            except (BadRequestError, OpenAIError) as exc:
                # Клиентские ошибки (например, schema violation / validation)
                logger.error(
                    "ClientError Llama (DeepInfra): url=%s, попытка=%s, ошибка=%s",
                    seller_url,
                    attempt_no,
                    exc,
                )
                return {
                    "verdict": "error",
                    "reason": f"Client error from Llama (DeepInfra): {exc}",
                    "confidence": 0.0,
                    "flags": ["llm_client_error"],
                    "items": [],
                    "llm_attempts": attempt_no,
                    "gemini_attempts": attempt_no,
                }
            except ValidationError as exc:
                logger.error(
                    "ValidationError Llama (DeepInfra): url=%s, попытка=%s, ошибка=%s",
                    seller_url,
                    attempt_no,
                    exc,
                )
                return {
                    "verdict": "error",
                    "reason": f"Response validation error: {exc}",
                    "confidence": 0.0,
                    "flags": ["llm_schema_error"],
                    "items": [],
                    "llm_attempts": attempt_no,
                    "gemini_attempts": attempt_no,
                }

        return {
            "verdict": "error",
            "reason": (
                "DeepInfra APIError after "
                f"{self.config.llm_max_retries} retries: {last_error}"
            ),
            "confidence": 0.0,
            "flags": ["llm_api_error"],
            "items": [],
            "llm_attempts": self.config.llm_max_retries,
            "gemini_attempts": self.config.llm_max_retries,
        }

    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        try:
            message = response.choices[0].message
            if isinstance(message.content, str):
                content = message.content
            else:
                parts = []
                for part in message.content:
                    text = getattr(part, "text", None)
                    if text is not None:
                        parts.append(text)
                content = "".join(parts)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:  # noqa: TRY003
            raise ValueError("OpenAI response не содержит пригодных данных") from exc
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        model = SellerVerdictSchema.model_validate_json(content)
        return model.model_dump()

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(10.0, 1.5 * math.pow(2, attempt - 1))
