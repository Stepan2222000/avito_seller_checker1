"""Обёртка над Gemini для анализа продавцов."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any, Dict, Sequence

from google import genai
from google.genai import errors, types
from google.oauth2 import service_account
from pydantic import BaseModel, Field, ValidationError

from .config import Config
from .prompt_builder import build_prompt_sections

logger = logging.getLogger("seller_validator.gemini")


class ItemDecisionSchema(BaseModel):
    item_id: str = Field(..., description="ID объявления Avito")
    decision: str = Field(..., description="passed|failed|uncertain")


class SellerVerdictSchema(BaseModel):
    verdict: str = Field(..., description="passed|failed|error")
    reason: str = Field(..., description="Сводное объяснение решения")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели")
    flags: list[str] = Field(default_factory=list, description="Произвольные флаги")
    items: list[ItemDecisionSchema] = Field(default_factory=list)


class GeminiAnalyzer:
    def __init__(self, config: Config) -> None:
        self.config = config

        # Load credentials from service account key file with required scopes
        credentials = service_account.Credentials.from_service_account_file(
            str(self.config.vertex_service_account_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # Initialize Vertex AI client
        self._client = genai.Client(
            vertexai=True,
            project=self.config.vertex_project_id,
            location=self.config.vertex_location,
            credentials=credentials,
        )
        self._async_client = self._client.aio
        self._semaphore = asyncio.Semaphore(self.config.llm_concurrency)

    async def close(self) -> None:
        await self._async_client.aclose()

    async def analyze(
        self,
        *,
        seller_url: str,
        seller_name: str | None,
        items: Sequence[Dict[str, Any]],
        attempt: int,
    ) -> Dict[str, Any]:
        payload = self._build_prompt(seller_url=seller_url, seller_name=seller_name, items=items)
        attempt_no = 0
        last_error: Exception | None = None
        while attempt_no < self.config.llm_max_retries:
            attempt_no += 1
            try:
                logger.info(
                    "Запрос к Gemini: url=%s, объявлений=%s, попытка=%s",
                    seller_url,
                    len(items),
                    attempt_no,
                )
                async with self._semaphore:
                    response = await self._async_client.models.generate_content(
                        model=self.config.vertex_model,
                        contents=payload,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=SellerVerdictSchema,
                            temperature=0.0,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                disable=True
                            ),
                        ),
                    )
                parsed = self._parse_response(response)
                logger.info(
                    "Ответ Gemini: url=%s, попытка=%s, verdict=%s, confidence=%.2f",
                    seller_url,
                    attempt_no,
                    parsed.get("verdict"),
                    float(parsed.get("confidence", 0.0)),
                )
                parsed["llm_attempts"] = attempt_no
                parsed["gemini_attempts"] = attempt_no
                return parsed
            except errors.APIError as exc:
                last_error = exc
                logger.warning(
                    "APIError Gemini: url=%s, попытка=%s, ошибка=%s",
                    seller_url,
                    attempt_no,
                    exc,
                )
                await asyncio.sleep(self._backoff(attempt_no))
                continue
            except errors.ClientError as exc:
                logger.error(
                    "ClientError Gemini: url=%s, попытка=%s, ошибка=%s",
                    seller_url,
                    attempt_no,
                    exc,
                )
                return {
                    "verdict": "error",
                    "reason": f"Client error from Gemini: {exc}",
                    "confidence": 0.0,
                    "flags": ["gemini_client_error"],
                    "items": [],
                    "llm_attempts": attempt_no,
                    "gemini_attempts": attempt_no,
                }
            except ValidationError as exc:
                logger.error(
                    "ValidationError Gemini: url=%s, попытка=%s, ошибка=%s",
                    seller_url,
                    attempt_no,
                    exc,
                )
                return {
                    "verdict": "error",
                    "reason": f"Response validation error: {exc}",
                    "confidence": 0.0,
                    "flags": ["gemini_schema_error"],
                    "items": [],
                    "llm_attempts": attempt_no,
                    "gemini_attempts": attempt_no,
                }

        return {
            "verdict": "error",
            "reason": f"Gemini APIError after {self.config.llm_max_retries} retries: {last_error}",
            "confidence": 0.0,
            "flags": ["gemini_api_error"],
            "items": [],
            "llm_attempts": self.config.llm_max_retries,
            "gemini_attempts": self.config.llm_max_retries,
        }

    def _build_prompt(
        self, *, seller_url: str, seller_name: str | None, items: Sequence[Dict[str, Any]]
    ) -> list[types.Content]:
        sections = build_prompt_sections(
            seller_url=seller_url, seller_name=seller_name, items=items
        )
        return [
            types.Content(
                role="user",
                parts=[
                    *[types.Part.from_text(text=section) for section in sections],
                ],
            )
        ]

    @staticmethod
    def _parse_response(response: types.GenerateContentResponse | Any) -> Dict[str, Any]:
        if hasattr(response, "parsed") and response.parsed is not None:
            data = response.parsed
            if isinstance(data, SellerVerdictSchema):
                return json.loads(data.model_dump_json())
            if isinstance(data, BaseModel):
                return json.loads(data.model_dump_json())
            if isinstance(data, dict):
                return data
        if hasattr(response, "text"):
            raw = response.text
            return json.loads(raw)
        raise ValueError("Gemini response не содержит пригодных данных")

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(10.0, 1.5 * math.pow(2, attempt - 1))
