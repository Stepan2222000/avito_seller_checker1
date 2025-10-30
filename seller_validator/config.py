"""Конфигурация агента проверки продавцов."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LLMProvider(str, Enum):
    GENAI = "genai"
    OPENAI = "openai"


@dataclass(slots=True)
class Config:
    """Базовые параметры пайплайна."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)

    urls_file: Path = field(init=False)
    proxies_file: Path = field(init=False)
    blocked_proxies_file: Path = field(init=False)
    results_json: Path = field(init=False)
    processed_urls_txt: Path = field(init=False)
    passed_urls_txt: Path = field(init=False)

    worker_count: int = 4
    queue_max_attempts: int = 4
    queue_idle_sleep: float = 0.5

    playwright_headless: bool = False
    playwright_slow_mo: float | None = None
    navigation_timeout_ms: int = 30_000

    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_concurrency: int = 30
    llm_request_timeout: float = 60.0
    llm_max_retries: int = 3

    gemini_api_key: str = "AIzaSyBYPiXtSWY3fSQjXhPnbSP0iCzBHngpetg"
    gemini_model: str = "gemini-2.5-flash"

    deepinfra_api_key: str = "ZyqsQeDmt4bFp19aNn6kyK9mGPDgId2H"
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/openai"
    deepinfra_model: str = "deepseek-ai/DeepSeek-V3.1-Terminus"
    deepinfra_reasoning_effort: str = "high"
    deepinfra_reasoning_enabled: bool = True
    deepinfra_temperature: float = 0.0

    max_items_per_seller: int = 100
    seller_schema: dict[str, object] = field(
        default_factory=lambda: {
            "id": "id",
            "title": "title",
            "description": "description",
            "priceDetailed": {
                "value": "priceDetailed.value",
                "currency": "priceDetailed.currency",
            },
            "category": {"name": "category.name"},
            "attributes": "attributes",
            "iva": {
                "AutoPartsManufacturerStep": "iva.AutoPartsManufacturerStep",
                "SparePartsParamsStep": "iva.SparePartsParamsStep",
            },
        }
    )

    worker_relaunch_delay: float = 3.0
    checkpoint_flush_interval: int = 1

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.urls_file = self.data_dir / "urls.txt"
        self.proxies_file = self.data_dir / "proxies.txt"
        self.blocked_proxies_file = self.data_dir / "blocked_proxies.txt"
        self.results_json = self.data_dir / "results.json"
        self.processed_urls_txt = self.data_dir / "processed_urls.txt"
        self.passed_urls_txt = self.data_dir / "passed_urls.txt"

        self.data_dir.mkdir(parents=True, exist_ok=True)
