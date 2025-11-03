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

    # Keyword filter settings
    keyword_filter_enabled: bool = True
    keyword_filter_threshold: int = 4  # Минимум вхождений для блокировки

    prohibited_keywords: list[str] = field(
        default_factory=lambda: [
            # Состояние
            "б/у", "бу", "с пробегом", "восстановленный", "контрактный",
            "был в использовании", "после эксплуатации", "бывший в употреблении",
            "бывшие в использовании", "бывшая в эксплуатации",

            # Не оригинал
            "аналог", "не оригинал", "неоригинал", "aftermarket",
            "копия", "реплика", "аналоговый", "аналогичный",

            # Разборка
            "разборка", "демонтаж", "снят с авто", "авторазбор",
            "с разборки", "разбор", "авто разбор",

            # Происхождение
            "китай", "китайский", "корея", "турция", "китайская",
        ]
    )

    prohibited_brands: list[str] = field(
        default_factory=lambda: [
            "ZF", "Lemförder", "Sachs", "TRW", "LuK", "INA", "FAG",
            "Monroe", "MOOG", "Ferodo", "Walker", "Champion", "Fel-Pro",
            "Goetze", "Glyco", "Payen", "Nural", "AE", "Sealed Power",
            "febi", "SWAG", "Blue Print", "VDO", "ATE", "Continental",
            "ContiTech", "MANN-FILTER", "WIX Filters", "FILTRON",
            "Purflux", "FRAM", "UFI Filters", "SOFIMA", "MAHLE",
            "Knecht", "Behr", "Hengst", "SKF", "NTN", "NSK", "GMB",
            "GSP", "CTR", "Sankei 555", "Sidem", "Teknorot",
            "Original Birth", "BIRTH", "Metalcaucho", "Herth+Buss Jakoparts",
            "TOPRAN", "Hans Pries", "MEYLE", "URO Parts", "KYB",
            "Bilstein", "Koni", "Brembo", "Textar", "Pagid", "Mintex",
            "Bendix", "Akebono", "EBC Brakes", "StopTech", "Centric",
            "PowerStop", "Remsa", "ICER", "LPR", "A.B.S.",
            "All Brake Systems", "Hella", "Magneti Marelli", "TYC",
            "DEPO", "Van Wezel", "Klokkerholm", "Prasco", "Nissens",
            "NRF", "Pierburg", "Carter", "Victor Reinz", "Elring",
            "Ajusa", "Corteco", "NPR", "Nippon Piston Ring", "Hastings",
            "Kolbenschmidt", "Dorman", "GKN", "VEMO", "VAICO", "Gates",
            "Hutchinson", "Four Seasons", "JP Group", "Metzger",
            "Comline", "Mapco", "Dayco", "Bando", "Optibelt", "Koyo",
            "SNR", "Timken", "Nisshinbo", "Zimmermann", "Raybestos",
            "Gabriel", "Delphi Technologies", "Beck/Arnley",
            "Standard Motor Products", "SMP", "Hepu", "Saleri", "Graf",
            "Dolz", "Airtex", "GKN Spidan", "King Engine Bearings",
            "Exedy", "AISIN", "Cloyes", "Cardone", "Baldwin Filters",
            "CoopersFiaam",
        ]
    )

    worker_relaunch_delay: float = 3.0
    checkpoint_flush_interval: int = 1

    # Database settings
    use_database: bool = True
    db_host: str = "81.30.105.134"
    db_port: int = 5414
    db_name: str = "avito_seller_checker"
    db_user: str = "admin"
    db_password: str = "Password123"
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.urls_file = self.data_dir / "urls.txt"
        self.proxies_file = self.data_dir / "proxies.txt"
        self.blocked_proxies_file = self.data_dir / "blocked_proxies.txt"
        self.results_json = self.data_dir / "results.json"
        self.processed_urls_txt = self.data_dir / "processed_urls.txt"
        self.passed_urls_txt = self.data_dir / "passed_urls.txt"

        self.data_dir.mkdir(parents=True, exist_ok=True)
