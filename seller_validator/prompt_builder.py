"""Утилиты для формирования промптов LLM."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence


BRAND_WHITELIST = [
    "Acura",
    "Alfa Romeo",
    "Aston Martin",
    "Audi",
    "Bentley",
    "BMW",
    "Bugatti",
    "Cadillac",
    "DS Automobiles",
    "Ferrari",
    "Genesis",
    "Hongqi",
    "Infiniti",
    "Jaguar",
    "Lamborghini",
    "Land Rover",
    "Lexus",
    "Lincoln",
    "Lotus",
    "Maserati",
    "Maybach",
    "McLaren",
    "Mercedes-Benz",
    "Nio",
    "Polestar",
    "Porsche",
    "Rolls-Royce",
    "Tesla",
    "Volvo",
    "Zeekr",
    # Дополнительно допускаем верхний мейнстрим
    "Toyota",
    "Volkswagen",
    "VW",
    "Seat",
    "Kia",
]


def build_prompt_sections(
    *,
    seller_url: str,
    seller_name: str | None,
    items: Sequence[Dict[str, Any]],
) -> list[str]:
    """Собирает одинаковые текстовые сегменты промпта для всех LLM."""
    intro = (
        "Ты — агент по валидации автозапчастей. "
        "Определи, продаёт ли продавец новые оригинальные запчасти премиальных брендов."
        "\n\n"
        "Позитивные признаки:\n"
        "- явно указанные премиальные бренды (список ниже) и верхний ценовой сегмент типа Toyota / Volkswagen / Kia,\n"
        "- акцент на оригинальные/сертифицированные детали,\n"
        "- состояние «новые».\n\n"
        "Негативные признаки:\n"
        "- упоминания б/у, восстановленных, контрактных деталей,\n"
        "- явные аналоги/aftermarket, безымянные или дешёвые бренды (например, ВАЗ, китайские noname),\n"
        "- массовая продажа дешёвых брендов."
        "также учитывай"
        "Может быть такое, что большиество объявлений в профиле продавца валидные, а меньшая часть невалидна, в таком случае можно пропустить продавца (но если количество оригинальных-новых явно перевосходит количество неоригинальных или бу)"
    )
    whitelist = (
        "Список допустимых брендов: премиальные марки и верхний мейнстрим (Toyota, Volkswagen, Kia, Seat). "
        "Низкосегментные/неизвестные бренды (ВАЗ, Lada, Chery, noname) считай негативным признаком.\n- "
        + "\n- ".join(BRAND_WHITELIST)
    )
    items_json = json.dumps(items, ensure_ascii=False, indent=2)
    seller_info = f"Профиль продавца: {seller_name or 'неизвестно'}\nURL: {seller_url}"
    instructions = (
        "Верни только JSON в виде, описанном в schema. "
        "В массиве items для каждого объявления укажи только item_id и решение decision без пояснений. "
        "Если данных явно недостаточно, выставь verdict=error и флаг `insufficient_data`. "
        "Структура JSON: {\n"
        '  "verdict": "passed|failed|uncertain|error",\n'
        '  "reason": "краткое объяснение",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "flags": ["optional_flag1", "optional_flag2"],\n'
        '  "items": [{"item_id": "...", "decision": "passed|failed|uncertain"}]\n'
        "}."
    )
    return [
        intro,
        whitelist,
        seller_info,
        "Объявления продавца:\n" + items_json,
        instructions,
    ]
