"""Модуль для фильтрации продавцов по запретным словам и брендам."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .config import Config


def _extract_text_from_item(item: Dict[str, Any]) -> str:
    """
    Извлекает весь текст из объявления для поиска.

    Args:
        item: Словарь с данными объявления

    Returns:
        Объединённый текст всех полей
    """
    texts = []

    # Основные текстовые поля
    for field in ["title", "description", "category", "condition"]:
        value = item.get(field)
        if isinstance(value, str) and value:
            texts.append(value)

    # Бренды
    brands = item.get("brand_candidates")
    if isinstance(brands, list):
        for brand in brands:
            if isinstance(brand, str) and brand:
                texts.append(brand)

    return " ".join(texts)


def count_prohibited_items(
    items: Sequence[Dict[str, Any]],
    seller_name: str | None,
    prohibited_keywords: List[str],
    prohibited_brands: List[str],
) -> Tuple[int, List[str]]:
    """
    Подсчитывает общее количество вхождений запретных слов/фраз и брендов.

    Особенности:
    - Case-insensitive поиск
    - Word boundaries для одиночных слов (чтобы "бу" не находил "бухгалтер")
    - Фразы ищутся как целые подстроки
    - Считаются все вхождения, не уникальные слова

    Args:
        items: Список объявлений продавца
        seller_name: Имя продавца (опционально)
        prohibited_keywords: Список запретных слов/фраз
        prohibited_brands: Список запретных брендов

    Returns:
        Кортеж: (количество_вхождений, список_найденных_слов)
    """
    # Собираем весь текст
    all_text = ""

    if seller_name:
        all_text += f"{seller_name} "

    for item in items:
        all_text += f"{_extract_text_from_item(item)} "

    all_text = all_text.lower()

    found_items: List[str] = []
    total_count = 0

    # Поиск запретных ключевых слов
    for keyword in prohibited_keywords:
        keyword_lower = keyword.lower()

        # Если ключевое слово содержит пробелы - это фраза, ищем как подстроку
        if " " in keyword_lower:
            # Фразы ищем как подстроки (без word boundaries)
            count = all_text.count(keyword_lower)
        else:
            # Одиночные слова ищем с word boundaries
            pattern = rf"\b{re.escape(keyword_lower)}\b"
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            count = len(matches)

        if count > 0:
            total_count += count
            # Добавляем каждое вхождение в список
            found_items.extend([keyword] * count)

    # Поиск запретных брендов (всегда с word boundaries)
    for brand in prohibited_brands:
        brand_lower = brand.lower()
        pattern = rf"\b{re.escape(brand_lower)}\b"
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        count = len(matches)

        if count > 0:
            total_count += count
            found_items.extend([brand] * count)

    return total_count, found_items


def check_keyword_filter(
    items: Sequence[Dict[str, Any]],
    seller_name: str | None,
    config: Config,
) -> Tuple[bool, int, List[str]]:
    """
    Проверяет, должен ли продавец быть заблокирован по keyword-фильтру.

    Args:
        items: Список объявлений продавца
        seller_name: Имя продавца
        config: Конфигурация с настройками фильтра

    Returns:
        Кортеж: (заблокирован, количество_вхождений, найденные_слова)
    """
    if not config.keyword_filter_enabled:
        return False, 0, []

    count, found = count_prohibited_items(
        items=items,
        seller_name=seller_name,
        prohibited_keywords=config.prohibited_keywords,
        prohibited_brands=config.prohibited_brands,
    )

    is_blocked = count >= config.keyword_filter_threshold

    return is_blocked, count, found
