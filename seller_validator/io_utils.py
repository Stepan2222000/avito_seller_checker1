"""Вспомогательные функции для чтения и записи данных."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional


class JsonArrayWriter:
    """Примитив для потоковой записи JSON-массива."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._opened = False
        self._count = 0
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        if self._path.exists():
            content = self._path.read_text(encoding="utf-8").strip()
            if not content:
                self._path.write_text("[]", encoding="utf-8")
                self._opened = True
                self._count = 0
                return
            if content == "[]":
                self._opened = True
                self._count = 0
                return
            # Попытаемся определить число записей для корректного добавления.
            try:
                items = json.loads(content)
                if isinstance(items, list):
                    self._count = len(items)
                else:
                    raise ValueError("results.json должен содержать массив JSON")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSON в файле {self._path}: {exc}") from exc
        else:
            self._path.write_text("[]", encoding="utf-8")
        self._opened = True

    async def append(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        async with self._lock:
            if not self._opened:
                self._ensure_initialized()
            await asyncio.to_thread(self._append_sync, data)
            self._count += 1

    def _append_sync(self, data: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("r+b") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"[]")
            fh.seek(-1, os.SEEK_END)
            tail = fh.read(1)
            if tail != b"]":
                raise ValueError(f"Файл {self._path} должен заканчиваться ']'")
            fh.seek(-1, os.SEEK_END)
            if self._count == 0:
                fh.write(f"{data}]".encode("utf-8"))
            else:
                fh.write(f",\n{data}]".encode("utf-8"))

    async def close(self) -> None:
        """Закрыть writer (ничего не делает, оставлено для совместимости)."""
        async with self._lock:
            self._opened = True


def read_processed_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def append_processed_url(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{url}\n")


def iter_urls(path: Path) -> Iterator[str]:
    if not path.exists():
        return iter(())
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            url = line.strip()
            if url:
                yield url


def append_valid_url(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{url}\n")


def filter_results_by_reason_codes(
    path: Path, codes: tuple[str, ...] = ("429", "503")
) -> tuple[int, int]:
    """Удалить из results.json записи, где reason содержит любой из кодов.

    Возвращает кортеж (сколько_удалено, сколько_осталось).
    Формат сохраняется: JSON-массив, по одной записи на строку.
    """
    if not path.exists():
        return 0, 0
    content = path.read_text(encoding="utf-8").strip() or "[]"
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("results.json должен содержать массив JSON")

    def is_bad(rec: dict) -> bool:
        if not isinstance(rec, dict):
            return False
        reason = rec.get("reason")
        if not isinstance(reason, str):
            return False
        return any(code in reason for code in codes)

    filtered = [r for r in data if not is_bad(r)]

    # Перезаписываем в компактном формате: один объект на строку
    if not filtered:
        path.write_text("[]", encoding="utf-8")
    else:
        lines = [
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in filtered
        ]
        text = "[\n" + ",\n".join(lines) + "\n]"
        path.write_text(text, encoding="utf-8")

    return (len(data) - len(filtered), len(filtered))


def read_results_urls(path: Path) -> set[str]:
    """Вернуть множество URL, присутствующих в results.json.

    Если файла нет — вернуть пустое множество.
    """
    if not path.exists():
        return set()
    content = path.read_text(encoding="utf-8").strip() or "[]"
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("results.json должен содержать массив JSON")
    results: set[str] = set()
    for rec in data:
        if isinstance(rec, dict):
            url = rec.get("seller_url")
            if isinstance(url, str) and url:
                results.add(url)
    return results
