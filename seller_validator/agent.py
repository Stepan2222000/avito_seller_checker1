"""Основной агента для проверки продавцов Авито."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Sequence

import avito_library
from avito_library import collect_seller_items, detect_page_state, resolve_captcha_flow
from avito_library.detectors.captcha_geetest_detector import DETECTOR_ID as CAPTCHA_DETECTOR_ID
from avito_library.detectors.continue_button_detector import DETECTOR_ID as CONTINUE_BUTTON_DETECTOR_ID
from avito_library.detectors.proxy_auth_407_detector import DETECTOR_ID as PROXY_AUTH_407_DETECTOR_ID
from avito_library.detectors.proxy_block_403_detector import DETECTOR_ID as PROXY_BLOCK_403_DETECTOR_ID
from avito_library.detectors.proxy_block_429_detector import DETECTOR_ID as PROXY_BLOCK_429_DETECTOR_ID
from avito_library.detectors.seller_profile_detector import DETECTOR_ID as SELLER_PROFILE_DETECTOR_ID
from avito_library.reuse_utils.proxy_pool import ProxyEndpoint, ProxyPool
from avito_library.reuse_utils.task_queue import ProcessingTask, TaskQueue
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import (
    Page,
    Response,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from .config import Config, LLMProvider
from .gemini_client import GeminiAnalyzer
from .openai_client import OpenAIAnalyzer
from .io_utils import (
    JsonArrayWriter,
    append_processed_url,
    append_valid_url,
    iter_urls,
    read_processed_urls,
    filter_results_by_reason_codes,
    read_results_urls,
)

logger = logging.getLogger("seller_validator.agent")


class TaskStatus(Enum):
    DONE = "done"
    RETRY = "retry"
    RETRY_NEW_PROXY = "retry_new_proxy"
    ABANDON = "abandon"


class SellerValidatorAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        avito_library.MAX_PAGE = 1
        from avito_library.parsers import seller_profile_parser

        seller_profile_parser.MAX_PAGE = 1
        self._queue = TaskQueue(max_attempts=self.config.queue_max_attempts)
        self._proxy_pool: ProxyPool | None = None
        self._analyzer = self._create_analyzer()
        self._json_writer = JsonArrayWriter(self.config.results_json)
        self._processed_urls = read_processed_urls(self.config.processed_urls_txt)
        self._active_lock = asyncio.Lock()
        self._active_tasks = 0
        self._shutdown_event = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        self._payload_logger = logging.getLogger("seller_validator.payload")

    async def run(self) -> None:
        # Очистка результатов Gemini с ошибками 429/503 перед формированием очереди
        removed, remained = filter_results_by_reason_codes(self.config.results_json, ("429", "503"))
        logger.info("Очищено записей с ошибками 429/503: %s, осталось: %s", removed, remained)

        await self._init_proxy_pool()
        await self._seed_queue()
        logger.info("DISPLAY env: %s", os.environ.get("DISPLAY"))
        pending = await self._queue.pending_count()
        if pending == 0:
            logger.info("Нет новых URL для обработки")
            await self._analyzer.close()
            return

        async with async_playwright() as playwright:
            self._workers = [
                asyncio.create_task(self._worker_loop(worker_id=i, playwright=playwright))
                for i in range(self.config.worker_count)
            ]
            await self._shutdown_event.wait()
            for task in self._workers:
                task.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)

        await self._analyzer.close()

    def _create_analyzer(self):
        if self.config.llm_provider is LLMProvider.GENAI:
            return GeminiAnalyzer(self.config)
        if self.config.llm_provider is LLMProvider.OPENAI:
            return OpenAIAnalyzer(self.config)
        raise ValueError(f"Неизвестный LLM провайдер: {self.config.llm_provider}")

    async def _init_proxy_pool(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self._proxy_pool = await ProxyPool.create(
            proxies_file=self.config.proxies_file, blocked_file=self.config.blocked_proxies_file
        )
        proxies = await self._proxy_pool.all_proxies()
        if not proxies:
            raise RuntimeError("Файл proxies.txt пуст или не найден")
        all_blocked = await self._proxy_pool.all_blocked()
        logger.info("Загружено %s прокси, все заблокированы: %s", len(proxies), all_blocked)

    async def _seed_queue(self) -> None:
        urls = list(dict.fromkeys(iter_urls(self.config.urls_file)))
        existing_in_results = read_results_urls(self.config.results_json)
        new_urls = [url for url in urls if url not in existing_in_results]
        if not new_urls:
            logger.info("Все URL уже присутствуют в results.json")
            return
        now = datetime.now(timezone.utc).isoformat()
        await self._queue.put_many(
            (url, {"source": "urls_file", "enqueued_at": now}) for url in new_urls
        )
        logger.info("В очередь добавлено %s URL", len(new_urls))

    async def _worker_loop(self, worker_id: int, playwright) -> None:
        logger.info("Воркер %s запущен", worker_id)
        proxy: ProxyEndpoint | None = None
        browser = context = page = None
        while not self._shutdown_event.is_set():
            if proxy is None or browser is None or page is None or page.is_closed():
                if browser or context or page:
                    await self._close_browser(browser, context, page)
                if proxy and self._proxy_pool:
                    await self._proxy_pool.release(proxy.address)
                proxy = None
                browser = context = page = None
                browser, context, page, proxy = await self._reset_playwright(worker_id, playwright)
                if proxy is None:
                    await asyncio.sleep(self.config.queue_idle_sleep)
                    continue

            task = await self._queue.get()
            if task is None:
                if await self._should_finish():
                    self._shutdown_event.set()
                    break
                await asyncio.sleep(self.config.queue_idle_sleep)
                continue

            await self._mark_active(+1)
            status = TaskStatus.RETRY
            try:
                status = await self._handle_task(task=task, page=page, proxy=proxy, worker_id=worker_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Необработанная ошибка воркера %s: %s", worker_id, exc)
                status = TaskStatus.RETRY
            finally:
                await self._mark_active(-1)

            if status == TaskStatus.DONE:
                await self._queue.mark_done(task.task_key)
                continue

            if status == TaskStatus.ABANDON:
                await self._queue.abandon(task.task_key)
                continue

            if status == TaskStatus.RETRY_NEW_PROXY:
                if proxy is not None and self._proxy_pool is not None:
                    logger.warning("Воркер %s переводит задачу на новый прокси", worker_id)
                await self._queue.retry(task.task_key, last_proxy=proxy.address if proxy else None)
                await self._close_browser(browser, context, page)
                browser = context = page = None
                proxy = None
                await asyncio.sleep(self.config.worker_relaunch_delay)
                continue

            if status == TaskStatus.RETRY:
                succeeded = await self._queue.retry(task.task_key, last_proxy=proxy.address if proxy else None)
                if not succeeded:
                    await self._queue.abandon(task.task_key)

        await self._close_browser(browser, context, page)
        if proxy and self._proxy_pool:
            await self._proxy_pool.release(proxy.address)
        logger.info("Воркер %s остановлен", worker_id)

    async def _reset_playwright(self, worker_id: int, playwright):
        if self._proxy_pool is None:
            raise RuntimeError("ProxyPool не инициализирован")
        proxy = await self._acquire_proxy(worker_id)
        if proxy is None:
            return None, None, None, None
        args = proxy.as_playwright_arguments()
        launch_env = None
        display_base = os.environ.get("PLAYWRIGHT_DISPLAY")
        if display_base:
            launch_env = dict(os.environ)
            launch_env["DISPLAY"] = display_base
        try:
            browser = await playwright.chromium.launch(
                headless=self.config.playwright_headless,
                slow_mo=self.config.playwright_slow_mo,
                proxy=args,
                env=launch_env,
            )
            context = await browser.new_context()
            page = await context.new_page()
            page.set_default_navigation_timeout(self.config.navigation_timeout_ms)
            logger.info("Воркер %s использует прокси %s", worker_id, proxy.address)
            return browser, context, page, proxy
        except Exception as exc:  # noqa: BLE001
            error_text = f"{exc}"
            logger.error(
                "Не удалось запустить Chromium с прокси %s: %s",
                proxy.address,
                error_text,
            )
            if self._is_environment_failure(error_text):
                logger.warning(
                    "Игнорируем блокировку прокси %s: ошибка окружения Playwright",
                    proxy.address,
                )
                await self._proxy_pool.release(proxy.address)
            else:
                await self._mark_proxy_blocked(proxy.address, reason="launch_failed")
            return None, None, None, None

    async def _acquire_proxy(self, worker_id: int) -> ProxyEndpoint | None:
        if self._proxy_pool is None:
            raise RuntimeError("ProxyPool не инициализирован")
        while not self._shutdown_event.is_set():
            proxy = await self._proxy_pool.acquire()
            if proxy:
                return proxy
            await self._proxy_pool.wait_for_unblocked()
            await asyncio.sleep(self.config.queue_idle_sleep)
        return None

    @staticmethod
    def _is_environment_failure(message: str) -> bool:
        lowered = message.lower()
        return any(
            marker in lowered
            for marker in (
                "missing x server",
                "x server",
                "failed to connect to the bus",
                "target page, context or browser has been closed",
                "browser has been closed",
                "xvfb",
            )
        )

    async def _handle_task(self, *, task: ProcessingTask, page: Page, proxy: ProxyEndpoint, worker_id: int) -> TaskStatus:
        url = task.task_key
        logger.info("Воркер %s обрабатывает %s (попытка %s)", worker_id, url, task.attempt)
        response: Response | None = None
        try:
            response = await page.goto(url, wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            logger.warning("Таймаут перехода на %s", url)
            return TaskStatus.RETRY
        except PlaywrightError as exc:
            logger.error("Playwright ошибка при переходе на %s: %s", url, exc)
            return TaskStatus.RETRY_NEW_PROXY

        state = await detect_page_state(page, last_response=response)
        if state in {CAPTCHA_DETECTOR_ID, CONTINUE_BUTTON_DETECTOR_ID, PROXY_BLOCK_429_DETECTOR_ID}:
            solved = await self._solve_captcha(page)
            if not solved:
                return TaskStatus.RETRY
            state = await detect_page_state(page)

        if state in {PROXY_BLOCK_403_DETECTOR_ID, PROXY_AUTH_407_DETECTOR_ID}:
            await self._mark_proxy_blocked(proxy.address, state)
            return TaskStatus.RETRY_NEW_PROXY

        if state != SELLER_PROFILE_DETECTOR_ID:
            logger.info("Не удалось подтвердить профиль для %s: %s", url, state)
            return TaskStatus.RETRY

        seller_result = await collect_seller_items(
            page,
            include_items=True,
            item_schema=self.config.seller_schema,
        )
        if seller_result.get("state") != SELLER_PROFILE_DETECTOR_ID:
            logger.info("collect_seller_items вернул состояние %s для %s", seller_result.get("state"), url)
            return TaskStatus.RETRY

        raw_items: dict = seller_result.get("items_by_id") or {}
        items = self._prepare_items(raw_items)
        seller_name = seller_result.get("seller_name")

        first_item = items[0] if items else {}
        log_entry = {
            "seller_url": url,
            "seller_name": seller_name,
            "attempt": task.attempt,
            "item_count": len(items),
            "first_item": first_item,
        }
        logger.info(
            "Данные для Gemini: url=%s, объявлений=%s, первое объявление=%s",
            url,
            len(items),
            json.dumps(first_item, ensure_ascii=False),
        )
        self._payload_logger.info(json.dumps(log_entry, ensure_ascii=False))

        gemini_result = await self._analyzer.analyze(
            seller_url=url,
            seller_name=seller_name,
            items=items,
            attempt=task.attempt,
        )

        record = {
            "seller_url": url,
            "verdict": gemini_result.get("verdict"),
            "reason": gemini_result.get("reason"),
        }
        await self._json_writer.append(record)
        append_processed_url(self.config.processed_urls_txt, url)
        self._processed_urls.add(url)
        if gemini_result.get("verdict") == "passed":
            append_valid_url(self.config.passed_urls_txt, url)
        logger.info("Обработка %s завершена", url)
        return TaskStatus.DONE

    async def _solve_captcha(self, page: Page) -> bool:
        try:
            await resolve_captcha_flow(page)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка решения капчи: %s", exc)
            return False

    def _prepare_items(self, raw_items: Dict[Any, Dict[str, Any]]) -> list[Dict[str, Any]]:
        prepared: list[Dict[str, Any]] = []
        for item_id, payload in list(raw_items.items())[: self.config.max_items_per_seller]:
            item = {
                "item_id": str(item_id),
                "title": self._safe_get(payload, "title"),
                "description": self._safe_get(payload, "description"),
                "price": self._extract_price(payload),
                "currency": self._safe_get(payload, ("priceDetailed", "currency")),
                "condition": self._extract_condition(payload),
                "brand_candidates": self._extract_brands(payload),
                "category": self._safe_get(payload, ("category", "name")),
            }
            prepared.append(item)
        return prepared

    @staticmethod
    def _safe_get(struct: Dict[str, Any], path: str | tuple[str, ...]) -> Any:
        if isinstance(path, tuple):
            current: Any = struct
            for key in path:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
            return current
        return struct.get(path) if isinstance(struct, dict) else None

    def _extract_price(self, payload: Dict[str, Any]) -> Any:
        price = self._safe_get(payload, ("priceDetailed", "value"))
        return price

    def _extract_condition(self, payload: Dict[str, Any]) -> Optional[str]:
        attributes = payload.get("attributes")
        if isinstance(attributes, list):
            for attr in attributes:
                if not isinstance(attr, dict):
                    continue
                title = attr.get("title") or attr.get("name") or attr.get("label")
                slug = attr.get("slug") or attr.get("code")
                if (title and "состояние" in title.lower()) or (slug and "condition" in slug.lower()):
                    return attr.get("value") or attr.get("payload", {}).get("title")
        return None

    def _extract_brands(self, payload: Dict[str, Any]) -> list[str]:
        results: list[str] = []
        iva = payload.get("iva")
        if isinstance(iva, dict):
            auto_parts = iva.get("AutoPartsManufacturerStep")
            if isinstance(auto_parts, list):
                for item in auto_parts:
                    if not isinstance(item, dict):
                        continue
                    value = item.get("payload", {}).get("value")
                    if isinstance(value, str):
                        results.append(value)
            spare_params = iva.get("SparePartsParamsStep")
            if isinstance(spare_params, list):
                for item in spare_params:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("payload", {}).get("text")
                    if isinstance(text, str):
                        results.append(text)
        return results

    async def _mark_proxy_blocked(self, address: str, reason: str) -> None:
        if self._proxy_pool is not None:
            await self._proxy_pool.mark_blocked(address, reason=reason)

    async def _mark_active(self, delta: int) -> None:
        async with self._active_lock:
            self._active_tasks += delta

    async def _should_finish(self) -> bool:
        async with self._active_lock:
            active = self._active_tasks
        pending = await self._queue.pending_count()
        return active == 0 and pending == 0

    async def _close_browser(self, browser, context, page) -> None:
        if page:
            try:
                await page.close()
            except Exception:  # noqa: BLE001
                pass
        if context:
            try:
                await context.close()
            except Exception:  # noqa: BLE001
                pass
        if browser:
            try:
                await browser.close()
            except Exception:  # noqa: BLE001
                pass


    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: SellerValidatorAgent._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [SellerValidatorAgent._sanitize_for_json(v) for v in value]
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return value


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = Config()
    payload_logger = logging.getLogger("seller_validator.payload")
    if not payload_logger.handlers:
        payload_log_path = config.data_dir / "gemini_payloads.log"
        handler = logging.FileHandler(payload_log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
        payload_logger.addHandler(handler)
        payload_logger.setLevel(logging.INFO)
        payload_logger.propagate = False

    agent = SellerValidatorAgent(config=config)
    await agent.run()
