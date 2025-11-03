"""Основной агента для проверки продавцов Авито."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime, timezone
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import avito_library
from avito_library import (
    NOT_DETECTED_STATE_ID,
    collect_seller_items,
    detect_page_state,
    resolve_captcha_flow,
)
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

# Database imports (conditionally used based on config)
import asyncpg
from .database import create_pool, init_database, close_pool, TaskRepository, ResultRepository, ProxyRepository
from .db_task_queue import DatabaseTaskQueue

logger = logging.getLogger("seller_validator.agent")


class TaskStatus(Enum):
    DONE = "done"
    RETRY = "retry"
    RETRY_NEW_PROXY = "retry_new_proxy"
    ABANDON = "abandon"
    DEFERRED = "deferred"


@dataclass(slots=True)
class ValidationJob:
    task: ProcessingTask
    seller_url: str
    seller_name: str | None
    items: list[Dict[str, Any]]
    attempt: int
    last_proxy: str | None


class ValidationExecutor:
    def __init__(
        self,
        *,
        config: Config,
        analyzer: GeminiAnalyzer | OpenAIAnalyzer,
        json_writer: JsonArrayWriter,
        task_queue: TaskQueue | DatabaseTaskQueue,
        processed_urls: set[str],
        processed_urls_lock: asyncio.Lock,
        processed_urls_path: os.PathLike[str] | str,
        passed_urls_path: os.PathLike[str] | str,
        result_repository: Optional[ResultRepository] = None,
        task_repository: Optional[TaskRepository] = None,
    ) -> None:
        self._config = config
        self._analyzer = analyzer
        self._json_writer = json_writer
        self._task_queue = task_queue
        self._processed_urls = processed_urls
        self._processed_urls_lock = processed_urls_lock
        self._processed_urls_path = Path(processed_urls_path)
        self._passed_urls_path = Path(passed_urls_path)
        self._jobs: asyncio.Queue[ValidationJob | None] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._concurrency = max(1, self._config.llm_concurrency)
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._started = False
        self._logger = logging.getLogger("seller_validator.validation")
        self._inflight = 0
        self._inflight_lock = asyncio.Lock()
        # Database mode support
        self._result_repository = result_repository
        self._task_repository = task_repository
        self._use_database = config.use_database

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for idx in range(self._concurrency):
            worker = asyncio.create_task(self._worker_loop(idx), name=f"validation-worker-{idx}")
            self._workers.append(worker)

    async def submit(self, job: ValidationJob) -> None:
        if not self._started:
            await self.start()
        async with self._inflight_lock:
            self._inflight += 1
        await self._jobs.put(job)

    async def shutdown(self) -> None:
        if not self._started:
            return
        # дождаться, пока вся очередь будет обработана
        await self._jobs.join()
        for _ in self._workers:
            await self._jobs.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False
        async with self._inflight_lock:
            self._inflight = 0

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            job = await self._jobs.get()
            if job is None:
                self._jobs.task_done()
                break
            async with self._semaphore:
                try:
                    await self._process_job(job, worker_id=worker_id)
                finally:
                    self._jobs.task_done()

    async def _process_job(self, job: ValidationJob, *, worker_id: int) -> None:
        try:
            self._logger.info(
                "Validation worker %s processing %s (attempt %s)",
                worker_id,
                job.seller_url,
                job.attempt,
            )
            try:
                result = await self._analyzer.analyze(
                    seller_url=job.seller_url,
                    seller_name=job.seller_name,
                    items=job.items,
                    attempt=job.attempt,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.exception(
                    "Validation worker %s failed for %s: %s",
                    worker_id,
                    job.seller_url,
                    exc,
                )
                await self._retry_task(job, reason="validation_exception")
                return

            verdict = result.get("verdict")
            if verdict == "error":
                self._logger.warning(
                    "Validation worker %s got error verdict for %s, scheduling retry",
                    worker_id,
                    job.seller_url,
                )
                await self._retry_task(job, reason="validation_error_verdict")
                return

            # Save result (database or file mode)
            if self._use_database and self._result_repository:
                # Database mode: save to results table
                # Get task_id from payload (set by DatabaseTaskQueue.get())
                task_id = job.task.payload.get('_db_task_id') if isinstance(job.task.payload, dict) else None
                if task_id:
                    await self._result_repository.save_result(
                        task_id=task_id,
                        seller_url=job.seller_url,
                        verdict=verdict,
                        reason=result.get("reason", ""),
                        confidence=float(result.get("confidence", 0.0)),
                        flags=result.get("flags", []),
                        items=result.get("items", []),
                        llm_attempts=job.attempt,
                    )
                    self._logger.info("Saved result to database for %s", job.seller_url)
                else:
                    self._logger.error("Task ID not found in payload for %s", job.seller_url)
            else:
                # File mode: save to JSON file
                record = {
                    "seller_url": job.seller_url,
                    "verdict": verdict,
                    "reason": result.get("reason"),
                }
                await self._json_writer.append(record)
                append_processed_url(self._processed_urls_path, job.seller_url)
                if verdict == "passed":
                    append_valid_url(self._passed_urls_path, job.seller_url)

            # Update processed URLs tracking (both modes)
            async with self._processed_urls_lock:
                self._processed_urls.add(job.seller_url)

            await self._task_queue.mark_done(job.task.task_key)
            self._logger.info(
                "Validation worker %s finished %s with verdict=%s",
                worker_id,
                job.seller_url,
                verdict,
            )
        finally:
            await self._job_finished()

    async def _retry_task(self, job: ValidationJob, *, reason: str) -> None:
        succeeded = await self._task_queue.retry(
            job.task.task_key, last_proxy=job.last_proxy
        )
        if not succeeded:
            self._logger.error(
                "Retry limit exceeded for %s, abandoning task", job.seller_url
            )
            await self._task_queue.abandon(job.task.task_key)

            # Save error result (database or file mode)
            if self._use_database and self._result_repository:
                # Database mode: save to results table
                # Get task_id from payload (set by DatabaseTaskQueue.get())
                task_id = job.task.payload.get('_db_task_id') if isinstance(job.task.payload, dict) else None
                if task_id:
                    await self._result_repository.save_result(
                        task_id=task_id,
                        seller_url=job.seller_url,
                        verdict="error",
                        reason=f"{reason}: retry limit exceeded",
                        confidence=0.0,
                        flags=[],
                        items=[],
                        llm_attempts=job.attempt,
                    )
                else:
                    self._logger.error("Task ID not found in payload for %s", job.seller_url)
            else:
                # File mode: save to JSON file
                record = {
                    "seller_url": job.seller_url,
                    "verdict": "error",
                    "reason": f"{reason}: retry limit exceeded",
                }
                await self._json_writer.append(record)
                append_processed_url(self._processed_urls_path, job.seller_url)

            # Update processed URLs tracking (both modes)
            async with self._processed_urls_lock:
                self._processed_urls.add(job.seller_url)
        else:
            self._logger.info(
                "Retry scheduled for %s due to %s", job.seller_url, reason
            )

    async def _job_finished(self) -> None:
        async with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    async def has_pending_jobs(self) -> bool:
        async with self._inflight_lock:
            return self._inflight > 0


class SellerValidatorAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        avito_library.MAX_PAGE = 1
        from avito_library.parsers import seller_profile_parser

        seller_profile_parser.MAX_PAGE = 1

        # Database mode initialization
        self._db_pool: Optional[asyncpg.Pool] = None
        self._task_repository: Optional[TaskRepository] = None
        self._result_repository: Optional[ResultRepository] = None
        self._proxy_repository: Optional[ProxyRepository] = None

        # Initialize task queue (database or in-memory)
        if self.config.use_database:
            # Database queue will be initialized in run() after pool creation
            self._queue: TaskQueue | DatabaseTaskQueue | None = None
        else:
            self._queue = TaskQueue(max_attempts=self.config.queue_max_attempts)

        self._proxy_pool: ProxyPool | None = None
        self._analyzer = self._create_analyzer()
        self._json_writer = JsonArrayWriter(self.config.results_json)
        self._processed_urls = read_processed_urls(self.config.processed_urls_txt) if not self.config.use_database else set()
        self._processed_urls_lock = asyncio.Lock()
        self._active_lock = asyncio.Lock()
        self._active_tasks = 0
        self._shutdown_event = asyncio.Event()
        self._workers: list[asyncio.Task[None]] = []
        self._payload_logger = logging.getLogger("seller_validator.payload")

        # ValidationExecutor will be initialized in run() after database setup (if needed)
        self._validation_executor: Optional[ValidationExecutor] = None

    async def run(self) -> None:
        # Initialize database connection if in database mode
        if self.config.use_database:
            logger.info("Initializing database connection...")
            self._db_pool = await create_pool(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                min_size=self.config.db_pool_min_size,
                max_size=self.config.db_pool_max_size,
            )
            await init_database(self._db_pool)

            # Initialize repositories
            self._task_repository = TaskRepository(self._db_pool)
            self._result_repository = ResultRepository(self._db_pool)
            self._proxy_repository = ProxyRepository(self._db_pool)

            # Initialize database task queue
            self._queue = DatabaseTaskQueue(
                pool=self._db_pool,
                max_attempts=self.config.queue_max_attempts
            )
            logger.info("Database mode initialized successfully")
        else:
            # File mode: clear results with errors
            removed, remained = filter_results_by_reason_codes(self.config.results_json, ("429", "503"))
            logger.info("Очищено записей с ошибками 429/503: %s, осталось: %s", removed, remained)

        # Initialize ValidationExecutor
        self._validation_executor = ValidationExecutor(
            config=self.config,
            analyzer=self._analyzer,
            json_writer=self._json_writer,
            task_queue=self._queue,
            processed_urls=self._processed_urls,
            processed_urls_lock=self._processed_urls_lock,
            processed_urls_path=self.config.processed_urls_txt,
            passed_urls_path=self.config.passed_urls_txt,
            result_repository=self._result_repository,
            task_repository=self._task_repository,
        )

        await self._init_proxy_pool()
        await self._seed_queue()
        logger.info("DISPLAY env: %s", os.environ.get("DISPLAY"))
        pending = await self._queue.pending_count()
        try:
            if pending == 0:
                logger.info("Нет новых URL для обработки")
                return

            await self._validation_executor.start()

            async with async_playwright() as playwright:
                self._workers = [
                    asyncio.create_task(self._worker_loop(worker_id=i, playwright=playwright))
                    for i in range(self.config.worker_count)
                ]
                await self._shutdown_event.wait()
                for task in self._workers:
                    task.cancel()
                await asyncio.gather(*self._workers, return_exceptions=True)
        finally:
            await self._validation_executor.shutdown()
            await self._analyzer.close()
            # Close database connection if in database mode
            if self.config.use_database:
                await close_pool()
                logger.info("Database connection closed")

    def _create_analyzer(self):
        if self.config.llm_provider is LLMProvider.GENAI:
            return GeminiAnalyzer(self.config)
        if self.config.llm_provider is LLMProvider.OPENAI:
            return OpenAIAnalyzer(self.config)
        raise ValueError(f"Неизвестный LLM провайдер: {self.config.llm_provider}")

    async def _init_proxy_pool(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_database and self._proxy_repository:
            # Database mode: load proxies from database
            proxies = await self._proxy_repository.get_all_proxies()
            if not proxies:
                raise RuntimeError("No proxies found in database. Please load proxies using scripts/load_proxies.py")

            # Create temporary files for ProxyPool (it still uses file-based API)
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "avito_seller_checker"
            temp_dir.mkdir(exist_ok=True, parents=True)
            temp_proxies_file = temp_dir / "proxies.txt"
            temp_blocked_file = temp_dir / "blocked_proxies.txt"

            # Write proxies to temp file
            with open(temp_proxies_file, 'w') as f:
                f.write('\n'.join(proxies))

            # Empty blocked file (blocked status is in database)
            temp_blocked_file.touch()

            self._proxy_pool = await ProxyPool.create(
                proxies_file=temp_proxies_file,
                blocked_file=temp_blocked_file
            )
            logger.info("Loaded %s proxies from database", len(proxies))
        else:
            # File mode: load from files
            self._proxy_pool = await ProxyPool.create(
                proxies_file=self.config.proxies_file,
                blocked_file=self.config.blocked_proxies_file
            )
            proxies = await self._proxy_pool.all_proxies()
            if not proxies:
                raise RuntimeError("Файл proxies.txt пуст или не найден")
            all_blocked = await self._proxy_pool.all_blocked()
            logger.info("Загружено %s прокси, все заблокированы: %s", len(proxies), all_blocked)

    async def _seed_queue(self) -> None:
        if self.config.use_database:
            # Database mode: tasks are loaded via scripts, skip seeding
            logger.info("Database mode: skipping queue seeding (use scripts/load_urls.py to load tasks)")
            return

        # File mode: load from urls.txt and filter by results.json
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

            if status == TaskStatus.DEFERRED:
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

        if state == NOT_DETECTED_STATE_ID:
            logger.info("Детекторы не сработали для %s, помечаем как not_detected", url)
            return TaskStatus.ABANDON

        if state != SELLER_PROFILE_DETECTOR_ID:
            logger.info("Не удалось подтвердить профиль для %s: %s", url, state)
            return TaskStatus.ABANDON

        seller_result = await collect_seller_items(
            page,
            include_items=True,
            item_schema=self.config.seller_schema,
        )
        seller_state = seller_result.get("state")
        if seller_state == NOT_DETECTED_STATE_ID:
            logger.info("collect_seller_items не нашёл профиль для %s (not_detected)", url)
            return TaskStatus.ABANDON
        if seller_state != SELLER_PROFILE_DETECTOR_ID:
            logger.info("collect_seller_items вернул состояние %s для %s", seller_state, url)
            return TaskStatus.ABANDON

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

        await self._validation_executor.submit(
            ValidationJob(
                task=task,
                seller_url=url,
                seller_name=seller_name,
                items=items,
                attempt=task.attempt,
                last_proxy=proxy.address if proxy else None,
            )
        )
        logger.info("Данные для %s отправлены на валидацию", url)
        return TaskStatus.DEFERRED

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
        validation_pending = await self._validation_executor.has_pending_jobs()
        return active == 0 and pending == 0 and not validation_pending

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
