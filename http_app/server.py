"""FastAPI application factory for DLNA control and media routes."""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from config import (
    PASSTHROUGH_LIVE_DEFAULT_PROFILE,
    REQUEST_HISTORY_DEBUG_ENDPOINT,
    REQUEST_HISTORY_DIR,
    REQUEST_HISTORY_ENABLED,
    REQUEST_HISTORY_FLUSH_EVERY,
    REQUEST_HISTORY_JSONL,
    REQUEST_HISTORY_MAX_RECORDS,
    REQUEST_HISTORY_REDACT,
)
from http_app.routes_control import router as control_router
from http_app.routes_dlna import router as dlna_router
from http_app.routes_media import router as media_router
from utils.logger import get
from utils.request_history import (
    configure_request_history,
    get_request_history,
    next_trace_id,
    record_request,
)
from utils.player_compat import clear_device_sessions


log = get("server")
StartupHook = Callable[[], Awaitable[Any] | Any]
_REQUEST_HISTORY_SKIP_PATHS = {"/", "/runtime_status"}
_REQUEST_HISTORY_SKIP_PREFIXES = ("/debug/",)


def _install_asyncio_noise_filter() -> None:
    """Suppress benign Windows disconnect noise while preserving real loop errors."""
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    def handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        handle = str(context.get("handle") or "")
        if (
            isinstance(exc, ConnectionResetError)
            and getattr(exc, "winerror", None) == 10054
            and "_ProactorBasePipeTransport._call_connection_lost" in handle
        ):
            log.debug("suppressed benign proactor disconnect: %s", exc)
            return
        if previous_handler is not None:
            previous_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


def _should_record_request(path: str) -> bool:
    if path in _REQUEST_HISTORY_SKIP_PATHS:
        return False
    return not any(path.startswith(prefix) for prefix in _REQUEST_HISTORY_SKIP_PREFIXES)


def create_app(startup_hook: StartupHook | None = None) -> FastAPI:
    """Create the HTTP app without starting network listeners."""
    configure_request_history(
        enabled=REQUEST_HISTORY_ENABLED,
        max_records=REQUEST_HISTORY_MAX_RECORDS,
        jsonl_enabled=REQUEST_HISTORY_JSONL,
        jsonl_dir=REQUEST_HISTORY_DIR,
        flush_every=REQUEST_HISTORY_FLUSH_EVERY,
        redact=REQUEST_HISTORY_REDACT,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _install_asyncio_noise_filter()
        if startup_hook is not None:
            result = startup_hook()
            if inspect.isawaitable(result):
                await result
        try:
            yield
        finally:
            try:
                from http_app.si_stream import shutdown_si_stream_service

                shutdown_si_stream_service()
            except Exception as e:
                log.warning("SI stream shutdown failed: %s", e)
            get_request_history().flush()

    app = FastAPI(title="PT VR Passthrough Server", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.middleware("http")
    async def request_history_middleware(request: Request, call_next):
        trace_id = next_trace_id()
        request.state.pt_trace_id = trace_id
        started = time.perf_counter()
        status_code = 500
        content_length = ""
        should_record = REQUEST_HISTORY_ENABLED and _should_record_request(request.url.path)
        try:
            response = await call_next(request)
            status_code = int(response.status_code)
            content_length = response.headers.get("content-length", "")
            response.headers["X-PT-Request-Trace-Id"] = trace_id
            return response
        except Exception as e:
            log.exception("request failed trace=%s method=%s path=%s: %s", trace_id, request.method, request.url.path, e)
            raise
        finally:
            if should_record:
                try:
                    record = record_request(
                        request=request,
                        trace_id=trace_id,
                        status_code=status_code,
                        started_at=started,
                        content_length=content_length,
                        default_profile=PASSTHROUGH_LIVE_DEFAULT_PROFILE,
                    )
                    log.debug(
                        "request trace=%s method=%s path=%s status=%d profile=%s intent=%s decision=%s rules=%s",
                        record.trace_id,
                        record.method,
                        record.path,
                        record.status_code,
                        record.route_profile,
                        record.intent,
                        record.decision,
                        ",".join(record.fired_rules),
                    )
                except Exception as e:
                    log.warning("request history record failed: %s", e)

    app.include_router(control_router)
    app.include_router(dlna_router)
    app.include_router(media_router)

    @app.get("/")
    async def index():
        return {"ok": True, "service": "pt-dlna"}

    if REQUEST_HISTORY_DEBUG_ENDPOINT:
        @app.get("/debug/request_history")
        async def debug_request_history(request: Request, limit: int = 100):
            client_host = request.client.host if request.client else ""
            if client_host not in {"127.0.0.1", "::1", "localhost"}:
                raise HTTPException(status_code=403, detail="local debug endpoint only")
            limit = max(0, min(int(limit), REQUEST_HISTORY_MAX_RECORDS or 500))
            snapshot = get_request_history().snapshot(limit=None)
            return {
                "ok": True,
                "count": len(snapshot),
                "records": snapshot[-limit:] if limit else [],
            }

        @app.post("/debug/clear_device_sessions")
        async def debug_clear_device_sessions(request: Request):
            client_host = request.client.host if request.client else ""
            if client_host not in {"127.0.0.1", "::1", "localhost"}:
                raise HTTPException(status_code=403, detail="local debug endpoint only")
            clear_device_sessions()
            return {"ok": True}

    return app
