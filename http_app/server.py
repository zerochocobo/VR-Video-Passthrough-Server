"""FastAPI application factory for DLNA control and media routes."""

import asyncio

from fastapi import FastAPI

from http_app.routes_control import router as control_router
from http_app.routes_dlna import router as dlna_router
from http_app.routes_media import router as media_router
from utils.logger import get


log = get("server")


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


def create_app() -> FastAPI:
    """Create the HTTP app without starting network listeners."""
    app = FastAPI(title="PT VR Passthrough Server", docs_url=None, redoc_url=None)
    app.include_router(control_router)
    app.include_router(dlna_router)
    app.include_router(media_router)

    @app.on_event("startup")
    async def startup() -> None:
        _install_asyncio_noise_filter()

    @app.get("/")
    async def index():
        return {"ok": True, "service": "pt-dlna"}

    return app
