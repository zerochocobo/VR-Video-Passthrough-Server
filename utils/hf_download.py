"""Shared Hugging Face download helpers (mirror-aware, no PyTorch/hub dep).

Used by the DA3 and NVDS ONNX model fetchers and the UI download dialog. Chinese
users default to hf-mirror.com; ``HF_ENDPOINT`` overrides everything.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable

HF_DEFAULT_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

ProgressFn = Callable[[int, int], None]


def hf_endpoints(language: str | None = None) -> list[str]:
    """Ordered endpoints to try. ``HF_ENDPOINT`` wins; zh prefers the mirror.

    ``language`` lets an in-process caller (e.g. the UI) pass the current UI
    language explicitly; otherwise the ``PT_UI_LANGUAGE`` env var is used.
    """
    override = str(os.environ.get("HF_ENDPOINT") or "").strip()
    if override:
        return [override.rstrip("/")]
    lang = str(language or os.environ.get("PT_UI_LANGUAGE") or "").lower()
    if lang.startswith("zh"):
        return [HF_MIRROR_ENDPOINT, HF_DEFAULT_ENDPOINT]
    return [HF_DEFAULT_ENDPOINT, HF_MIRROR_ENDPOINT]


def hf_resolve_urls(repo: str, filename: str, language: str | None = None) -> list[str]:
    return [f"{ep}/{repo}/resolve/main/{filename}" for ep in hf_endpoints(language)]


def remote_size(urls: list[str], timeout: float = 15.0) -> int:
    """Best-effort Content-Length across mirrors (0 if it cannot be determined).

    Opens the stream and reads only the headers (no body), following the HF 302
    redirect to the CDN so the length reflects the actual file.
    """
    for url in urls:
        req = urllib.request.Request(url, headers={"User-Agent": "PTMediaServer"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                size = int(resp.headers.get("Content-Length") or 0)
                if size > 0:
                    return size
        except Exception:
            continue
    return 0


def download_file(
    urls: list[str],
    dest: Path,
    progress: ProgressFn | None = None,
    log: Callable[[str], None] = print,
    chunk: int = 1 << 20,
    timeout: float = 30.0,
) -> Path:
    """Stream ``urls`` (mirror fallback) to ``dest`` via a ``.part`` temp file.

    ``progress(done, total)`` is called as bytes arrive (total may be 0 when the
    server omits Content-Length). Raises the last error if every mirror fails.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for index, url in enumerate(urls):
        log(f"download: {dest.name} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "PTMediaServer"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        if progress:
                            progress(done, total)
            tmp.replace(dest)
            log(f"download: {dest.name} done ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except Exception as exc:
            last_error = exc
            try:
                tmp.unlink()
            except OSError:
                pass
            if index + 1 < len(urls):
                log(f"download failed from {url}: {type(exc).__name__}: {exc}; trying next mirror")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"no download endpoint configured for {dest.name}")
