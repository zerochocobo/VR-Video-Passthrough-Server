"""Fire-and-forget live control PUTs to the running server's /control endpoints."""
from __future__ import annotations

import json
import os
import threading
import urllib.request


def send_control(path: str, payload: dict, timeout: float = 0.35) -> None:
    port = str(os.environ.get("PT_HTTP_PORT") or "8200").strip() or "8200"
    url = f"http://127.0.0.1:{port}/control/{path}"

    def worker() -> None:
        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                method="PUT",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read(2048)
        except Exception:
            pass

    threading.Thread(target=worker, name=f"live-control-{path}", daemon=True).start()
