"""Local UI control endpoints for runtime server settings."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from utils.runtime_settings import get_light_match, get_si_mix, set_light_match, set_si_mix
from utils.logger import get


router = APIRouter()
log = get("control")
_LIGHT_MATCH_KEYS = {
    "enabled",
    "temp_k",
    "tint",
    "exposure_ev",
    "contrast",
    "gamma",
    "saturation",
    "preset",
}
_SI_MIX_KEYS = {
    "enabled",
    "mix_channel",
    "original_volume_percent",
    "si_volume_percent",
    "si_delay_seconds",
    "duck_original",
}


def _is_local_request(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


@router.get("/control/light_match")
async def get_light_match_control(request: Request):
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="local control only")
    payload = get_light_match().to_dict()
    payload["ok"] = True
    return payload


@router.put("/control/light_match")
async def put_light_match_control(request: Request):
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="local control only")
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid json") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="json object required")
    unknown = sorted(str(key) for key in data.keys() if str(key) not in _LIGHT_MATCH_KEYS)
    if unknown:
        log.warning("light_match control ignored unknown keys: %s", ", ".join(unknown))
    payload = set_light_match(data).to_dict()
    payload["ok"] = True
    return payload


@router.get("/control/si_mix")
async def get_si_mix_control(request: Request):
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="local control only")
    payload = get_si_mix().to_dict()
    payload["ok"] = True
    return payload


@router.put("/control/si_mix")
async def put_si_mix_control(request: Request):
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="local control only")
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid json") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="json object required")
    unknown = sorted(str(key) for key in data.keys() if str(key) not in _SI_MIX_KEYS)
    if unknown:
        log.warning("si_mix control ignored unknown keys: %s", ", ".join(unknown))
    before = get_si_mix()
    payload = set_si_mix(data)
    if payload.version != before.version:
        try:
            from http_app.si_stream import reload_si_stream_service

            reload_si_stream_service(payload)
        except Exception as exc:
            log.warning("si_mix stream reload failed: %s", exc)
        try:
            from dlna.content_directory import clear_dir_items_cache

            clear_dir_items_cache()
        except Exception as exc:
            log.warning("si_mix DLNA cache clear failed: %s", exc)
    result = payload.to_dict()
    result["ok"] = True
    return result
