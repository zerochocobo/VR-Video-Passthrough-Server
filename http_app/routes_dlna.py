"""HTTP endpoints for UPnP device XML, service XML, and SOAP control."""
from __future__ import annotations

import re

from fastapi import APIRouter, Request, Response

from dlna.connection_manager import handle_soap as handle_cm_soap
from dlna.content_directory import handle_soap as handle_cds_soap
from dlna.descriptions import cds_scpd, cm_scpd, device_description
from utils.request_history import annotate_request

router = APIRouter()
XML_MEDIA_TYPE = "text/xml; charset=utf-8"
_SOAP_FIELD_RE = re.compile(rb"<(?:\w+:)?(ObjectID|BrowseFlag|RequestedCount|StartingIndex)>(.*?)</(?:\w+:)?\1>", re.IGNORECASE | re.DOTALL)


def _soap_history_fields(body: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _SOAP_FIELD_RE.finditer(body[:64 * 1024]):
        key = match.group(1).decode("ascii", "ignore")
        value = match.group(2).decode("utf-8", "ignore").strip()
        if value:
            fields[key] = value
    return fields


@router.get("/description.xml")
async def get_description():
    return Response(content=device_description(), media_type=XML_MEDIA_TYPE)


@router.get("/cds.xml")
async def get_cds_scpd():
    return Response(content=cds_scpd(), media_type=XML_MEDIA_TYPE)


@router.get("/cm.xml")
async def get_cm_scpd():
    return Response(content=cm_scpd(), media_type=XML_MEDIA_TYPE)


@router.post("/control/cds")
async def control_cds(request: Request):
    soap_action = request.headers.get("SOAPAction", "")
    body = await request.body()
    annotate_request(request, soap_action=soap_action, **_soap_history_fields(body))
    payload, status = handle_cds_soap(soap_action, body)
    return Response(
        content=payload,
        status_code=status,
        media_type=XML_MEDIA_TYPE,
    )


@router.post("/control/cm")
async def control_cm(request: Request):
    soap_action = request.headers.get("SOAPAction", "")
    body = await request.body()
    annotate_request(request, soap_action=soap_action)
    payload, status = handle_cm_soap(soap_action, body)
    return Response(
        content=payload,
        status_code=status,
        media_type=XML_MEDIA_TYPE,
    )
