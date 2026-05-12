"""HTTP endpoints for UPnP device XML, service XML, and SOAP control."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response

from dlna.connection_manager import handle_soap as handle_cm_soap
from dlna.content_directory import handle_soap as handle_cds_soap
from dlna.descriptions import cds_scpd, cm_scpd, device_description

router = APIRouter()
XML_MEDIA_TYPE = "text/xml; charset=utf-8"


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
    payload, status = handle_cm_soap(soap_action, body)
    return Response(
        content=payload,
        status_code=status,
        media_type=XML_MEDIA_TYPE,
    )
