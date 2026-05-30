from __future__ import annotations

import unittest
from xml.etree import ElementTree as ET

import dlna.connection_manager as cm
from dlna.connection_manager import handle_soap
from dlna.descriptions import cm_scpd


NS = {"s": "urn:schemas-upnp-org:service-1-0"}


class ConnectionManagerTests(unittest.TestCase):
    def test_scpd_includes_required_actions(self) -> None:
        root = ET.fromstring(cm_scpd())
        actions = {elem.text for elem in root.findall(".//s:action/s:name", NS)}

        self.assertEqual(
            {"GetProtocolInfo", "GetCurrentConnectionIDs", "GetCurrentConnectionInfo"},
            actions,
        )

    def test_scpd_includes_required_state_variables(self) -> None:
        root = ET.fromstring(cm_scpd())
        states = {elem.text for elem in root.findall(".//s:stateVariable/s:name", NS)}

        self.assertTrue(
            {
                "SourceProtocolInfo",
                "SinkProtocolInfo",
                "CurrentConnectionIDs",
                "A_ARG_TYPE_ConnectionStatus",
                "A_ARG_TYPE_ConnectionManager",
                "A_ARG_TYPE_Direction",
                "A_ARG_TYPE_ProtocolInfo",
                "A_ARG_TYPE_ConnectionID",
                "A_ARG_TYPE_AVTransportID",
                "A_ARG_TYPE_RcsID",
            }.issubset(states)
        )

    def test_get_protocol_info(self) -> None:
        payload, status = handle_soap('"urn:schemas-upnp-org:service:ConnectionManager:1#GetProtocolInfo"', b"")

        self.assertEqual(status, 200)
        text = payload.decode("utf-8")
        self.assertIn("<u:GetProtocolInfoResponse", text)
        self.assertIn("<Source>http-get:*:video/mp4:*", text)
        self.assertNotIn("http-get:*:image/jpeg:*", text)
        self.assertIn("<Sink></Sink>", text)

    def test_source_protocol_info_can_include_images_when_enabled(self) -> None:
        original = cm.DLNA_IMAGE_ENABLED
        try:
            cm.DLNA_IMAGE_ENABLED = True
            text = cm._source_protocol_info()
        finally:
            cm.DLNA_IMAGE_ENABLED = original

        self.assertIn("http-get:*:image/jpeg:*", text)
        self.assertIn("http-get:*:image/png:*", text)

    def test_get_current_connection_ids(self) -> None:
        payload, status = handle_soap(
            '"urn:schemas-upnp-org:service:ConnectionManager:1#GetCurrentConnectionIDs"',
            b"",
        )

        self.assertEqual(status, 200)
        self.assertIn("<ConnectionIDs>0</ConnectionIDs>", payload.decode("utf-8"))

    def test_get_current_connection_info(self) -> None:
        payload, status = handle_soap(
            '"urn:schemas-upnp-org:service:ConnectionManager:1#GetCurrentConnectionInfo"',
            b"",
        )

        self.assertEqual(status, 200)
        text = payload.decode("utf-8")
        self.assertIn("<RcsID>-1</RcsID>", text)
        self.assertIn("<AVTransportID>-1</AVTransportID>", text)
        self.assertIn("<Direction>Output</Direction>", text)
        self.assertIn("<Status>Unknown</Status>", text)


if __name__ == "__main__":
    unittest.main()
