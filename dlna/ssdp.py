"""Minimal SSDP responder and notifier for UPnP/DLNA discovery.

The thread listens for M-SEARCH on 239.255.255.250:1900 and periodically sends
ssdp:alive notifications for the media-server device and its two services. On
shutdown it broadcasts ssdp:byebye so clients can drop stale entries quickly.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
import random

from config import (
    DEVICE_USN,
    HTTP_PORT,
    LAN_IP,
    SERVER_NAME,
    SSDP_INTERVAL_SEC,
)
from utils.logger import get

log = get("ssdp")

MCAST_GRP = "239.255.255.250"
MCAST_PORT = 1900

# Search targets advertised by this server.
TARGETS = [
    "upnp:rootdevice",
    DEVICE_USN,  # uuid:<uuid>
    "urn:schemas-upnp-org:device:MediaServer:1",
    "urn:schemas-upnp-org:service:ContentDirectory:1",
    "urn:schemas-upnp-org:service:ConnectionManager:1",
]


def _location() -> str:
    return f"http://{LAN_IP}:{HTTP_PORT}/description.xml"


def _server_header() -> str:
    # OS/UPnP/Product
    return f"Windows/10 UPnP/1.0 {SERVER_NAME}/1.0"


def _usn_for(nt: str) -> str:
    if nt == DEVICE_USN:
        return DEVICE_USN
    return f"{DEVICE_USN}::{nt}"


def _build_response(st: str) -> bytes:
    msg = (
        "HTTP/1.1 200 OK\r\n"
        "CACHE-CONTROL: max-age=1800\r\n"
        "EXT:\r\n"
        f"LOCATION: {_location()}\r\n"
        f"SERVER: {_server_header()}\r\n"
        f"ST: {st}\r\n"
        f"USN: {_usn_for(st)}\r\n"
        "DATE: " + time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()) + "\r\n"
        "\r\n"
    )
    return msg.encode("utf-8")


def _build_notify(nt: str, alive: bool = True) -> bytes:
    if alive:
        msg = (
            "NOTIFY * HTTP/1.1\r\n"
            f"HOST: {MCAST_GRP}:{MCAST_PORT}\r\n"
            "CACHE-CONTROL: max-age=1800\r\n"
            f"LOCATION: {_location()}\r\n"
            f"SERVER: {_server_header()}\r\n"
            "NTS: ssdp:alive\r\n"
            f"NT: {nt}\r\n"
            f"USN: {_usn_for(nt)}\r\n"
            "\r\n"
        )
    else:
        msg = (
            "NOTIFY * HTTP/1.1\r\n"
            f"HOST: {MCAST_GRP}:{MCAST_PORT}\r\n"
            "NTS: ssdp:byebye\r\n"
            f"NT: {nt}\r\n"
            f"USN: {_usn_for(nt)}\r\n"
            "\r\n"
        )
    return msg.encode("utf-8")


class SSDPServer(threading.Thread):
    """Background SSDP service used by the FastAPI process."""
    def __init__(self):
        super().__init__(name="ssdp", daemon=True)
        self._stop = threading.Event()
        self._sock: socket.socket | None = None

    # ---- M-SEARCH ----
    def _recv_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)  # type: ignore
        except (AttributeError, OSError):
            pass
        s.bind(("", MCAST_PORT))
        mreq = struct.pack("=4s4s", socket.inet_aton(MCAST_GRP), socket.inet_aton(LAN_IP))
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(LAN_IP))
        self._sock = s
        log.info("SSDP listening on %s:%d (iface %s)", MCAST_GRP, MCAST_PORT, LAN_IP)

        while not self._stop.is_set():
            try:
                data, addr = s.recvfrom(2048)
            except OSError:
                break
            try:
                self._handle(data, addr)
            except Exception as e:
                log.warning("ssdp handle error: %s", e)

    def _handle(self, data: bytes, addr):
        text = data.decode("utf-8", errors="ignore")
        if not text.startswith("M-SEARCH"):
            return
        # Parse the requested search target from the M-SEARCH packet.
        st = ""
        mx = 1.0
        for line in text.split("\r\n"):
            lower = line.lower()
            if lower.startswith("st:"):
                st = line.split(":", 1)[1].strip()
            elif lower.startswith("mx:"):
                try:
                    mx = float(line.split(":", 1)[1].strip())
                except ValueError:
                    mx = 1.0
        if not st:
            return
        # ssdp:all expects one response for each target we advertise.
        replies: list[str] = []
        if st == "ssdp:all":
            replies = TARGETS
        elif st in TARGETS or st.startswith("uuid:"):
            replies = [st if st in TARGETS else DEVICE_USN]
        if not replies:
            return
        delay = random.uniform(0.0, max(0.0, min(mx, 2.0)))
        if delay > 0:
            if self._stop.wait(delay):
                return
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for s in replies:
            out.sendto(_build_response(s), addr)
        out.close()
        log.debug("M-SEARCH from %s ST=%s -> replied %d", addr, st, len(replies))

    # ---- NOTIFY alive ----
    def _notify_loop(self):
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
        sender.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(LAN_IP))
        # Send a burst at startup; many clients only listen briefly.
        for _ in range(3):
            self._broadcast(sender, alive=True)
            time.sleep(0.3)
        while not self._stop.wait(SSDP_INTERVAL_SEC):
            self._broadcast(sender, alive=True)
        # bye
        self._broadcast(sender, alive=False)
        sender.close()

    def _broadcast(self, sender: socket.socket, alive: bool):
        for nt in TARGETS:
            try:
                sender.sendto(_build_notify(nt, alive), (MCAST_GRP, MCAST_PORT))
            except OSError as e:
                log.debug("notify error: %s", e)

    def run(self):
        threading.Thread(target=self._notify_loop, name="ssdp-notify", daemon=True).start()
        self._recv_loop()

    def stop(self):
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
