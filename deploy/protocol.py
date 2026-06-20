"""Tiny JSON-over-UDP messaging for the deploy bridge (pure stdlib).

Messages are small JSON objects. UDP is fine here: at 30 Hz on localhost loss is
negligible and we always act on the latest packet (stale packets are dropped by
draining the socket buffer).

Client -> bridge (state):
    {"seq": int, "t": float, "q_rad": [6 floats]}   # joints in Isaac order/units

Bridge -> client (command):
    {"seq": int, "target_rad": [5 floats], "estop": bool}  # arm joint targets (Isaac)
"""

from __future__ import annotations

import json
import socket


def make_socket(bind_port: int, timeout: float = 0.0) -> socket.socket:
    """UDP socket bound to ``bind_port`` on all interfaces."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", bind_port))
    sock.setblocking(False)
    if timeout > 0:
        sock.settimeout(timeout)
    return sock


def send(sock: socket.socket, host: str, port: int, msg: dict) -> None:
    sock.sendto(json.dumps(msg).encode("utf-8"), (host, port))


def recv_latest(sock: socket.socket, bufsize: int = 4096) -> dict | None:
    """Return the most recent pending message, draining older ones. None if empty."""
    latest = None
    while True:
        try:
            data, _ = sock.recvfrom(bufsize)
        except (BlockingIOError, socket.timeout):
            break
        except OSError:
            break
        if not data:
            break
        try:
            latest = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            continue
    return latest
