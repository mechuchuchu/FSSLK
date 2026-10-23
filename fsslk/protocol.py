"""Small newline-delimited JSON protocol used by the prototype."""

import json
import socket
from typing import Any, BinaryIO


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
    sock.sendall(payload)


def receive_message(stream: BinaryIO) -> dict[str, Any] | None:
    line = stream.readline()
    if not line:
        return None
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("Protocol messages must be JSON objects")
    return message
