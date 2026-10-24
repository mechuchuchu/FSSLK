"""Length-prefixed message framing with an optional binary payload."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, BinaryIO


MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_HEADER_SIZE = struct.Struct("!I")


def send_frame(sock: socket.socket, message: dict[str, Any], payload: bytes = b"") -> None:
    if "payload_size" in message:
        raise ValueError("payload_size is reserved by the FSSLK frame protocol")
    if not isinstance(payload, bytes):
        raise TypeError("frame payload must be bytes")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"frame payload exceeds the {MAX_PAYLOAD_BYTES}-byte limit")

    header = dict(message)
    header["payload_size"] = len(payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError(f"frame header exceeds the {MAX_HEADER_BYTES}-byte limit")

    sock.sendall(_HEADER_SIZE.pack(len(header_bytes)) + header_bytes + payload)


def receive_frame(stream: BinaryIO) -> tuple[dict[str, Any], bytes] | None:
    encoded_header_size = _read_exact(stream, _HEADER_SIZE.size, allow_clean_eof=True)
    if encoded_header_size is None:
        return None

    (header_size,) = _HEADER_SIZE.unpack(encoded_header_size)
    if not 1 <= header_size <= MAX_HEADER_BYTES:
        raise ValueError(f"invalid frame header size: {header_size}")

    encoded_header = _read_exact(stream, header_size)
    try:
        header = json.loads(encoded_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frame header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("frame header must be a JSON object")

    payload_size = header.pop("payload_size", 0)
    if (
        not isinstance(payload_size, int)
        or isinstance(payload_size, bool)
        or not 0 <= payload_size <= MAX_PAYLOAD_BYTES
    ):
        raise ValueError(f"invalid frame payload size: {payload_size!r}")

    payload = _read_exact(stream, payload_size)
    return header, payload


def send_message(sock: socket.socket, message: dict[str, Any]) -> None:
    """Send a metadata-only frame for control messages."""
    send_frame(sock, message)


def receive_message(stream: BinaryIO) -> dict[str, Any] | None:
    """Receive a metadata-only frame, rejecting unexpected binary data."""
    frame = receive_frame(stream)
    if frame is None:
        return None
    message, payload = frame
    if payload:
        raise ValueError("unexpected binary payload in metadata-only message")
    return message


def _read_exact(
    stream: BinaryIO,
    size: int,
    *,
    allow_clean_eof: bool = False,
) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            if allow_clean_eof and not chunks:
                return None
            raise ConnectionError("connection closed before the full frame was received")
        chunks.extend(chunk)
    return bytes(chunks)
