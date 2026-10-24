"""Small authenticated UDP echo datagrams used to estimate probe loss."""

from __future__ import annotations

import struct


UDP_PROBE_MAGIC = b"FSLK"
UDP_PROBE_PACKET = struct.Struct("!4s16s16sI")


def encode_probe_packet(token: bytes, probe_id: bytes, sequence: int) -> bytes:
    if len(token) != 16 or len(probe_id) != 16:
        raise ValueError("UDP probe tokens and IDs must be 16 bytes")
    return UDP_PROBE_PACKET.pack(UDP_PROBE_MAGIC, token, probe_id, sequence)


def decode_probe_packet(packet: bytes) -> tuple[bytes, bytes, int] | None:
    if len(packet) != UDP_PROBE_PACKET.size:
        return None
    magic, token, probe_id, sequence = UDP_PROBE_PACKET.unpack(packet)
    if magic != UDP_PROBE_MAGIC:
        return None
    return token, probe_id, sequence
