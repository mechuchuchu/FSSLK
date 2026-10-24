"""Tests for length-prefixed protocol frames."""

from __future__ import annotations

import json
import socket
import struct
import unittest

from fsslk.protocol import MAX_HEADER_BYTES, MAX_PAYLOAD_BYTES, receive_frame, receive_message, send_frame


class ProtocolFrameTests(unittest.TestCase):
    def exchange(self, message: dict[str, object], payload: bytes = b"") -> tuple[dict[str, object], bytes]:
        sender, receiver = socket.socketpair()
        stream = receiver.makefile("rb")
        try:
            send_frame(sender, message, payload)
            sender.shutdown(socket.SHUT_WR)
            frame = receive_frame(stream)
            self.assertIsNotNone(frame)
            assert frame is not None
            self.assertIsNone(receive_frame(stream))
            return frame
        finally:
            stream.close()
            sender.close()
            receiver.close()

    def test_frame_round_trip_preserves_metadata_and_binary_payload(self) -> None:
        message, payload = self.exchange(
            {"type": "tensor", "dtype": "int64", "shape": [3]},
            b"\x01\x02\x03\x04",
        )
        self.assertEqual(message, {"type": "tensor", "dtype": "int64", "shape": [3]})
        self.assertEqual(payload, b"\x01\x02\x03\x04")

    def test_truncated_payload_is_rejected(self) -> None:
        sender, receiver = socket.socketpair()
        stream = receiver.makefile("rb")
        header = json.dumps({"payload_size": 4}).encode("utf-8")
        try:
            sender.sendall(struct.pack("!I", len(header)) + header + b"ab")
            sender.shutdown(socket.SHUT_WR)
            with self.assertRaisesRegex(ConnectionError, "full frame"):
                receive_frame(stream)
        finally:
            stream.close()
            sender.close()
            receiver.close()

    def test_oversized_frame_header_and_payload_are_rejected(self) -> None:
        sender, receiver = socket.socketpair()
        stream = receiver.makefile("rb")
        header = json.dumps({"payload_size": MAX_PAYLOAD_BYTES + 1}).encode("utf-8")
        try:
            sender.sendall(struct.pack("!I", len(header)) + header)
            sender.shutdown(socket.SHUT_WR)
            with self.assertRaisesRegex(ValueError, "payload size"):
                receive_frame(stream)
        finally:
            stream.close()
            sender.close()
            receiver.close()

        sender, receiver = socket.socketpair()
        stream = receiver.makefile("rb")
        try:
            sender.sendall(struct.pack("!I", MAX_HEADER_BYTES + 1))
            sender.shutdown(socket.SHUT_WR)
            with self.assertRaisesRegex(ValueError, "header size"):
                receive_frame(stream)
        finally:
            stream.close()
            sender.close()
            receiver.close()

    def test_metadata_only_receiver_rejects_binary_payload(self) -> None:
        sender, receiver = socket.socketpair()
        stream = receiver.makefile("rb")
        try:
            send_frame(sender, {"type": "control"}, b"data")
            sender.shutdown(socket.SHUT_WR)
            with self.assertRaisesRegex(ValueError, "unexpected binary payload"):
                receive_message(stream)
        finally:
            stream.close()
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
