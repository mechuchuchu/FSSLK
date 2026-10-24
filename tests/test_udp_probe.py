"""Tests for the UDP packet-loss probe wire format."""

from __future__ import annotations

import unittest

from fsslk.udp_probe import decode_probe_packet, encode_probe_packet


class UdpProbeTests(unittest.TestCase):
    def test_probe_packet_round_trip(self) -> None:
        token = b"t" * 16
        probe_id = b"p" * 16
        packet = encode_probe_packet(token, probe_id, 7)

        self.assertEqual(decode_probe_packet(packet), (token, probe_id, 7))

    def test_invalid_probe_packet_is_ignored(self) -> None:
        self.assertIsNone(decode_probe_packet(b"invalid"))
        packet = bytearray(encode_probe_packet(b"t" * 16, b"p" * 16, 7))
        packet[0] ^= 0xFF
        self.assertIsNone(decode_probe_packet(bytes(packet)))


if __name__ == "__main__":
    unittest.main()
