"""Tests for tensor metadata and raw-byte conversion."""

from __future__ import annotations

import unittest

import torch

from fsslk.tensor import decode_tensor, encode_tensor


class TensorCodecTests(unittest.TestCase):
    def test_int64_round_trip_copies_only_the_selected_view(self) -> None:
        source = torch.arange(12, dtype=torch.int64)[3:8]

        metadata, payload = encode_tensor(source)
        decoded = decode_tensor(metadata, payload)

        self.assertEqual(metadata, {"dtype": "int64", "shape": [5]})
        self.assertEqual(len(payload), 5 * torch.tensor([], dtype=torch.int64).element_size())
        self.assertTrue(torch.equal(decoded, source))

    def test_float32_round_trip_preserves_shape(self) -> None:
        source = torch.tensor([[1.25, -2.5], [3.0, 4.75]], dtype=torch.float32)

        metadata, payload = encode_tensor(source)
        decoded = decode_tensor(metadata, payload)

        self.assertEqual(metadata, {"dtype": "float32", "shape": [2, 2]})
        self.assertTrue(torch.equal(decoded, source))

    def test_unsupported_dtype_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported tensor dtype"):
            encode_tensor(torch.tensor([1], dtype=torch.int32))

    def test_inconsistent_shape_or_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload size mismatch"):
            decode_tensor({"dtype": "int64", "shape": [2]}, b"\x00" * 8)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            decode_tensor({"dtype": "int64", "shape": [True]}, b"\x00" * 8)


if __name__ == "__main__":
    unittest.main()
