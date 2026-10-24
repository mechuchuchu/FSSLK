"""Tests for binary bundles used by the stage protocol."""

from __future__ import annotations

import unittest

import torch

from fsslk.bundle import decode_bundle, encode_bundle, split_bundles


class BundleTests(unittest.TestCase):
    def test_round_trip_preserves_named_tensors_and_noncontiguous_views(self) -> None:
        source = {
            "matrix": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
            "indices": torch.tensor([3, 5, 8], dtype=torch.int64),
        }

        metadata, payload = encode_bundle(source)
        decoded = decode_bundle(metadata, payload)

        self.assertEqual(set(decoded), set(source))
        for name in source:
            self.assertTrue(torch.equal(decoded[name], source[name]))
            self.assertEqual(decoded[name].device.type, "cpu")

    def test_rejects_duplicate_names_and_payload_size_mismatches(self) -> None:
        metadata, payload = encode_bundle({"x": torch.arange(3, dtype=torch.float32)})

        with self.assertRaisesRegex(ValueError, "duplicate"):
            decode_bundle([metadata[0], metadata[0]], payload + payload)
        with self.assertRaisesRegex(ValueError, "shorter"):
            decode_bundle(metadata, payload[:-1])
        with self.assertRaisesRegex(ValueError, "longer"):
            decode_bundle(metadata, payload + b"x")

    def test_split_bundles_respects_byte_and_item_limits(self) -> None:
        tensors = {
            "a": torch.zeros(2, dtype=torch.float32),
            "b": torch.zeros(2, dtype=torch.float32),
            "c": torch.zeros(1, dtype=torch.float32),
        }

        bundles = split_bundles(tensors, max_bytes=12, max_items=2)

        self.assertEqual([list(bundle) for bundle in bundles], [["a"], ["b", "c"]])

    def test_tensor_larger_than_target_bundle_is_kept_alone(self) -> None:
        tensor = torch.zeros(4, dtype=torch.float32)

        bundles = split_bundles({"large": tensor}, max_bytes=8)

        self.assertEqual(len(bundles), 1)
        self.assertEqual(list(bundles[0]), ["large"])


if __name__ == "__main__":
    unittest.main()
