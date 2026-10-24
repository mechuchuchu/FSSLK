"""In-process tests for graph deployment and two-stage microbatch execution."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from fsslk.bundle import encode_bundle
from fsslk.pipeline import PipelineError, _ActivationSpool, run_pipeline
from fsslk.stage import StageRuntime


class ActivationSpoolTests(unittest.TestCase):
    def test_activation_bundle_is_written_read_once_and_removed(self) -> None:
        metadata, payload = encode_bundle(
            {"activation": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
        )

        with _ActivationSpool() as spool:
            activation = spool.write(metadata, payload)
            root = activation.path.parent
            self.assertTrue(activation.path.is_file())
            restored = spool.read(activation)
            expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            self.assertTrue(torch.equal(restored["activation"], expected))
            self.assertFalse(activation.path.exists())

        self.assertFalse(root.exists())


class PipelineTests(unittest.TestCase):
    def test_two_stage_pipeline_preserves_output_and_streams_microbatches(self) -> None:
        model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)).eval()
        example = torch.randn(8, 3)
        runtimes: dict[tuple[str, str], StageRuntime] = {}
        stage_calls: list[tuple[str, str]] = []

        def fake_deploy(host, port, worker, spec, weights):
            runtime = StageRuntime(spec)
            runtime.put_params(weights)
            runtimes[(worker, spec.stage_id)] = runtime
            return 0.001

        def fake_run_stage(host, port, worker, stage_id, inputs):
            stage_calls.append((worker, stage_id))
            if worker == "worker-b":
                time.sleep(0.01)
            outputs = runtimes[(worker, stage_id)].run(inputs)
            sent = sum(t.numel() * t.element_size() for t in inputs.values())
            received = sum(t.numel() * t.element_size() for t in outputs.values())
            return outputs, {
                "wall": 0.01,
                "compute": 0.005,
                "sent_bytes": sent,
                "received_bytes": received,
            }

        def fake_run_stage_payload(host, port, worker, stage_id, inputs):
            outputs, record = fake_run_stage(host, port, worker, stage_id, inputs)
            metadata, payload = encode_bundle(outputs)
            return metadata, payload, record

        def fake_call_worker(host, port, worker, command, fields=None, payload=b""):
            self.assertEqual(command, "stage_drop")
            runtimes.pop((worker, fields["stage_id"]), None)
            return {"ok": True}, b""

        original_spool_write = _ActivationSpool.write
        spool_lock = threading.Lock()
        max_spooled_files = [0]

        def tracked_spool_write(spool, metadata, payload):
            activation = original_spool_write(spool, metadata, payload)
            with spool_lock:
                current_files = sum(1 for _path in spool._root.glob("*.bin"))
                max_spooled_files[0] = max(max_spooled_files[0], current_files)
            return activation

        statuses = {
            "worker-a": {"rtt_ms": 2.0, "bandwidth_mbps": 100.0, "packet_loss_pct": 0.0},
            "worker-b": {"rtt_ms": 8.0, "bandwidth_mbps": 50.0, "packet_loss_pct": 1.0},
        }
        with (
            patch("fsslk.pipeline.wait_for_workers", return_value=statuses),
            patch("fsslk.pipeline.deploy_stage", side_effect=fake_deploy),
            patch("fsslk.pipeline.run_stage", side_effect=fake_run_stage),
            patch("fsslk.pipeline.run_stage_payload", side_effect=fake_run_stage_payload),
            patch("fsslk.pipeline._ActivationSpool.write", new=tracked_spool_write),
            patch("fsslk.pipeline.call_worker", side_effect=fake_call_worker) as call_worker,
        ):
            result = run_pipeline(
                model,
                example,
                workers=("worker-a", "worker-b"),
                microbatches=8,
                repeats=1,
                log=lambda _message: None,
            )

        with torch.no_grad():
            expected = model(example)
        torch.testing.assert_close(result.output, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(result.microbatches, 8)
        self.assertEqual(result.workers, ("worker-a", "worker-b"))
        self.assertEqual(len(result.run_seconds), 1)
        self.assertEqual(len(result.calls), 16)  # A and B for each of eight chunks.
        self.assertEqual(len(stage_calls), 32)  # warm-up plus one measured pass.
        self.assertLessEqual(max_spooled_files[0], 3)  # reading, queued, and producer payloads.
        self.assertEqual(call_worker.call_count, 2)  # both deployed stages are dropped.
        self.assertEqual(runtimes, {})

    def test_rejects_training_models_and_batches_that_do_not_split_evenly(self) -> None:
        training_model = nn.Linear(3, 2).train()
        with self.assertRaisesRegex(PipelineError, "eval mode"):
            run_pipeline(training_model, torch.randn(4, 3))

        eval_model = nn.Linear(3, 2).eval()
        with self.assertRaisesRegex(PipelineError, "does not split"):
            run_pipeline(eval_model, torch.randn(3, 3), microbatches=2)
        with self.assertRaisesRegex(PipelineError, "repeats must be at least 1"):
            run_pipeline(eval_model, torch.randn(4, 3), repeats=0)

    def test_drops_both_stages_if_stage_deployment_fails(self) -> None:
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
        statuses = {
            "worker-a": {"rtt_ms": 1.0, "bandwidth_mbps": 100.0},
            "worker-b": {"rtt_ms": 1.0, "bandwidth_mbps": 100.0},
        }
        drops: list[tuple[str, str]] = []

        def fail_deploy(host, port, worker, spec, weights):
            raise RuntimeError("deployment failed")

        def record_drop(host, port, worker, command, fields=None, payload=b""):
            drops.append((worker, fields["stage_id"]))
            return {"ok": True}, b""

        with (
            patch("fsslk.pipeline.wait_for_workers", return_value=statuses),
            patch("fsslk.pipeline.deploy_stage", side_effect=fail_deploy),
            patch("fsslk.pipeline.call_worker", side_effect=record_drop),
        ):
            with self.assertRaisesRegex(RuntimeError, "deployment failed"):
                run_pipeline(
                    model,
                    torch.randn(4, 3),
                    workers=("worker-a", "worker-b"),
                    repeats=1,
                    log=lambda _message: None,
                )

        dropped_workers = {worker for worker, _stage_id in drops}
        self.assertEqual(dropped_workers, {"worker-a", "worker-b"})
        self.assertEqual(len(drops), 2)

    def test_stage_b_failure_stops_producer_and_cleans_spool_files(self) -> None:
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
        example = torch.randn(8, 3)
        runtimes: dict[tuple[str, str], StageRuntime] = {}

        def fake_deploy(host, port, worker, spec, weights):
            runtime = StageRuntime(spec)
            runtime.put_params(weights)
            runtimes[(worker, spec.stage_id)] = runtime
            return 0.001

        def fake_run_stage_payload(host, port, worker, stage_id, inputs):
            outputs = runtimes[(worker, stage_id)].run(inputs)
            metadata, payload = encode_bundle(outputs)
            return metadata, payload, {
                "wall": 0.01,
                "compute": 0.005,
                "sent_bytes": 1,
                "received_bytes": len(payload),
            }

        def failed_stage_b(host, port, worker, stage_id, inputs):
            raise RuntimeError("stage B failed")

        def fake_drop(host, port, worker, command, fields=None, payload=b""):
            runtimes.pop((worker, fields["stage_id"]), None)
            return {"ok": True}, b""

        statuses = {
            "worker-a": {"rtt_ms": 1.0, "bandwidth_mbps": 100.0},
            "worker-b": {"rtt_ms": 1.0, "bandwidth_mbps": 100.0},
        }
        with tempfile.TemporaryDirectory() as spool_root:
            with (
                patch.dict(os.environ, {"FSSLK_ACTIVATION_SPOOL_DIR": spool_root}),
                patch("fsslk.pipeline.wait_for_workers", return_value=statuses),
                patch("fsslk.pipeline.deploy_stage", side_effect=fake_deploy),
                patch("fsslk.pipeline.run_stage_payload", side_effect=fake_run_stage_payload),
                patch("fsslk.pipeline.run_stage", side_effect=failed_stage_b),
                patch("fsslk.pipeline.call_worker", side_effect=fake_drop),
            ):
                with self.assertRaisesRegex(PipelineError, "stage B failed"):
                    run_pipeline(
                        model,
                        example,
                        workers=("worker-a", "worker-b"),
                        microbatches=8,
                        repeats=1,
                        log=lambda _message: None,
                    )

            self.assertEqual(os.listdir(spool_root), [])
        self.assertEqual(runtimes, {})


if __name__ == "__main__":
    unittest.main()
