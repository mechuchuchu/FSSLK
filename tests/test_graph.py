import contextlib
import dataclasses
import io
import json
import logging
import unittest
import warnings

import torch
from torch import nn

import fsslk
from fsslk.graph import (
    CaptureError,
    CapturedGraph,
    GraphNode,
    LinkProfile,
    TensorInfo,
    capture_graph,
    wrap,
)
from fsslk.graph_demo import Block
from fsslk.protocol import MAX_PAYLOAD_BYTES
from fsslk.scheduler import NodeCapacity


def mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).eval()


class Residual(nn.Module):
    """x flows around fc1/relu/fc2, so a cut inside keeps two tensors live."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.relu(self.fc1(x)))


class Branchy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:  # data-dependent control flow cannot be exported
            return x + 1
        return x - 1


class WrapTests(unittest.TestCase):
    def test_wrap_does_not_change_the_output(self) -> None:
        model, x = mlp(), torch.randn(2, 8)
        wrapped = wrap(model, x)
        with torch.no_grad():
            self.assertTrue(torch.equal(wrapped(x), model(x)))

    def test_wrap_is_exposed_lazily_on_the_package(self) -> None:
        self.assertIs(fsslk.wrap, wrap)
        with self.assertRaises(AttributeError):
            fsslk.does_not_exist

    def test_graph_is_captured_on_first_forward_without_examples(self) -> None:
        wrapped = wrap(mlp())
        with self.assertRaises(RuntimeError):
            wrapped.graph
        wrapped(torch.randn(2, 8))
        self.assertEqual(len(wrapped.graph.compute_nodes), 3)

    def quiet_torch_export_logs(self) -> None:
        # torch itself logs a long diagnostic when export hits data-dependent control flow.
        logger = logging.getLogger("torch.fx.experimental.symbolic_shapes")
        previous = logger.level
        logger.setLevel(logging.CRITICAL)
        self.addCleanup(logger.setLevel, previous)

    def test_failed_lazy_capture_warns_once_and_still_runs_the_model(self) -> None:
        self.quiet_torch_export_logs()
        wrapped = wrap(Branchy())
        x = torch.ones(3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(torch.equal(wrapped(x), x + 1))
            self.assertTrue(torch.equal(wrapped(x), x + 1))
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(wrapped.capture_error, CaptureError)

    def test_failed_explicit_capture_raises(self) -> None:
        self.quiet_torch_export_logs()
        with self.assertRaises(CaptureError):
            wrap(Branchy(), torch.ones(3))

    def test_failed_capture_does_not_dump_torchs_graph_to_stderr(self) -> None:
        self.quiet_torch_export_logs()
        console = io.StringIO()
        with contextlib.redirect_stderr(console), self.assertRaises(CaptureError) as caught:
            wrap(Branchy(), torch.ones(3))
        self.assertNotIn("def forward", console.getvalue())
        self.assertIsInstance(caught.exception.diagnostics, str)

    def test_wrapping_a_training_module_leaves_its_state_alone(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).train()
        batch_norm = model[1]
        mean = batch_norm.running_mean.clone()
        tracked = batch_norm.num_batches_tracked.clone()
        x = torch.randn(5, 4)

        graph = wrap(model, x).graph

        self.assertTrue(torch.equal(batch_norm.running_mean, mean))
        self.assertTrue(torch.equal(batch_norm.num_batches_tracked, tracked))
        self.assertIsNone(graph.reference_flops)  # no real forward pass in training mode

    def test_capture_does_not_consume_global_randomness(self) -> None:
        class Noisy(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + torch.rand_like(x)  # draws random numbers even in eval mode

        x = torch.randn(3, 4)
        torch.manual_seed(7)
        expected_next = torch.rand(1)
        torch.manual_seed(7)

        graph = wrap(Noisy().eval(), x).graph

        self.assertTrue(torch.equal(torch.rand(1), expected_next))
        self.assertIsNotNone(graph.reference_flops)  # the real pass did run


class CaptureTests(unittest.TestCase):
    def test_nodes_describe_shapes_dtypes_bytes_and_flops(self) -> None:
        graph = capture_graph(mlp(), (torch.randn(2, 8),))
        first = graph.compute_nodes[0]
        self.assertEqual(first.op, "aten.linear.default")
        self.assertEqual(first.outputs, (TensorInfo((2, 16), "float32", 128),))
        self.assertEqual(first.flops, 2 * 2 * 16 * 8)
        kinds = [node.kind for node in graph.nodes]
        self.assertEqual(kinds.count("param"), 4)
        self.assertEqual(kinds.count("input"), 1)
        self.assertEqual(kinds[-1], "output")
        self.assertEqual(graph.output_bytes, 2 * 4 * 4)

    def test_flops_match_flop_counter_mode(self) -> None:
        torch.manual_seed(0)
        conv = nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 4, 3, padding=1)
        ).eval()
        cases = {
            "mlp": (mlp(), torch.randn(2, 8)),
            "conv": (conv, torch.randn(2, 3, 10, 10)),
            "attention": (nn.Sequential(Block(32, 4), Block(32, 4)).eval(), torch.randn(2, 6, 32)),
        }
        for name, (model, x) in cases.items():
            with self.subTest(name):
                graph = capture_graph(model, (x,))
                self.assertGreater(graph.total_flops, 0)
                self.assertEqual(graph.total_flops, graph.reference_flops)

    def test_multi_output_ops_are_costed_by_all_their_outputs(self) -> None:
        graph = capture_graph(nn.Sequential(Block(32, 4)).eval(), (torch.randn(2, 6, 32),))
        unbind = [node for node in graph.nodes if node.op == "aten.unbind.int"]
        self.assertEqual(len(unbind), 1)
        self.assertEqual(len(unbind[0].outputs), 3)

    def test_wire_protocol_checks(self) -> None:
        half = nn.Linear(4, 4).eval().to(torch.float16)
        graph = capture_graph(half, (torch.randn(2, 4, dtype=torch.float16),), reference=False)
        self.assertEqual(graph.unsupported_dtypes(), ["float16"])
        self.assertEqual(capture_graph(mlp(), (torch.randn(2, 8),)).unsupported_dtypes(), [])

        huge = TensorInfo((1,), "float32", MAX_PAYLOAD_BYTES + 1)
        node = GraphNode("x", "input", "placeholder", (), (huge,))
        self.assertEqual(
            CapturedGraph("m", (node,)).oversized_tensors(), [("x", MAX_PAYLOAD_BYTES + 1)]
        )


class CutTests(unittest.TestCase):
    def test_cut_inside_a_residual_keeps_two_tensors_live(self) -> None:
        graph = capture_graph(Residual().eval(), (torch.randn(2, 8),))
        linear, relu, linear_1, add = graph.compute_nodes
        cuts = {cut.after: cut for cut in graph.cut_points()}
        self.assertEqual(set(cuts), {linear.name, relu.name, linear_1.name})

        # after relu: the (2, 32) activation plus the residual input (2, 8)
        self.assertEqual(cuts[relu.name].transfer_bytes, 2 * 32 * 4 + 2 * 8 * 4)
        self.assertEqual(len(cuts[relu.name].crossing), 2)
        # after fc2: fc2's output plus the residual input
        self.assertEqual(cuts[linear_1.name].transfer_bytes, 2 * 8 * 4 + 2 * 8 * 4)

    def test_flops_and_weights_are_split_per_stage(self) -> None:
        graph = capture_graph(Residual().eval(), (torch.randn(2, 8),))
        cut = next(cut for cut in graph.cut_points() if cut.after == graph.compute_nodes[1].name)
        self.assertEqual(cut.flops_before + cut.flops_after, graph.total_flops)
        self.assertEqual(cut.param_bytes_before, (32 * 8 + 32) * 4)  # fc1 only
        self.assertEqual(cut.param_bytes_after, (8 * 32 + 8) * 4)  # fc2 only

    def test_cheapest_balanced_cut_of_a_transformer_is_the_residual_stream(self) -> None:
        batch, seq, dim = 2, 6, 32
        model = nn.Sequential(Block(dim, 4), Block(dim, 4)).eval()
        graph = capture_graph(model, (torch.randn(batch, seq, dim),))
        balanced = graph.cut_points(min_share=0.3)
        self.assertTrue(balanced)
        cheapest = min(balanced, key=lambda cut: cut.transfer_bytes)
        self.assertEqual(cheapest.transfer_bytes, batch * seq * dim * 4)
        self.assertEqual(len(cheapest.crossing), 1)

    def test_min_share_filters_unbalanced_cuts(self) -> None:
        graph = capture_graph(nn.Sequential(Block(32, 4), Block(32, 4)).eval(), (torch.randn(2, 6, 32),))
        everything = graph.cut_points()
        balanced = graph.cut_points(min_share=0.4)
        self.assertLess(len(balanced), len(everything))
        for cut in balanced:
            self.assertGreaterEqual(cut.flops_share_before, 0.4)
            self.assertLessEqual(cut.flops_share_before, 0.6)
        with self.assertRaises(ValueError):
            graph.cut_points(min_share=0.7)


class LinkTests(unittest.TestCase):
    def test_transfer_time_uses_the_scheduler_loss_model(self) -> None:
        link = LinkProfile("l", bandwidth_mbps=8, rtt_ms=50, packet_loss_pct=10)
        # 1 MB at 8 Mbps = 1 s, stretched by 1 / 0.9, plus one RTT
        self.assertAlmostEqual(link.transfer_seconds(1_000_000), 0.05 + 1 / 0.9)

    def test_loss_never_drops_throughput_below_one_percent(self) -> None:
        link = LinkProfile("l", bandwidth_mbps=8, packet_loss_pct=100)
        self.assertAlmostEqual(link.transfer_seconds(1_000_000), 100.0)

    def test_from_capacity(self) -> None:
        capacity = NodeCapacity("n1", 1.0, bandwidth_mbps=50, packet_loss_pct=1, rtt_ms=25)
        self.assertEqual(LinkProfile.from_capacity(capacity), LinkProfile("n1", 50, 25, 1))
        with self.assertRaises(ValueError):
            LinkProfile.from_capacity(NodeCapacity("n2", 1.0))


class SerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = capture_graph(Residual().eval(), (torch.randn(2, 8),))

    def test_json_round_trip(self) -> None:
        self.assertEqual(CapturedGraph.from_json(self.graph.to_json()), self.graph)

    def test_rejects_malformed_graphs(self) -> None:
        def mutated(change) -> dict:
            data = json.loads(self.graph.to_json())
            change(data)
            return data

        bad = {
            "version": lambda d: d.update(version=99),
            "kind": lambda d: d["nodes"][0].update(kind="exec"),
            "negative flops": lambda d: d["nodes"][-2].update(flops=-1),
            "bool flops": lambda d: d["nodes"][-2].update(flops=True),
            "duplicate name": lambda d: d["nodes"][1].update(name=d["nodes"][0]["name"]),
            "forward reference": lambda d: d["nodes"][0].update(inputs=[d["nodes"][-1]["name"]]),
            "bad shape": lambda d: d["nodes"][-2]["outputs"][0].update(shape=[2, True]),
            "too many dims": lambda d: d["nodes"][-2]["outputs"][0].update(shape=[1] * 33),
            "nodes not a list": lambda d: d.update(nodes="x"),
        }
        for label, change in bad.items():
            with self.subTest(label), self.assertRaises(ValueError):
                CapturedGraph.from_dict(mutated(change))
        with self.assertRaises(ValueError):
            CapturedGraph.from_dict([])


class ReportTests(unittest.TestCase):
    def test_report_lists_cuts_and_links(self) -> None:
        graph = capture_graph(nn.Sequential(Block(32, 4), Block(32, 4)).eval(), (torch.randn(2, 6, 32),))
        text = graph.report()
        for expected in (
            "FSSLK graph: Sequential",
            "matches FlopCounterMode",
            "Heaviest nodes",
            "Cheapest 2-stage cuts",
            "fast",
            "medium",
            "slow",
            "Wire protocol: ok",
        ):
            self.assertIn(expected, text)

    def test_report_flags_disagreement_and_wire_problems(self) -> None:
        graph = capture_graph(mlp(), (torch.randn(2, 8),))
        skewed = dataclasses.replace(graph, reference_flops=graph.total_flops + 1)
        self.assertIn("disagree", skewed.report())
        no_check = dataclasses.replace(graph, reference_flops=None)
        self.assertIn("no FlopCounterMode cross-check", no_check.report())

        half = nn.Linear(4, 4).eval().to(torch.float16)
        graph16 = capture_graph(half, (torch.randn(2, 4, dtype=torch.float16),), reference=False)
        self.assertIn("NOT ready", graph16.report())


if __name__ == "__main__":
    unittest.main()
