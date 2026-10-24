"""Tests for stage spec validation and execution on a CPU Worker."""

from __future__ import annotations

import copy
import json
import unittest

import torch
from torch import nn

from fsslk.bundle import decode_bundle, encode_bundle
from fsslk.graph import capture_graph
from fsslk.stage import (
    StageError,
    StageNode,
    StageRuntime,
    StageSpec,
    TensorSpec,
    choose_cut,
    split_two_stage,
)
from fsslk.worker import _handle_stage


def matmul_stage() -> StageSpec:
    return StageSpec(
        stage_id="matmul-stage",
        inputs=(TensorSpec("x", (2, 2), "float32"),),
        params=(TensorSpec("weight", (2, 2), "float32"),),
        nodes=(
            StageNode(
                "result",
                "aten.mm.default",
                {"args": [{"ref": "x"}, {"ref": "weight"}], "kwargs": {}},
            ),
        ),
        outputs=("result",),
    )


class StageSpecTests(unittest.TestCase):
    def test_round_trip_and_runtime_execute_a_full_matrix_operation(self) -> None:
        spec = matmul_stage()
        rebuilt = StageSpec.from_dict(spec.to_dict())
        runtime = StageRuntime(rebuilt)
        weight = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        runtime.put_params({"weight": weight})
        result = runtime.run({"x": inputs})["result"]

        self.assertTrue(torch.equal(result, inputs @ weight))

    def test_autograd_returns_input_and_weight_gradients(self) -> None:
        runtime = StageRuntime(matmul_stage())
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        weight = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
        output_gradient = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
        runtime.put_params({"weight": weight})

        with torch.no_grad():
            execution = runtime.run_with_grad({"x": inputs})
        self.assertTrue(execution.outputs["result"].requires_grad)
        gradients = execution.backward({"result": output_gradient})

        torch.testing.assert_close(gradients.inputs["x"], output_gradient @ weight.t())
        torch.testing.assert_close(gradients.params["weight"], inputs.t() @ output_gradient)
        self.assertIsNone(runtime.params["weight"].grad)
        with self.assertRaisesRegex(StageError, "already been backpropagated"):
            execution.backward({"result": output_gradient})

    def test_gradients_can_be_chained_across_a_two_stage_cut(self) -> None:
        stage_a = StageRuntime(matmul_stage())
        stage_b = StageRuntime(
            StageSpec(
                stage_id="second-stage",
                inputs=(TensorSpec("activation", (2, 2), "float32"),),
                params=(TensorSpec("weight_b", (2, 2), "float32"),),
                nodes=(
                    StageNode(
                        "prediction",
                        "aten.mm.default",
                        {
                            "args": [
                                {"ref": "activation"},
                                {"ref": "weight_b"},
                            ],
                            "kwargs": {},
                        },
                    ),
                ),
                outputs=("prediction",),
            )
        )
        weight_a = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
        weight_b = torch.tensor([[0.5, 1.0], [-1.0, 0.75]])
        inputs = torch.tensor([[1.0, 2.0], [-2.0, 1.5]])
        stage_a.put_params({"weight": weight_a})
        stage_b.put_params({"weight_b": weight_b})

        execution_a = stage_a.run_with_grad({"x": inputs})
        execution_b = stage_b.run_with_grad(
            {"activation": execution_a.outputs["result"]}
        )
        prediction = execution_b.outputs["prediction"]
        gradients_b = execution_b.backward(
            {"prediction": 2 * prediction / prediction.numel()}
        )
        gradients_a = execution_a.backward(
            {"result": gradients_b.inputs["activation"]}
        )

        reference_input = inputs.detach().requires_grad_()
        reference_weight_a = weight_a.detach().requires_grad_()
        reference_weight_b = weight_b.detach().requires_grad_()
        reference_loss = ((reference_input @ reference_weight_a) @ reference_weight_b).square().mean()
        expected_input, expected_weight_a, expected_weight_b = torch.autograd.grad(
            reference_loss,
            (reference_input, reference_weight_a, reference_weight_b),
        )
        torch.testing.assert_close(gradients_a.inputs["x"], expected_input)
        torch.testing.assert_close(gradients_a.params["weight"], expected_weight_a)
        torch.testing.assert_close(gradients_b.params["weight_b"], expected_weight_b)

    def test_captured_linear_model_gradients_match_full_model_autograd(self) -> None:
        torch.manual_seed(4)
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
        inputs = torch.randn(4, 3)
        graph = capture_graph(model, (inputs,), reference=False)
        spec_a, spec_b = split_two_stage(graph, choose_cut(graph).after, "autograd")
        sources = {
            node.name: node.source
            for node in graph.nodes
            if node.kind in ("param", "buffer")
        }
        state = model.state_dict()
        weights_a = {
            item.name: state[sources[item.name]].detach().contiguous()
            for item in spec_a.params
        }
        weights_b = {
            item.name: state[sources[item.name]].detach().contiguous()
            for item in spec_b.params
        }
        runtime_a, runtime_b = StageRuntime(spec_a), StageRuntime(spec_b)
        runtime_a.put_params(weights_a)
        runtime_b.put_params(weights_b)

        execution_a = runtime_a.run_with_grad({spec_a.inputs[0].name: inputs})
        execution_b = runtime_b.run_with_grad(
            {
                tensor_spec.name: execution_a.outputs[tensor_spec.name]
                for tensor_spec in spec_b.inputs
            }
        )
        output_name = spec_b.outputs[0]
        prediction = execution_b.outputs[output_name]
        grad_prediction = 2 * prediction / prediction.numel()
        gradients_b = execution_b.backward({output_name: grad_prediction})
        gradients_a = execution_a.backward(
            {name: gradients_b.inputs[name] for name in spec_a.outputs}
        )

        reference_input = inputs.detach().requires_grad_()
        reference_output = model(reference_input)
        reference_loss = reference_output.square().mean()
        parameter_names, parameter_values = zip(*model.named_parameters())
        reference_gradients = torch.autograd.grad(
            reference_loss, (reference_input, *parameter_values)
        )
        expected_params = dict(zip(parameter_names, reference_gradients[1:]))

        torch.testing.assert_close(gradients_a.inputs[spec_a.inputs[0].name], reference_gradients[0])
        for spec, gradients in ((spec_a, gradients_a.params), (spec_b, gradients_b.params)):
            for name, gradient in gradients.items():
                torch.testing.assert_close(gradient, expected_params[sources[name]])

    def test_autograd_checks_output_gradient_names_and_shapes(self) -> None:
        runtime = StageRuntime(matmul_stage())
        runtime.put_params({"weight": torch.eye(2)})
        execution = runtime.run_with_grad({"x": torch.eye(2)})

        with self.assertRaisesRegex(StageError, "expected output gradients"):
            execution.backward({"missing": torch.eye(2)})
        with self.assertRaisesRegex(StageError, "expected"):
            execution.backward({"result": torch.ones(1)})

    def test_rejects_unknown_ops_forward_references_and_undefined_outputs(self) -> None:
        mutations = (
            lambda data: data["nodes"][0].update(op="builtins.eval"),
            lambda data: data["nodes"][0]["call"]["args"][0].update(ref="later"),
            lambda data: data.update(outputs=["missing"]),
        )
        for mutate in mutations:
            data = copy.deepcopy(matmul_stage().to_dict())
            mutate(data)
            with self.subTest(data=data), self.assertRaises(StageError):
                StageSpec.from_dict(data)

    def test_runtime_rejects_missing_weights_unknown_names_and_wrong_shapes(self) -> None:
        runtime = StageRuntime(matmul_stage())
        inputs = {"x": torch.eye(2)}

        with self.assertRaisesRegex(StageError, "weights not loaded"):
            runtime.run(inputs)
        with self.assertRaisesRegex(StageError, "unknown weight"):
            runtime.put_params({"unexpected": torch.eye(2)})
        with self.assertRaisesRegex(StageError, "expected"):
            runtime.put_params({"weight": torch.ones(3, 2)})

    def test_worker_stage_lifecycle_define_put_run_drop(self) -> None:
        spec = matmul_stage()
        stages: dict[str, StageRuntime] = {}
        defined, empty = _handle_stage(
            {"op": "define", "stage_id": spec.stage_id},
            json.dumps(spec.to_dict()).encode(),
            stages,
        )
        self.assertTrue(defined["ok"])
        self.assertEqual(defined["params_expected"], 1)
        self.assertEqual(empty, b"")

        weight_meta, weight_payload = encode_bundle(
            {"weight": torch.tensor([[2.0, 0.0], [1.0, 3.0]])}
        )
        put, _ = _handle_stage(
            {"op": "put", "stage_id": spec.stage_id, "tensors": weight_meta},
            weight_payload,
            stages,
        )
        self.assertTrue(put["ok"])
        self.assertEqual(put["params_missing"], 0)

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        input_meta, input_payload = encode_bundle({"x": inputs})
        result, result_payload = _handle_stage(
            {"op": "run", "stage_id": spec.stage_id, "tensors": input_meta},
            input_payload,
            stages,
        )
        self.assertTrue(result["ok"])
        outputs = decode_bundle(result["tensors"], result_payload)
        expected = inputs @ torch.tensor([[2.0, 0.0], [1.0, 3.0]])
        self.assertTrue(torch.equal(outputs["result"], expected))

        dropped, _ = _handle_stage(
            {"op": "drop", "stage_id": spec.stage_id}, b"", stages
        )
        self.assertTrue(dropped["ok"])
        self.assertNotIn(spec.stage_id, stages)

    def test_worker_returns_errors_without_raising_or_removing_stage(self) -> None:
        spec = matmul_stage()
        stages = {spec.stage_id: StageRuntime(spec)}

        reply, payload = _handle_stage(
            {"op": "run", "stage_id": spec.stage_id, "tensors": []}, b"", stages
        )

        self.assertFalse(reply["ok"])
        self.assertIn("weights not loaded", reply["error"])
        self.assertEqual(payload, b"")
        self.assertIn(spec.stage_id, stages)


if __name__ == "__main__":
    unittest.main()
