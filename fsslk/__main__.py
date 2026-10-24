"""Command line entry point for the FSSLK prototype."""

import argparse
import json
import socket
import sys
from typing import Any

import torch

from .coordinator import MAX_TASK_ATTEMPTS, TASK_TIMEOUT_SECONDS, Coordinator
from .protocol import receive_message, send_frame
from .tensor import encode_tensor
from .worker import run_worker


def _request(
    host: str,
    port: int,
    request: dict[str, Any],
    binary_payload: bytes = b"",
) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=5) as conn:
        conn.settimeout(TASK_TIMEOUT_SECONDS * MAX_TASK_ATTEMPTS + 10)
        stream = conn.makefile("rb")
        try:
            send_frame(conn, {"role": "client", **request}, binary_payload)
            response = receive_message(stream)
            if response is None:
                raise ConnectionError("Coordinator closed the connection without a response")
            return response
        finally:
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="fsslk", description="FSSLK CPU prototype")
    commands = parser.add_subparsers(dest="command", required=True)

    coordinator_parser = commands.add_parser("coordinator", help="start a coordinator")
    coordinator_parser.add_argument("--host", default="0.0.0.0")
    coordinator_parser.add_argument("--port", type=int, default=27494)
    coordinator_parser.add_argument("--udp-port", type=int)

    worker_parser = commands.add_parser("worker", help="join a coordinator as a CPU worker")
    worker_parser.add_argument("--host", default="coordinator")
    worker_parser.add_argument("--port", type=int, default=27494)
    worker_parser.add_argument("--node-id")

    status_parser = commands.add_parser("status", help="show connected workers")
    status_parser.add_argument("--host", default="127.0.0.1")
    status_parser.add_argument("--port", type=int, default=27494)

    sum_parser = commands.add_parser("run-sum", help="sum a CPU tensor across workers")
    sum_parser.add_argument("--size", type=int, default=16)
    sum_parser.add_argument("--host", default="127.0.0.1")
    sum_parser.add_argument("--port", type=int, default=27494)

    from .pipeline import add_arguments as add_pipeline_arguments

    pipeline_parser = commands.add_parser(
        "run-pipeline",
        help="run a captured demo model as a two-stage pipeline on two workers",
    )
    add_pipeline_arguments(pipeline_parser)

    args = parser.parse_args()
    try:
        if args.command == "coordinator":
            Coordinator(args.host, args.port, udp_port=args.udp_port).serve_forever()
        elif args.command == "worker":
            run_worker(args.host, args.port, args.node_id)
        elif args.command == "status":
            response = _request(args.host, args.port, {"command": "status"})
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "status request failed"))
            print(json.dumps(response["nodes"], indent=2))
        elif args.command == "run-sum":
            if args.size < 1:
                raise ValueError("--size must be at least 1")
            values = torch.arange(args.size, dtype=torch.int64, device="cpu")
            tensor_metadata, binary_payload = encode_tensor(values)
            response = _request(
                args.host,
                args.port,
                {"command": "sum", **tensor_metadata},
                binary_payload,
            )
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "distributed task failed"))
            expected = values.sum().item()
            print(f"Input elements: {args.size}")
            print(f"Distributed result: {response['result']}")
            print(f"Expected result: {expected}")
            print(f"Workers: {', '.join(response['workers'])}")
        elif args.command == "run-pipeline":
            from .pipeline import main as run_pipeline_main

            run_pipeline_main(args)
        else:
            parser.error("unknown command")
    except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
        print(f"[FSSLK] Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
