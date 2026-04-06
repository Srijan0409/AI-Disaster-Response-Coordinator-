# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the AI Disaster Response Coordinator environment.

Exposes DisasterEnvironment over HTTP and WebSocket endpoints,
compatible with EnvClient and the OpenEnv evaluation harness.

Endpoints:
    POST /reset   — Reset the environment (accepts difficulty + seed)
    POST /step    — Execute an action (allocate rescue team to a victim)
    GET  /state   — Get current environment state
    GET  /schema  — Get action/observation schemas
    WS   /ws      — WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import DisasterAction, DisasterObservation
    from .disaster_env_environment import DisasterEnvironment
except ModuleNotFoundError:
    from models import DisasterAction, DisasterObservation
    from server.disaster_env_environment import DisasterEnvironment


# Create the FastAPI app — OpenEnv wires up all endpoints automatically
app = create_app(
    DisasterEnvironment,
    DisasterAction,
    DisasterObservation,
    env_name="disaster_env",
    max_concurrent_envs=1,  # increase for more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

        uv run --project . server
        uv run --project . server --port 8001
        python -m disaster_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Disaster Response Coordinator server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    main(host=args.host, port=args.port)