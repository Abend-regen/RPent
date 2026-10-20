# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboDojo Awomo05 endpoint contract; the official adapter owns preprocessing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from robots.robodojo.awomo05_contract import (
    CAMERAS,
    STATE_FIELDS,
    validate_model_meta,
    validate_native_state,
)
from rpent.robots.components.xpolicylab_vla_client import XPolicyLabVLAClient
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc.http_rpc import HttpRpcClient


class Awomo05Client(XPolicyLabVLAClient):
    """Validated stateful client; no image transforms or action cache."""

    def __init__(self, client) -> None:
        super().__init__(client)
        try:
            meta = client.call("vla.get_model_meta")
        except Exception as exc:
            raise RuntimeError(
                "Awomo05 endpoint must expose vla.get_model_meta; use the RoboDojo Awomo05 bridge"
            ) from exc
        self.model_meta = validate_model_meta(meta, bridge=True)

    def update_obs(self, obs: dict[str, Any], step: int) -> None:
        validate_native_state(obs.get("state", {}))
        for camera in CAMERAS:
            image = np.asarray(obs.get("vision", {}).get(camera, {}).get("color"))
            if (
                image.ndim != 3
                or image.shape[-1] != 3
                or min(image.shape[:2]) == 0
                or image.dtype != np.uint8
            ):
                raise ValueError(f"Awomo05 {camera} requires HWC uint8 RGB")
        if (
            not isinstance(obs.get("instruction"), str)
            or not obs["instruction"].strip()
        ):
            raise ValueError("Awomo05 requires the official task instruction")
        super().update_obs(obs, step)

    def get_action(self) -> list[dict[str, Any]]:
        actions = super().get_action()
        if (
            not isinstance(actions, (list, tuple))
            or not 1 <= len(actions) <= self.model_meta["replan_steps"]
        ):
            raise ValueError("Awomo05 returned an invalid action chunk")
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("Awomo05 requires native joint action dictionaries")
            if set(action) != set(STATE_FIELDS):
                raise ValueError("Awomo05 returned unexpected action fields")
            validate_native_state(action)
        return actions


def spawn_bridge(
    args: argparse.Namespace, output_dir: Path
) -> tuple[ProcessDaemon, HttpRpcClient]:
    """Start only the CPU bridge, borrowing an independently managed model server."""
    interpreter = Path(args.pi05_python).expanduser()
    if not interpreter.is_file():
        raise ValueError(f"Policy bridge Python not found: {interpreter}")
    root = (
        Path(args.xpolicylab_root).expanduser()
        if args.xpolicylab_root
        else (
            Path(args.source_root).expanduser() / "XPolicyLab"
            if args.source_root
            else None
        )
    )
    if root is None or not (root / "client_server/ws/model_client.py").is_file():
        raise ValueError(
            "Awomo05 WebSocket connection requires --xpolicylab-root or --source-root"
        )
    root = root.resolve()
    host, port = "127.0.0.1", pick_free_port()
    daemon = ProcessDaemon(
        name="robodojo_awomo05_bridge",
        cmd=[
            str(interpreter),
            "-u",
            "-m",
            "robots.robodojo.awomo05_bridge",
            "--task",
            args.task,
            "--policy-server-url",
            args.policy_server_url,
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ],
        env_overrides={
            "PYTHONPATH": os.pathsep.join(
                [str(get_repo_root()), str(root.parent), str(root)]
                + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
            )
        },
        log_path=str(output_dir / "awomo05_bridge.log"),
    )
    try:
        daemon.start()
    except BaseException:
        daemon.stop()
        raise
    return daemon, HttpRpcClient(f"http://{host}:{port}", enable_sessions=True)
