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

"""Awomo endpoint validation and native observation/action contracts without weights."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import robot_spec
from robots.robodojo.awomo05 import Awomo05Client
from robots.robodojo.awomo05_contract import (
    CAMERAS,
    POLICY_METHODS,
    SESSION_METHODS,
    STATE_FIELDS,
    validate_model_meta,
)
from robots.robodojo.awomo05_policy_server import ModelWithMetadata


def metadata():
    return {
        "policy_name": "Awomo05",
        "env_cfg_type": "arx_x5",
        "action_type": "joint",
        "action_dim": 14,
        "action_layout": "native_joint_dict",
        "camera_order": list(CAMERAS),
        "input_color_order": "bgr",
        "control_hz": 25,
        "history_enabled": True,
        "action_horizon": 16,
        "replan_steps": 8,
        "methods": list(POLICY_METHODS),
        "stateful": True,
        "rpc_methods": list(SESSION_METHODS),
    }


def args(*flags):
    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, True)
    return parser.parse_args(flags)


def test_awomo_requires_exactly_one_endpoint_and_native_joint_robot():
    for flags in [
        [],
        ["--action-type", "ee", "--vla-endpoint", "http://bridge"],
        ["--env-cfg-type", "other", "--vla-endpoint", "http://bridge"],
        ["--policy-server-url", "http://wrong"],
    ]:
        with pytest.raises(ValueError):
            robot_spec._validate_args(args("--policy-backend", "awomo05", *flags))
    with pytest.raises(SystemExit):
        args("--vla-endpoint", "http://bridge", "--policy-server-url", "ws://model")
    with pytest.raises(ValueError, match="requires --policy-backend"):
        robot_spec._validate_args(args("--policy-server-url", "ws://model"))
    for flag, url in [
        ("--vla-endpoint", "http://bridge"),
        ("--policy-server-url", "ws://model"),
    ]:
        config = args("--policy-backend", "awomo05", flag, url)
        robot_spec._validate_args(config)
        assert robot_spec._uses_stateful_policy(config)
    assert args("--policy-python", "/custom/python").pi05_python == "/custom/python"


@pytest.mark.parametrize(
    "key,value",
    [
        ("policy_name", "Pi_05"),
        ("stateful", False),
        ("rpc_methods", []),
        ("methods", ["reset"]),
        ("action_dim", 12),
        ("camera_order", list(reversed(CAMERAS))),
        ("input_color_order", "rgb"),
        ("history_enabled", False),
        ("control_hz", 30),
        ("replan_steps", 0),
        ("replan_steps", 17),
        ("replan_steps", True),
        ("action_layout", "qpos14"),
    ],
)
def test_incompatible_endpoint_is_rejected_without_reset(key, value):
    meta = {**metadata(), key: value}
    calls = []

    def call(method):
        calls.append(method)
        return meta

    with pytest.raises(ValueError):
        Awomo05Client(SimpleNamespace(call=call))
    assert calls == ["vla.get_model_meta"]


def test_missing_metadata_is_actionable_and_does_not_probe_reset():
    def call(method):
        assert method == "vla.get_model_meta"
        raise ValueError("unknown method")

    with pytest.raises(RuntimeError, match="Awomo05 bridge"):
        Awomo05Client(SimpleNamespace(call=call))


def test_native_payload_keeps_camera_colors_and_named_joint_order():
    calls = []
    state = {
        key: np.full(size, index, np.float32)
        for index, (key, size) in enumerate(STATE_FIELDS.items())
    }
    image = np.array([[[255, 17, 3]]], dtype=np.uint8)
    obs = {
        "state": state,
        "vision": {camera: {"color": image.copy()} for camera in CAMERAS},
        "instruction": "stack blocks",
    }
    actions = [state]

    def call(method, **kwargs):
        calls.append((method, kwargs))
        return metadata() if method == "vla.get_model_meta" else actions

    client = Awomo05Client(SimpleNamespace(call=call))
    client.update_obs(obs, 0)
    assert calls[-1][1]["args"][0] is obs
    np.testing.assert_array_equal(obs["vision"][CAMERAS[0]]["color"], image)
    assert client.get_action() is actions
    obs["vision"].pop(CAMERAS[-1])
    with pytest.raises(ValueError, match="cam_right_wrist"):
        client.update_obs(obs, 1)


@pytest.mark.parametrize(
    "actions",
    [
        [],
        [[0.0] * 14],
        [{"left_arm_joint_state": [0] * 6}],
        [{key: [float("nan")] * size for key, size in STATE_FIELDS.items()}],
    ],
)
def test_invalid_native_actions_are_rejected(actions):
    def call(method, **kwargs):
        return metadata() if method == "vla.get_model_meta" else actions

    client = Awomo05Client(SimpleNamespace(call=call))
    with pytest.raises(ValueError):
        client.get_action()


def test_standalone_wrapper_delegates_without_action_or_history_cache():
    events = []
    native = SimpleNamespace(
        run_cfg=SimpleNamespace(data=SimpleNamespace(train={"history": {"fps": 25}})),
        env_cfg_type="arx_x5",
        action_type="joint",
        expected_action_dim=14,
        cameras=CAMERAS,
        input_color_order="bgr",
        history_grid=object(),
        action_horizon=16,
        replan_steps=8,
        reset=lambda: events.append("reset"),
        update_obs=lambda obs: events.append(obs),
        get_action=lambda: events.append("get_action"),
    )
    wrapped = ModelWithMetadata(native, {})
    validate_model_meta(wrapped.get_model_meta())
    assert not events
    wrapped.reset()
    wrapped.update_obs({"frame": 0})
    wrapped.get_action()
    assert events == ["reset", {"frame": 0}, "get_action"]
    with pytest.raises(ValueError, match="25 Hz"):
        ModelWithMetadata(native, {"history_fps": 10})


def test_owned_bridge_uses_selected_interpreter_and_preserves_borrowed_service(
    monkeypatch, tmp_path
):
    from robots.robodojo import awomo05

    root = tmp_path / "XPolicyLab"
    module = root / "client_server/ws/model_client.py"
    module.parent.mkdir(parents=True)
    module.touch()
    interpreter = tmp_path / "python"
    interpreter.touch()
    recorded = {}

    class Daemon:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def start(self):
            recorded["started"] = True

    monkeypatch.setattr(awomo05, "ProcessDaemon", Daemon)
    config = args(
        "--policy-backend",
        "awomo05",
        "--policy-server-url",
        "ws://model:9000",
        "--xpolicylab-root",
        str(root),
        "--policy-python",
        str(interpreter),
    )
    daemon, rpc = robot_spec._spawn_vla_server(config, tmp_path)
    try:
        assert recorded["started"]
        assert recorded["cmd"][0] == str(interpreter)
        assert "robots.robodojo.awomo05_bridge" in recorded["cmd"]
        assert "ws://model:9000" in recorded["cmd"]
        assert "CUDA_VISIBLE_DEVICES" not in recorded["env_overrides"]
        assert "--ckpt" not in recorded["cmd"]
    finally:
        rpc.close()


def test_bridge_checks_upstream_without_state_mutation_and_releases_transport(
    monkeypatch,
):
    from robots.robodojo.awomo05_bridge import Awomo05Facade
    from rpent.robots.components import xpolicylab_vla_server as shared

    calls = []
    closed = []

    def call(func_name):
        calls.append(func_name)
        return metadata()

    model = SimpleNamespace(call=call, close=lambda: closed.append(True))
    monkeypatch.setattr(shared, "_connect_policy", lambda *args: model)
    parser = argparse.ArgumentParser()
    shared.add_backend_args(parser)
    facade = Awomo05Facade(
        parser.parse_args(["--task", "stack", "--policy-server-url", "ws://model"])
    )
    try:
        validate_model_meta(facade.get_model_meta(), bridge=True)
        assert calls == ["get_model_meta"]
        assert set(SESSION_METHODS).issubset(facade._rpc)
    finally:
        facade.close()
    assert closed == [True]


def test_rejected_upstream_closes_bridge_without_reset(monkeypatch):
    from robots.robodojo.awomo05_bridge import Awomo05Facade
    from rpent.robots.components import xpolicylab_vla_server as shared

    closed = []

    def call(func_name):
        assert func_name == "get_model_meta"
        return {**metadata(), "policy_name": "Pi_05"}

    monkeypatch.setattr(
        shared,
        "_connect_policy",
        lambda *args: SimpleNamespace(call=call, close=lambda: closed.append(True)),
    )
    parser = argparse.ArgumentParser()
    shared.add_backend_args(parser)
    with pytest.raises(ValueError, match="policy_name"):
        Awomo05Facade(
            parser.parse_args(["--task", "stack", "--policy-server-url", "ws://model"])
        )
    assert closed == [True]


def test_runtime_validates_awomo_before_resetting_borrowed_environment(
    monkeypatch, tmp_path
):
    events = []
    config = args(
        "--policy-backend",
        "awomo05",
        "--vla-endpoint",
        "http://bridge",
        "--env-endpoint",
        "http://env",
    )
    monkeypatch.setattr(
        robot_spec, "try_spawn_server", lambda owned, sink, name, starter: (None, name)
    )

    def wait(owned, sink, name, rpc, daemon, timeout, post_fn):
        events.append(name)
        raise ValueError("incompatible model")

    monkeypatch.setattr(robot_spec, "try_wait_server", wait)
    with pytest.raises(ValueError, match="incompatible model"):
        robot_spec._init_runtime(config, tmp_path, None, {"env", "vla"})
    assert events == ["vla"]
