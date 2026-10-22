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

"""CPU-only execution through real RPC transports and a fake XPolicyLab model."""

import argparse
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.execution import ExecutionSyncError
from robots.robodojo.policy_adapter import RoboDojoPolicyAdapter
from robots.robodojo.stateful_policy import StatefulPolicyEnv
from rpent.robots.components import xpolicylab_vla_server as bridge
from rpent.robots.components.xpolicylab_vla_client import XPolicyLabVLAClient
from rpent.utils.rpc import RpcError, make_rpc_client, wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcServer
from rpent.utils.rpc.socket_rpc import SocketRpcServer


class FakeModel:
    def __init__(self):
        self.history = []
        self.inferences = []
        self.resets = 0
        self.failure = None
        self.empty = False
        self.closed = False

    def call(self, func_name, obs=None):
        if func_name == "reset":
            self.resets += 1
            self.history.clear()
        elif func_name == "update_obs":
            # Check the actual image payload survived the RPC boundary.
            frame = obs["state"]["frame"]
            image = obs["vision"]["cam_head"]["color"]
            np.testing.assert_array_equal(image, np.full((2, 2, 3), frame, np.uint8))
            assert obs["instruction"] == "stack the blocks"
            self.history.append(frame)
        elif func_name == "get_action":
            self.inferences.append(list(self.history))
        else:
            raise AssertionError(func_name)
        if self.failure == func_name:
            raise RuntimeError("lost response after model mutation")
        if func_name == "get_action":
            return {
                "actions": []
                if self.empty
                else [{"expected_step": self.history[-1] + i} for i in range(1, 4)]
            }

    def close(self):
        self.closed = True


class FakeEnv(RoboDojoEnvClient):
    def __init__(self, limit=20, done_at=None):
        self.frame = 0
        self.limit = limit
        self.done_at = done_at
        self.closed = False
        self.after_step = lambda: None
        super().__init__(SimpleNamespace(call=self._rpc), expected_meta={})

    def get_obs(self):
        return {
            "instruction": self.get_task_language(),
            "state": {"frame": self.frame},
            "vision": {
                name: {
                    "color": np.full((2, 2, 3), self.frame % 256, np.uint8),
                    "depth": np.ones((2, 2)),
                    "intrinsic_matrix": np.eye(3),
                    "extrinsic_matrix": np.eye(4),
                }
                for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
            },
        }

    def get_task_language(self):
        return "stack the blocks"

    def get_status(self):
        return {"step": self.frame, "step_limit": self.limit}

    def _native_step(self, action, *, expected_step=None):
        before = self.frame
        if expected_step is not None:
            assert expected_step == before
        if "expected_step" in action:
            assert action["expected_step"] == self.frame + 1
        self.frame += 1
        self.after_step()
        return (
            self.get_obs(),
            0.0,
            self.frame == self.done_at,
            {
                "status": self.get_status(),
                "step_before": before,
                "step_after": self.frame,
                "executed_actions": 1,
                "terminated": self.frame == self.done_at,
                "truncated": self.frame >= self.limit,
            },
        )

    def _native_reset(self):
        self.frame = 0
        return self.get_obs()

    def get_motion_state(self):
        return {f"{arm}_gripper": 0.5 for arm in ("left", "right")}

    def _rpc(self, method, *, kwargs=None, **options):
        kwargs = kwargs or {}
        if method == "env.get_env_meta":
            return {}
        if method == "env.reset":
            return self._native_reset()
        if method == "env.close":
            self.closed = True
            return None
        if method == "env.step":
            return self._native_step(
                kwargs["flat_action"], expected_step=kwargs.get("expected_step")
            )
        if method in ("env.step_joint_update", "env.hold_step"):
            obs, _, _, info = self._native_step(
                {}, expected_step=kwargs.get("expected_step")
            )
            return {**info, "episode_status": info["status"], "observation": obs}
        raise AssertionError(method)


def _execution_env(env, policy, check_cancelled):
    return StatefulPolicyEnv(
        env, RoboDojoPolicyAdapter(policy, stateful=True), check_cancelled
    )


@pytest.fixture(params=["http", "socket"])
def live_bridge(request, monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(bridge, "_connect_policy", lambda *args: model)
    parser = argparse.ArgumentParser()
    bridge.add_backend_args(parser)
    facade = bridge.XPolicyLabVLAFacade(
        parser.parse_args(
            ["--task", "stack", "--policy-server-url", "ws://fake", "--stateful"]
        )
    )
    server_cls = HttpRpcServer if request.param == "http" else SocketRpcServer
    server = server_cls(("127.0.0.1", 0), facade._dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"{request.param}://127.0.0.1:{server.server_address[1]}"
    clients = []

    def connect():
        rpc = make_rpc_client(endpoint, enable_sessions=True)
        clients.append(rpc)
        wait_for_ready(rpc, timeout_s=2)
        return XPolicyLabVLAClient(rpc)

    try:
        yield SimpleNamespace(
            connect=connect, model=model, facade=facade, endpoint=endpoint
        )
    finally:
        for client in clients:
            client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        facade.close()
        assert not thread.is_alive()
        assert model.closed


def test_chunks_scripted_steps_and_reads_preserve_exact_history(live_bridge):
    native = FakeEnv()
    env = _execution_env(native, live_bridge.connect(), lambda: None)
    assert live_bridge.model.history == [0]
    assert env.run(5) == {
        "executed_steps": 5,
        "episode_ended": False,
        "observation_step": 5,
    }
    # A scripted action between tool calls must also reach the model.
    env.step({"scripted": True})
    for _ in range(3):
        env.get_obs()
        env.get_status()
    env.run(2)
    assert live_bridge.model.history == list(range(9))
    assert live_bridge.model.inferences == [[0], list(range(4)), list(range(7))]
    assert live_bridge.model.resets == 1
    env.close()
    env.close()
    assert native.closed


def test_environment_reset_clears_history_and_cadence(live_bridge):
    env = _execution_env(FakeEnv(), live_bridge.connect(), lambda: None)
    env.run(4)
    env.reset()
    assert live_bridge.model.history == [0]
    env.run(1)
    assert live_bridge.model.inferences[-1] == [0]
    assert live_bridge.model.resets == 2


def test_sequential_toolkits_can_reuse_the_policy_connection(live_bridge):
    policy = live_bridge.connect()
    first = _execution_env(FakeEnv(), policy, lambda: None)
    first.run(2)
    first.close()
    with pytest.raises(RpcError, match="acquire"):
        policy.get_action()
    second = _execution_env(FakeEnv(), policy, lambda: None)
    second.run(1)
    assert live_bridge.model.history == [0, 1]
    assert live_bridge.model.resets == 2
    second.close()


@pytest.mark.parametrize("limit,done_at", [(2, None), (20, 2)])
def test_episode_end_discards_remaining_actions(live_bridge, limit, done_at):
    native = FakeEnv(limit=limit, done_at=done_at)
    env = _execution_env(native, live_bridge.connect(), lambda: None)
    assert env.run(9)["executed_steps"] == 2
    assert env.run(9)["episode_ended"]
    assert live_bridge.model.history == [0, 1, 2]
    assert len(live_bridge.model.inferences) == 1
    with pytest.raises(RuntimeError, match="ended"):
        env.step({})
    assert native.frame == 2


def test_session_ownership_and_release(live_bridge):
    owner, other = live_bridge.connect(), live_bridge.connect()
    env = _execution_env(FakeEnv(), owner, lambda: None)
    with pytest.raises(RpcError, match="another session"):
        other.reset_session()
    with pytest.raises(RpcError, match="acquire"):
        other.update_obs({}, 1)
    with pytest.raises(RpcError, match="acquire"):
        other.get_action()
    other.end_episode()
    env.run(1)
    assert live_bridge.model.resets == 1
    assert live_bridge.model.history == [0, 1]
    env.close()
    other.reset_session()
    assert live_bridge.model.resets == 2
    assert live_bridge.model.history == []


def test_missing_duplicate_and_out_of_order_observations(live_bridge):
    policy = live_bridge.connect()
    policy.reset_session()
    with pytest.raises(RpcError, match="new observation"):
        policy.get_action()
    obs = {**FakeEnv().get_obs(), "instruction": "stack the blocks"}
    for step in [1, -1, True, 0.0]:
        with pytest.raises(RpcError, match="Expected observation step 0"):
            policy.update_obs(obs, step)
    policy.update_obs(obs, 0)
    with pytest.raises(RpcError, match="Expected observation step 1"):
        policy.update_obs(obs, 0)
    policy.get_action()
    with pytest.raises(RpcError, match="new observation"):
        policy.get_action()
    assert live_bridge.model.history == [0]
    assert live_bridge.model.inferences == [[0]]


@pytest.mark.parametrize("method", ["reset", "vla.predict"])
def test_legacy_mutations_are_rejected_in_stateful_mode(live_bridge, method):
    policy = live_bridge.connect()
    policy.reset_session()
    args = ({},) if method == "vla.predict" else ()
    with pytest.raises(RpcError, match="stateful mode"):
        policy._client.call(method, args=args)
    assert live_bridge.model.resets == 1


def test_unbound_clients_cannot_mutate_policy(live_bridge):
    rpc = make_rpc_client(live_bridge.endpoint)
    with pytest.raises(RpcError, match="bound session"):
        rpc.call("reset")
    assert live_bridge.model.resets == 0


@pytest.mark.parametrize("failure", ["update_obs", "get_action"])
def test_ambiguous_model_failure_requires_new_episode(live_bridge, failure):
    native = FakeEnv()
    policy = live_bridge.connect()
    env = _execution_env(native, policy, lambda: None)
    live_bridge.model.failure = failure
    with pytest.raises(
        ExecutionSyncError if failure == "update_obs" else RpcError,
        match="lost response",
    ):
        env.run(3)
    frame = native.frame
    with pytest.raises(RuntimeError, match="unavailable"):
        env.run(3)
    with pytest.raises(RuntimeError, match="unavailable"):
        env.step({})
    with pytest.raises(RpcError, match="uncertain"):
        policy.get_action()
    assert native.frame == frame
    live_bridge.model.failure = None
    env.reset()
    env.run(1)
    assert live_bridge.model.history == [0, 1]


def test_empty_actions_stop_without_advancing_environment(live_bridge):
    native = FakeEnv()
    env = _execution_env(native, live_bridge.connect(), lambda: None)
    live_bridge.model.empty = True
    with pytest.raises(RuntimeError, match="empty action chunk"):
        env.run(3)
    with pytest.raises(RuntimeError, match="unavailable"):
        env.run(3)
    assert native.frame == 0


def test_cancellation_stops_mid_chunk_without_extra_observations(live_bridge):
    native = FakeEnv()

    def check_cancelled():
        if native.frame >= 2:
            raise RuntimeError("cancelled")

    env = _execution_env(native, live_bridge.connect(), check_cancelled)
    with pytest.raises(RuntimeError, match="cancelled"):
        env.run(10)
    assert native.frame == 2
    assert live_bridge.model.history == [0, 1, 2]
    with pytest.raises(RuntimeError, match="unavailable"):
        env.run(1)


def test_environment_failure_after_action_blocks_further_control(live_bridge):
    native = FakeEnv()
    policy = live_bridge.connect()
    env = _execution_env(native, policy, lambda: None)

    def fail():
        raise RuntimeError("environment response lost")

    native.after_step = fail
    with pytest.raises(RuntimeError, match="response lost"):
        env.run(3)
    assert native.frame == 1
    assert live_bridge.model.history == [0]
    with pytest.raises(RuntimeError, match="unavailable"):
        env.step({})
    native.after_step = lambda: None
    env.reset()
    env.run(1)
    assert live_bridge.model.history == [0, 1]


def test_initial_observation_failure_releases_session(live_bridge):
    live_bridge.model.failure = "update_obs"
    with pytest.raises(ExecutionSyncError, match="lost response"):
        _execution_env(FakeEnv(), live_bridge.connect(), lambda: None)
    live_bridge.model.failure = None
    env = _execution_env(FakeEnv(), live_bridge.connect(), lambda: None)
    env.run(1)
    assert live_bridge.model.history == [0, 1]


def test_session_expiry_requires_reset_by_a_new_owner(live_bridge):
    policy = live_bridge.connect()
    env = _execution_env(FakeEnv(), policy, lambda: None)
    env.run(1)
    facade = live_bridge.facade
    facade._session_timeout_s = -1
    for sid in list(facade._sessions):
        facade._expire_session(sid)
    with pytest.raises(RpcError, match="session not found"):
        policy.get_action()
    other = live_bridge.connect()
    other.reset_session()
    assert live_bridge.model.history == []


def test_toolkit_runs_policy_and_scripted_gripper_with_one_history(
    live_bridge, monkeypatch, tmp_path
):
    from robots.robodojo.toolkit import RoboDojoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    predictions = []
    predict = RoboDojoPolicyAdapter.predict

    def tracked_predict(self, obs, options=None, *, step=None):
        predictions.append(step)
        return predict(self, obs, options, step=step)

    monkeypatch.setattr(RoboDojoPolicyAdapter, "predict", tracked_predict)
    native = FakeEnv()
    toolkit = RoboDojoToolkit(
        primitives_kwargs={
            "env": native,
            "vla_client": live_bridge.connect(),
            "vla_backend": "awomo05",
            "vla_agent": True,
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    from tests.unit_tests.robots.robodojo.test_vla_agent import call, plan

    plan(toolkit)
    try:
        names = {spec.name for spec in toolkit.list_tools()}
        assert "run_policy" in names
        assert "pi0_pick" not in names
        result = call(toolkit, "run_policy", max_steps=4)
        assert not result.is_error, result.data
        assert result.data["log"]["result"]["executed_steps"] == 4
        assert not call(toolkit, "view_env_state").is_error
        call(toolkit, "set_gripper", arm="right", val=0.0, steps=1)
        result = call(toolkit, "run_policy", max_steps=1)
        assert not result.is_error, result.data
        assert result.data["log"]["result"]["executed_steps"] == 1
        assert live_bridge.model.history == list(range(7))
        assert predictions == [0, 3, 5]
    finally:
        toolkit.close()
    assert not native.closed  # Runtime retains environment ownership.


def test_native_toolkit_keeps_legacy_pick_unregistered(monkeypatch, tmp_path):
    from robots.robodojo.toolkit import RoboDojoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.memory import MemoryManager
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)

    class PickEnv(FakeEnv):
        def get_obs(self):
            obs = super().get_obs()
            for arm in ("left", "right"):
                obs["state"][f"{arm}_ee_pose"] = [0, 0, 1.0]
                obs["state"][f"{arm}_ee_joint_state"] = [1.0]
            return obs

    native = PickEnv()
    adapter_calls, backend_calls = [], []
    predict = RoboDojoPolicyAdapter.predict

    def tracked_predict(self, obs, options=None, *, step=None):
        adapter_calls.append((obs, step))
        return predict(self, obs, options, step=step)

    def backend_predict(obs):
        backend_calls.append(obs)
        return [{"expected_step": native.frame + 1}]

    monkeypatch.setattr(RoboDojoPolicyAdapter, "predict", tracked_predict)
    toolkit = RoboDojoToolkit(
        primitives_kwargs={
            "env": native,
            "vla_client": SimpleNamespace(predict=backend_predict),
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    try:
        names = {spec.name for spec in toolkit.list_tools()}
        assert "pi0_pick" not in names
        assert "run_policy" not in names
        result = toolkit.execute_tool("pi0_pick", {"max_chunks": 1})
        assert result.is_error
        assert native.frame == 0
        assert adapter_calls == backend_calls == []
    finally:
        toolkit.close()
