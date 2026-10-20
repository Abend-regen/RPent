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

"""Full-episode policy tools without weights or autonomous planning decisions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo.awomo05 import Awomo05Client
from robots.robodojo.policy_adapter import RoboDojoPolicyAdapter
from robots.robodojo.prompt_bundle import system_prompt, user_prompt
from robots.robodojo.stateful_policy import StatefulPolicyEnv, run_policy
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools.toolkit import ToolCancelled
from rpent.utils import logging
from tests.unit_tests.robots.robodojo.test_awomo05_contracts import metadata
from tests.unit_tests.robots.robodojo.test_stateful_policy_loopback import FakeEnv


class EpisodeEnv(FakeEnv):
    def get_obs(self):
        obs = (
            super().get_obs()
            if self.frame < 256
            else {
                "state": {"frame": self.frame},
                "vision": {"cam_head": {"color": np.zeros((2, 2, 3), np.uint8)}},
            }
        )
        return obs


class EpisodePolicy:
    def __init__(self):
        self.events = []
        self.step = None
        self.last_action_info = {}

    def reset_session(self):
        self.events.append(("reset",))

    def update_obs(self, obs, step):
        assert obs["instruction"] == "stack the blocks"
        assert obs["state"]["frame"] == step
        self.step = step
        self.events.append(("observe", step))

    def get_action(self):
        self.events.append(("infer", self.step))
        self.last_action_info = {"upstream": {"latency_ms": 123.5}}
        return [{"expected_step": self.step + i} for i in range(1, 9)]

    def end_episode(self):
        self.events.append(("end",))


@pytest.mark.parametrize("budgets", [(400,), (3, 5, 2, 31)])
@pytest.mark.parametrize("done_at", [None, 137])
def test_complete_episode_preserves_observations_inference_and_termination(
    budgets, done_at
):
    native, client = EpisodeEnv(400, done_at), EpisodePolicy()
    env = StatefulPolicyEnv(
        native,
        RoboDojoPolicyAdapter(client, stateful=True),
        lambda: None,
        resume_chunks=True,
    )
    reports = []
    try:
        while not reports or not reports[-1]["episode_ended"]:
            before = native.frame
            env.get_obs()  # read-only tool boundary never advances the timeline
            assert native.frame == before
            reports.append(env.run(budgets[len(reports) % len(budgets)]))
        count = done_at or 400
        assert native.frame == count
        assert sum(r["executed_steps"] for r in reports) == count
        assert client.events == [("reset",)] + [
            event
            for step in range(count)
            for event in (
                [("observe", step), ("infer", step)]
                if step % 8 == 0
                else [("observe", step)]
            )
        ]
        inferences = [i for r in reports for i in r["inferences"]]
        assert [i["observation_step"] for i in inferences] == list(range(0, count, 8))
        assert all(i["info"]["upstream"]["latency_ms"] == 123.5 for i in inferences)
        assert reports[-1]["stop_reason"] == "episode_end"
        assert reports[-1]["pending_actions"] == 0
        assert env.run(4)["executed_steps"] == 0
        assert native.frame == count
    finally:
        env.close()


@pytest.mark.parametrize(
    "failure", [ToolCancelled("cancelled"), RuntimeError("lost observation")]
)
def test_tool_error_preserves_partial_execution_and_discards_actions(failure):
    native, client = EpisodeEnv(), EpisodePolicy()
    env = StatefulPolicyEnv(
        native,
        RoboDojoPolicyAdapter(client, stateful=True),
        lambda: None,
        resume_chunks=True,
    )
    original = client.update_obs

    def observe(obs, step):
        original(obs, step)
        if step == 2:
            raise failure

    client.update_obs = observe
    try:
        result = run_policy(
            primitives=SimpleNamespace(env=env), state=None, max_steps=8
        )
        assert result.is_error
        assert result.data["executed_steps"] == 2
        assert result.data["pending_actions"] == 0
        assert len(result.data["inferences"]) == 1
        assert result.data["stop_reason"] == (
            "cancelled" if isinstance(failure, ToolCancelled) else "error"
        )
        assert client.events.count(("reset",)) == 1
        with pytest.raises(RuntimeError, match="unavailable"):
            env.run(1)
    finally:
        env.close()


def test_optional_action_info_preserves_public_action_list_without_second_rpc():
    actions = [
        {
            key: np.zeros(width)
            for key, width in {
                "left_arm_joint_state": 6,
                "right_arm_joint_state": 6,
                "left_ee_joint_state": 1,
                "right_ee_joint_state": 1,
            }.items()
        }
    ]
    calls = []
    info = {"upstream": {"latency_ms": 12.5}, "model": {"custom": {"x": 1}}}

    def call(method, **kwargs):
        calls.append((method, kwargs))
        if method == "vla.get_model_meta":
            return {**metadata(), "action_info": True}
        assert kwargs["kwargs"] == {"include_info": True}
        return {"actions": actions, "info": info}

    client = Awomo05Client(SimpleNamespace(call=call))
    assert client.get_action() is actions
    assert client.last_action_info is info
    assert len(calls) == 2


def test_awomo_prompt_does_not_inherit_task_specific_recovery():
    variables = {
        "policy_backend": "awomo05",
        "stateful_policy": True,
        "task": "put_bottles_into_dustbin",
    }
    text = str(system_prompt(variables)) + str(user_prompt(variables))
    for name in ("pi0_pick", "place_in_bin", "stabilize", "segment", "back_project"):
        assert name not in text
    assert "run_policy" in text
    assert "place_in_bin" in str(user_prompt({"task": "put_bottles_into_dustbin"}))


def test_fixed_planner_uses_registered_tool_until_native_episode_end(
    monkeypatch, tmp_path
):
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from rpent.planner.api_loop import ApiAgentLoop

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    native, client = EpisodeEnv(400, 11), EpisodePolicy()
    toolkit = RoboDojoToolkit(
        primitives_kwargs={
            "env": native,
            "vla_client": client,
            "stateful_policy": True,
            "policy_backend": "awomo05",
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    script = iter(
        [
            ("view_env_state", {}),
            ("run_policy", {"max_steps": 3}),
            ("run_policy", {"max_steps": 5}),
            ("run_policy", {"max_steps": 32}),
            (
                "finish",
                {
                    "status": "stuck",
                    "summary": "native episode ended; success is not inferred",
                },
            ),
        ]
    )

    async def stream(messages, info):
        name, arguments = next(script)
        if name == "finish":
            assert native.frame == 11
            assert toolkit._primitives.env.last_run["episode_ended"]
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(arguments))}

    try:
        for name in ("pi0_pick", "move_to", "set_gripper", "place_in_bin", "stabilize"):
            assert name not in toolkit._tools
        invalid = toolkit.execute_tool(
            "run_policy", {"max_steps": 3, "prompt": "pick another object"}
        )
        assert invalid.is_error
        assert native.frame == 0
        result = ApiAgentLoop(
            model=FunctionModel(stream_function=stream),
            dashboard_events=NullDashboardEventSink(),
            no_images=True,
        ).solve(
            system_prompt="Execute the original instruction until the environment ends.",
            user_message="stack the blocks",
            toolkit=toolkit,
            max_turns=5,
        )
        assert result.error is None
        assert result.finish_result["_finish"]
        records = toolkit._state.records()
        reports = [
            r.result
            for r in records
            if r.command and r.command["action"] == "run_policy"
        ]
        assert [r["executed_steps"] for r in reports] == [3, 5, 3]
        assert [len(r["inferences"]) for r in reports] == [1, 0, 1]
        assert client.events.count(("reset",)) == 1
    finally:
        toolkit.close()


def test_bridge_envelope_is_optional_and_never_repeats_inference(monkeypatch):
    import argparse

    from robots.robodojo.awomo05_bridge import Awomo05Facade
    from rpent.robots.components import xpolicylab_vla_server as shared

    calls = []
    actions = [{"joint": [1, 2]}]
    upstream = {"ok": True, "latency_ms": 987.25, "future_metric": 17}
    result = {"actions": actions, "model_stats": {"decode_s": 0.5}}

    def connect(self, url, args):
        self._observed_client = SimpleNamespace(call_info=upstream)

        def call(func_name, obs=None):
            calls.append(func_name)
            if func_name == "get_model_meta":
                return metadata()
            if func_name == "get_action":
                return result

        return SimpleNamespace(call=call, close=lambda: None)

    monkeypatch.setattr(Awomo05Facade, "_connect_policy", connect)
    parser = argparse.ArgumentParser()
    shared.add_backend_args(parser)
    facade = Awomo05Facade(
        parser.parse_args(["--task", "stack", "--policy-server-url", "ws://test"])
    )
    try:
        facade.reset_session(session_id="owner")
        facade.update_obs({}, 0, session_id="owner")
        with pytest.raises(ValueError):
            facade.get_action(session_id="owner", include_info="yes")
        envelope = facade.get_action(session_id="owner", include_info=True)
        assert envelope["actions"] is actions
        assert envelope["info"]["upstream"] == upstream
        assert envelope["info"]["model"] == {"model_stats": {"decode_s": 0.5}}
        assert envelope["info"]["bridge_rpc_s"] >= 0
        with pytest.raises(RuntimeError, match="new observation"):
            facade.get_action(session_id="owner", include_info=True)
        assert calls.count("get_action") == 1
        facade.update_obs({}, 1, session_id="owner")
        assert facade.get_action(session_id="owner") is actions
        assert calls.count("get_action") == 2
    finally:
        facade.close()


def test_cancel_between_calls_discards_pending_actions_without_model_reset(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    native, client = EpisodeEnv(), EpisodePolicy()
    toolkit = RoboDojoToolkit(
        primitives_kwargs={
            "env": native,
            "vla_client": client,
            "stateful_policy": True,
            "policy_backend": "awomo05",
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    try:
        first = toolkit.execute_tool("run_policy", {"max_steps": 3})
        assert not first.is_error
        assert first.data["log"]["result"]["pending_actions"] == 5
        assert first.data["log"]["result"]["inference_count"] == 1
        assert "inferences" not in first.data["log"]["result"]
        toolkit.cancel_active_and_wait()
        toolkit.resume_calls()
        assert toolkit.execute_tool("run_policy", {"max_steps": 1}).is_error
        assert native.frame == 3
        assert client.events.count(("reset",)) == 1
        assert not toolkit._primitives.env._pending
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    "backend,components,expected",
    [
        ("awomo05", None, {"env", "vla"}),
        ("awomo05", {"sam3"}, {"sam3"}),
        ("rlinf", None, {"env", "vla", "sam3"}),
    ],
)
def test_runtime_starts_only_requested_services(
    monkeypatch, tmp_path, backend, components, expected
):
    from robots.robodojo import robot_spec
    from tests.unit_tests.robots.robodojo.test_awomo05_contracts import args

    started = []

    def spawn(owned, sink, name, starter):
        started.append(name)
        return None, name

    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *a, **kw: {})
    config = args("--policy-backend", backend, "--vla-endpoint", "http://bridge")
    robot_spec._init_runtime(config, tmp_path, None, components)
    assert set(started) == expected
