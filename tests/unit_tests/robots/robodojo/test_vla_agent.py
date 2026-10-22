# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Real Toolkit/Planner dispatch, with deterministic control and model feedback."""

import json

import numpy as np
import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.robodojo.access import CAMERAS
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.utils import logging
from tests.unit_tests.robots.robodojo.test_awomo05_policy_tool import (
    EpisodeEnv,
    EpisodePolicy,
)


class AgentEnv(EpisodeEnv):
    def get_obs(self):
        obs = super().get_obs()
        camera = obs["vision"]["cam_head"]
        camera.update(
            depth=np.ones((2, 2), np.float32),
            intrinsic_matrix=np.eye(3),
            extrinsic_matrix=np.eye(4),
        )
        obs["vision"] = {key: dict(camera) for key in CAMERAS}
        return obs

    def get_motion_state(self):
        return {
            **super().get_motion_state(),
            **{
                f"{arm}_eef_pose": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                for arm in ("left", "right")
            },
        }

    def plan_arm_path(self, arm, target):
        return {"status": "Failure", "position": []}

    def is_success(self):
        return self.frame == self.done_at


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    instances = []

    def build(limit=20, done_at=None, **kwargs):
        env, policy = AgentEnv(limit, done_at), EpisodePolicy()
        kit = RoboDojoToolkit(
            primitives_kwargs=dict(
                env=env,
                vla_client=policy,
                stateful_policy=True,
                policy_backend="awomo05",
                vla_agent=True,
                **kwargs,
            ),
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            state_output_dir=tmp_path,
        )
        instances.append(kit)
        return env, policy, kit

    yield build
    for kit in instances:
        kit.close()


def plan(kit):
    result = kit.execute_tool(
        "set_stage_plan",
        {
            "stages": [
                {
                    "name": "goal",
                    "completion_condition": "Visible target relation persists",
                }
            ]
        },
    )
    assert not result.is_error


def call(kit, name, **kwargs):
    decision = {
        "stage": "goal",
        "expected_change": "Inspect visible progress toward the original task",
    }
    if kit._agent.decisions:
        decision["review"] = {
            "decision_id": kit._agent.decisions[-1]["decision_id"],
            "evidence": "Read returned images and confirmed action count",
            "stage_status": "in_progress",
            "next_step": name,
        }
    return kit.execute_tool(name, {**kwargs, "decision": decision})


def report(result):
    return result.data.get("log", {}).get("result", result.data)


def test_actual_planner_dispatch_mixed_control_and_cleanup(setup, tmp_path):
    env, policy, kit = setup()
    script = iter(
        [
            ("view_env_state", {}),
            (
                "set_stage_plan",
                {
                    "stages": [
                        {
                            "name": "goal",
                            "completion_condition": "Visible target relation persists",
                        }
                    ]
                },
            ),
            ("run_policy", {"max_steps": 3}),
            ("run_policy", {"max_steps": 2}),
            ("set_gripper", {"arm": "left", "val": 0.5, "steps": 1}),
            ("hold", {"steps": 2}),
            ("run_policy", {"max_steps": 2}),
            (
                "finish",
                {
                    "status": "uncertain",
                    "summary": "Attempt complete; no native verdict inferred",
                },
            ),
        ]
    )

    async def stream(messages, info):
        name, args = next(script)
        if name not in {"view_env_state", "set_stage_plan"}:
            args["decision"] = {
                "stage": "goal",
                "expected_change": "Observe progress",
                "review": {
                    "decision_id": kit._agent.decisions[-1]["decision_id"],
                    "evidence": "Current image and feedback reviewed",
                    "stage_status": "in_progress",
                    "next_step": name,
                },
            }
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    result = ApiAgentLoop(
        model=FunctionModel(stream_function=stream),
        dashboard_events=NullDashboardEventSink(),
        no_images=True,
    ).solve(
        system_prompt="Solve through the registered tools.",
        user_message="stack the blocks",
        toolkit=kit,
        max_turns=8,
    )
    assert result.error is None
    assert result.finish_result["agent_assessment"] == "uncertain"
    assert "native_environment_success" not in result.finish_result
    assert env.frame == 10
    assert [e for e in policy.events if e[0] == "infer"] == [("infer", 0), ("infer", 8)]
    assert [e[1] for e in policy.events if e[0] == "observe"] == list(range(11))
    assert policy.events.count(("reset",)) == 1
    assert policy.events.count(("end",)) == 1
    kit.close()
    assert policy.events.count(("end",)) == 1
    journal = json.loads((tmp_path / "agent-decisions.json").read_text())
    assert journal["closed"] and journal["finished"]
    assert all(d["feedback"] and d["review"] for d in journal["decisions"][:-1])
    assert all(
        d["image_count"] == 3
        for d in journal["decisions"]
        if d["tool"] in {"run_policy", "hold", "set_gripper"}
    )
    assert not {"read_file", "write_file", "reset", "chunk_step"} & kit._tools.keys()


def test_reads_and_zero_execution_failures_preserve_actions(setup):
    env, policy, kit = setup()
    plan(kit)
    assert report(call(kit, "run_policy", max_steps=3))["pending_actions"] == 5
    before = list(policy.events)
    for name, args in [
        ("view_env_state", {}),
        ("render", {}),
        ("move_delta", {"arm": "right"}),
        ("move_to", {"arm": "left", "xyz": [0.0, 0.0, 1.0]}),
        ("hold", {"steps": -1}),
        ("run_policy", {"max_steps": 1, "instruction": "changed"}),
    ]:
        call(kit, name, **args)
        assert env.frame == 3
        assert policy.events == before
    result = report(call(kit, "run_policy", max_steps=2))
    assert result["pending_actions"] == 3
    assert len([e for e in policy.events if e[0] == "infer"]) == 1


@pytest.mark.parametrize("done_at", [None, 6])
def test_shared_native_budget_and_termination(setup, done_at):
    env, policy, kit = setup(limit=8, done_at=done_at)
    plan(kit)
    call(kit, "run_policy", max_steps=3)
    result = report(call(kit, "hold", steps=10))
    assert env.frame == (done_at or 8)
    assert result["executed_steps"] == env.frame - 3
    assert result["terminated"] if done_at else result["truncated"]
    call(kit, "run_policy", max_steps=3)
    assert env.frame == (done_at or 8)
    assert [e[1] for e in policy.events if e[0] == "observe"] == list(range(env.frame))


@pytest.mark.parametrize("failure", ["rpc", "feedback", "capture", "counter"])
def test_uncertainty_blocks_later_actions_without_retry(setup, monkeypatch, failure):
    env, policy, kit = setup()
    plan(kit)
    call(kit, "run_policy", max_steps=2)
    if failure == "rpc":

        def lost():
            raise TimeoutError("result uncertain")

        env.after_step = lost
    elif failure == "feedback":

        def lost(obs, step):
            raise RuntimeError("model feedback unavailable")

        policy.update_obs = lost
    elif failure == "capture":
        monkeypatch.setattr(
            kit,
            "get_env_state",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("delivery failed")),
        )
    else:
        original = env._native_step

        def wrong(*args, **kwargs):
            obs, reward, done, info = original(*args, **kwargs)
            info["step_after"] += 1
            return obs, reward, done, info

        env._native_step = wrong
    call(kit, "hold", steps=1)
    assert env.frame == 3
    infer_count = policy.events.count(("infer", 0))
    call(kit, "run_policy", max_steps=1)
    call(kit, "hold", steps=1)
    assert env.frame == 3
    assert policy.events.count(("infer", 0)) == infer_count
    assert policy.events.count(("reset",)) == 1


def test_partial_cancel_discards_actions_and_closes(setup):
    env, policy, kit = setup()
    plan(kit)
    call(kit, "run_policy", max_steps=2)

    def cancel():
        if env.frame == 4:
            kit._scheduler.cancel()

    env.after_step = cancel
    result = report(call(kit, "hold", steps=8))
    assert env.frame == 4
    assert result["executed_steps"] == 2
    assert result["interrupted"]
    kit.close()
    assert policy.events.count(("end",)) == 1


def test_agent_budgets_allow_finish_without_more_motion(setup):
    env, policy, kit = setup(agent_max_calls=2)
    plan(kit)
    call(kit, "hold", steps=1)
    assert call(kit, "run_policy", max_steps=1).data["code"] == "agent_budget"
    assert env.frame == 1
    assert call(kit, "finish", status="budget", summary="Call limit reached").data[
        "_finish"
    ]


def test_wall_clock_budget_and_initial_primitive(setup):
    env, policy, kit = setup()
    plan(kit)
    assert not call(kit, "hold", steps=2).is_error
    assert env.frame == 2
    assert not [e for e in policy.events if e[0] == "infer"]
    kit._agent.started -= kit._agent.timeout_s + 1
    assert call(kit, "run_policy", max_steps=1).data["code"] == "agent_budget"
    assert call(kit, "finish", status="budget", summary="Time limit").data["_finish"]


def test_requires_rgbd_and_no_model_reset_on_failed_start(setup, monkeypatch):
    monkeypatch.setattr(AgentEnv, "get_obs", EpisodeEnv.get_obs)
    with pytest.raises(ValueError, match="same-frame"):
        setup()


def test_reads_cannot_hide_an_unreviewed_control(setup):
    env, policy, kit = setup()
    plan(kit)
    call(kit, "hold", steps=1)
    kit.execute_tool("view_env_state", {})
    assert call(kit, "hold", steps=1).is_error
    assert env.frame == 1


def test_execution_progress_is_reported_on_uncertain_rpc(setup):
    env, policy, kit = setup()
    plan(kit)
    result = call(kit, "hold", steps=2)
    assert result.data["execution"]["confirmed_actions"] == 2
    env.after_step = lambda: (_ for _ in ()).throw(TimeoutError("unknown"))
    result = call(kit, "hold", steps=1)
    assert result.data["execution"]["confirmed_step"] == 2
    assert not result.data["execution"]["synchronization_trusted"]
    assert env.frame == 3


def test_model_input_excludes_agent_geometry():
    from types import SimpleNamespace

    from robots.robodojo.awomo05 import Awomo05Client
    from robots.robodojo.awomo05_contract import STATE_FIELDS
    from tests.unit_tests.robots.robodojo.test_awomo05_contracts import metadata

    calls = []

    def rpc(method, **kwargs):
        calls.append((method, kwargs))
        return metadata() if method == "vla.get_model_meta" else None

    obs = AgentEnv().get_obs()
    obs["state"] = {key: np.zeros(size) for key, size in STATE_FIELDS.items()}
    client = Awomo05Client(SimpleNamespace(call=rpc))
    client.update_obs(obs, 0)
    sent = calls[-1][1]["args"][0]
    assert sent["instruction"] == obs["instruction"]
    for camera in CAMERAS:
        assert set(sent["vision"][camera]) == {"color"}
        assert sent["vision"][camera]["color"] is obs["vision"][camera]["color"]
    assert "depth" in obs["vision"]["cam_head"]


def test_explicit_tool_groups_and_backend_requirements(setup):
    env, policy, kit = setup(agent_tools=["policy", "geometry"])
    assert "run_policy" in kit._tools and "sample_world_xyz" in kit._tools
    assert not {"move_delta", "hold", "segment"} & kit._tools.keys()
    with pytest.raises(ValueError, match="SAM3"):
        setup(agent_tools=["policy", "sam3"])


def test_cli_opt_in_has_no_historical_memory(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from robots.robodojo.robot_spec import _prepare_memory, _validate_args
    from tests.unit_tests.robots.robodojo.test_awomo05_contracts import args

    parsed = args(
        "--policy-backend", "awomo05", "--vla-endpoint", "http://bridge", "--vla-agent"
    )
    parsed.planner = "codex"
    _validate_args(parsed)
    config = SimpleNamespace(output_dir=tmp_path, prompt_vars={})
    monkeypatch.setattr(
        MemoryManager,
        "sync",
        lambda **kwargs: pytest.fail("historical memory must not load"),
    )
    _prepare_memory(parsed, config)
    assert config.prompt_vars["memory_dir"] == str(tmp_path / "agent-memory")
    parsed.memory_repo = "example/recipes"
    with pytest.raises(ValueError, match="historical memory"):
        _validate_args(parsed)
