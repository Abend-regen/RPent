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

"""Exercise Awomo action-cache ownership through phase-one real RPC transports."""

import pytest

from robots.robodojo.policy_adapter import RoboDojoPolicyAdapter
from robots.robodojo.stateful_policy import StatefulPolicyEnv
from rpent.tools.toolkit import ToolCancelled
from tests.unit_tests.robots.robodojo import (
    test_stateful_policy_loopback as stateful_tests,
)

FakeEnv = stateful_tests.FakeEnv
live_bridge = stateful_tests.live_bridge


def execution(live_bridge, native=None, cancel=lambda: None):
    native = native or FakeEnv()
    return StatefulPolicyEnv(
        native,
        RoboDojoPolicyAdapter(live_bridge.connect(), stateful=True),
        cancel,
        resume_chunks=True,
    )


def test_split_calls_resume_one_chunk_without_extra_inference(live_bridge):
    env = execution(live_bridge)
    env.run(1)
    env.get_obs()
    env.run(1)
    env.run(3)
    assert live_bridge.model.history == list(range(6))
    assert live_bridge.model.inferences == [[0], list(range(4))]
    assert live_bridge.model.resets == 1
    env.close()


def test_external_action_discards_remainder_without_resetting_model(live_bridge):
    env = execution(live_bridge)
    env.run(1)
    env.step({"scripted": True})
    env.run(1)
    assert live_bridge.model.inferences == [[0], [0, 1, 2]]
    assert live_bridge.model.resets == 1
    env.close()


@pytest.mark.parametrize("cancel_at", [0, 1, 3])
def test_cancelled_chunk_resumes_without_repeating_inference(live_bridge, cancel_at):
    native = FakeEnv()
    model = live_bridge.model
    cancelled = True

    def check():
        if cancelled and native.frame == cancel_at and model.inferences:
            # At 0 and 3, wait for the new chunk's inference to return. At 1,
            # cancel after one acknowledged action from the first chunk.
            if cancel_at == 1 or model.inferences[-1][-1] == cancel_at:
                raise ToolCancelled("cancelled before action")

    env = execution(live_bridge, native, check)
    try:
        with pytest.raises(ToolCancelled):
            env.run(8)
        assert env.last_run["stop_reason"] == "cancelled"
        assert env.last_run["executed_steps"] == native.frame == cancel_at
        assert env.last_run["pending_actions"] == (0 if cancel_at == 1 else 3)
        assert model.history == list(range(cancel_at + 1))
        with pytest.raises(RuntimeError, match="resume a pause"):
            env.run(1)
        cancelled = False
        env.resume_execution()
        result = env.run(4)
        assert result["executed_steps"] == 4
        assert result["observation_step"] == native.frame == cancel_at + 4
        assert model.history == list(range(cancel_at + 5))
        assert native._history_step == cancel_at + 4
        assert [step for step, _ in native._rgb_history] == model.history
        assert [history[-1] for history in model.inferences] == {
            0: [0, 3],
            1: [0, 1, 4],
            3: [0, 3, 6],
        }[cancel_at]
        assert model.resets == 1
    finally:
        env.close()


def test_cancelled_unstarted_chunk_cannot_resume_uncertain_execution(live_bridge):
    native = FakeEnv()
    model = live_bridge.model
    cancelled = True

    def check():
        if cancelled and model.inferences:
            raise ToolCancelled("cancelled after inference")

    env = execution(live_bridge, native, check)
    try:
        with pytest.raises(ToolCancelled):
            env.run(2)
        cancelled = False
        env.resume_execution()

        def fail():
            raise RuntimeError("response lost")

        native.after_step = fail
        with pytest.raises(RuntimeError, match="response lost"):
            env.run(1)
        assert native.frame == 1
        assert env.last_run["executed_steps"] == 0
        assert not env._pending
        assert model.history == [0]
        assert native._history_step == 0
        native.after_step = lambda: None
        env.resume_execution()
        with pytest.raises(RuntimeError, match="unavailable"):
            env.run(1)
        assert native.frame == 1
        assert model.inferences == [[0]]
        assert model.resets == 1
    finally:
        env.close()


@pytest.mark.parametrize("change", ["advance", "reset", "instruction", "cancel"])
def test_context_changes_discard_old_actions_without_implicit_model_reset(
    live_bridge, change
):
    native = FakeEnv()
    cancelled = False

    def check():
        if cancelled:
            raise RuntimeError("cancelled")

    env = execution(live_bridge, native, check)
    env.run(2)
    if change == "advance":
        native.frame += 1
    elif change == "reset":
        native._native_reset()  # Bypass RPent to simulate an external reset.
    elif change == "instruction":
        native.get_task_language = lambda: "new instruction"
    else:
        cancelled = True
    frame = native.frame
    with pytest.raises(RuntimeError):
        env.run(1)
    assert not env._pending
    assert native.frame == frame
    assert live_bridge.model.resets == 1
    env.close()


def test_reset_starts_fresh_action_chunk(live_bridge):
    env = execution(live_bridge)
    env.run(1)
    env.reset()
    env.run(1)
    assert live_bridge.model.history == [0, 1]
    assert live_bridge.model.inferences == [[0], [0]]
    assert live_bridge.model.resets == 2
    env.close()


@pytest.mark.parametrize("limit,done_at", [(2, None), (20, 2)])
def test_terminal_frame_matches_official_deploy_loop(live_bridge, limit, done_at):
    env = execution(live_bridge, FakeEnv(limit, done_at))
    result = env.run(10)
    assert result["executed_steps"] == 2
    assert result["episode_ended"]
    assert live_bridge.model.history == [0, 1]
    assert not env._pending
    env.close()


def test_failure_discards_cache_without_reset_or_retry(live_bridge):
    native = FakeEnv()
    env = execution(live_bridge, native)
    env.run(1)

    def fail():
        raise RuntimeError("response lost")

    native.after_step = fail
    with pytest.raises(RuntimeError, match="response lost"):
        env.run(1)
    assert not env._pending
    assert live_bridge.model.resets == 1
    with pytest.raises(RuntimeError, match="unavailable"):
        env.run(1)
    env.close()
