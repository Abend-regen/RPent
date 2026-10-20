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

"""RoboDojo control-step observation delivery for a stateful policy."""

from __future__ import annotations

import time
from collections import deque
from typing import Annotated, Any

from pydantic import Field

from robots.robodojo.policy_adapter import RoboDojoPolicyAdapter
from robots.robodojo.tasks import validate_instruction
from rpent.tools import ToolResult, tool
from rpent.tools.toolkit import ToolCancelled
from rpent.utils.logging import get_logger

logger = get_logger("stateful_policy")


@tool(readonly=False, exclude=("primitives", "state"))
def run_policy(
    primitives: Any,
    state: Any,
    max_steps: Annotated[int, Field(gt=0)] = 32,
) -> ToolResult:
    """Run the configured policy on the official task instruction for at most max_steps control steps.

    Inspect the resulting images before continuing. Episode end does not itself
    establish task success.
    """
    try:
        return ToolResult(data=primitives.env.run(max_steps))
    except Exception as exc:
        if not primitives.env.last_run:
            raise
        result = {**primitives.env.last_run, "error": str(exc)}
        if isinstance(exc, ToolCancelled):
            result.update(code="tool_cancelled", interrupted=True)
        else:
            logger.exception("Policy execution failed")
        return ToolResult(data=result)


class StatefulPolicyEnv:
    """Wrap one environment and one exclusive policy session.

    All action tools use this boundary so scripted actions also advance model
    history. Read-only environment calls pass through without recording frames.
    """

    def __init__(
        self,
        env,
        policy: RoboDojoPolicyAdapter,
        check_cancelled,
        *,
        resume_chunks: bool = False,
    ) -> None:
        self._resume_chunks = resume_chunks
        self._pending = deque()
        self._env = env
        self._policy = policy
        self._check_cancelled = check_cancelled
        self._failed = True
        self._closed = False
        self.last_run: dict[str, Any] = {}
        self._recording_run = False
        try:
            self._start_episode()
        except Exception:
            try:
                policy.end_episode()
            except Exception:
                logger.warning(
                    "Failed to release policy after startup error", exc_info=True
                )
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _start_episode(self) -> None:
        self._discard_actions()
        self._failed = True
        self._native_step = self._env.get_status()["step"]
        self._instruction = validate_instruction(self._env.get_task_language())
        self._step = 0
        self._done = False
        self._obs = {
            **self._env.get_obs(),
            "instruction": self._instruction,
        }
        self._policy.begin_episode(self._obs)
        self._failed = False

    def _observe(self, obs: dict) -> None:
        # Use the same task instruction for scripted and learned control.
        self._obs = {**obs, "instruction": self._instruction}
        started = time.perf_counter()
        try:
            self._policy.observe(self._obs, step=self._step)
        finally:
            if self._recording_run:
                self.last_run["observation_rpc_s"] += time.perf_counter() - started

    def _discard_actions(self) -> None:
        # This is the sole cache of returned actions. It never resets the model.
        self._pending.clear()

    def _check_context(self) -> None:
        if not self._resume_chunks:
            return
        if self._env.get_status()["step"] != self._native_step:
            raise RuntimeError(
                "Environment advanced outside the policy boundary; start a new episode"
            )
        if validate_instruction(self._env.get_task_language()) != self._instruction:
            raise RuntimeError("Task instruction changed; start a new episode")

    def _invalidate_episode(self) -> None:
        self._discard_actions()
        self._failed = True
        self._policy.invalidate_episode()

    def interrupt(self) -> None:
        """Invalidate pending control after tool scheduling has been paused."""
        self._invalidate_episode()

    def _require_ready(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("Policy episode is unavailable; start a new episode")

    def reset(self) -> dict:
        """Reset the environment and policy together for a new episode."""
        if self._closed:
            raise RuntimeError("Policy session is closed")
        self._invalidate_episode()
        obs = self._env.reset()
        self._start_episode()
        return obs

    def step(self, action):
        """Apply an external tool action, invalidating any cached policy remainder."""
        self._discard_actions()
        return self._step_action(action)

    def _step_action(self, action):
        self._require_ready()
        try:
            self._check_cancelled()
            self._check_context()
            status = self._env.get_status()
            if self._done or status["step"] >= status["step_limit"]:
                self._done = True
                self._discard_actions()
                raise RuntimeError("Episode has ended")
            started = time.perf_counter()
            try:
                result = self._env.step(action)
            finally:
                if self._recording_run:
                    self.last_run["env_step_s"] += time.perf_counter() - started
            obs, _reward, done, info = result
            if self._resume_chunks and info["status"]["step"] != self._native_step + 1:
                raise RuntimeError(
                    "Environment did not execute exactly one control step"
                )
            self._step += 1
            self._native_step = info["status"]["step"]
            self._done = bool(done) or (
                self._native_step >= info["status"]["step_limit"]
            )
            # Official Awomo deploy.py does not record a terminal observation.
            if not (self._resume_chunks and self._done):
                self._check_context()
                self._observe(obs)
            if self._done:
                self._discard_actions()
            return result
        except Exception:
            # Applied actions / lost observations can leave history uncertain.
            self._invalidate_episode()
            raise

    def run(self, max_steps: int) -> dict[str, Any]:
        """Execute policy chunks up to a control-step budget or episode end."""
        self.last_run = {}
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self._require_ready()
        if self._resume_chunks:
            return self._run_continuous(max_steps)
        executed = 0
        while executed < max_steps and not self._done:
            self._check_cancelled()
            status = self._env.get_status()
            if status["step"] >= status["step_limit"]:
                self._done = True
                break
            try:
                actions = self._policy.predict(self._obs, step=self._step)
                if actions is None or len(actions) == 0:
                    raise RuntimeError("Policy returned an empty action chunk")
            except Exception:
                self._invalidate_episode()
                raise
            for action in actions:
                try:
                    self.step(action)
                except Exception:
                    # Inference may itself advance model state. Discarding an
                    # unexecuted chunk on cancellation requires a new episode.
                    self._invalidate_episode()
                    raise
                executed += 1
                if executed == max_steps or self._done:
                    break
        return {
            "executed_steps": executed,
            "episode_ended": self._done,
            "observation_step": self._step,
        }

    def _run_continuous(self, max_steps: int) -> dict[str, Any]:
        executed = 0
        initial_step = self._step
        started = time.perf_counter()
        self.last_run = {
            "inferences": [],
            "policy_rpc_s": 0.0,
            "env_step_s": 0.0,
            "observation_rpc_s": 0.0,
            "stop_reason": "budget",
        }
        self._recording_run = True
        try:
            self._check_cancelled()
            self._check_context()
            while executed < max_steps and not self._done:
                self._check_cancelled()
                self._check_context()
                status = self._env.get_status()
                if status["step"] >= status["step_limit"]:
                    self._done = True
                    self._discard_actions()
                    break
                if not self._pending:
                    infer_started = time.perf_counter()
                    try:
                        actions = self._policy.predict(self._obs, step=self._step)
                    finally:
                        self.last_run["policy_rpc_s"] += (
                            time.perf_counter() - infer_started
                        )
                    if actions is None or len(actions) == 0:
                        raise RuntimeError("Policy returned an empty action chunk")
                    self.last_run["inferences"].append(
                        {
                            "observation_step": self._step,
                            "action_count": len(actions),
                            "info": self._policy.last_action_info,
                        }
                    )
                    self._pending.extend(actions)
                self._step_action(self._pending.popleft())
                executed += 1
        except Exception as exc:
            self.last_run["stop_reason"] = (
                "cancelled" if isinstance(exc, ToolCancelled) else "error"
            )
            self._invalidate_episode()
            raise
        finally:
            self._recording_run = False
            if self._done and self.last_run["stop_reason"] == "budget":
                self.last_run["stop_reason"] = "episode_end"
            self.last_run.update(
                executed_steps=self._step - initial_step,
                episode_ended=self._done,
                observation_step=self._step,
                pending_actions=len(self._pending),
                elapsed_s=time.perf_counter() - started,
            )
        return self.last_run

    def close(self) -> None:
        """Release this episode and close its environment, retaining the VLA transport."""
        if self._closed:
            return
        self._closed = True
        self._discard_actions()
        try:
            self._policy.end_episode()
        finally:
            self._env.close()
