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

"""RoboDojo env client — thin RPC layer over the Isaac Sim env server."""

from __future__ import annotations

from threading import RLock
from typing import Any

import numpy as np

from robots.robodojo.execution import (
    ActionRejected,
    ExecutionObserver,
    ExecutionSyncError,
)
from rpent.robots.components.env_client_base import BaseEnvClient


class RoboDojoEnvClient(BaseEnvClient):
    """Unified env client for the RoboDojo Isaac Sim backend.

    Inherits :class:`BaseEnvClient` (``env.get_env_meta`` / ``env.reset`` /
    ``env.step`` / ``env.chunk_step``) and adds the RoboDojo-specific read
    surface used by the toolkit primitives (``env.get_obs``,
    ``env.get_reward_details``, ``env.solve_ik_position``, ...). The backend
    is single-environment, so no ``env_idx`` is exposed.
    """

    def __init__(self, client, *, expected_meta: dict[str, Any]):
        self._execution = None
        self._execution_lock = RLock()
        self.eval_fair = expected_meta.get("mode") == "eval-fair"
        super().__init__(
            client,
            expected_meta=expected_meta,
            reset_on_connect=expected_meta.get("mode") != "eval-fair",
        )

    def get_obs(self) -> dict[str, Any]:
        return self._client.call("env.get_obs")

    def get_status(self) -> dict[str, Any]:
        return self._client.call("env.get_status")

    def get_motion_state(self) -> dict[str, Any]:
        return self._client.call("env.get_motion_state")

    def plan_arm_path(self, arm: str, target_pose: list[float]) -> dict[str, Any]:
        return self._client.call(
            "env.plan_arm_path",
            kwargs={"arm": arm, "target_pose": target_pose},
            timeout_s=120,
        )

    def register_execution_observer(self, callback) -> None:
        """Attach one synchronous adapter; no model or environment reset is implied."""
        with self._execution_lock:
            if self._execution is not None:
                raise RuntimeError("an execution observer is already registered")
            self._execution = ExecutionObserver(callback)
            self._initialize_execution(reset=False)

    def _initialize_execution(self, *, reset, observation=None):
        monitor = self._execution
        try:
            before = self.get_status()
            obs = self.get_obs() if observation is None else observation
            after = self.get_status()
            if before["step"] != after["step"]:
                raise RuntimeError("environment advanced while capturing initial state")
            self.last_obs = obs
            monitor.initialize(obs, after, reset=reset)
        except ExecutionSyncError:
            raise
        except Exception as exc:
            raise monitor.lost(f"initial feedback unavailable: {exc}") from exc

    def require_policy_ready(self) -> None:
        """Adapters must call this before inference and use step_policy for actions."""
        with self._execution_lock:
            monitor = self._execution
            if monitor is None:
                raise RuntimeError(
                    "register an execution observer before policy inference"
                )
            if (
                monitor.delivering
                or monitor.closed
                or monitor.paused
                or not monitor.trusted
            ):
                monitor.require_ready({}, None)
            try:
                status = self.get_status()
                instruction = self.get_task_language()
            except Exception as exc:
                raise monitor.lost(f"execution context unavailable: {exc}") from exc
            monitor.require_ready(status, instruction)

    def pause_execution(self) -> None:
        with self._execution_lock:
            if self._execution is not None:
                if self._execution.delivering:
                    raise RuntimeError("execution callbacks cannot re-enter control")
                self._execution.paused = True
                self._execution.notify("actions_invalidated", reason="cancelled")

    def resume_execution(self) -> None:
        with self._execution_lock:
            if self._execution is not None:
                if self._execution.delivering:
                    raise RuntimeError("execution callbacks cannot re-enter control")
                self._execution.paused = False

    def end_execution(self) -> None:
        with self._execution_lock:
            if self._execution is not None:
                if self._execution.delivering:
                    raise RuntimeError("execution callbacks cannot re-enter control")
                if self._execution.closed:
                    return
                self._execution.closed = True
                self._execution.notify("actions_invalidated", reason="closed")
                self._execution.notify("closed")

    def reset(self):
        """Explicit recovery: reset the environment, then deliver a new episode."""
        with self._execution_lock:
            monitor = self._execution
            if monitor is not None:
                if monitor.closed or monitor.delivering:
                    raise RuntimeError(
                        "cannot reset a closed or re-entered execution boundary"
                    )
                monitor.trusted = False
                monitor.notify("actions_invalidated", reason="reset")
            try:
                obs = super().reset()
            except Exception as exc:
                if monitor is not None:
                    raise monitor.lost(
                        f"reset RPC failed: {exc}", execution_uncertain=True
                    ) from exc
                raise
            if monitor is not None:
                self._initialize_execution(reset=True, observation=obs)
            return obs

    def _call_action(self, method, *, kwargs, source, tuple_result=False):
        with self._execution_lock:
            monitor = self._execution
            if monitor is not None:
                self.require_policy_ready()
                if source != "policy":
                    monitor.notify("actions_invalidated", reason=source)
                kwargs = {**kwargs, "expected_step": monitor.step}
            try:
                result = self._client.call(method, kwargs=kwargs, timeout_s=60)
                if tuple_result:
                    obs, _, _, info = result
                    feedback = {
                        **info,
                        "episode_status": info["status"],
                        "observation": obs,
                    }
                else:
                    feedback = dict(result)
                    obs = feedback.get("observation")
                if monitor is not None:
                    # Validate the complete receipt before publishing to an adapter.
                    monitor.accept({**feedback, "source": source})
            except ExecutionSyncError as exc:
                if exc.feedback is not None and isinstance(
                    exc.feedback.get("observation"), dict
                ):
                    self.last_obs = exc.feedback["observation"]
                raise
            except Exception as exc:
                if monitor is not None:
                    raise monitor.lost(
                        f"action RPC/receipt failed: {exc}", execution_uncertain=True
                    ) from exc
                raise
            if feedback["executed_actions"] and obs is not None:
                self.last_obs = obs
            if feedback.get("rejection"):
                raise ActionRejected(feedback["rejection"])
            if tuple_result:
                return result
            # Large observations travel in the observer event, not tool-result text.
            result.pop("observation", None)
            return result

    def step(self, flat_action):
        """Execute an external action; invalidate policy actions before takeover."""
        return self._call_action(
            "env.step",
            kwargs={"flat_action": flat_action},
            source="external",
            tuple_result=True,
        )

    def step_policy(self, flat_action):
        """Execute one caller-owned policy action without discarding its remainder."""
        if self._execution is None:
            raise RuntimeError("register an execution observer before policy actions")
        return self._call_action(
            "env.step",
            kwargs={"flat_action": flat_action},
            source="policy",
            tuple_result=True,
        )

    def chunk_step(self, flat_actions, *, return_all_frames=False):
        """RoboDojo requires acknowledged single actions; batches are unsupported."""
        raise NotImplementedError(
            "RoboDojo chunk_step is disabled; use acknowledged single-action calls"
        )

    def step_joint_update(self, arm, arm_qpos=None, gripper=None):
        if arm not in ("left", "right"):
            raise ValueError("arm must be left or right")
        if arm_qpos is None and gripper is None:
            raise ValueError("set arm_qpos and/or gripper")
        if arm_qpos is not None:
            vector = np.asarray(arm_qpos)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError("arm_qpos must be a finite vector")
        if gripper is not None and (not np.isfinite(gripper) or not 0 <= gripper <= 1):
            raise ValueError("gripper must be finite and in [0,1]")
        return self._call_action(
            "env.step_joint_update",
            kwargs={"arm": arm, "arm_qpos": arm_qpos, "gripper": gripper},
            source="primitive",
        )

    def hold_step(self):
        """Advance one native action while retaining both arms' controller targets."""
        return self._call_action("env.hold_step", kwargs={}, source="hold")

    def get_reward_details(self) -> dict[str, Any]:
        return self._client.call("env.get_reward_details")

    def solve_ik_position(self, arm: str, xyz: list) -> dict[str, Any]:
        return self._client.call(
            "env.solve_ik_position", kwargs={"arm": arm, "xyz": xyz}
        )

    def get_safety_status(self) -> dict[str, Any]:
        return self._client.call("env.get_safety_status")

    def is_success(self) -> bool:
        return bool(self._client.call("env.is_success"))

    def close(self) -> None:
        try:
            self.end_execution()
        finally:
            try:
                self._client.call("env.close", timeout_s=10)
            except Exception:  # noqa: BLE001
                pass
