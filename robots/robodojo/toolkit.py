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

"""RoboDojo tool registration and observation capture."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robodojo.execution import ExecutionSyncError
from robots.robodojo.primitives import RoboDojoPrimitives
from rpent.dashboard.events import DashboardEventSink
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, iter_tools, tool
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger(__name__)

OBSERVATION_TOOLS = frozenset(
    {"view_env_state", "segment", "back_project", "sample_world_xyz", "query_world_map"}
)
MOTION_TOOLS = frozenset(
    {
        "move_to",
        "move_delta",
        "hold",
        "rotate_wrist",
        "set_gripper",
        "release",
        "vpp2_act",
    }
)

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager


class RoboDojoToolkit(Toolkit):
    """Toolkit for the RoboDojo (Isaac Sim) environment.

    Legacy policy code and composites are not exposed. ``eval_fair`` requires a mode-validated
    environment, removes file tools, and disables development perception
    recording. Scoring and safety diagnostics are never planner tools.
    """

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        eval_fair: bool = False,
        state_output_dir: Path | str | None = None,
    ) -> None:
        self.eval_fair = eval_fair
        self._agent = None
        self._finalizing = False
        self.vla_agent = primitives_kwargs.get("vla_agent", False)
        self.registered_tools_only = self.vla_agent
        self._stateful_policy = (
            primitives_kwargs.get("vla_backend", "vpp2") == "awomo05"
        )
        self._resume_chunks = primitives_kwargs.get("vla_backend") == "awomo05"
        self._policy_only = self._resume_chunks and not self.vla_agent
        if self.vla_agent and not self._stateful_policy:
            raise ValueError("VLA Agent requires a stateful policy backend")
        if self._stateful_policy and eval_fair:
            raise ValueError("Stateful policy execution is not supported in eval-fair")
        if eval_fair and not getattr(primitives_kwargs.get("env"), "eval_fair", False):
            raise ValueError("eval-fair requires a mode-validated environment client")
        state = EnvState(Path(state_output_dir or get_output_dir()))
        super().__init__(
            dashboard_events=dashboard_events,
            state=state,
            memory=memory,
        )
        self.init_primitives(primitives_kwargs=primitives_kwargs)
        self._register_robodojo_tools()
        if self.vla_agent:
            try:
                self._configure_agent(primitives_kwargs, state_output_dir)
            except Exception:
                self._primitives.env.release()
                raise
        if eval_fair:
            allowed = OBSERVATION_TOOLS | MOTION_TOOLS | {"finish", "render"}
            self._tools = {
                name: entry for name, entry in self._tools.items() if name in allowed
            }

    def execute_tool(self, name, input_dict):
        if self._agent is not None:
            try:
                return self._agent.dispatch(
                    name,
                    input_dict,
                    self._execute_registered,
                    mutating=name in MOTION_TOOLS | {"run_policy"},
                )
            except Exception as exc:
                # The action may already have completed when journal/feedback
                # delivery fails. Preserve confirmed progress and fail closed.
                self._primitives.env.invalidate_execution(str(exc))
                return ToolResult(
                    data={
                        "error": str(exc),
                        **self._primitives.env.get_execution_state(),
                    }
                )
        return self._execute_registered(name, input_dict)

    def flash_observation(self) -> dict:
        """Refresh the public observation used by live Flash grounding."""
        from robots.robodojo.access import public_observation

        if not self.eval_fair:
            raise RuntimeError("Flash requires eval-fair toolkit")
        obs = public_observation(self._primitives.env.get_obs())
        self._primitives._last_obs = obs
        return obs

    def init_primitives(self, *, primitives_kwargs: dict[str, Any]) -> None:
        """Start a new trace from the runtime-owned environment."""
        from robots.robodojo import tools as robodojo_tools

        self._pending_perception = []
        env = primitives_kwargs["env"]
        policy = primitives_kwargs.get("vla_client")
        if self._stateful_policy:
            from robots.robodojo.policy_adapter import RoboDojoPolicyAdapter
            from robots.robodojo.stateful_policy import StatefulPolicyEnv

            if policy is None:
                raise ValueError("Stateful policy execution requires a VLA client")
            env = StatefulPolicyEnv(
                env,
                RoboDojoPolicyAdapter(policy, stateful=True),
                self.raise_if_cancelled,
                resume_chunks=self._resume_chunks,
            )
        self._state.reset()
        self._primitives = RoboDojoPrimitives(
            env=env,
            sam3_client=primitives_kwargs.get("sam3_client"),
            vla_client=None
            if self._stateful_policy
            else primitives_kwargs.get("vla_client"),
            action_type=primitives_kwargs.get("action_type", "joint"),
            check_cancelled=self.raise_if_cancelled,
        )
        self._primitives.require_rgbd = self.vla_agent
        try:
            self._publish_step(
                robodojo_tools.dump_state(self._primitives, self._state, log=None)
            )
        except Exception:
            if self._stateful_policy:
                env.release()
            raise

    def solved(self) -> bool:
        """Read official success for the runner after planner execution."""
        if self.eval_fair:
            return False  # Plan completion is not an official task verdict.
        return self._primitives.env.is_success()

    def _register_robodojo_tools(self) -> None:
        from robots.robodojo import tools as robodojo_tools

        for definition in iter_tools(robodojo_tools):
            if self._policy_only and definition.name != "view_env_state":
                continue
            if definition.name not in OBSERVATION_TOOLS:
                continue
            resources = {"state": self._state}
            if definition.name in {"sample_world_xyz", "query_world_map"}:
                resources = {"env_state": self._state}
            elif definition.name == "segment":
                resources["primitives"] = self._primitives
            self.add_tool(definition.with_handler(partial(definition, **resources)))
        for definition in iter_tools(self._primitives):
            if self._policy_only and definition.name != "finish":
                continue
            if definition.name == "vpp2_act" and self._primitives.vla_client is None:
                continue
            self.add_tool(definition, replace=definition.name == "finish")
        if self._stateful_policy:
            from robots.robodojo.stateful_policy import run_policy

            self.add_tool(
                run_policy.with_handler(
                    partial(run_policy, primitives=self._primitives, state=self._state)
                )
            )
        if not self._policy_only:
            self.add_tool(self.render)

    @tool
    def render(self) -> ToolResult:
        """Refresh the synchronized RGB-D, calibration and robot state record."""
        self.raise_if_cancelled()
        return ToolResult(data={"success": True})

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        from robots.robodojo import tools as robodojo_tools

        if self.vla_agent:
            result = dict(result)
            if "episode_status" in result:
                result["episode_status"] = {
                    k: v
                    for k, v in result["episode_status"].items()
                    if k in {"step", "step_limit"}
                }
            result.pop("inferences", None)
        record = robodojo_tools.dump_state(
            self._primitives,
            self._state,
            log={"command": command, "result": result, "elapsed_s": elapsed_s},
            perception=self._pending_perception,
        )
        self._pending_perception = []
        observed = robodojo_tools.view_env_state(record.step_idx, state=self._state)
        if self._policy_only and command.get("action") == "run_policy":
            summary = {k: v for k, v in result.items() if k != "inferences"}
            summary["inference_count"] = len(result.get("inferences", []))
            observed.data["log"] = {**observed.data["log"], "result": summary}
        return observed

    def close(self) -> None:
        """Drain tools; the runtime owns the environment across planner sessions."""
        drained = False
        try:
            super().close()
            drained = True
        finally:
            try:
                end = getattr(self._primitives.env, "end_execution", None)
                if end is not None:
                    try:
                        end()
                    except ExecutionSyncError:
                        if drained:
                            raise
                        logger.exception(
                            "Failed to close execution after tool cleanup failed"
                        )
            finally:
                try:
                    if self._stateful_policy:
                        self._primitives.env.release()
                finally:
                    if self._agent is not None:
                        self._agent.close()

    def cancel_active_and_wait(self) -> None:
        super().cancel_active_and_wait()
        pause = getattr(self._primitives.env, "pause_execution", None)
        if pause is not None:
            pause()

    def resume_calls(self) -> None:
        resume = getattr(self._primitives.env, "resume_execution", None)
        if resume is not None:
            resume()
        super().resume_calls()

    def _execute_registered(self, name, input_dict):
        before = self._primitives.env.get_execution_state() if self.vla_agent else None
        self._finalizing = name == "finish"
        try:
            result = super().execute_tool(name, input_dict)
        finally:
            self._finalizing = False
        if self.vla_agent and result.data.get("state_capture_error"):
            self._primitives.env.invalidate_execution(
                "Agent observation delivery failed"
            )
            result.data["synchronization_trusted"] = False
        if name == "finish" and result.data.get("_finish"):
            end = getattr(self._primitives.env, "end_execution", None)
            if end is not None:
                try:
                    end()
                except ExecutionSyncError as exc:
                    logger.exception("Failed to close execution after finish")
                    result.data.setdefault("error", str(exc))
        if self.vla_agent:
            after = self._primitives.env.get_execution_state()
            result.data.pop("inferences", None)
            result.data["execution"] = {
                **after,
                "step_before": before["confirmed_step"],
                "step_after": after["confirmed_step"],
                "confirmed_actions": after["confirmed_step"] - before["confirmed_step"],
            }
        if not self.eval_fair and name in {
            "segment",
            "back_project",
        }:
            import copy

            if isinstance(result.to_dict(), dict):
                # Read-only queries do not create EnvState steps. Attach them
                # to the next action for Flash grounding; actions themselves
                # are already recorded by the shared Toolkit/EnvState path.
                self._pending_perception.append(
                    copy.deepcopy(
                        {
                            "action": name,
                            "arguments": input_dict,
                            "result": result.to_dict(),
                        }
                    )
                )
        return result

    def raise_if_cancelled(self):
        super().raise_if_cancelled()
        if self._agent is not None and not self._finalizing:
            self._agent.check_deadline()

    def _configure_agent(self, kwargs, state_output_dir):
        from robots.robodojo.collaboration import (
            AgentCollaboration,
            finish,
            with_decision,
        )

        groups = set(kwargs.get("agent_tools", ("policy", "motion", "geometry")))
        choices = {
            "policy": {"run_policy"},
            "motion": MOTION_TOOLS,
            "geometry": {"back_project", "sample_world_xyz", "query_world_map"},
            "sam3": {"segment"},
        }
        if not groups or groups - choices.keys():
            raise ValueError("Unknown or empty Agent tool groups")
        if "sam3" in groups and self._primitives.sam3_client is None:
            raise ValueError("The sam3 tool group requires a SAM3 service")
        obs = self._primitives._last_obs
        from robots.robodojo.access import validate_rgbd

        validate_rgbd(obs)
        allowed = {"view_env_state", "render", "finish"}.union(
            *(choices[g] for g in groups)
        )
        self._tools = {
            name: with_decision(entry)
            for name, entry in self._tools.items()
            if name in allowed
        }
        self.add_tool(with_decision(finish), replace=True)
        self._agent = AgentCollaboration(
            Path(state_output_dir or get_output_dir()),
            max_calls=kwargs.get("agent_max_calls", 120),
            timeout_s=kwargs.get("agent_timeout_s", 1800),
        )
        self.add_tools(iter_tools(self._agent))
