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

"""Opt-in full-episode passthrough with a fixed planner and original instruction."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils import logging
from tests.e2e_tests.common import parse_runtime_args, runtime_phase

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_AWOMO05_COMPLETE") != "1",
        reason="requires exclusive Awomo05 and RoboDojo endpoints for a full episode",
    ),
    pytest.mark.timeout(7200),
]


def test_complete_episode_through_fixed_planner(monkeypatch):
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from rpent.planner.api_loop import ApiAgentLoop

    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(logging, "_output_dir", output)
    spec = get_robot_spec()
    upstream = os.environ.get("RPENT_AWOMO05_POLICY_SERVER_URL")
    bridge = os.environ.get("RPENT_AWOMO05_VLA_ENDPOINT")
    assert bool(upstream) != bool(bridge), "choose exactly one connection mode"
    budgets = [
        int(n) for n in os.environ.get("RPENT_POLICY_BUDGETS", "3,5,31").split(",")
    ]
    assert budgets and all(n > 0 for n in budgets)
    argv = [
        "--task",
        os.environ.get("RPENT_ROBODOJO_TASK", "stack_blocks_by_language"),
        "--layout",
        os.environ.get("RPENT_ROBODOJO_LAYOUT", "0"),
        "--policy-backend",
        "awomo05",
        "--env-endpoint",
        os.environ["RPENT_ROBODOJO_ENV_ENDPOINT"],
        "--max-episode-steps",
        os.environ.get("RPENT_ROBODOJO_STEP_LIMIT", "400"),
    ]
    if upstream:
        argv += [
            "--policy-server-url",
            upstream,
            "--xpolicylab-root",
            os.environ["RPENT_XPOLICYLAB_ROOT"],
            "--policy-python",
            sys.executable,
        ]
    else:
        argv += ["--vla-endpoint", bridge]
    args = parse_runtime_args(spec, argv)
    report = {
        "task": args.task,
        "layout": args.layout,
        "budgets": budgets,
        "calls": [],
        "passed": False,
        "predicted_actions": [],
        "executed_actions": [],
    }
    with runtime_phase(spec, args, output, {"env", "vla"}) as runtime:
        policy, native = runtime["vla_client"], runtime["env"]
        instruction = native.get_task_language()
        report.update(
            instruction=instruction,
            model_meta=policy.model_meta,
            initial_status=native.get_status(),
        )
        update_obs, get_action, reset = (
            policy.update_obs,
            policy.get_action,
            policy.reset_session,
        )

        def observe(obs, step):
            assert obs["instruction"] == instruction
            report["calls"].append(["observe", step])
            return update_obs(obs, step)

        def infer():
            report["calls"].append(["infer"])
            actions = get_action()
            report["predicted_actions"].extend(action_digest(a) for a in actions)
            return actions

        def action_digest(action):
            encoded = json.dumps(
                action, sort_keys=True, default=lambda value: value.tolist()
            )
            return hashlib.sha256(encoded.encode()).hexdigest()

        step = native.step_policy

        def execute(action):
            value = step(action)
            report["executed_actions"].append(action_digest(action))
            return value

        def begin():
            report["calls"].append(["reset"])
            return reset()

        monkeypatch.setattr(policy, "update_obs", observe)
        monkeypatch.setattr(policy, "get_action", infer)
        monkeypatch.setattr(policy, "reset_session", begin)
        monkeypatch.setattr(native, "step_policy", execute)
        toolkit = None
        try:
            toolkit = RoboDojoToolkit(
                primitives_kwargs=runtime,
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(output / "memory"),
            )
            stage, index = "view", 0

            async def stream(messages, info):
                nonlocal stage, index
                if stage == "view":
                    name, arguments = "view_env_state", {}
                    stage = "run"
                elif stage == "run" and not toolkit._primitives.env.last_run.get(
                    "episode_ended"
                ):
                    assert not toolkit._primitives.env._failed, (
                        "execution failed; no automatic retry"
                    )
                    assert (
                        native.get_status()["step"]
                        == toolkit._primitives.env._native_step
                    )
                    name, arguments = (
                        "run_policy",
                        {"max_steps": budgets[index % len(budgets)]},
                    )
                    index += 1
                elif stage == "run":
                    name, arguments = (
                        "write_text_file",
                        {
                            "path": str(output / "audit.json"),
                            "content": json.dumps(
                                {
                                    "instruction": instruction,
                                    "reason": "native episode ended",
                                    "strategy": "fixed policy budgets; no goal rewrite or recovery",
                                }
                            ),
                        },
                    )
                    stage = "finish"
                else:
                    assert toolkit._primitives.env.last_run["episode_ended"]
                    name, arguments = (
                        "finish",
                        {
                            "status": "stuck",
                            "summary": "Native episode ended; runner records official task success separately.",
                        },
                    )
                yield {0: DeltaToolCall(name=name, json_args=json.dumps(arguments))}

            planner = ApiAgentLoop(
                model=FunctionModel(stream_function=stream),
                dashboard_events=NullDashboardEventSink(),
                timeout_s=7200,
            )
            # The test planner uses no learned decisions. Prompts remain the
            # actual Awomo tool instructions, with the official goal verbatim.
            prompt = spec.prompts.render(
                "system",
                variables={
                    "policy_backend": "awomo05",
                    "output_dir": str(output),
                    "recipe_tag": f"awomo05_l{args.layout}",
                },
            )
            result = planner.solve(
                system_prompt=prompt,
                user_message=instruction,
                toolkit=toolkit,
                max_turns=args.max_episode_steps + 3,
            )
            assert result.error is None, result.error
            assert result.finish_result["_finish"]
            assert (output / "audit.json").is_file()
            reports = [
                r.result
                for r in toolkit._state.records()
                if r.command and r.command["action"] == "run_policy"
            ]
            assert reports and not any(r.get("error") for r in reports)
            assert reports[-1]["episode_ended"]
            assert reports[-1]["stop_reason"] == "episode_end"
            steps = sum(r["executed_steps"] for r in reports)
            assert steps == native.get_status()["step"]
            assert report["executed_actions"] == report["predicted_actions"][:steps]
            assert len(report["executed_actions"]) == steps
            assert [c[1] for c in report["calls"] if c[0] == "observe"] == list(
                range(steps)
            )
            assert report["calls"].count(["reset"]) == 1
            inferences = [i for r in reports for i in r["inferences"]]
            assert [i["observation_step"] for i in inferences] == list(
                range(0, steps, policy.model_meta["replan_steps"])
            )
            assert report["calls"].count(["infer"]) == len(inferences)
            if policy.model_meta.get("action_info"):
                assert all(i["info"]["upstream"]["latency_ms"] >= 0 for i in inferences)
            assert native.get_task_language() == instruction
            report.update(
                passed=True,
                executed_steps=steps,
                segments=reports,
                native_success=native.is_success(),
                finish=result.finish_result,
                transcript=result.messages,
            )
        finally:
            (output / "complete-result.json").write_text(
                json.dumps(report, indent=2, default=str)
            )
            if toolkit is not None:
                toolkit.close()
            policy.close()
            native.close()  # This test owns the dedicated per-episode simulator.
