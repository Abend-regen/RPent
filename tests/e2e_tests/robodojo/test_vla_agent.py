# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in autonomous collaboration on dedicated native episode endpoints."""

import os
import sys
from pathlib import Path

import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import write_json_atomic
from rpent.memory import MemoryManager
from rpent.planner.codex import CodexPlanner
from rpent.utils import logging
from tests.e2e_tests.common import parse_runtime_args, runtime_phase

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_VLA_AGENT") != "1",
        reason="requires dedicated simulator, model, and logged-in Codex",
    ),
    pytest.mark.timeout(4200),
]


def test_agent_episode(monkeypatch):
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(logging, "_output_dir", output)
    for key in (
        "CODEX_API_KEY",
        "CODEX_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "RPENT_CODEX_PROVIDER_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    spec = get_robot_spec()
    args = parse_runtime_args(
        spec,
        [
            "--task",
            os.environ["RPENT_ROBODOJO_TASK"],
            "--layout",
            os.environ["RPENT_ROBODOJO_LAYOUT"],
            "--policy-backend",
            "awomo05",
            "--vla-agent",
            "--agent-tools",
            "policy",
            "motion",
            "geometry",
            "--env-endpoint",
            os.environ["RPENT_ROBODOJO_ENV_ENDPOINT"],
            "--policy-server-url",
            os.environ["RPENT_AWOMO05_POLICY_SERVER_URL"],
            "--xpolicylab-root",
            os.environ["RPENT_XPOLICYLAB_ROOT"],
            "--policy-python",
            sys.executable,
            "--max-episode-steps",
            "400",
            "--agent-max-calls",
            "120",
            "--agent-timeout-s",
            "1800",
        ],
    )
    report = {"layout": args.layout, "calls": [], "events": [], "passed": False}
    path = output / "agent-result.json"
    with runtime_phase(spec, args, output, {"env", "vla"}) as runtime:
        native, policy = runtime["env"], runtime["vla_client"]
        instruction = native.get_task_language()
        report.update(instruction=instruction, model_meta=policy.model_meta)
        observe, infer, reset = (
            policy.update_obs,
            policy.get_action,
            policy.reset_session,
        )
        register = native.register_execution_observer

        def record_observe(obs, step):
            assert obs["instruction"] == instruction
            result = observe(obs, step)
            report["calls"].append(["observe", step])
            write_json_atomic(path, report)
            return result

        def record_infer():
            report["calls"].append(["infer", native.get_status()["step"]])
            return infer()

        def record_reset():
            report["calls"].append(["reset"])
            return reset()

        def record_register(callback):
            def delivery(event):
                report["events"].append(
                    {
                        k: v
                        for k, v in event.items()
                        if k not in {"observation", "status", "episode_status"}
                    }
                )
                return callback(event)

            return register(delivery)

        monkeypatch.setattr(policy, "update_obs", record_observe)
        monkeypatch.setattr(policy, "get_action", record_infer)
        monkeypatch.setattr(policy, "reset_session", record_reset)
        monkeypatch.setattr(native, "register_execution_observer", record_register)
        kit = None
        try:
            kit = RoboDojoToolkit(
                primitives_kwargs=runtime,
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(output / "empty-memory"),
                state_output_dir=output,
            )
            variables = {
                "vla_agent": True,
                "agent_tools": ["policy", "motion", "geometry"],
            }
            system = spec.prompts.render("system", variables=variables)
            user = spec.prompts.render("user", variables=variables)
            write_json_atomic(output / "prompts.json", {"system": system, "user": user})
            planner = CodexPlanner(
                output_dir=str(output),
                repo_root=output,
                output_path=output / "codex.out",
                model=os.environ["RPENT_PLANNER_MODEL"],
                reasoning_effort="high",
                timeout_s=1800,
                dashboard_events=NullDashboardEventSink(),
            )
            result = planner.solve(
                system_prompt=system, user_message=user, toolkit=kit, max_turns=240
            )
            status = native.get_status()
            execution = native.get_execution_state()
            report.update(
                planner_error=result.error,
                finish=result.finish_result,
                planner_stats=result.stats,
                native_success=native.is_success(),
                native_status=status,
                execution=execution,
                transcript=result.messages,
            )
            steps = status["step"]
            observed = [c[1] for c in report["calls"] if c[0] == "observe"]
            feedback = [
                e
                for e in report["events"]
                if e["kind"] == "action_feedback" and e["executed_actions"]
            ]
            checks = {
                "planner_completed": result.error is None,
                "agent_finish": bool(
                    result.finish_result and result.finish_result.get("_finish")
                ),
                "instruction_unchanged": native.get_task_language() == instruction,
                "one_reset": report["calls"].count(["reset"]) == 1,
                "synchronized": execution["synchronization_trusted"],
                "feedback_once": [e["step_after"] for e in feedback]
                == list(range(1, steps + 1)),
                "continuous_history": observed
                == list(range(steps if execution["episode_ended"] else steps + 1)),
                "native_budget": 0 < steps <= 400,
            }
            report.update(checks=checks, passed=all(checks.values()))
            assert report["passed"], checks
        finally:
            try:
                if kit is not None:
                    kit.close()
            finally:
                try:
                    policy.close()
                finally:
                    native.close()
                    write_json_atomic(path, report)
