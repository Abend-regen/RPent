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

"""Opt-in real Awomo05/Isaac bounded execution; requires exclusive test endpoints."""

import json
import os
from pathlib import Path

import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils import logging
from tests.e2e_tests.common import parse_runtime_args, runtime_phase

pytestmark = pytest.mark.skipif(
    os.environ.get("RPENT_RUN_AWOMO05_INTEGRATION") != "1",
    reason="requires exclusive Awomo05 and RoboDojo test endpoints",
)


def test_awomo05_bounded_tool_execution_and_reset(monkeypatch):
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(logging, "_output_dir", output)
    spec = get_robot_spec()
    upstream = os.environ.get("RPENT_AWOMO05_POLICY_SERVER_URL")
    bridge = os.environ.get("RPENT_AWOMO05_VLA_ENDPOINT")
    assert bool(upstream) != bool(bridge), "choose exactly one model connection mode"
    argv = [
        "--task",
        os.environ.get("RPENT_ROBODOJO_TASK", "stack_blocks_by_language"),
        "--policy-backend",
        "awomo05",
        "--env-endpoint",
        os.environ["RPENT_ROBODOJO_ENV_ENDPOINT"],
        "--max-episode-steps",
        os.environ.get("RPENT_ROBODOJO_STEP_LIMIT", "40"),
    ]
    if upstream:
        argv += [
            "--policy-server-url",
            upstream,
            "--xpolicylab-root",
            os.environ["RPENT_XPOLICYLAB_ROOT"],
        ]
    else:
        argv += ["--vla-endpoint", bridge]
    args = parse_runtime_args(spec, argv)
    report = {
        "task": args.task,
        "purpose": "bounded integration, not an official success-rate baseline",
        "calls": [],
    }
    with runtime_phase(spec, args, output, {"env", "vla"}) as runtime:
        policy = runtime["vla_client"]
        report["model_meta"] = policy.model_meta
        update_obs, get_action, reset = (
            policy.update_obs,
            policy.get_action,
            policy.reset_session,
        )

        def observe(obs, step):
            report["calls"].append(["observe", step])
            return update_obs(obs, step)

        def infer():
            report["calls"].append(["infer"])
            return get_action()

        def begin():
            report["calls"].append(["reset"])
            return reset()

        monkeypatch.setattr(policy, "update_obs", observe)
        monkeypatch.setattr(policy, "get_action", infer)
        monkeypatch.setattr(policy, "reset_session", begin)
        toolkit = None
        try:
            toolkit = RoboDojoToolkit(
                primitives_kwargs=runtime,
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(output / "memory"),
            )
            results = []
            for budget in (3, 5, 2):
                result = toolkit.execute_tool("run_policy", {"max_steps": budget})
                assert not result.is_error, result.data
                results.append(result.data["log"]["result"])
            report["segments"] = results
            assert [r["executed_steps"] for r in results] == [3, 5, 2]
            assert report["calls"].count(["infer"]) == 2
            assert [c[1] for c in report["calls"] if c[0] == "observe"] == list(
                range(11)
            )
            toolkit._primitives.env.reset()
            result = toolkit.execute_tool("run_policy", {"max_steps": 2})
            assert not result.is_error, result.data
            assert report["calls"].count(["reset"]) == 2
            assert report["calls"][-2:] == [["observe", 1], ["observe", 2]]
            report["native_success"] = runtime["env"].is_success()
            report["passed"] = True
        finally:
            (output / "awomo05-result.json").write_text(json.dumps(report, indent=2))
            if toolkit is not None:
                toolkit.close()
            policy.close()
