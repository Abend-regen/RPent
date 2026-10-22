# Copyright 2026 The RPent Authors.
"""Real Awomo inference and bounded public CLI mixed-control verification."""

import json
import os
import sys
from pathlib import Path

import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.e2e_tests.common import (
    ScriptedToolCall,
    parse_runtime_args,
    publish_check,
    run_scripted_policy_chain,
    runtime_phase,
)

pytestmark = [
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_AWOMO05_INTEGRATION") != "1",
        reason="requires real Awomo weights, RoboDojo and reserved GPUs",
    ),
]


def runtime_argv():
    return [
        "--task",
        os.environ.get("RPENT_ROBODOJO_TASK", "general_pickup"),
        "--vla-backend",
        "awomo05",
        "--policy-server-url",
        os.environ["RPENT_AWOMO05_POLICY_SERVER_URL"],
        "--xpolicylab-root",
        os.environ["RPENT_XPOLICYLAB_ROOT"],
        "--policy-python",
        sys.executable,
        "--max-episode-steps",
        "40",
    ]


def test_awomo_real_inference_continuation_takeover_and_reset(monkeypatch):
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "awomo-component"
    spec = get_robot_spec()
    args = parse_runtime_args(spec, runtime_argv())
    events = []
    with runtime_phase(spec, args, output / "runtime", {"env", "vla"}) as runtime:
        policy, native = runtime["vla_client"], runtime["env"]
        observe, infer = policy.update_obs, policy.get_action

        def record_obs(obs, step):
            assert obs["instruction"] == native.get_task_language()
            events.append(("observe", step))
            return observe(obs, step)

        def record_infer():
            events.append(("infer", native.get_status()["step"]))
            return infer()

        monkeypatch.setattr(policy, "update_obs", record_obs)
        monkeypatch.setattr(policy, "get_action", record_infer)
        toolkit = None
        try:
            toolkit = RoboDojoToolkit(
                primitives_kwargs=runtime,
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(output / "memory"),
                state_output_dir=output / "states",
            )
            for budget in (3, 2):
                result = toolkit.execute_tool("run_policy", {"max_steps": budget})
                assert not result.is_error, result.data
                assert result.data["log"]["result"]["executed_steps"] == budget
            assert len(toolkit._primitives.env._pending) == 3
            native.hold_step()
            assert not toolkit._primitives.env._pending
            result = toolkit.execute_tool("run_policy", {"max_steps": 1})
            assert not result.is_error, result.data
            assert [step for kind, step in events if kind == "infer"] == [0, 6]
            assert [step for kind, step in events if kind == "observe"] == list(
                range(8)
            )
            assert native.get_status()["step"] == 7
            native_success = native.is_success()
            toolkit._primitives.env.reset()
            result = toolkit.execute_tool("run_policy", {"max_steps": 1})
            assert not result.is_error, result.data
            assert native.get_status()["step"] == 1
            toolkit.execute_tool(
                "finish", {"status": "stuck", "summary": "bounded inference verified"}
            )
            publish_check(
                "robodojo_awomo_component",
                {
                    "events": events,
                    "executed_actions": 8,
                    "native_success": native_success,
                    "model_meta": policy.model_meta,
                },
            )
        finally:
            if toolkit is not None:
                toolkit.close()
            policy.close()


def decision(previous=None):
    value = {
        "stage": "bounded check",
        "expected_change": "Fresh feedback after a bounded control call",
    }
    if previous is not None:
        value["review"] = {
            "decision_id": previous,
            "evidence": "The preceding call returned fresh images and confirmed native action counts.",
            "stage_status": "in_progress",
            "next_step": "Continue the bounded mixed-control check",
        }
    return value


def test_awomo_public_cli_policy_primitive_policy_finish():
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "awomo-cli"
    result = run_scripted_policy_chain(
        robot="robodojo",
        robot_argv=[
            *runtime_argv(),
            "--vla-agent",
            "--agent-max-calls",
            "8",
            "--agent-timeout-s",
            "600",
        ],
        output_dir=output,
        before_action_calls=(
            ScriptedToolCall("view_env_state", {}),
            ScriptedToolCall(
                "set_stage_plan",
                {
                    "stages": [
                        {
                            "name": "bounded check",
                            "completion_condition": "Two policy calls separated by a primitive return confirmed feedback.",
                        }
                    ]
                },
            ),
            ScriptedToolCall("run_policy", {"max_steps": 2, "decision": decision(1)}),
            ScriptedToolCall("hold", {"steps": 1, "decision": decision(2)}),
        ),
        action=ScriptedToolCall(
            "run_policy", {"max_steps": 2, "decision": decision(3)}
        ),
        action_count_field="executed_steps",
        include_images=True,
        use_memory=False,
    )
    assert result["post_action_new_image_count"] > 0, result
    journal = json.loads((output / "agent-decisions.json").read_text())
    controls = [d for d in journal["decisions"] if d["control"]]
    assert [d["tool"] for d in controls] == ["run_policy", "hold", "run_policy"]
    assert [d["feedback"]["execution"]["confirmed_actions"] for d in controls] == [
        2,
        1,
        2,
    ]
    assert all(d["image_count"] > 0 for d in controls)
    assert journal["closed"] and journal["finished"]
    native_result = json.loads((output / "result.json").read_text())
    result["native_success"] = native_result["success"]
    assert not native_result["agent_error_present"], native_result
    result["definition"] = (
        "public CLI -> real Awomo -> primitive -> real Awomo -> finish"
    )
    publish_check("robodojo_awomo_public_cli", result)
