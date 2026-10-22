# Copyright 2026 The RPent Authors.
"""Bounded real-component check for relative motion, hold and VPP2 history."""

import os
from pathlib import Path

import numpy as np
import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.e2e_tests.common import parse_runtime_args, publish_check, runtime_phase


@pytest.mark.timeout(900)
@pytest.mark.skipif(
    os.environ.get("RPENT_RUN_ROBODOJO_INTEGRATION") != "1",
    reason="requires real RoboDojo and a reserved GPU",
)
def test_relative_motion_hold_history():
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "primitive-extensions"
    spec = get_robot_spec()
    args = parse_runtime_args(spec, ["--task", "put_bottles_into_dustbin"])
    with runtime_phase(spec, args, output / "runtime", {"env"}) as runtime:
        env = runtime["env"]
        toolkit = RoboDojoToolkit(
            primitives_kwargs=runtime,
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(output / "memory"),
            state_output_dir=output / "states",
        )
        try:
            before = env.get_motion_state()["left_eef_pose"]
            initial = env.get_status()["step"]
            env.policy_observation()
            actions = []
            for name, arguments in [
                ("move_delta", {"arm": "left", "dxyz": [0, 0, 0.01], "substeps": 8}),
                ("hold", {"steps": 2}),
            ]:
                result = toolkit.execute_tool(name, arguments)
                assert not result.is_error, result.data
                execution = result.data["log"]["result"]
                assert execution["completed"] and execution["executed_actions"] > 0, (
                    execution
                )
                actions.append(execution["executed_actions"])
            after = env.get_motion_state()["left_eef_pose"]
            assert after[2] > before[2] + 0.004, (before, after)
            assert env.get_status()["step"] - initial == sum(actions)
            history = env.policy_observation()
            env.get_obs()
            assert env._history_step == sum(actions)
            np.testing.assert_array_equal(
                history["history"][-1][0], env.last_obs["vision"]["cam_head"]["color"]
            )
            assert toolkit._state.latest_step == 2
            assert toolkit._state.load("cam_head.png", step=2).shape[-1] == 3
            result = toolkit.execute_tool(
                "finish",
                {
                    "status": "stuck",
                    "summary": "bounded relative motion and hold check",
                },
            )
            assert result.data["_finish"]
            publish_check(
                "robodojo_primitive_extensions",
                {
                    "executed_actions": sum(actions),
                    "native_success": env.is_success(),
                    "definition": "real native move_delta and hold with existing VPP2 history sampling; no VPP2 inference",
                },
            )
        finally:
            toolkit.close()
