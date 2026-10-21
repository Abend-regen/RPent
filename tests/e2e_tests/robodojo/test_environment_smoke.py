# Copyright 2026 The RPent Authors.
"""Opt-in real RoboDojo primitive checks; no VLA or SAM3 is needed."""

import os
from pathlib import Path

import numpy as np
import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.e2e_tests.common import parse_runtime_args, require_rgb, runtime_phase

pytestmark = [
    pytest.mark.timeout(900),
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_ROBODOJO_INTEGRATION") != "1",
        reason="requires an explicitly reserved GPU and RoboDojo assets",
    ),
]


def check_pose_transition(before, after, *, z_delta=0.0, yaw_deg=0.0):
    """Check measured motion; a no-op must not pass the small-move check."""
    before, after = np.asarray(before), np.asarray(after)
    np.testing.assert_allclose(after[:3], before[:3] + [0, 0, z_delta], atol=0.005)
    half = np.deg2rad(yaw_deg) / 2
    c, s = np.cos(half), np.sin(half)
    w, x, y, z = before[3:]
    expected = np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])
    expected /= np.linalg.norm(expected)
    actual = after[3:] / np.linalg.norm(after[3:])
    angle_error = 2 * np.arccos(np.clip(abs(np.dot(expected, actual)), 0, 1))
    assert angle_error < np.deg2rad(0.5), angle_error


def test_environment_primitives(monkeypatch):
    from rpent.utils import logging

    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"])
    monkeypatch.setattr(logging, "_output_dir", output)
    spec = get_robot_spec()
    flags = [
        "--task",
        os.environ.get("RPENT_ROBODOJO_TASK", "put_bottles_into_dustbin"),
        "--seed",
        "0",
        "--layout",
        os.environ.get("RPENT_ROBODOJO_LAYOUT", "0"),
    ]
    for variable, flag in (
        ("RPENT_ROBODOJO_SIM_PYTHON", "--sim-python"),
        ("RPENT_ROBODOJO_SOURCE_ROOT", "--source-root"),
    ):
        if os.environ.get(variable):
            flags.extend([flag, os.environ[variable]])
    args = parse_runtime_args(spec, flags)
    with runtime_phase(spec, args, output / "runtime", {"env"}) as kwargs:
        env = kwargs["env"]
        toolkit = RoboDojoToolkit(
            primitives_kwargs=kwargs,
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(output / "memory"),
            state_output_dir=output / "states",
        )
        try:
            initial = env.get_motion_state()
            record = toolkit._state.latest_record()
            for view in ("head", "left_wrist", "right_wrist"):
                rgb = require_rgb(toolkit._state.load(f"cam_{view}.png"), view)
                world = toolkit._state.load(f"{view}_world_xyz.npy")
                assert world.shape == (*rgb.shape[:2], 3)
                valid = np.argwhere(np.isfinite(world).all(axis=-1))
                assert len(valid)
                pixel = valid[len(valid) // 2].tolist()
                result = toolkit.execute_tool(
                    "sample_world_xyz",
                    {"view": view, "pixels": [pixel], "neighborhood": 0},
                ).data
                np.testing.assert_allclose(
                    result["samples"][0]["xyz"], world[tuple(pixel)]
                )
            for arm in ("left", "right"):
                measurements = []
                for value in (0.0, 0.5, 1.0):
                    before = env.get_motion_state()["targets"]
                    result = toolkit.execute_tool(
                        "set_gripper", {"arm": arm, "val": value}
                    )
                    assert not result.is_error, result.data
                    execution = result.data["log"]["result"]
                    assert execution["completed"] and execution["executed_steps"] == 10
                    measurements.append(env.get_motion_state()[f"{arm}_gripper"])
                    after = env.get_motion_state()["targets"]
                    for key in before:
                        if key != f"{arm}_ee_joint_state":
                            np.testing.assert_allclose(
                                after[key], before[key], atol=1e-6
                            )
                assert measurements[0] < measurements[1] < measurements[2], measurements
                assert measurements[2] - measurements[0] > 0.5, measurements
                pose = env.get_motion_state()[f"{arm}_eef_pose"]
                target = np.asarray(pose[:3]) + [0, 0, 0.015]
                result = toolkit.execute_tool(
                    "move_to", {"arm": arm, "xyz": target.tolist(), "substeps": 25}
                )
                assert not result.is_error, result.data
                execution = result.data["log"]["result"]
                assert execution["completed"] and execution["executed_steps"] > 0, (
                    execution
                )
                moved = env.get_motion_state()[f"{arm}_eef_pose"]
                check_pose_transition(pose, moved, z_delta=0.015)
                assert execution["final_dist_m"] < 0.005, execution
                result = toolkit.execute_tool(
                    "rotate_wrist", {"arm": arm, "delta_yaw_deg": 3}
                )
                assert not result.is_error, result.data
                assert result.data["log"]["result"]["completed"], result.data
                check_pose_transition(
                    moved, env.get_motion_state()[f"{arm}_eef_pose"], yaw_deg=3
                )
            # Hold advances actual simulator time without replacing held targets
            # with measurements; check both without and with an attached observer.
            events = []
            for attached in (False, True):
                if attached:
                    env.register_execution_observer(events.append)
                before = env.get_motion_state()
                count = env.get_status()["step"]
                result = toolkit.execute_tool("hold", {"steps": 3})
                assert not result.is_error, result.data
                receipt = result.data["log"]["result"]
                after = env.get_motion_state()
                assert receipt["executed_actions"] == 3 and receipt["completed"]
                assert receipt["step_before"] == count
                assert receipt["step_after"] == env.get_status()["step"] == count + 3
                assert after["simulation_time_s"] > before["simulation_time_s"]
                for key in before["targets"]:
                    np.testing.assert_array_equal(
                        before["targets"][key], after["targets"][key]
                    )
            for arm in ("left", "right"):
                before = env.get_motion_state()
                count = env.get_status()["step"]
                result = toolkit.execute_tool(
                    "move_delta",
                    {
                        "arm": arm,
                        "dxyz": [0, 0, 0.005],
                        "axis": [1, 2, 3],
                        "angle_deg": -2.0,
                        "frame": "world",
                        "substeps": 25,
                    },
                )
                assert not result.is_error, result.data
                receipt = result.data["log"]["result"]
                assert receipt["completed"] and receipt["executed_actions"] > 0, receipt
                assert env.get_status()["step"] - count == receipt["executed_actions"]
                after = env.get_motion_state()
                other = "right" if arm == "left" else "left"
                for suffix in ("arm_joint_state", "ee_joint_state"):
                    key = f"{other}_{suffix}"
                    np.testing.assert_array_equal(
                        before["targets"][key], after["targets"][key]
                    )
                key = f"{arm}_ee_joint_state"
                np.testing.assert_array_equal(
                    before["targets"][key], after["targets"][key]
                )
                assert not np.array_equal(
                    before["targets"][f"{arm}_arm_joint_state"],
                    after["targets"][f"{arm}_arm_joint_state"],
                )
            receipts = [event for event in events if event["kind"] == "action_feedback"]
            assert len(receipts) == env.get_status()["step"] - events[0]["step"]
            assert len({event["step_after"] for event in receipts}) == len(receipts)
            assert all(
                event["executed_actions"] == 1 and event["observation"]["vision"]
                for event in receipts
            )
            assert not any(event["kind"] == "reset" for event in events)
            env.reset()
            assert toolkit._state.get(record.step_idx).step_idx == record.step_idx
            final = env.get_motion_state()
            for arm in ("left", "right"):
                np.testing.assert_allclose(
                    final[f"{arm}_eef_pose"][:3],
                    initial[f"{arm}_eef_pose"][:3],
                    atol=0.02,
                )
            finish = toolkit.execute_tool(
                "finish", {"status": "stuck", "summary": "bounded primitive checks"}
            )
            assert finish.data["_finish"]
        finally:
            toolkit.close()
