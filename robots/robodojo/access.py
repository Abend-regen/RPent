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

"""Positive observation allowlist for frozen-plan replay."""

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = tuple(
    f"{arm}_{field}"
    for arm in ("left", "right")
    for field in ("ee_pose", "ee_joint_state", "arm_joint_state")
)
CAMERA_KEYS = (
    "color",
    "depth",
    "distance_to_image_plane",
    "intrinsic_matrix",
    "extrinsic_matrix",
    "shape",
)


def public_observation(obs: dict) -> dict:
    """Keep RGB-D, calibration, proprioception and the public instruction only."""
    return {
        "instruction": obs.get("instruction", ""),
        "state": {
            key: obs["state"][key] for key in STATE_KEYS if key in obs.get("state", {})
        },
        "vision": {
            name: {key: cam[key] for key in CAMERA_KEYS if key in cam}
            for name, cam in obs.get("vision", {}).items()
            if name in CAMERAS
        },
    }


def validate_rgbd(obs: dict) -> None:
    """Require synchronized, calibrated RGB-D for every public camera."""
    import numpy as np

    for name in CAMERAS:
        camera = obs.get("vision", {}).get(name, {})
        rgb = np.asarray(camera.get("color"))
        depth = camera.get("distance_to_image_plane")
        if depth is None:
            depth = camera.get("depth")
        intrinsic = np.asarray(camera.get("intrinsic_matrix"), dtype=float)
        extrinsic = np.asarray(camera.get("extrinsic_matrix"), dtype=float)
        if (
            rgb.ndim != 3
            or rgb.shape[-1] != 3
            or rgb.dtype != np.uint8
            or np.asarray(depth).shape != rgb.shape[:2]
            or intrinsic.shape != (3, 3)
            or extrinsic.shape != (4, 4)
            or not np.isfinite(intrinsic).all()
            or not np.isfinite(extrinsic).all()
            or intrinsic[0, 0] <= 0
            or intrinsic[1, 1] <= 0
        ):
            raise ValueError(
                f"VLA Agent requires same-frame RGB-D and calibration: {name}"
            )
