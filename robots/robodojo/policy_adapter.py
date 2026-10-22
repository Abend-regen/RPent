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

"""Local policy interface for RoboDojo tools and control loops."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class RoboDojoPolicyAdapter:
    """Adapt legacy prediction and explicit stateful observation delivery.

    In stateful mode a step identifies one observation within the active episode.
    Repeating the latest step does not send it again; callers must associate the
    same control state and instruction with that step. Image contents are not
    compared. Legacy mode preserves ``predict(obs, options)`` and performs no
    observation or episode RPCs.
    """

    def __init__(self, client: Any, *, stateful: bool = False) -> None:
        self._client = client
        self._stateful = stateful
        self._active = False
        self._failed = False
        self._observed_step = -1

    def begin_episode(self, obs: dict[str, Any]) -> None:
        """Reset stateful history and deliver the initial observation at step zero."""
        if not self._stateful:
            return
        self._active = True
        self._failed = False
        self._observed_step = -1
        self._call(self._client.reset_session)
        self.observe(obs, step=0)

    def _require_episode(self) -> None:
        if not self._active or self._failed:
            raise RuntimeError("Policy episode is unavailable; begin a new episode")

    def _call(self, method: Callable[..., Any], *args: Any) -> Any:
        try:
            return method(*args)
        except Exception:
            # A failed response does not establish whether the model mutated.
            self.invalidate_episode()
            raise

    def observe(self, obs: dict[str, Any], *, step: int) -> None:
        """Deliver the next control observation, or reuse the already recorded step."""
        if not self._stateful:
            return
        self._require_episode()
        if type(step) is not int or step < 0:
            raise ValueError(
                "Stateful observations require a non-negative integer step"
            )
        if step == self._observed_step:
            return
        if step != self._observed_step + 1:
            raise ValueError(
                f"Expected observation step {self._observed_step + 1} "
                f"or current step {self._observed_step}, got {step}"
            )
        self._call(self._client.update_obs, obs, step)
        self._observed_step = step

    def predict(
        self,
        obs: dict[str, Any],
        options: dict | None = None,
        *,
        step: int | None = None,
    ) -> Any:
        """Request an action chunk, recording a new stateful observation if needed."""
        if not self._stateful:
            if options is None:
                return self._client.predict(obs)
            return self._client.predict(obs, options)
        if options is not None:
            raise ValueError("Stateful prediction does not accept inference options")
        if step is None:
            raise ValueError("Stateful prediction requires a control step")
        self.observe(obs, step=step)
        return self._call(self._client.get_action)

    def invalidate_episode(self) -> None:
        """Prevent further policy use after uncertain control or a discarded inference."""
        if self._stateful:
            self._failed = True

    @property
    def last_action_info(self) -> dict[str, Any]:
        """Metadata for the last inference, if the backend provides it."""
        return getattr(self._client, "last_action_info", {})

    def end_episode(self) -> None:
        """Release stateful history ownership, retaining the underlying transport."""
        if not self._stateful or not self._active:
            return
        try:
            self._call(self._client.end_episode)
        finally:
            self._active = False
