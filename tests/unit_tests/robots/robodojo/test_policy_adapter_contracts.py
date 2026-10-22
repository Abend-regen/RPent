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

"""RoboDojo's local prediction and observation-delivery contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo.policy_adapter import RoboDojoPolicyAdapter


class StatefulClient:
    def __init__(self):
        self.calls = []
        self.history = []
        self.actions = [{"joint": [1.0]}]
        self.failure = None

    def _record(self, method, *args):
        self.calls.append((method, *args))
        if self.failure == method:
            raise RuntimeError(f"{method} response lost")

    def reset_session(self):
        self.history.clear()
        self._record("reset")

    def update_obs(self, obs, step):
        self.history.append((obs, step))
        self._record("observe", step)

    def get_action(self):
        self._record("infer")
        return self.actions

    def end_episode(self):
        self._record("end")


@pytest.mark.parametrize("options", [None, {"horizon": 8}])
def test_legacy_predict_preserves_payload_result_and_call_count(options):
    obs = {"image": np.zeros((2, 2, 3), dtype=np.uint8)}
    actions = np.zeros((8, 14), dtype=np.float32)
    calls = []

    def predict(*args):
        calls.append(args)
        return actions

    # Only predict is available. Episode and observation hooks must be no-ops.
    adapter = RoboDojoPolicyAdapter(SimpleNamespace(predict=predict))
    adapter.begin_episode(obs)
    adapter.observe(obs, step=0)
    assert adapter.predict(obs, options) is actions
    adapter.invalidate_episode()
    adapter.end_episode()
    assert len(calls) == 1
    assert calls[0][0] is obs
    assert len(calls[0]) == (1 if options is None else 2)
    if options is not None:
        assert calls[0][1] is options


def test_legacy_predict_preserves_backend_exception():
    failure = RuntimeError("legacy backend failed")

    def predict(obs):
        raise failure

    adapter = RoboDojoPolicyAdapter(SimpleNamespace(predict=predict))
    with pytest.raises(RuntimeError) as result:
        adapter.predict({})
    assert result.value is failure


@pytest.mark.parametrize("observe_first", [False, True])
def test_stateful_predict_sends_each_control_observation_once(observe_first):
    client = StatefulClient()
    adapter = RoboDojoPolicyAdapter(client, stateful=True)
    initial, next_obs = {"frame": 0}, {"frame": 1}
    adapter.begin_episode(initial)
    adapter.observe(initial, step=0)
    assert adapter.predict(initial, step=0) is client.actions
    if observe_first:
        adapter.observe(next_obs, step=1)
    assert adapter.predict(next_obs, step=1) is client.actions
    adapter.observe(next_obs, step=1)
    assert client.calls == [
        ("reset",),
        ("observe", 0),
        ("infer",),
        ("observe", 1),
        ("infer",),
    ]
    assert client.history == [(initial, 0), (next_obs, 1)]


@pytest.mark.parametrize("step", [-1, 0, 3, True, 2.0, None])
def test_invalid_steps_do_not_reach_backend(step):
    client = StatefulClient()
    adapter = RoboDojoPolicyAdapter(client, stateful=True)
    adapter.begin_episode({})
    adapter.observe({}, step=1)
    calls = list(client.calls)
    with pytest.raises(ValueError, match="step"):
        adapter.predict({}, step=step)
    assert client.calls == calls
    # Invalid input has not changed history; the next correct step is usable.
    adapter.predict({}, step=2)
    assert client.calls[-2:] == [("observe", 2), ("infer",)]


def test_stateful_options_are_rejected_before_observation_delivery():
    client = StatefulClient()
    adapter = RoboDojoPolicyAdapter(client, stateful=True)
    adapter.begin_episode({})
    with pytest.raises(ValueError, match="options"):
        adapter.predict({}, {"horizon": 1}, step=1)
    assert client.calls == [("reset",), ("observe", 0)]


@pytest.mark.parametrize("failure", ["reset", "observe", "infer"])
def test_failed_model_mutation_cannot_be_retried_without_new_episode(failure):
    client = StatefulClient()
    adapter = RoboDojoPolicyAdapter(client, stateful=True)
    if failure != "reset":
        adapter.begin_episode({})
    client.failure = failure
    with pytest.raises(RuntimeError, match="response lost"):
        if failure == "reset":
            adapter.begin_episode({})
        else:
            adapter.predict({}, step=1)
    calls = list(client.calls)
    for step in (0, 1, 2):
        with pytest.raises(RuntimeError, match="unavailable"):
            adapter.predict({}, step=step)
        with pytest.raises(RuntimeError, match="unavailable"):
            adapter.observe({}, step=step)
    assert client.calls == calls
    client.failure = None
    adapter.begin_episode({"new_episode": True})
    assert client.history == [({"new_episode": True}, 0)]
    adapter.predict({"new_episode": True}, step=0)


def test_episode_release_and_restart_reuse_transport_and_reset_step_identity():
    client = StatefulClient()
    adapter = RoboDojoPolicyAdapter(client, stateful=True)
    with pytest.raises(RuntimeError, match="unavailable"):
        adapter.predict({}, step=0)
    adapter.begin_episode({"frame": 0})
    adapter.predict({"frame": 1}, step=1)
    adapter.end_episode()
    adapter.end_episode()
    assert client.calls.count(("end",)) == 1
    with pytest.raises(RuntimeError, match="unavailable"):
        adapter.predict({}, step=2)
    adapter.begin_episode({"second": 0})
    assert client.history == [({"second": 0}, 0)]


def test_execution_failure_invalidates_adapter_until_next_episode():
    client = StatefulClient()
    adapter = RoboDojoPolicyAdapter(client, stateful=True)
    adapter.begin_episode({})
    adapter.invalidate_episode()
    with pytest.raises(RuntimeError, match="unavailable"):
        adapter.predict({}, step=0)
    adapter.end_episode()
    adapter.begin_episode({})
    adapter.predict({}, step=0)
