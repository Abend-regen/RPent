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

"""Stateful RPent bridge for a separately managed official Awomo05 service."""

from __future__ import annotations

import argparse
import signal

from robots.robodojo.awomo05_contract import SESSION_METHODS, validate_model_meta
from rpent.robots.components.xpolicylab_vla_server import (
    XPolicyLabVLAFacade,
    add_backend_args,
)


class Awomo05Facade(XPolicyLabVLAFacade):
    """Reuse the phase-one session protocol and validate upstream capabilities."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not args.policy_server_url:
            raise ValueError("Awomo05 bridge requires --policy-server-url")
        args.stateful = True
        super().__init__(args)
        try:
            try:
                meta = self._model_client.call(func_name="get_model_meta")
            except Exception as exc:
                raise RuntimeError(
                    "Upstream Awomo05 must expose get_model_meta; start it with "
                    "robots/robodojo/awomo05_policy_server.py"
                ) from exc
            self._meta = {
                **validate_model_meta(meta),
                "stateful": True,
                "rpc_methods": list(SESSION_METHODS),
            }
        except BaseException:
            self.close()
            raise

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc["vla.get_model_meta"] = self.get_model_meta
        self._readonly_methods.add("vla.get_model_meta")

    def get_model_meta(self, *, session_id: str | None = None) -> dict:
        return dict(self._meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    facade = None
    try:
        facade = Awomo05Facade(args)
        facade.serve(
            transport="http",
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
            session_sweep_s=60.0,
        )
    finally:
        if facade is not None:
            facade.close()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
