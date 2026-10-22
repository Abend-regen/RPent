# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in stage planning and evidence journal for existing Toolkit dispatch.

This layer owns no robot control, policy scheduling, action cache or observations.
The planner supplies stages and reviews; the Toolkit supplies actual results.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from rpent.evaluation import write_json_atomic
from rpent.tools import Tool, ToolResult, tool
from rpent.tools.toolkit import ToolCancelled

ShortText = Annotated[str, Field(min_length=1, max_length=1200)]


@tool(readonly=True)
def finish(*, status: str, summary: str) -> ToolResult:
    """End this attempt with your observed assessment, not a native task verdict."""
    return ToolResult(
        data={
            "_finish": True,
            "status": status,
            "summary": summary,
            "agent_assessment": status,
        }
    )


class Stage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: ShortText
    completion_condition: ShortText


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: int
    evidence: ShortText
    stage_status: Literal["pending", "in_progress", "complete", "blocked", "uncertain"]
    next_step: ShortText


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: ShortText
    expected_change: ShortText
    review: Review | None = None


def with_decision(declaration: Tool) -> Tool:
    """Add planner intent to a declaration, preserving its native handler."""
    schema = create_model(
        declaration.args_schema.__name__ + "Decision",
        __base__=declaration.args_schema,
        decision=(Decision | None, None),
    )

    def invoke(*, decision=None, **kwargs):
        return declaration(**kwargs)

    return replace(
        declaration,
        args_schema=schema,
        _input_schema=schema.model_json_schema(),
        handler=invoke,
        _unbound_method=False,
        description=declaration.description
        + " Supply decision(stage, expected_change, review) from your current plan; "
        "review the previous decision using its returned decision_id and visible evidence.",
    )


class AgentCollaboration:
    """Persist planner intent, actual feedback and later review for each call."""

    def __init__(self, directory: Path, *, max_calls: int, timeout_s: float):
        if max_calls < 1 or timeout_s <= 0:
            raise ValueError("Agent call and wall-clock budgets must be positive")
        self.path = directory / "agent-decisions.json"
        self.max_calls, self.timeout_s = max_calls, timeout_s
        self.started = time.monotonic()
        self.calls = 0
        self.plans: list[dict] = []
        self.decisions: list[dict] = []
        self.closed = False
        self.finished = False
        self.lock = RLock()
        self._save()

    def _save(self):
        write_json_atomic(
            self.path,
            {
                "plans": self.plans,
                "decisions": self.decisions,
                "calls": self.calls,
                "max_calls": self.max_calls,
                "timeout_s": self.timeout_s,
                "closed": self.closed,
                "finished": self.finished,
            },
        )

    def check_deadline(self):
        if time.monotonic() - self.started >= self.timeout_s:
            raise ToolCancelled("Agent wall-clock budget exhausted")

    @tool(readonly=True)
    def set_stage_plan(
        self, stages: Annotated[list[Stage], Field(min_length=1, max_length=12)]
    ) -> ToolResult:
        """Create or revise a concise plan with observable completion conditions.

        Stage text guides your tool choices only; it is never sent to the policy.
        Keep names unique. This call does not move the robot or reset the model.
        """
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError("Stage names must be unique")
        self.plans.append(
            {
                "revision": len(self.plans) + 1,
                "stages": [s.model_dump() for s in stages],
            }
        )
        self._save()
        return ToolResult(data=self.plans[-1])

    @tool(readonly=True)
    def review_execution(
        self,
        decision_id: int,
        evidence: ShortText,
        stage_status: Literal[
            "pending", "in_progress", "complete", "blocked", "uncertain"
        ],
        next_step: ShortText,
    ) -> ToolResult:
        """Record visible evidence, stage status and next step after a tool return."""
        self._review(
            Review(
                decision_id=decision_id,
                evidence=evidence,
                stage_status=stage_status,
                next_step=next_step,
            )
        )
        self._save()
        return ToolResult(data={"reviewed_decision_id": decision_id})

    def _review(self, review: Review):
        if not 1 <= review.decision_id <= len(self.decisions):
            raise ValueError("Unknown decision_id")
        entry = self.decisions[review.decision_id - 1]
        entry.update(
            review=review.model_dump(),
            stage_status=review.stage_status,
            next_step=review.next_step,
        )

    def _budget_rejection(self, name, arguments, message):
        intent = arguments.get("decision") or {}
        result = ToolResult(
            data={"error": message, "code": "agent_budget", "next_step": "finish"}
        )
        entry = {
            "decision_id": len(self.decisions) + 1,
            "stage": intent.get("stage") if isinstance(intent, dict) else None,
            "expected_change": intent.get("expected_change")
            if isinstance(intent, dict)
            else None,
            "tool": name,
            "arguments": deepcopy(arguments),
            "control": False,
            "feedback": deepcopy(result.to_dict()),
            "review": None,
            "stage_status": "blocked",
            "next_step": "finish",
            "image_count": 0,
        }
        self.decisions.append(entry)
        result.data["decision_id"] = entry["decision_id"]
        self._save()
        return result

    def dispatch(
        self, name: str, arguments: dict, execute, *, mutating: bool
    ) -> ToolResult:
        """Serialize the journal around the existing Toolkit, without choosing tools."""
        with self.lock:
            if self.closed or self.finished:
                return ToolResult(data={"error": "Agent run has ended"})
            if name != "finish":
                if self.calls >= self.max_calls:
                    return self._budget_rejection(
                        name, arguments, "Agent tool-call budget exhausted"
                    )
                try:
                    self.check_deadline()
                except ToolCancelled as exc:
                    return self._budget_rejection(name, arguments, str(exc))
            self.calls += 1
            intent = None
            error = None
            try:
                if arguments.get("decision") is not None:
                    intent = Decision.model_validate(arguments["decision"], strict=True)
                    if intent.review is not None:
                        self._review(intent.review)
                    names = (
                        {s["name"] for s in self.plans[-1]["stages"]}
                        if self.plans
                        else set()
                    )
                    if intent.stage not in names:
                        raise ValueError(
                            "decision.stage must name a stage in set_stage_plan"
                        )
                requires_intent = name not in {
                    "set_stage_plan",
                    "review_execution",
                    "finish",
                } and not (name == "view_env_state" and not self.decisions)
                if requires_intent and intent is None:
                    raise ValueError(
                        "Set a stage plan and supply decision for this tool call"
                    )
                if (
                    mutating
                    and self.decisions
                    and (
                        self.decisions[-1]["review"] is None
                        or any(
                            d["control"] and d["review"] is None for d in self.decisions
                        )
                    )
                ):
                    raise ValueError(
                        "Review the previous tool result before the next robot action"
                    )
            except ValueError as exc:
                error = str(exc)
            journal_tool = name in {"set_stage_plan", "review_execution"}
            entry = None
            if not journal_tool:
                entry = {
                    "decision_id": len(self.decisions) + 1,
                    "stage": intent.stage if intent else None,
                    "expected_change": intent.expected_change if intent else None,
                    "tool": name,
                    "arguments": deepcopy(
                        {k: v for k, v in arguments.items() if k != "decision"}
                    ),
                    "control": mutating,
                    "feedback": None,
                    "review": None,
                    "stage_status": "awaiting_review",
                    "next_step": None,
                }
                self.decisions.append(entry)
                self._save()  # Persist intent before any mutation.
            result = (
                ToolResult(data={"error": error}) if error else execute(name, arguments)
            )
            if entry is not None:
                entry["feedback"] = deepcopy(result.to_dict())
                entry["image_count"] = len(result.images)
                entry["elapsed_s"] = round(time.monotonic() - self.started, 3)
                result.data["decision_id"] = entry["decision_id"]
            if result.data.get("_finish"):
                self.finished = True
                if entry is not None:
                    entry.update(stage_status="closed", next_step="none")
            result.data["agent_budget"] = {
                "calls_remaining": max(0, self.max_calls - self.calls),
                "wall_seconds_remaining": max(
                    0, round(self.timeout_s - (time.monotonic() - self.started), 1)
                ),
            }
            self._save()
            return result

    def close(self):
        with self.lock:
            self.closed = True
            self._save()
