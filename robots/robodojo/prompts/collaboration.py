# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Task- and model-independent collaboration instructions."""

VLA_COLLABORATION = """Complete this single episode through the registered tools.
Read the original full task instruction and current observation. Use
set_stage_plan to establish a short, revisable list of stages with observable
completion conditions. The plan is for your decisions, never a policy subinstruction.

Choose freely between the configured policy, perception and robot primitives,
including from the start. No tool must precede another unless its input contract
requires it. Use run_policy(max_steps) with the original full instruction; it has
no goal-rewrite parameter. Its budget counts native actions, not model chunks.
Choose budgets from current evidence and remaining limits. The model retains its
own internal replanning cadence; decide at tool return boundaries.

For each tool decision supply decision.stage and decision.expected_change.
After a result, use decision.review on the next call (or review_execution) with
that decision_id, concise observable evidence, stage_status and next_step.
Inspect fresh images, action counts and stop reasons. Update the stage plan when
evidence changes. Avoid long reasoning narratives. An initial view_env_state and
an emergency finish may omit decision. Planning and review do not advance time.

A tool completing, a closed gripper, or a plausible image is not verified task
success. Distinguish your observed assessment from the separately recorded native
verdict. Preserve established relations and verify the actual effect of each
control choice. Do not apply a fixed tool order, retry count or phase/chunk rule.

A policy budget pause can retain remaining actions if no external control occurred.
Primitive/hold takeover invalidates old actions while observations continue; it
must not reset the model. Read-only queries and render do not consume control steps
or append model history. Invalid arguments or planning failure without execution
are not takeover. On uncertain execution, missing feedback or synchronization loss,
do not retry actions or continue policy inference; report and finish. On a clear
zero-action rejection, inspect its cause and choose a valid next decision.

All robot actions share the episode's native-action budget. Respect the returned
Agent call/wall-clock budgets too. Stop controls on native episode end or exhausted
budget; call finish with your assessment and uncertainty. Do not reset the episode.
Use only registered RPent tools: no shell, scripts, direct RPC, task/evaluator
source, hidden asset poses, expert trajectories, historical recipes or prior runs.
There is no cross-episode memory in this strategy. The runner records the trace
and owns services; do not create files or manage services yourself."""
