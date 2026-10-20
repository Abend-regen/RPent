RoboDojo
========

Run bimanual tabletop tasks with RPent in RoboDojo. The simulator uses Isaac
Sim / IsaacLab with an ARX-X5 dual-arm platform and Pi0.5 as the default action model.
The integration is experimental; success rates across a complete task suite
have not been established.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Pi0.5

   .. grid-item-card:: Planners

      ``api``, ``claude_code``, ``codex``

   .. grid-item-card:: Tasks

      Bottle disposal, pen-holder filling, bowl stacking, and other tabletop tasks.

   .. grid-item-card:: Hardware

      Linux, NVIDIA GPU; Python 3.11; Isaac Sim 5.1.

Tasks
~~~~~

The `RoboDojo simulation task catalog <https://robodojo-benchmark.com/doc/sim-tasks/>`_
groups 42 benchmark tasks into five capability categories:

.. list-table::
   :header-rows: 1

   * - Category
     - Tasks
     - Focus
   * - Generalization
     - 12
     - Varied objects and layouts, including random variants.
   * - Memory
     - 6
     - State tracking and sequence recall.
   * - Precision
     - 8
     - Manipulation with tight spatial constraints.
   * - Long-Horizon
     - 8
     - Multi-step tasks with several subgoals.
   * - Open
     - 8
     - Open-ended or language- and image-conditioned tasks.

The catalog also lists one separate DLC task. Use ``--task`` to select a task
from the installed runtime and ``--layout`` to select its scene layout. For
example, ``stack_bowls_random`` is a generalization task, while
``fill_pen_holder`` and ``put_bottles_into_dustbin`` are long-horizon tasks.
These are examples, not the complete task list.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Description
   * - Observation
     - Head and left/right wrist RGB-D images, camera calibration, and joint states, end-effector poses, and gripper states for both arms. Pi0.5 uses three RGB views, robot state, and task text.
   * - Action
     - The planner calls ``pi0_pick`` or primitives such as ``move_to`` and ``set_gripper``. Pi0.5 outputs 14-dimensional joint and gripper actions for both arms.
   * - Reward / success
     - RoboDojo's native environment determines task success, recorded in the ``success`` field of ``result.json``.
   * - Task prompt
     - The current scene's RoboDojo task description. ``pi0_pick`` uses this instruction when its ``prompt`` argument is omitted.

Installation and Resources
--------------------------

For NVIDIA driver recommendations, see the `Isaac Sim 5.1 system requirements
<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_.
The host also needs the CUDA toolkit/NVCC and a compiler toolchain to build
cuRobo. From the RPent root, create an environment and install the dependencies:

.. code-block:: bash

   uv venv --python 3.11 .venv-robodojo
   source .venv-robodojo/bin/activate
   uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

.. note::

   Blackwell GPUs need the CUDA 12.8 build of PyTorch. The pinned PyTorch 2.7.0
   also has a CUDA 12.6 build, but that build does not support Blackwell. On
   these GPUs, use the following command for both initial installation and
   reinstallation; ``--torch-backend=cu128`` selects the CUDA 12.8 build:

   .. code-block:: bash

      uv pip install -e ".[robodojo]" --extra-index-url https://pypi.nvidia.com \
        --override requirements/robodojo-override.txt \
        --torch-backend=cu128

   If ``torch==2.7.0+cu126`` is already installed, uv considers it compatible
   with the version requirement and keeps it. Append
   ``--reinstall-package torch --reinstall-package torchvision`` to the command
   above to replace it with the CUDA 12.8 build.

The pinned IsaacLab revision needs an editable installation to locate its
extension configuration. Clone it into a writable directory and install it
in the same environment:

.. code-block:: bash

   git clone --no-checkout https://github.com/yuechen0614/IsaacLab.git /path/to/IsaacLab
   git -C /path/to/IsaacLab checkout afca7b09d60d8beb9c1cb28b43066499940b969b
   uv pip install --no-deps \
     -e /path/to/IsaacLab/source/isaaclab \
     -e /path/to/IsaacLab/source/isaaclab_assets \
     -e /path/to/IsaacLab/source/isaaclab_tasks

Keep this checkout after installation; the runtime still uses it.

Download Simulator Resources
----------------------------

Download the robot, object, material, and layout files from the
`RoboDojo dataset <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_
and set the resource directory:

.. code-block:: bash

   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
     --include 'Assets/**' --local-dir /data/robodojo
   export ROBODOJO_ASSETS_ROOT=/data/robodojo

The download should contain ``Assets/Robots``, ``Assets/Object``,
``Assets/Material``, and ``Assets/Eval_Layout``.

Generate a ``*.yml`` configuration beside each ``*_tmp.yml`` template under
``Assets/Robots``, substituting absolute local resource paths. The simulator
reads ``Assets/Robots/x5/curobo.yml`` for x5:

.. code-block:: bash

   python - <<'PY'
   import os
   import runpy
   from robodojo_runtime import source_root

   script = source_root() / "utils/update_embodiment_config_path.py"
   os.chdir(os.environ["ROBODOJO_ASSETS_ROOT"])
   runpy.run_path(str(script), run_name="__main__")
   PY
   test -s "$ROBODOJO_ASSETS_ROOT/Assets/Robots/x5/curobo.yml"

Run this once after the initial download; subsequent task runs do not need it.
Moving the resource directory invalidates the paths saved in these configurations,
so run the command again afterward. Regeneration overwrites existing ``*.yml``
configurations.

Some NVIDIA resources referenced by the scene load online by default.
For offline use or repeated resource-loading failures, see
:ref:`robodojo-local-nvidia-assets` at the end of this page.

VLA Configuration
-----------------

Use ``--policy-backend`` to select the Pi0.5 backend. The default is ``rlinf``;
``xpolicylab`` is available for a separately prepared runtime.

RLinf (Default)
~~~~~~~~~~~~~~~

``--policy-backend rlinf`` uses the RLinf runtime installed above.
``PI05_CHECKPOINT_PATH`` selects its PyTorch weights and normalization statistics.

The released weights use JAX format and need conversion to PyTorch.
This example downloads only the complete ``params/`` directory and matching
``norm_stats.json`` from ``RoboDojo-sim-arx_x5-joint-0/59999``, about 12.4 GB
in total. Training state and other models are unnecessary. The converted
weights are about 7.23 GB; allow extra space for conversion and caches.

.. code-block:: bash

   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
     --include \
       'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/params/**' \
       'ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets/arx_x5_sim/norm_stats.json' \
     --local-dir /data/robodojo/pi05_jax
   export PI05_JAX_CHECKPOINT=/data/robodojo/pi05_jax/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999

.. note::

   Give the ``--local-dir`` directory a lowercase ``pi05_`` prefix, as in
   ``pi05_jax`` above. The current converter requires lowercase ``pi05`` in
   its input path; the release directory name ``Pi_05`` does not satisfy this.

   If your existing download path has no lowercase ``pi05``, you do not need
   to download again. Create a symlink with a ``pi05_`` prefix to the model
   directory containing ``59999/``, then set ``PI05_JAX_CHECKPOINT`` to
   ``59999`` under that link. For a download rooted at ``/data/robodojo``:

   .. code-block:: bash

      ln -s /data/robodojo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0 \
        /data/robodojo/pi05_robodojo_arx_x5
      export PI05_JAX_CHECKPOINT=/data/robodojo/pi05_robodojo_arx_x5/59999

Convert the weights in the same environment and set the converted model path:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/data/robodojo/pi05_robodojo_arx_x5_torch

   python -m rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python \
     --checkpoint-dir "$PI05_JAX_CHECKPOINT" \
     --config-name pi05_aloha \
     --output-path "$PI05_CHECKPOINT_PATH"

``--checkpoint-dir`` must point to the directory containing ``params/``.

Add ``--inspect_only`` to view parameter information without converting.

The model also needs the ``norm_stats.json`` file supplied with its weights.
The converter does not write this file to
``$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5/norm_stats.json``, so copy it explicitly:

.. code-block:: bash

   mkdir -p "$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5"
   cp "$PI05_JAX_CHECKPOINT/assets/arx_x5_sim/norm_stats.json" \
     "$PI05_CHECKPOINT_PATH/RoboDojo/arx_x5/norm_stats.json"

XPolicyLab (Optional)
~~~~~~~~~~~~~~~~~~~~~

Use the RLinf setup above unless you need XPolicyLab. The default backend is
``--policy-backend rlinf``, so no XPolicyLab installation is needed. RPent also
provides ``--policy-backend xpolicylab`` for users who already have that
environment or want its Pi0.5 implementation.

XPolicyLab is another software stack for loading and running Pi0.5. Clone its
source separately and prepare its Python environment following the
`Pi0.5 installation instructions
<https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/README.md#installation>`_.
You can reuse the original JAX weights and statistics downloaded above,
but must configure their location as XPolicyLab expects.
After configuring SAM3 and your planner as described below, run a task with
the XPolicyLab checkout and its Python environment:

.. code-block:: bash

   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> \
     --policy-backend xpolicylab --xpolicylab-root /path/to/XPolicyLab \
     --pi05-python /path/to/xpolicylab-env/bin/python \
     --output-dir /path/to/run-output

``--xpolicylab-root`` points to the separate XPolicyLab checkout.
If the Python packages needed for XPolicyLab's Pi0.5 inference are installed in
a separate environment, point ``--pi05-python`` to that environment's Python
interpreter; otherwise omit the option.

Stateful XPolicyLab Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a policy that needs control-step history, prepare a dedicated XPolicyLab
WebSocket server implementing ``reset``, ``update_obs`` and ``get_action``.
The model owns its history buffer, preprocessing and action-chunk length.
Use a separate server and port from any ongoing evaluation.

In a Python environment containing RPent and XPolicyLab's client dependencies,
start the bridge with the XPolicyLab checkout on ``PYTHONPATH``:

.. code-block:: bash

   PYTHONPATH=/path/to/XPolicyLab python -m rpent.robots.components.xpolicylab_vla_server \
     --task stack_blocks_by_language --stateful \
     --policy-server-url ws://127.0.0.1:19105 \
     --transport http --host 127.0.0.1 --port 19106

With the simulator, SAM3 and planner configured as described on this page,
connect RPent to that bridge:

.. code-block:: bash

   rpent --robot robodojo --task stack_blocks_by_language --layout 0 \
     --planner codex --model <planner-model> \
     --policy-backend xpolicylab --stateful-policy \
     --vla-endpoint http://127.0.0.1:19106 \
     --output-dir /path/to/stateful-run

This opt-in path requires an external bridge endpoint. The default RLinf and
XPolicyLab Pi0.5 launch paths retain their existing behavior.
``run_policy(max_steps)`` replaces ``pi0_pick`` in the tool list. It executes
bounded action chunks using the official task instruction. Each completed
control step, including scripted motions, sends one observation; reading or
recording the scene sends none. Chunk boundaries do not duplicate observations.
The model is reset once when the toolkit starts, and environment resets also
reset model history. Repeated tool calls in an episode preserve that history.

The bridge uses RPent RPC sessions and allows one history owner at a time.
Another session cannot reset or infer through the occupied bridge. Closing or
expiring the owner session releases ownership; toolkit close also releases the
episode while retaining the VLA connection for later tasks. A new episode must
reset before use. This does not isolate clients that bypass the bridge: dedicate the
upstream model server to this bridge. Flash replay is not supported in this
mode.

Observation steps start at zero and must be contiguous. Observation or inference
failures invalidate the episode because a lost response may follow a successful
model mutation. The execution loop also stops after an uncertain environment
action or cancellation within an inferred chunk. Do not automatically retry
these calls; start a new episode. Planner tools do not expose an episode reset.
Episode end and action execution are not task-success evidence.

Within RoboDojo, both ``pi0_pick`` and the stateful execution loop use
``RoboDojoPolicyAdapter`` from ``robots.robodojo.policy_adapter``. Legacy
``predict(obs, options)`` calls pass through unchanged; episode and observation
hooks make no backend calls in legacy mode. Stateful callers use
``begin_episode(obs)`` to reset and record step zero, then
``observe(obs, step=t)`` and ``predict(obs, step=t)``. Prediction also records
the next observation if it has not yet been delivered, so either call order
sends each control step once. Repeating the latest observation step is a no-op;
older or skipped steps are rejected. Step identity refers to the same control
state and instruction within one episode, not equality of image contents.
Repeated inference without a new observation remains disallowed by the
stateful bridge. ``end_episode()`` releases ownership for the next task.
The adapter is local to RoboDojo; other robots and the shared VLA interfaces
keep their existing contracts.

CPU-only tests exercise the bridge, both RPC transports, toolkit execution,
ownership, reset, cancellation and failure handling with a fake model and
environment:

.. code-block:: bash

   pytest tests/unit_tests/robots/robodojo/test_policy_adapter_contracts.py \
     tests/unit_tests/robots/robodojo/test_stateful_policy_loopback.py -v

Awomo-0.5 backend
^^^^^^^^^^^^^^^^^

``--policy-backend awomo05`` uses the official XPolicyLab ``Awomo05`` model
through the stateful RoboDojo adapter. It requires ``arx_x5`` and absolute
``joint`` actions, and automatically enables ``run_policy``. Flash replay is
not supported. The default RLinf/Pi0.5 path is unchanged.

Prepare the official model environment, checkpoint and matching statistics
using XPolicyLab's Awomo05 installation instructions. Keep model and simulator
interpreters separate. In the model environment, start this thin launcher from
the RPent checkout; it uses the official model and adds read-only metadata:

.. code-block:: bash

   export XPL_ROOT=/path/to/RoboDojo/XPolicyLab
   export PYTHONPATH="$(dirname "$XPL_ROOT"):$XPL_ROOT:${PYTHONPATH:-}"
   CUDA_VISIBLE_DEVICES=<free-model-gpu> /path/to/awomo/python \
     robots/robodojo/awomo05_policy_server.py \
     --config_path "$XPL_ROOT/policy/Awomo05/deploy.yml" \
     --host 127.0.0.1 --port 19105 \
     --overrides policy_name=Awomo05 bench_name=RoboDojo \
     task_name=stack_blocks_by_language env_cfg_type=arx_x5 action_type=joint \
     seed=0 ckpt=model.pt input_color_order=bgr replan_steps=8

Use the official weight-path environment variables and model configuration.
The launcher does not copy model code or alter inference, normalization,
history sampling or AR state. An unmodified older service without
``get_model_meta`` is rejected before reset; launch a dedicated compatible
service rather than probing it with state-changing calls.

Choose exactly one connection mode. For an existing official WebSocket service,
RPent starts only a lightweight bridge. The bridge interpreter needs RPent and
XPolicyLab's client dependencies (including WebSockets, msgpack and
msgpack-numpy); it does not load model weights:

.. code-block:: bash

   rpent --robot robodojo --task stack_blocks_by_language --layout 0 \
     --planner codex --model <planner-model> --policy-backend awomo05 \
     --policy-server-url ws://127.0.0.1:19105 \
     --xpolicylab-root "$XPL_ROOT" --policy-python /path/to/bridge/python \
     --env-endpoint http://127.0.0.1:19106 --output-dir /path/to/awomo-run

Alternatively, start a reusable RPent Awomo bridge with the XPolicyLab checkout
and its parent on ``PYTHONPATH``:

.. code-block:: bash

   /path/to/bridge/python -m robots.robodojo.awomo05_bridge \
     --task stack_blocks_by_language --policy-server-url ws://127.0.0.1:19105 \
     --host 127.0.0.1 --port 19107

Then replace ``--policy-server-url``, ``--xpolicylab-root`` and
``--policy-python`` in the RPent command with
``--vla-endpoint http://127.0.0.1:19107``. The two endpoint options are mutually
exclusive. ``--policy-python`` aliases the existing ``--pi05-python`` option.
Both modes validate model identity, native action layout, cameras, history,
replanning and stateful capabilities before starting an episode. Configure
the planner as described elsewhere on this page. Awomo runs do not start SAM3
unless that component is explicitly requested.

An independently managed model service is borrowed: RPent does not stop it.
Give one active run exclusive access to both the environment and model service.
The bridge's session ownership cannot prevent another official client or a
second bridge from resetting the same upstream model.

The wire observation keeps RGB ``uint8`` HWC images in the order ``cam_head``,
``cam_left_wrist``, ``cam_right_wrist``. The released checkpoint's BGR conversion
occurs once inside the official adapter. State packing there is left arm (6),
left gripper (1), right arm (6), right gripper (1). Returned actions retain those
four named fields; they never pass through Pi0.5's different flattened layout.
Joint targets are absolute radians, with normalized gripper opening (0 closed,
1 open) mapped by the native environment. Each control target represents 0.04 s
of simulated time, not a wall-clock delay imposed by network inference.

For Awomo, the local execution loop is the sole owner of unexecuted returned
actions. Calls such as ``run_policy(3)`` followed by ``run_policy(5)`` consume one
8-action chunk when the episode, instruction and environment step remain
consistent. A scripted action drops the remainder and sends its new observation
without resetting model history. Cancellation, errors, instruction changes or
unobserved environment advancement discard the remainder and block further
execution until an explicit episode reset. Episode end drops remaining actions
and does not send a terminal observation, matching the official deploy loop.
Cache invalidation itself never invokes model reset. Same-step out-of-band
resets cannot be detected from the current environment API; use exclusive
endpoints and perform resets through the execution boundary.

Run the offline regression tests without model weights:

.. code-block:: bash

   pytest tests/unit_tests/robots/robodojo/test_awomo05_contracts.py \
     tests/unit_tests/robots/robodojo/test_awomo05_loopback.py \
     tests/unit_tests/robots/robodojo/test_awomo05_policy_tool.py -v

For a bounded real-model check, prepare exclusive test endpoints and run:

.. code-block:: bash

   RPENT_RUN_AWOMO05_INTEGRATION=1 \
   RPENT_E2E_OUTPUT_DIR=/path/to/new-evidence \
   RPENT_ROBODOJO_ENV_ENDPOINT=http://127.0.0.1:19106 \
   RPENT_ROBODOJO_STEP_LIMIT=40 \
   RPENT_AWOMO05_POLICY_SERVER_URL=ws://127.0.0.1:19105 \
   RPENT_XPOLICYLAB_ROOT="$XPL_ROOT" \
     pytest tests/e2e_tests/robodojo/test_awomo05.py -v -s

The environment must be configured for the selected task, layout 0 and the
specified step limit. ``RPENT_ROBODOJO_TASK`` defaults to
``stack_blocks_by_language``, an official open task. To test a borrowed bridge,
set ``RPENT_AWOMO05_VLA_ENDPOINT`` instead of
``RPENT_AWOMO05_POLICY_SERVER_URL``. The test executes 3+5+2 control steps, resets
and executes two more; it writes ``awomo05-result.json`` and closes the test
environment. Early native success is a distinct outcome from passing this fixed
cadence check. This test is opt-in and is not part of the GPU suite runner.

Full-episode policy tool validation
"""""""""""""""""""""""""""""""""""

The Awomo toolkit exposes ``run_policy`` for robot actions, alongside
``view_env_state`` and common audit/finish tools. It does not expose grasp,
placement or scripted recovery tools. Its prompt keeps the original task
instruction; RPent does not rewrite subtasks or stop when a grasp is detected.
A budget return pauses execution while preserving valid pending actions.
Continue until ``episode_ended`` or an execution error. A cancelled or failed
call must not be retried automatically.

For Awomo calls, ``log.result`` reports ``executed_steps``, ``observation_step``,
``episode_ended``, ``pending_actions`` and ``stop_reason`` (``budget``,
``episode_end``, ``cancelled`` or ``error``). Successful execution does not imply
task success. Complete per-inference metadata is recorded in the action's
``result.inferences`` in ``states.json``; the immediate tool response includes
``inference_count`` instead of those details. Each inference is recorded only
in the call that generated its chunk, even when later calls consume its tail.

A bridge advertising ``action_info`` accepts ``include_info=True`` on
``vla.get_action``. The default still returns actions, preserving existing
clients. The Awomo client also continues to return an action list. Metadata
contains the upstream response fields (including ``latency_ms`` when supplied),
model result metadata and the measured ``bridge_rpc_s``. The execution report
separately measures ``policy_rpc_s``, ``observation_rpc_s``, ``env_step_s`` and
``elapsed_s`` in seconds. These nested wall-clock durations must not be added
together as independent costs; ``elapsed_s`` excludes subsequent state capture
and planner work. Setup/reset and the initial observation are outside the call.
An older bridge without this capability remains usable and yields empty
inference ``info``. Missing upstream timings are unavailable, not zero.

To check complete task execution, use the same task, layout, checkpoint,
configuration and native step limit as the official reference. Start exclusive
endpoints, then run the fixed planner test:

.. code-block:: bash

   RPENT_RUN_AWOMO05_COMPLETE=1 \
   RPENT_E2E_OUTPUT_DIR=/path/to/new-complete-evidence \
   RPENT_ROBODOJO_TASK=stack_blocks_by_language \
   RPENT_ROBODOJO_LAYOUT=13 RPENT_ROBODOJO_STEP_LIMIT=400 \
   RPENT_POLICY_BUDGETS=3,5,31 \
   RPENT_ROBODOJO_ENV_ENDPOINT=http://127.0.0.1:19106 \
   RPENT_AWOMO05_POLICY_SERVER_URL=ws://127.0.0.1:19105 \
   RPENT_XPOLICYLAB_ROOT="$XPL_ROOT" \
     pytest tests/e2e_tests/robodojo/test_awomo05_complete.py -v -s

The endpoint's layout and step limit must match these values. This opt-in test allows up to two hours. The test uses
RPent's actual API planner and registered tools with a deterministic model;
it retains the official instruction and permits ``finish`` only after native
episode end. It exercises segmented continuation, checks every observation
and inference step, writes an audit and ``complete-result.json``, and closes
the test environment. No planner credentials or autonomous goal/recovery
choices are involved. ``RPENT_POLICY_BUDGETS=400`` checks a single tool budget;
``RPENT_AWOMO05_VLA_ENDPOINT`` selects a borrowed bridge as in the bounded test.
Run each variant with a fresh episode. Compare the native termination, action
sequence and success outcome with the official reference; a matching task name
alone does not establish equivalent simulator semantics or model quality.

Record the simulator/runtime and model source revisions separately. RPent's
current RLinf runtime requires the packaged ``src.collect_client`` interface;
an official source tree without that module cannot replace it using only
``--source-root``. Runtime integration results are not official baseline success
rates. The bounded check alone does not establish full-episode passthrough.


The CPU regressions require neither Awomo weights nor Isaac Sim. They do not
establish Awomo inference quality, simulator camera cadence or benchmark success.

SAM3 Configuration
------------------

SAM3 provides object segmentation. Request access on the
`facebook/sam3 model page <https://huggingface.co/facebook/sam3>`_, then
download ``sam3.pt`` and set its path:

.. code-block:: bash

   hf download facebook/sam3 sam3.pt --local-dir /data/sam3
   export SAM3_CHECKPOINT_PATH=/data/sam3/sam3.pt

Alternatively, download the same file from
`ModelScope <https://modelscope.cn/models/facebook/sam3>`_ and set
``SAM3_CHECKPOINT_PATH`` to the local file.

Run a Task
----------

Follow :doc:`../guides/configure_planner` to choose and configure the planner
and model you want to use. The following Codex example checks connectivity in
the terminal that will run the task:

.. code-block:: bash

   rpent-check-llm --planner codex --model <planner-model> --json

Replace ``<planner-model>`` with the selected model, then run the
bottle-disposal task:

.. code-block:: bash

   # Workaround: RPent currently prioritizes a sibling rlinf checkout, which can shadow the installed RoboDojo integration.
   # Point to the active environment's site-packages so child services use its installed RLinf.
   # Remove this setting once RPent's default import-path selection is fixed.
   export RPENT_RLINF_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000  # Recommended: let objects settle; the default is 20 steps
   export OMNI_KIT_ACCEPT_EULA=YES  # Recommended for the first non-interactive start; omit if the Kit license is already accepted
   # Isaac Sim takes tens of seconds to start; the first run also compiles shaders
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --reasoning-effort medium \
     --max-turns 40 --planner-timeout-s 1500 \
     --output-dir /path/to/run-output

View Results
------------

The terminal shows service startup, planner output, and tool calls. After the
run, inspect these files in the directory selected by ``--output-dir``:

- ``result.json``: the environment's task verdict and run error status.
- ``states.json``: action records and post-action observations.
- ``videos/``: head, left-wrist, and right-wrist camera videos.
- ``robodojo_env_server.log``, ``robodojo_vla_server.log``, and
  ``sam3_server.log``: service startup and error logs.
- ``run.log`` and ``transcript_*.json``: main-process logs, planner conversations,
  and errors.

Check ``environment_result_available=true`` and
``agent_error_present=false`` first, then inspect whether ``success`` is
``true``. A clean service exit or a planner call to ``finish`` alone does not
establish task success.

Without ``--output-dir``, results go to
``logs/<timestamp>_robodojo_<task>_l<layout>/``. Use different directories for
concurrent runs. For live monitoring, see :doc:`../guides/dashboard`.

Common Options
--------------

RPent starts the simulator, policy, and SAM3 services in the current Python
environment by default. Specify ``--sim-python`` and ``--pi05-python`` only
when the simulator and policy are installed in different environments.

- ``--task`` and ``--layout`` select the task and fixed layout. ``--random``
  ignores ``--layout`` and samples a new random scene on each reset.
- ``--cuda-device`` selects the GPU for locally started services. If omitted,
  services inherit ``CUDA_VISIBLE_DEVICES``.
- ``--env-endpoint``, ``--vla-endpoint``, and ``--sam3-endpoint`` connect to
  existing services. The VLA service must match ``--policy-backend``.
- ``--source-root`` selects a local RoboDojo checkout instead of the packaged code.

Child services inherit the current shell environment. The CLI does not load
``config/runtime.env`` automatically. Run ``rpent --robot robodojo --help``
for the full option list.

Task Memory and Flash Replay
----------------------------

RoboDojo does not currently support Flash Mode or ``--explore``.

.. _robodojo-local-nvidia-assets:

Prepare Local NVIDIA Scene Assets (Optional)
--------------------------------------------

Download NVIDIA scene assets in advance if either of these applies:

- You need to run the simulator on a machine without network access.
- NVIDIA resource URLs are unreliable and the logs show repeated download failures.

RoboDojo scenes also reference NVIDIA 3D models, materials, and textures.
These files are separate from the RoboDojo resources downloaded above and
load over the network by default. Skip this section if scenes load normally;
you do not need a local asset pack or ``ROBODOJO_USD_ASSET_PREFIX``.

The steps below prepare local resources for ``Assets/Isaac/5.0`` paths.
The three parts of the complete asset pack total about 150 GB; this is not
the download size for a single RoboDojo scene. Allow extra space for merging
and extraction. For an offline machine, download the parts on a connected
machine first, then copy them to ``/data/nvidia``.

.. code-block:: bash

   mkdir -p /data/nvidia

   # Download all three parts; interrupted downloads can resume
   wget -c -P /data/nvidia \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.001 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.002 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.003

   # Merge and extract, preserving the Assets/Isaac/5.0 directory structure
   cat /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.001 \
       /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.002 \
       /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.003 \
     > /data/nvidia/isaac-sim-assets-complete-5.0.0.zip
   unzip /data/nvidia/isaac-sim-assets-complete-5.0.0.zip -d /data/nvidia

   test -d /data/nvidia/Assets/Isaac/5.0/Isaac
   test -d /data/nvidia/Assets/Isaac/5.0/NVIDIA

Set this variable in the terminal that will run RPent, then run the task:

.. code-block:: bash

   export ROBODOJO_USD_ASSET_PREFIX=/data/nvidia

This changes the matching RoboDojo 5.0 resource URLs to local paths. It does
not make every Isaac Sim resource available offline. If other online resources
still fail to load, such as ``Assets/Isaac/5.1`` paths, prepare assets for that
version and configure local loading separately. See the
`NVIDIA local asset instructions
<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`_.
