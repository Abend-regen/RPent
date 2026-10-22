RoboDojo
========

通过 RPent 工具控制 RoboDojo 双臂桌面场景。仿真器基于 Isaac Sim / IsaacLab，采用 ARX-X5 双机械臂平台。

.. note::

   解析式工具调用原生 cuRobo 位姿规划，并执行关节目标。规划器默认障碍物模型只有简化桌面，未包含全部物体或另一只机械臂。路径规划及抽样不能保证执行时无碰撞。组件检查和有界动作执行检查不能说明基准任务成功率。

   ``pi0_pick``、``stabilize`` 和 ``place_in_bin`` 不注册。常规启动仅加载 Env 和 SAM3，不需要动作策略权重；Flash 重放仍不可用。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      解析式位姿和夹爪 Primitive

   .. grid-item-card:: 规划器

      ``api``、``claude_code``、``codex``

   .. grid-item-card:: 任务

      瓶子入桶、笔筒整理、碗堆叠等桌面操作。

   .. grid-item-card:: 硬件

      Linux、NVIDIA GPU；Python 3.11；Isaac Sim 5.1。

任务
~~~~

`RoboDojo 仿真任务目录 <https://robodojo-benchmark.com/doc/sim-tasks/>`_ 将 42 个基准任务分为五类能力：

.. list-table::
   :header-rows: 1

   * - 类别
     - 任务数
     - 考察内容
   * - 泛化
     - 12
     - 不同物体和布局，包括随机变体。
   * - 记忆
     - 6
     - 状态追踪和顺序记忆。
   * - 精细操作
     - 8
     - 空间约束严格的操作。
   * - 长程任务
     - 8
     - 需要完成多个子目标的多步骤任务。
   * - 开放任务
     - 8
     - 开放式或依赖语言、图像指令的任务。

目录还单列了一个 DLC 任务。通过 ``--task`` 选择已安装运行时中的任务，``--layout`` 选择场景布局。\
例如，``stack_bowls_random`` 属于泛化任务，``fill_pen_holder`` 和 ``put_bottles_into_dustbin``\
属于长程任务。这三个名称只是示例，并非完整任务列表。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 说明
   * - 观测
     - 头部和左右腕部相机的 RGB-D 图像、相机标定，以及双臂关节状态、末端执行器位姿和夹爪开合状态。
   * - 动作
     - ``move_to``、``move_delta``、``hold``、``rotate_wrist``、``set_gripper`` 和 ``release``。``finish`` 不能覆盖原生任务判定。
   * - 奖励与成功判定
     - 任务是否成功由 RoboDojo 原生环境判定，结果记录在 ``result.json`` 的 ``success`` 字段中。
   * - 任务指令
     - 来自当前场景的 RoboDojo 任务描述，通过 ``view_env_state`` 读取。

安装与资源准备
--------------

NVIDIA 驱动版本建议参考 `Isaac Sim 5.1 官方系统要求 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>`_。\
宿主机还需具备构建 cuRobo 所需的 CUDA toolkit/NVCC 和编译工具链。从 RPent 根目录创建独立环境并安装依赖：

.. code-block:: bash

   uv venv --python 3.11 .venv-robodojo
   source .venv-robodojo/bin/activate
   uv pip install -e ".[robodojo-sim,sam3]" --extra-index-url https://pypi.nvidia.com \
     --override requirements/robodojo-override.txt

.. note::

   Blackwell GPU 需要 CUDA 12.8 版 PyTorch。本项目固定的 PyTorch 2.7.0 也提供 CUDA 12.6 构建，\
   但该构建不支持 Blackwell。使用这类 GPU 时，首次安装及后续重新安装都使用以下命令，\
   通过 ``--torch-backend=cu128`` 选择 CUDA 12.8 构建：

   .. code-block:: bash

      uv pip install -e ".[robodojo-sim,sam3]" --extra-index-url https://pypi.nvidia.com \
        --override requirements/robodojo-override.txt \
        --torch-backend=cu128

   如果环境中已安装 ``torch==2.7.0+cu126``，uv 会认为它满足版本要求，继续保留原有安装。\
   此时需在上面的命令末尾再添加 ``--reinstall-package torch --reinstall-package torchvision``，\
   强制替换为 CUDA 12.8 构建。

当前固定的 IsaacLab 版本需要 editable 安装，才能找到扩展配置。克隆到一个可写目录，并在同一环境中执行：

.. code-block:: bash

   git clone --no-checkout https://github.com/yuechen0614/IsaacLab.git /path/to/IsaacLab
   git -C /path/to/IsaacLab checkout afca7b09d60d8beb9c1cb28b43066499940b969b
   uv pip install --no-deps \
     -e /path/to/IsaacLab/source/isaaclab \
     -e /path/to/IsaacLab/source/isaaclab_assets \
     -e /path/to/IsaacLab/source/isaaclab_tasks

安装后保留该源码目录，运行时仍会使用它。

下载仿真资源
------------

从 `RoboDojo 数据集 <https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo>`_ 下载机器人、物体、材质和布局文件，\
并设置资源目录：

.. code-block:: bash

   hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
     --include 'Assets/**' --local-dir /data/robodojo
   export ROBODOJO_ASSETS_ROOT=/data/robodojo

下载后，目录中应有 ``Assets/Robots``、``Assets/Object``、``Assets/Material`` 和 ``Assets/Eval_Layout``。

运行以下命令，将 ``Assets/Robots`` 下的 ``*_tmp.yml`` 模板生成为同目录下的 ``*.yml`` 配置，并填入本机资源的绝对路径。\
仿真器使用的 x5 配置为 ``Assets/Robots/x5/curobo.yml``：

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

首次下载后需要执行一次，之后运行任务时无需重复执行。若移动了资源目录，配置中保存的路径就会失效，需要重新运行此命令。\
重新生成会覆盖已有的 ``*.yml`` 配置。

场景引用的部分 NVIDIA 素材默认在线加载。离线运行或相关资源持续加载失败时，参见文末的 :ref:`robodojo-local-nvidia-assets`。

SAM3 配置
---------

SAM3 用于物体分割。先在 `facebook/sam3 模型页面 <https://huggingface.co/facebook/sam3>`_ 申请访问权限，\
再下载 ``sam3.pt`` 并设置路径：

.. code-block:: bash

   hf download facebook/sam3 sam3.pt --local-dir /data/sam3
   export SAM3_CHECKPOINT_PATH=/data/sam3/sam3.pt

也可以从 `ModelScope <https://modelscope.cn/models/facebook/sam3>`_ 下载同一文件，\
再将 ``SAM3_CHECKPOINT_PATH`` 指向本地文件。

运行一个任务
------------

先按 :doc:`../guides/configure_planner` 选择并配置要使用的 planner 和模型。以下以 Codex 为例，\
在运行任务的同一终端检查连接：

.. code-block:: bash

   rpent-check-llm --planner codex --model <planner-model> --json

将 ``<planner-model>`` 替换为所选模型名，配置好 SAM3 后运行瓶子入桶任务。此命令会让 Agent 控制双臂动作：

.. code-block:: bash

   # 临时处理：当前 RPent 会优先导入仓库旁的 rlinf，可能覆盖已安装的 RoboDojo 集成版。
   # 指向当前已激活环境的 site-packages，让子服务使用该环境安装的 RLinf。
   # 后续修正 RPent 的默认导入路径后，可移除此设置。
   export RPENT_RLINF_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

   export ROBODOJO_PLACEMENT_SETTLE_STEPS=1000  # 推荐：让物体充分稳定；不设置时默认等待 20 步
   export OMNI_KIT_ACCEPT_EULA=YES  # 推荐用于首次非交互启动，提前接受 Kit 许可；已接受时可省略
   # Isaac Sim 启动需数十秒，首次运行还会编译 shader
   rpent --robot robodojo --task put_bottles_into_dustbin --layout 0 \
     --planner codex --model <planner-model> --reasoning-effort medium \
     --max-turns 40 --planner-timeout-s 1500 \
     --output-dir /path/to/run-output

查看结果
--------

终端会显示服务启动信息、planner 输出和工具调用。运行结束后，查看 ``--output-dir`` 指定目录中的文件：

- ``result.json``：环境的任务成功判定及运行错误状态。
- ``states.json``：工具命令、执行结果及同帧观测记录。
- 各相机逐步保存的 RGB、米制深度、标定和世界坐标三维图。
- ``videos/``：头部、左腕和右腕相机的视频。
- ``robodojo_env_server.log`` 和 ``sam3_server.log``：\
  本地启动服务的日志和报错信息。
- ``run.log`` 和 ``transcript_*.json``：主进程日志、planner 对话和错误。

先检查 ``environment_result_available=true`` 和\
``agent_error_present=false``，再看 ``success`` 是否为 ``true``。服务正常退出或 planner\
调用 ``finish``，均不能单独说明任务成功。

未指定输出目录时，结果保存在 ``logs/<timestamp>_robodojo_<task>_s<seed>_l<layout>/``。并发运行应使用不同目录。\
实时查看方式见 :doc:`../guides/dashboard`。

常用参数
--------

默认由 RPent 在当前 Python 环境中启动仿真和 SAM3 服务。仿真器安装在独立环境时，指定 ``--sim-python``。旧 Pi0.5 诊断辅助代码保留在源码中，但 CLI 不启动该服务，也不向规划器开放其工具。

- ``--task``、``--seed``（默认 0）和 ``--layout``：选择任务、官方布局组和组内布局编号；传入 ``--random`` 时忽略 ``--layout``，\
  每次重置重新采样随机场景。
- ``--cuda-device``：为本地启动的服务选择 GPU；省略时沿用 ``CUDA_VISIBLE_DEVICES``。
- ``--env-endpoint`` 和 ``--sam3-endpoint``：连接已有服务。
- ``--source-root``：使用本地 RoboDojo 源码替代包内源码。

子服务继承当前终端的环境变量；CLI 不会自动读取 ``config/runtime.env``。\
完整参数可运行 ``rpent --robot robodojo --help`` 查看。

工具
----

规划器不能重置回合；探索模式和 Flash 重放不可用。

``view_env_state`` 读取保存的图像和状态，``render`` 刷新观测。``sample_world_xyz(view, pixels, step=None, neighborhood=1)`` 查询 [行,列] 像素，默认取 3×3 邻域。``query_world_map`` 查询半开区间 [r0,c0,r1,c1]，默认最多返回 256 个点。view 使用 ``head``、``left_wrist`` 或 ``right_wrist``，必须与选点图像的视角及步号一致。查询读取保存的同帧深度，不暗中刷新；无效深度被排除，返回的是可见表面坐标，不是物体中心。

``segment`` 和 ``back_project`` 也接受 ``step``。分割返回质心 ``centroid_rc`` 和掩码内像素 ``point_rc``；凹形掩码的质心可能在物体外，反投影应优先使用 ``point_rc`` 及返回的 ``step_idx``。这两个工具保留相机名 ``cam_head``、``cam_left_wrist``、``cam_right_wrist``。

调用 ``segment(text_prompt=...)`` 时，描述可见的颜色、形状和位置。工具返回红色掩码叠加图；使用定位点前先检查叠加图，高分不代表选中了正确物体。分割失败时可修改描述，或在同一历史帧中手动选取可见像素查询。叠加图和几何结果保存在对应帧中。与 LIBERO 的接口不同，这里不接受点提示，也不返回掩码区域的世界坐标中位数；选定掩码内像素后需调用 ``back_project``。

``move_to(arm, xyz, quat=None, gripper=None, substeps=25)`` 使用世界坐标、米及 wxyz 四元数；省略姿态或夹爪参数则保持对应状态／目标。``substeps`` 限制路径抽样数，0 保留完整路径，1 仅取最后路点，不代表速度或物理步数。``rotate_wrist`` 绕世界 Z 轴相对旋转，单位为度。``final_dist_m`` 是实测位置误差，不是到达保证。检查 ``completed``、``executed_steps`` 和 ``stop_reason``；超时的动作可能已在服务端执行，不会自动重试。

``set_gripper(arm, val, steps=10)`` 使用 0=闭、1=开，``release`` 默认开到 1。中间值是连续控制目标，不保证对应精确物理宽度；反馈读取实际关节，而非命令字段。仅当实测值因物理超调越界时，将插值起点限制在合法范围，用户非法输入仍被拒绝。其他关节目标保持不变，每次下发目标还会经过底层插值。

相对运动与保持
~~~~~~~~~~~~~~

``move_delta(arm, dxyz=[0,0,0], axis=[0,0,1], angle_deg=0, frame="world", gripper=None, substeps=25)`` 从调用时实测的末端位姿构造目标，复用 ``move_to`` 的原生规划器。平移单位为米，旋转单位为度，任意非零旋转轴穿过末端原点。``world`` 使用世界坐标平移并左乘旋转；``tool`` 使用起始工具坐标系并右乘旋转。平移不会使末端绕世界原点公转，等价四元数 ``q`` 和 ``-q`` 产生相同的目标位姿。这不承诺直线、固定旋转轴轨迹或多圈旋转。零平移、零角度且未指定夹爪目标时返回零动作的 ``no_op``；显式指定夹爪目标仍属于动作请求。

``hold(steps)`` 要求正整数，按原生动作次数重复双臂及夹爪当前保持的控制目标，受回合剩余预算和取消约束。仿真时间会推进，实际位姿和物体不一定静止。一个动作可能包含多个物理子步；``steps`` 和反馈计数均不代表物理子步、观测采样周期或墙钟时间。``substeps`` 只控制路径采样数量，运动 primitive 不另外提供 ``max_steps`` 预算。

执行结果包含 ``step_before``、``step_after``、``executed_actions``、``terminated`` 和 ``truncated``。原有 ``executed_steps`` 同样表示原生动作数量。中断或结果不确定时只报告已确认的进度，不重试动作。``get_motion_state()`` 中的 ``simulation_time_s`` 是独立于动作计数的仿真时间。

有状态适配器协作
~~~~~~~~~~~~~~~~

默认 Env + SAM3 工具链不需要 VLA。模型适配器可通过 ``env.register_execution_observer(callback)`` 注册一个同步回调，或通过 toolkit 的运行时 kwargs 传入 ``execution_observer``。注册时发送包含当前观测和状态的 ``attached`` 通知，不重置环境或模型。协作期间只能由一个 RPent 客户端持有环境控制权；其他客户端不得重置或修改同一服务。

适配器每次推理前必须调用 ``env.require_policy_ready()``，并通过 ``env.step_policy(action)`` 执行自身动作。指令切换、剩余动作缓存、推理串行化及模型历史采样由适配器负责。此边界不提供 Policy 调度器、动作缓存或模型历史采样器。RoboDojo 客户端和服务端均明确禁止 ``chunk_step``；支持的动作入口逐个动作获取确认。

回调接收以下事件：

* ``actions_invalidated``：在外部动作、primitive 或 hold 接管前，以及取消、终止、关闭或失同步时发送。丢弃缓存的 Policy 动作，不自动重置模型历史。
* ``action_feedback``：每个确认动作的权威 ``step_before``、``step_after``、``executed_actions``、对应的 ``observation``、``episode_status``、``terminated``、``truncated`` 和 ``source``。只为实际执行的动作追加历史；零动作拒绝及只读查询不重复追加帧。回调应在接受事件后返回，不得重入推理或环境修改操作。
* ``episode_end`` 与 ``reset`` 分开通知。只有显式环境重置成功后才发送 ``reset``，并附带初始观测。
* ``synchronization_lost``：RPC 结果不确定、计数不一致、观测缺失或反馈交付失败时发送。保留已确认的动作进度，同时阻止继续进入推理或执行。

Policy 预算暂停只需停止调用方的循环，不调用 ``pause_execution()``；同回合、同指令且环境未额外推进时可保留剩余动作。``pause_execution()`` 表示取消，会使动作缓存失效；``resume_execution()`` 只解除暂停，不能修复失同步。执行前的参数拒绝或规划失败不会使模型历史失去可信状态。环境指令变化或检测到边界外动作计数变化时，协作边界会停止接纳推理及执行。

当前显式恢复方式是成功调用 ``env.reset()``，并成功交付重置事件，不会自动恢复。如果取消同时暂停了控制，还需显式恢复暂停。Eval-fair 服务禁止 reset，需开启新回合／服务。关闭 toolkit 会结束本次协作边界，后续会话须建立新的客户端／适配器。丢失响应后不重试动作。离线假适配器与仿真 primitive 检查只验证公共契约，不代表所有 VLA 已兼容。下文的 Awomo 接入消费此契约，完整 episode 对照仍需单独验收。

有状态 XPolicyLab 策略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

对于需要逐控制步历史观测的策略，先准备独立的 XPolicyLab WebSocket 服务，\
实现 ``reset``、``update_obs`` 和 ``get_action``。历史缓存、预处理和动作块长度由模型适配器负责。
请使用独立的服务和端口，避免与正在进行的评测共用。

在已安装 RPent 和 XPolicyLab 客户端依赖的 Python 环境中，将 XPolicyLab 源码目录\
加入 ``PYTHONPATH``，启动桥接：

.. code-block:: bash

   PYTHONPATH=/path/to/XPolicyLab python -m rpent.robots.components.xpolicylab_vla_server \
     --task stack_blocks_by_language --stateful \
     --policy-server-url ws://127.0.0.1:19105 \
     --transport http --host 127.0.0.1 --port 19106

按本页说明配置仿真、SAM3 和 planner 后，将 RPent 连接到桥接服务：

.. code-block:: bash

   rpent --robot robodojo --task stack_blocks_by_language --layout 0 \
     --planner codex --model <planner-model> \
     --policy-backend xpolicylab --stateful-policy \
     --vla-endpoint http://127.0.0.1:19106 \
     --output-dir /path/to/stateful-run

此模式需显式启用，并提供外部桥接端点。默认仍只启动 Env + SAM3，不启动策略服务。
工具列表在原生 primitive 之外新增 ``run_policy(max_steps)``，使用官方任务指令执行有限步数的动作块。
每完成一个控制步，包括脚本动作，都会发送一次观测；读取场景或记录状态不会发送额外观测，\
动作块边界也不会重复发送。工具包启动时重置一次模型；环境回合重置时同步清空模型历史，\
同一回合内多次调用工具则保留历史。

桥接复用 RPent RPC 会话机制，同一时刻只允许一个会话持有模型历史。\
其他会话无法通过已占用的桥接重置模型或请求推理。持有者会话关闭或过期后释放占用；\
工具包关闭时也会结束回合并释放占用，同时保留 VLA 连接供后续任务复用。新回合使用前必须重置。
此机制无法隔离绕过桥接的客户端，因此上游模型服务也应专供该桥接使用。
此模式暂不支持 Flash 回放。

观测步号从零开始，必须连续。观测更新或推理失败后，该回合状态被标记为不可继续使用，\
因为响应丢失前模型可能已经更新。环境动作结果不确定时同样停止执行，须显式重置后再使用。取消只清空剩余动作并暂停控制，不重置模型历史。不要自动重试不确定调用；planner 工具不提供回合重置操作。
回合结束和动作执行完成均不能作为任务成功的证据。

有状态执行循环使用 ``robots.robodojo.policy_adapter`` 中的 ``RoboDojoPolicyAdapter``。
默认模式原样转发 ``predict(obs, options)``，回合与观测钩子不会额外调用后端。
有状态模式通过 ``begin_episode(obs)`` 重置并记录第零步，随后调用 ``observe(obs, step=t)`` 和 ``predict(obs, step=t)``。若下一步观测尚未发送，\
预测入口会先发送观测，因此两种调用顺序都只会记录一次该控制步。
再次提交最新观测步号时不重复发送；旧步号或跳步会被拒绝。
步号表示同一回合中的控制状态和指令，不根据图像内容判断是否重复。
没有新观测时重复推理，仍会被有状态桥接拒绝。``end_episode()`` 释放占用，供后续任务使用。
该适配器仅属于 RoboDojo，其他机器人和共享 VLA 接口保持现有契约。

以下 CPU 测试使用假模型和假环境，覆盖桥接、两种 RPC 传输、工具执行、\
会话占用、回合重置、取消和异常处理：

.. code-block:: bash

   pytest tests/unit_tests/robots/robodojo/test_policy_adapter_contracts.py \
     tests/unit_tests/robots/robodojo/test_stateful_policy_loopback.py -v

Awomo-0.5 后端
^^^^^^^^^^^^^^^^

``--policy-backend awomo05`` 通过 Robodojo 有状态适配层调用 XPolicyLab 官方 ``Awomo05`` 模型，要求使用 ``arx_x5`` 和绝对 ``joint`` 动作，并自动启用 ``run_policy``。该路径不支持 Flash replay；未选择 Awomo 时默认仍启动 Env + SAM3。

先按 XPolicyLab 的 Awomo05 安装说明准备独立模型环境、权重及匹配的统计量，模型与仿真使用不同解释器。在 RPent 仓库根目录，用模型环境启动薄包装服务：

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

沿用官方权重路径环境变量和模型配置。该启动器复用官方模型，仅增加只读元信息，不复制模型代码，也不改变推理、归一化、历史采样或 AR 状态。缺少 ``get_model_meta`` 的旧服务会在 reset 前被拒绝；应启动独立的兼容服务，不使用改变状态的调用探测能力。

连接模式二选一。连接已有官方 WebSocket 服务时，RPent 只启动轻量桥接；桥接解释器需要安装 RPent 和 XPolicyLab 客户端依赖，包括 WebSockets、msgpack 和 msgpack-numpy，无需加载模型权重：

.. code-block:: bash

   rpent --robot robodojo --task stack_blocks_by_language --layout 0 \
     --planner codex --model <planner-model> --policy-backend awomo05 \
     --policy-server-url ws://127.0.0.1:19105 \
     --xpolicylab-root "$XPL_ROOT" --policy-python /path/to/bridge/python \
     --env-endpoint http://127.0.0.1:19106 --output-dir /path/to/awomo-run

也可预先启动可复用的 RPent Awomo 桥接；将 XPolicyLab 目录及其父目录加入 ``PYTHONPATH``：

.. code-block:: bash

   /path/to/bridge/python -m robots.robodojo.awomo05_bridge \
     --task stack_blocks_by_language --policy-server-url ws://127.0.0.1:19105 \
     --host 127.0.0.1 --port 19107

随后将 RPent 命令中的 ``--policy-server-url``、``--xpolicylab-root`` 和 ``--policy-python`` 替换为 ``--vla-endpoint http://127.0.0.1:19107``。两类端点参数互斥。``--policy-python`` 是现有 ``--pi05-python`` 的别名。两种模式都会在回合开始前检查模型身份、原生动作布局、相机、历史、重规划和有状态能力。使用 CLI 时，按本页其他章节配置 planner；Awomo 默认不启动 SAM3。

独立模型服务属于借用资源，RPent 退出时不会停止它。一个运行任务应独占环境和模型服务；桥接的会话锁无法阻止另一官方客户端或另一桥接重置同一个上游模型。

观测在线路上传输为 RGB、``uint8``、HWC 图像，顺序为 ``cam_head``、``cam_left_wrist``、``cam_right_wrist``。发布权重所需的 BGR 换序仅在官方 adapter 内发生一次。官方状态打包顺序为左臂 6 维、左夹爪 1 维、右臂 6 维、右夹爪 1 维；返回动作保留这四个具名字段，不经过 Pi0.5 的不同扁平向量布局。关节目标为绝对弧度值，夹爪开度为 0 关闭、1 张开，由原生环境映射为实际关节量。每个控制目标对应 0.04 秒仿真时间，不以网络推理耗时决定控制频率。

Awomo 路径中，本地执行层是已返回、尚未执行动作的唯一缓存持有者。在回合、指令和环境步数保持一致时，先 ``run_policy(3)`` 再 ``run_policy(5)`` 会执行同一组 8 步动作。脚本动作清空剩余动作并补送新观测，保留模型历史。取消会清空剩余动作并暂停控制，显式恢复暂停后保留已确认历史。结果不确定的异常、换指令或检测到环境绕过执行层推进时，阻止继续执行，直到显式重置回合。回合结束时清空动作且不再发送终止观测，与官方部署循环一致。清空缓存本身不会调用模型 reset。现有环境接口无法识别步数未变的外部 reset，因此必须使用独占端点并通过执行层重置。

无需模型权重即可运行离线回归：

.. code-block:: bash

   pytest tests/unit_tests/robots/robodojo/test_awomo05_contracts.py \
     tests/unit_tests/robots/robodojo/test_awomo05_loopback.py \
     tests/unit_tests/robots/robodojo/test_awomo05_policy_tool.py -v

准备独占的测试端点后，运行真实模型的有界验证：

.. code-block:: bash

   RPENT_RUN_AWOMO05_INTEGRATION=1 \
   RPENT_E2E_OUTPUT_DIR=/path/to/new-evidence \
   RPENT_ROBODOJO_ENV_ENDPOINT=http://127.0.0.1:19106 \
   RPENT_ROBODOJO_STEP_LIMIT=40 \
   RPENT_AWOMO05_POLICY_SERVER_URL=ws://127.0.0.1:19105 \
   RPENT_XPOLICYLAB_ROOT="$XPL_ROOT" \
     pytest tests/e2e_tests/robodojo/test_awomo05.py -v -s

环境需使用所选任务、layout 0 和指定步数上限。``RPENT_ROBODOJO_TASK`` 默认为官方 open 任务 ``stack_blocks_by_language``。验证已有桥接时，用 ``RPENT_AWOMO05_VLA_ENDPOINT`` 替代 ``RPENT_AWOMO05_POLICY_SERVER_URL``。测试执行 3+5+2 步、重置后再执行 2 步，写出 ``awomo05-result.json`` 并关闭测试环境。任务提前成功与通过该固定节奏检查是不同结果。此测试需显式启用，未接入通用 GPU 套件运行器。

完整回合的 Policy 工具验证
""""""""""""""""""""""""""""""""""""""""""""""""""""

Awomo 工具包通过 ``run_policy`` 执行动作，同时保留 ``view_env_state`` 和公共审计、结束工具，不暴露抓取、放置或脚本恢复工具。提示词保留原始任务指令，RPent 不改写子任务，也不因检测到抓取而停止。步数预算用尽后暂停，保留仍有效的剩余动作，后续调用继续执行，直到 ``episode_ended`` 或执行出错。取消或失败后不得自动重试。Awomo 默认不启动 SAM3，只有显式请求该组件时才启动。

Awomo 工具返回的 ``log.result`` 包含 ``executed_steps``、``observation_step``、``episode_ended``、``pending_actions`` 和 ``stop_reason``。停止原因分别为预算用尽（``budget``）、回合结束（``episode_end``）、取消（``cancelled``）或错误（``error``）。执行完成不等于任务成功。
每次推理的完整元数据保存在 ``states.json`` 对应动作的 ``result.inferences`` 中；即时工具响应以 ``inference_count`` 提供简要计数。动作块跨工具调用续接时，仅在生成该动作块的调用中记录一次推理。

声明 ``action_info`` 能力的桥接支持在 ``vla.get_action`` 中传入 ``include_info=True``。默认仍返回动作，Awomo 客户端的动作列表返回方式也保持兼容。元数据保留上游响应字段（包括服务提供的 ``latency_ms``）、模型结果元数据及桥接测量的 ``bridge_rpc_s``。
执行报告另行记录以秒为单位的 ``policy_rpc_s``、``observation_rpc_s``、``env_step_s`` 和 ``elapsed_s``。这些墙钟耗时存在包含关系，不能作为独立开销相加；``elapsed_s`` 不包括之后的状态记录和 planner 工作，初始化、reset 与初始观测也不属于本次调用。旧桥接未声明该能力时仍可使用，推理 ``info`` 为空。缺失的上游耗时表示不可用，不补零。

完整任务验证应采用与官方参考一致的任务、layout、权重、配置和原生步数上限。启动独占端点后运行固定 planner 测试：

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

端点的 layout 和步数上限需与命令一致。此显式启用的测试允许最多两小时。测试使用实际 RPent API planner 和工具分发链，以确定性模型固定调度，保留官方指令，仅在原生回合结束后调用 ``finish``。它验证分段续接、逐步观测和推理时点，写出审计文件与 ``complete-result.json``，随后关闭测试环境。此过程不需要 planner 凭证，不包含自主目标选择或恢复决策。设 ``RPENT_POLICY_BUDGETS=400`` 可验证单次工具预算；与有界测试一样，``RPENT_AWOMO05_VLA_ENDPOINT`` 可选择已有桥接。每种方式均使用新回合，对照官方参考的终止、动作序列和任务结果；仅任务名相同不能证明模拟器语义等价或模型质量。

分别记录仿真运行时和模型源码版本。RPent 当前 RLinf 运行时要求打包源码中的 ``src.collect_client`` 接口；缺少该模块的官方源码不能仅通过 ``--source-root`` 替换。运行时接入结果不等于官方基线成功率。仅有界验证不足以证明完整回合透传或官方基线成功率。


CPU 回归测试不需要 Awomo 权重或 Isaac Sim，也不能证明 Awomo 推理质量、\
仿真相机刷新频率或基准任务成功率。

.. _robodojo-local-nvidia-assets:

NVIDIA 场景素材的本地准备（可选）
------------------------------------------------------------

以下两类用户需要预先下载 NVIDIA 场景素材：

- 需要在无法联网的机器上运行仿真的用户。
- 运行时无法稳定访问 NVIDIA 资源地址，日志中相关文件反复下载失败的用户。

RoboDojo 场景还会引用 NVIDIA 提供的三维模型、材质和贴图。这些文件与前面下载的 RoboDojo 资源分开提供，默认在加载场景时通过网络获取。\
场景能正常加载时，可跳过本节，无需下载本地资产包，也不用设置 ``ROBODOJO_USD_ASSET_PREFIX``。

以下步骤用于准备 ``Assets/Isaac/5.0`` 路径对应的本地资源。完整资产包的三个分卷合计约 150 GB，并非单个 RoboDojo 场景所需的下载量；\
合并和解压还需额外空间。离线机器可先在能联网的机器上下载，再将分卷复制到 ``/data/nvidia``。

.. code-block:: bash

   mkdir -p /data/nvidia

   # 下载三个分卷，已有下载可断点续传
   wget -c -P /data/nvidia \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.001 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.002 \
     https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-complete-5.0.0.zip.003

   # 合并并解压，保留 Assets/Isaac/5.0 目录结构
   cat /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.001 \
       /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.002 \
       /data/nvidia/isaac-sim-assets-complete-5.0.0.zip.003 \
     > /data/nvidia/isaac-sim-assets-complete-5.0.0.zip
   unzip /data/nvidia/isaac-sim-assets-complete-5.0.0.zip -d /data/nvidia

   test -d /data/nvidia/Assets/Isaac/5.0/Isaac
   test -d /data/nvidia/Assets/Isaac/5.0/NVIDIA

在运行 RPent 的终端中设置以下变量，再运行任务：

.. code-block:: bash

   export ROBODOJO_USD_ASSET_PREFIX=/data/nvidia

这会将 RoboDojo 匹配的 5.0 在线资源地址改为本地路径。它并不代表所有 Isaac Sim 资源均可离线使用；\
若仍有其他在线资源加载失败（例如 ``Assets/Isaac/5.1`` 路径），需另外准备对应版本的资产并配置本地加载，\
见 `NVIDIA 本地资产说明 <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`_。

VLA Agent 协作
--------------

在有状态 Policy 运行中加入 ``--vla-agent``，可通过现有 Planner 和 Toolkit
启用阶段规划、主动混合控制和执行后检查。Agent 可以从 primitive 或
``run_policy`` 开始。阶段文本仅用于 Agent 决策；每次 Policy 观测都保留原始
完整任务指令，不改写为子指令。此模式显式启用，与纯 Policy 透传和其他模型
专属工作流分别保留。

例如，连接为本次运行准备的独立仿真和官方 Policy 服务：

.. code-block:: bash

   rpent --robot robodojo --planner codex --model gpt-6-astra \
     --reasoning-effort high --task stack_blocks_by_language --layout 0 \
     --policy-backend awomo05 --policy-server-url ws://127.0.0.1:9000 \
     --env-endpoint http://127.0.0.1:9001 --vla-agent \
     --agent-tools policy motion geometry --agent-max-calls 120 \
     --agent-timeout-s 1800 --planner-timeout-s 1800

桥接需要独立解释器或源码目录时，沿用 ``--policy-python`` 和
``--xpolicylab-root``。也可以用 ``--vla-endpoint`` 连接经能力验证的 RPent
桥接端点，它与 ``--policy-server-url`` 互斥。同一回合内不切换模型。

``policy`` 提供 ``run_policy``，``motion`` 提供六项机器人 primitive，
``geometry`` 提供保存帧上的深度反投影与世界坐标查询。额外选择 ``sam3``
后注册分割工具并启动或连接其服务。观测、render、计划、检查和 finish 始终
注册。三个相机均需提供 RGB、匹配的深度及有效标定。定位使用同一保存观测
索引，该索引不同于原生动作计数。Awomo 适配层过滤额外感知字段，保持官方
RGB、关节状态与语言输入。

Agent 使用 ``set_stage_plan``，控制调用通过 ``decision`` 提供阶段与预期
变化；``decision.review`` 或 ``review_execution`` 记录前次结果 ID、可观察
依据、阶段状态与下一步。``agent-decisions.json`` 将这些记录与实际工具反馈
一起保存。相机和深度产物仍保存在原有状态轨迹中。``finish`` 记录 Agent 判断，
运行器单独记录原生成功结果。日志已关闭但 ``finished`` 不为 ``true``，表示
运行器清理时尚未收到被接受的 Agent finish。

所有控制共享回合的原生动作预算，另有 Agent 调用数与墙钟限制。逐动作检查
墙钟期限，Planner 超时后取消执行；正在进行的推理或 RPC 需要返回或超时后
才能完成清理。Policy 预算暂停后，仅在环境未额外推进时保留剩余动作。
primitive 或 hold 实际接管时使旧动作失效，逐步反馈观测，不 reset 模型。
未执行的错误保留同步状态。RPC 不确定、反馈缺失或观测交付失败会阻止后续
执行；Agent 应报告并 finish，不得自动重试。返回的 ``execution`` 计数表示
已确认动作，不把结果不确定的远程动作算作已确认执行。

此策略不提供配方或历史记忆工具。API Planner 使用新会话；Codex 使用临时
线程，禁用记忆、项目技能、外部应用以及原生文件和 shell 工具，仅注册本次
运行的 RPent MCP 服务，沿用现有登录和 provider。此模式当前支持 ``api``
与 ``codex`` Planner。离线契约测试和确定性 Planner 测试验证执行行为；模型
兼容性与任务成功率仍需单独的真实 episode 评测。
