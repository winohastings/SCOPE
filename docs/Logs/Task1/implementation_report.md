# Task 1：SCOPE 多分辨率采集实现与执行报告

日期：2026-11-05。角色：Exec Agent（用户明确指定执行者）。根目录 `AGENTS.md`
要求的 `agents/exec-agent.md` 不存在，`agents/` 为空，因此依据任务文件、项目索引、
`docs/index.md`、协议与现有基线运行日志执行，没有假造角色文档或交付规范。

## 结论与边界

- 完成 512 原图保留、256 兼容路径、四路 128 Glance、16 个无缩放的 256 ROI、
  Fixed-grid / seeded Random / 顺序选择器 / STOP 接口，以及原始 checkpoint 推理。
- 未修改服务先完成 R2R-CE 与 RxR-CE 各两集完整闭环；新分支在两任务六模式下完成
  **24 次 episode 冒烟运行、48 次策略预测**，全部无执行失败。
- 合成固定图和真实 Habitat 固定观测均比较两个连续预测（覆盖历史输入），
  原始输出 `[8,4]` 的最大绝对差均为 **0.0**。
- Glance 原生网格 `[1,8,8]` → 16 token，Focus `[1,16,16]` → 64 token；
  四 Glance 加两 Focus = **192 个当前观测 token**，每 token 2560 维。
- **R2R 与 RxR 按官方协议共用 RXR step-8000 checkpoint**。官方仓库
  `docs/guides/VLNCE_TRAINING_AND_EVALUATION.md` 明确说明，同一份在 RxR 上训练的
  step-8000 权重用于两个任务。无需独立 R2R 权重；此前将其列为缺项的说法已撤回。
  R2R 独立复跑、验收与成本见 `R2R/implementation_report.md`。
- **没有已训练 SCOPE selector 权重**，也未启动训练。learned 两模式通过明确标记的
  `SCRIPTED_INTERFACE_SMOKE_NOT_LEARNED` 选择器完成接口测试；不是已学习策略结果。
- 每集两步仅是显式冒烟截断；没有进行完整六模式评测，不将其 SR/SPL 当论文性能。

## 实际运行入口与兼容性

原有模型服务：`/home/data/czh/Third_Party/baselines/SimpleNav/deployment/model_server/legacy_rxr_server.py`。
现有评测入口仍为同一仓库的 `benchmark/vlnce/r2r/eval.py` 和 `benchmark/vlnce/rxr/eval.py`。
模型环境为 SimpleNav `.venv/bin/python`；Habitat 环境为项目
`Assumption/assumption1/envs/correctnav_hab033/bin/python`。

原服务占用 GPU 1 / 10093，未停止或修改；本次新增服务占用 GPU 0 / 10094，
Habitat worker 仍为物理 GPU 0。checkpoint 为
`local/checkpoints/vlnce/navvla_qwen35_cpm_rxr/checkpoints/steps_8000_pytorch_model.pt`，
SHA256 保存于 `checkpoint.sha256`，未写入权重。

实际旧 checkpoint 加载 `/tmp/SimpleNav-RXR-official` 的模型实现。主干同名模型已经重构，
因此没有仅修改主干却让实际评测继续加载未修改旧代码；而是将旧运行时持久化至
`Experiment/SCOPE/legacy_runtime/` 后定点扩展。`runtime_provenance.json` 记录逐文件
来源与 SHA256，`legacy_runtime_changes.patch` 给出相对原运行时的差异。

原始 `original` 分支不启用可变分辨率开关，保留原 processor、历史选择、TVI、
scatter、M-RoPE、权重加载和动作预测。四图原始输入是 256 时不再缩放，512 时使用原有
BICUBIC 兼容缩放得到 256。固定输入比较给旧服务相同的 256 图、给新服务其来源 512 图。
这证明**相同模型输入的输出等价性**；不宣称 Habitat 原生 256 渲染与 512 渲染再缩放逐像素相同。

动作仍为 `anchor_relative_body_frame_xyz_yaw`，`vln_train_train` 归一化统计量，
每步最多执行 8 个 waypoint；停止仍为相邻 waypoint 平均平移量 ≤0.03。
未改动作适配器、相机高度/HFOV/朝向、碰撞控制或导航停止协议。

## 修改文件及职责

模型文件均相对于 `Experiment/SCOPE/legacy_runtime/`；完整复制文件、代码、报告和
原始运行产物列表见 `files_manifest.txt`，不是将整个运行时副本都声称为新实现。

| 文件 | 相比原版本的变化 |
|---|---|
| `starVLA/model/modules/input_context.py` | 新建原图保留、兼容缩放、Glance、16 ROI、六模式顺序选择及 STOP |
| `starVLA/model/modules/navvla_context.py` | 每块独立 token 预算；追加 Focus，相机/ROI/focus 索引元数据；复用相机 TVI |
| `starVLA/model/modules/vlm/QWen3_5.py` | 仅 SCOPE 分支禁用 processor 隐式 resize，原始分支保持原调用 |
| `starVLA/model/modules/qwen35_vision.py` | 增加混合网格预算验证，保留 postmerge 与 BF16 缓存函数 |
| `starVLA/model/framework/VLM4A/navvla_qwen35_cpm.py` | 原生 128/256 输入、逐块特征形状校验；Focus 不进入四相机历史缓存 |
| `Experiment/SCOPE/scope_server.py` | 新服务；加载旧权重，六模式共享推理与动作反归一化，逐阶段同步计时和峰值显存 |
| `Experiment/SCOPE/run_smoke.py` | 共享 episode/config 的两任务六模式运行、原始结果校验、CSV 聚合 |
| `Experiment/SCOPE/verify_fixed.py` | 旧/新服务同固定观测两次连续预测比较，验证所有新模式含历史输入 |
| `Experiment/SCOPE/capture_observation.py` | 从实际 worker plan 采集真实 Habitat 原始固定观测 |
| `tests/test_scope_sampling.py` | 原图/ROI、尺寸拒绝、确定性随机、顺序选择、STOP 和非法动作测试 |
| `tests/test_scope_vision.py` | 16/64/192 预算、2560 维历史 pooling 与 BF16 bits 往返测试 |
| 外部 SimpleNav `benchmark/common/runner/worker.py` | 透传 policy_mode，保存逐步预测/实际 world waypoint、渲染和端到端计时，含停止预测 |
| 外部 SimpleNav `benchmark/common/simulators/habitat/backend.py` | 透传实际观测采集计时 |
| 外部 SimpleNav `benchmark/common/simulators/habitat/vlnce031_runtime.py` | 嵌套去重计时 Habitat 观测采集接口，不修改返回观测或动作 |
| `Experiment/SCOPE/README.md`、`docs/index.md`、本报告 | 入口、复现、结果索引与边界说明 |

## 张量、网格与历史契约

| 块/模式 | 原生尺寸 | 原生 image_grid_thw | 合并后特征 | 当前观测 token |
|---|---|---|---|---:|
| original 每视图 | 256×256 | `[1,16,16]` | `[64,2560]` | 四视图256 |
| Glance 每视图 | 128×128 | `[1,8,8]` | `[16,2560]` | 四视图64 |
| Focus 每裁剪 | 256×256 | `[1,16,16]` | `[64,2560]` | 每裁剪64 |
| fixed_2/random_2/learned_2 | 四 Glance + 两 Focus | 按块分别保留 | `[192,2560]` 当前块拼接 | 192 |
| learned_adaptive 冒烟选择器 | 四 Glance + 一 Focus + STOP | 同上 | `[128,2560]` | 128 |
| 历史缓存每相机 | 原契约 pool4 | 上下文为 `[1,4,4]` | `[4,2560]`，BF16 bits 存储不变 | 不计入当前预算 |

ROI 坐标使用 512 原图上的 `[x0,y0,x1,y1]`，右/下边界为排除边界。
相机顺序 front/left/right/rear，ID 0–3，每相机 R1/R2/R3/R4 按行优先编号。
Fixed-grid 取 front.R1、front.R4；Random 无重复采样，按 seed/episode/step 可复现。
learned selector 按序接收已经获得的裁剪；STOP=16，仅 adaptive 可用，最多两次选择。

既有 `pixel_values` 按原生 patch 网格打包，经原视觉 encoder / merger 后进入既有多模态
scatter。逐图 `_nav_original_image_grid_thw`、online grid 与 context grid 保留；
位置仍由既有 `compute_3d_position_ids` 计算，未以纯文本拼接或均匀补齐取代 M-RoPE。
当前块形状在实际 forward 中校验，不是仅根据输入尺寸估算 token。

历史预算配置和 BATS 选择契约保持原值，没有利用 Glance 节省的 token 擅自增加历史帧。
Focus 不覆盖同名相机的历史记录；若捕获在线当前缓存，只缓存四个全视野 Glance 的 pool4。
最终 CSV 同时区分 `current_visual_tokens` 与含历史的 `total_visual_tokens`。

## 实测成本

单位 ms，四个预测步均值（两集×两步）；显存为该组所有预测步中的最大 PyTorch
`max_memory_allocated`，单位 GiB，包含模型驻留，不含 Habitat/驱动的非 PyTorch 分配。
GPU 型号 RTX 4090。表格不是稳态微基准，不能从四步样本断言显著加速。

| 任务 | 模式 | 当前token | Vision | LLM | DiT-B | 服务总计 | Habitat采集 | 客户端闭环 | 峰值GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R2R | original | 256 | 81.57 | 292.70 | 51.23 | 579.83 | 37.10 | 786.62 | 9.696 |
| R2R | glance_only | 64 | 78.23 | 267.23 | 51.27 | 564.72 | 35.61 | 766.31 | 9.661 |
| R2R | fixed_2 | 192 | 84.28 | 287.59 | 51.23 | 600.39 | 36.42 | 800.77 | 9.688 |
| R2R | random_2 | 192 | 84.78 | 286.64 | 51.63 | 598.03 | 36.26 | 799.01 | 9.688 |
| R2R | learned_2（脚本） | 192 | 83.64 | 283.64 | 51.59 | 590.76 | 35.25 | 794.00 | 9.688 |
| R2R | learned_adaptive（脚本） | 128 | 67.62 | 206.57 | 42.29 | 475.55 | 37.24 | 684.13 | 9.674 |
| RxR | original | 256 | 81.25 | 297.37 | 51.30 | 591.29 | 40.95 | 789.01 | 9.696 |
| RxR | glance_only | 64 | 78.62 | 272.42 | 51.03 | 574.57 | 39.24 | 777.10 | 9.661 |
| RxR | fixed_2 | 192 | 72.11 | 207.54 | 43.10 | 484.54 | 40.03 | 671.96 | 9.688 |
| RxR | random_2 | 192 | 71.85 | 208.33 | 42.97 | 487.37 | 41.02 | 685.38 | 9.688 |
| RxR | learned_2（脚本） | 192 | 71.81 | 207.92 | 42.53 | 510.05 | 41.42 | 725.26 | 9.688 |
| RxR | learned_adaptive（脚本） | 128 | 68.55 | 201.29 | 42.05 | 467.46 | 41.23 | 663.54 | 9.674 |

Vision 计时包含本步历史图像编码；LLM 与 DiT-B 前后 CUDA 同步。服务总计还包含选择、
processor、输入构建、TVI、cache pooling 等，因此不等于三模块时间简单相加。
Habitat 数值是 `get_sensor_observations` / `get_observations_at` 的同步墙钟时间，嵌套调用
去重；它是实际渲染/观测采集接口成本，不是仅 GPU 光栅化 kernel 的事件时间。
客户端闭环包含 RPC、执行 8-waypoint 与模拟器操作，到动作返回为止，不含之后的日志落盘。
首次观测采集时间另存 `observation_render_ms`，不重复加到每步动作后的采集计时中。

## 验证记录

1. 修改前原服务：R2R episode 1/2、RxR episode 4/5，各完成2、失败0；均 SR=1。
   R2R 平均 NE=0.943318、SPL=0.825799、nDTW=0.737396；
   RxR 平均 NE=1.429361、SPL=0.552052、nDTW=0.420924。
   原始完整结果已归档 `before_baselines/`，不是历史聊天中继承的数字。
2. `fixed_equivalence.json`、`fixed_habitat_equivalence.json`：同固定图两次连续输出严格一致。
   真实原始观测保存为 `fixed_habitat_observation.npz`，没有使用目标位置或评测指标作为模型输入。
3. `scope_tests.log`：6项单元测试通过；`baseline_regression_tests.log`：既有5项路径/nDTW回归通过。
   初次回归缺少 PYTHONPATH 的失败日志单独保留，补正确环境后重跑成功，没有改测试掩盖失败。
4. `smoke_verification.json`：12个任务/模式组，每组相同两集，共24集/48预测；
   逐集无 failure，原始 eval_info、trajectory、step 和 scope JSON 全部保留在 `runs/`。
5. `aggregate.csv`：每任务/模式一行，包含原始 episode 导航指标和成本均值/最大值。
   所有集均显式受两步上限约束，不能用该表的导航指标宣称完整评测性能。
6. `compile.log`：新增入口与运行时编译检查通过。保留 Gym、processor 弃用提示与
   readiness 探针造成的 WebSocket HTTP 426 提示；它们未导致 episode 推理失败。

## 复现与未完成项

完整启动、测试、重跑命令见 `Experiment/SCOPE/README.md`。
原始逐组执行参数保存在 `*_command.json`，解析后配置保存在各组 `runs/*/config.yaml`。
`benchmark_instrumentation.patch` 仅含本次外部3文件插桩差异，原文件副本在 `before_sources/`。
所有新增/修改文件（不含自动生成 Python 字节码）逐项列在 `files_manifest.txt`。

未完成且不可冒充完成的事项：真实已学习 selector 的评测、
六模式完整 episode/全量 split 评测、充分预热与多次重复的稳态成本统计。
本任务不启动训练、不改 checkpoint、不提交 git commit；以上边界不影响采集与选择器接口的可运行验证。

执行结束已停止本次新增的 10094 服务，释放 GPU 0 模型显存；原有 10093 基线服务保持运行。
需要继续推理时按 README 命令重新启动，而非假设临时服务仍常驻。
