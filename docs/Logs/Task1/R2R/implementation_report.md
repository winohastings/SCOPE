# Task1 R2R 专项执行报告

日期：2026-11-05。角色：执行者（Exec Agent）；根目录要求的角色文档仍缺失，
按用户指定任务、项目索引及已有适配文档执行。

## 1. 纠正模型口径

官方仓库本地文件：
`/home/data/czh/Third_Party/baselines/SimpleNav/docs/guides/VLNCE_TRAINING_AND_EVALUATION.md`。
其开头明确说明：在 RxR LeRobot 数据上训练一个模型，将同一份 step-8000 checkpoint
分别用于 R2R `val_unseen` 与 RxR `val_unseen`。第4–7节给出共享 checkpoint、
统计量、动作协议及两个任务入口。原始指南的路径及 SHA256 已保存于 `protocol_audit.json`。

因此，**SimpleNav-RXR 权重评测 R2R 就是官方共享权重协议，不需要额外 R2R checkpoint**。
上一版总报告将“独立 R2R 权重基线”列为未完成项不准确，已经修正；不是新增另一份模型，
也不应将已完成的 R2R 适配重复标记成模型不兼容。

已有适配和单集基线见 `../../2026-11-05-simplenav-r2r-rxr-baseline-smoke.md`。
本次继续使用该入口、数据路径和 nDTW 修复，不改评测指标或动作协议。

## 2. 本次独立执行结果

使用 `Experiment/SCOPE/run_smoke.py --task r2r`，在本目录创建全新运行产物，
没有复用此前 `Task1/runs/r2r_*` 的 episode 结果。

| 检查 | 实测结果 |
|---|---|
| 模型 | RXR `steps_8000_pytorch_model.pt`，Qwen3.5-4B；与已有 R2R 适配一致 |
| 数据 | R2R-CE `val_unseen`，episode 1、2，场景 `zsNo4HB9uLZ` |
| 六模式 | original、glance_only、fixed_2、random_2、learned_2、learned_adaptive |
| 执行 | 每模式两集、每集两步，共12次 episode 运行、24次预测，无执行失败 |
| 原始图像 | 四个相机 512×512；固定真实观测 NPZ 中四图均为 `[512,512,3]` |
| 固定观测兼容性 | 与10093原服务比较连续两步，最大绝对差均为0.0，动作形状 `[8,4]` |
| Glance | 实际当前块 grid `[1,8,8]`，每图16 token，四图64 |
| Focus | 实际当前块 grid `[1,16,16]`，每个64 token；四Glance＋两Focus为192 |
| 自适应 STOP | 本次脚本选择器每步 trace `[0,16]`，一个Focus后STOP，当前128 token |
| 形状与元数据 | 每步预测 body/world 和实际 waypoint 均为 `[8,4]`；相机ID、ROI、focus索引保存 |
| 测试 | SCOPE 9项通过，既有路径/nDTW回归5项通过；修改脚本编译通过 |
| 权重 | 复核既有 SHA256 成功；未写入、重新训练或替换 checkpoint |

固定观测来自真实 R2R 初始状态，使用 `../fixed_habitat_observation.npz`。
旧服务接收同一原图经BICUBIC得到的256图，新服务original分支接收512原图并做相同缩放。
这验证同模型输入的严格输出等价性，不声称512渲染再缩放与原生256渲染逐像素相同。
两次预测覆盖历史输入路径；实际多分辨率模式也分别验证了连续两次推理。

### 既有未修改基线

Task1修改前已运行同一RXR权重在R2R两集的完整闭环，原始结果保存在
`../before_baselines/task1_before_r2r_20261105/`。本次重新核对这些磁盘结果，
**没有将其冒充本轮新跑的完整基线**：完成2、失败0，平均SR=1、NE=0.943318m、
SPL=0.825799、nDTW=0.737396、policy steps=10.5。
本轮新增的是六模式独立冒烟及固定观测复验，而非完整重评整个R2R split。

## 3. 协议一致性与实现复用

`protocol_audit.json` 比较六组解析后的配置：`benchmark`、`input`、`env`、`model`、
`stop_rule`、`stop_threshold`、`observation` 均一致，只有模式和输出目录不同。

- 相机顺序 front/left/right/rear，相同相机高度、HFOV与朝向，统一512渲染。
- 模型seed=42；Habitat seed沿用已有R2R配置；每模式相同两集。
- `vln_train_train` 动作统计量不变，`anchor_relative_body_frame_xyz_yaw` 动作不变。
- 每步执行8个waypoint，`collision_slide_pose_delta`碰撞控制不变。
- 导航停止仍为 `mean_adjacent_waypoint_translation <= 0.03`。
- 所有模式统一显式两步冒烟上限；不是修改导航STOP，也不是官方完整评测步数。

Task1多分辨率实现由两个任务共享，不再复制一份R2R专用模型代码：
原始512图在裁剪前保留；兼容图为256；Glance为128；16个ROI均直接从512图裁出256区域。
Fixed-grid取front.R1/R4，Random按seed/episode/step确定性无重复采样。
顺序选择器最多调用两次，STOP=16仅用于adaptive。
`learned_2` / `learned_adaptive` 当前是**显式脚本选择器接口测试，不是已学习策略**。

共享模型继续保留2560维postmerge接口、逐图grid、M-RoPE、既有scatter和pool4历史缓存。
当前/历史token分开计数；Focus不覆盖相机全视野历史缓存；详细模型改动见上级总报告。

## 4. 本轮实测成本

下表每模式为两集×两步的4次预测均值，延迟单位ms；显存为组内最大PyTorch allocated
显存GiB，含模型驻留，不含Habitat/驱动分配。GPU为RTX4090，模型及Habitat均在GPU0；
原基线服务在GPU1。没有额外预热与多次重复，不能由这些小样本宣称显著加速。

| 模式 | 当前token | Vision | LLM | DiT-B | 服务总计 | Habitat采集 | 客户端闭环 | 峰值GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 256 | 81.91 | 296.30 | 51.85 | 579.44 | 37.02 | 781.64 | 9.696 |
| glance_only | 64 | 79.74 | 270.43 | 53.66 | 572.16 | 35.31 | 775.77 | 9.661 |
| fixed_2 | 192 | 85.74 | 288.98 | 51.80 | 600.87 | 36.73 | 805.29 | 9.688 |
| random_2 | 192 | 85.02 | 288.28 | 52.07 | 600.08 | 36.55 | 817.30 | 9.688 |
| learned_2（脚本） | 192 | 72.26 | 202.11 | 42.16 | 475.54 | 37.37 | 680.98 | 9.688 |
| learned_adaptive（脚本） | 128 | 80.76 | 277.62 | 52.02 | 579.74 | 36.62 | 780.09 | 9.674 |

Vision包含当前及历史图编码；模块前后CUDA同步。服务总计还含预处理、选择、TVI等。
Habitat指标是观测采集接口墙钟时间，嵌套调用去重，不是纯GPU光栅化kernel耗时。
客户端闭环含RPC和waypoint执行，不含之后的日志落盘。各组时钟/缓存及短样本存在波动，
不能解释成脚本learned策略比固定策略更快或更优。

## 5. 修改与产物

| 文件 | 本轮变化 |
|---|---|
| `Experiment/SCOPE/run_smoke.py` | 新增 `--task r2r/rxr/both`；按任务聚合与计算组数；默认仍双任务；拒绝非正步数；CSV不再固定标为two-step |
| `tests/test_scope_runner.py` | 新建R2R单任务、双任务默认兼容、episode不一致拒绝3项测试 |
| `Experiment/SCOPE/README.md` | 补R2R独立命令与官方共享权重说明 |
| `docs/Logs/Task1/implementation_report.md` | 撤回独立R2R权重缺项，链接本报告 |
| `docs/Logs/2026-11-05-simplenav-r2r-rxr-baseline-smoke.md` | 保留原结果，补官方口径与Task1 R2R后续链接 |
| `docs/index.md` | 增加本专项日志入口 |
| 本报告及本目录运行产物 | 六组原始结果、固定观测比较、成本CSV、配置审计与测试日志 |
| `docs/Logs/Task1/files_manifest.txt` | 补本轮脚本、文档与产物清单 |

所有本轮文件逐项见 `files_manifest.txt`。关键产物：

- `aggregate.csv`：六行R2R模式成本及原始episode导航指标；只供冒烟分析，不作正式分数。
- `runs/r2r_*/`：每组配置、episode计划、原始eval_info/trajectory/动作及scope逐步JSON。
- `*_command.json`：每组实际执行参数；`smoke_verification.json`：六组通过记录。
- `fixed_habitat_equivalence.json`：本轮真实固定观测连续两步等价性与新模式推理记录。
- `protocol_audit.json`：共享权重出处、配置一致性、网格/ROI/waypoint/计时检查。
- `scope_tests.log`、`baseline_regression_tests.log`、`compile.log`、`checkpoint_integrity.log`。

## 6. 具体复现命令

以下在AdaPercVLN根目录执行。先在独立终端启动共享权重服务：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/scope_server.py \
  --checkpoint /home/data/czh/Third_Party/baselines/SimpleNav/local/checkpoints/vlnce/navvla_qwen35_cpm_rxr/checkpoints/steps_8000_pytorch_model.pt \
  --telemetry /home/data/czh/AdaPercVLN/docs/Logs/Task1/R2R_recheck/server_steps.jsonl \
  --smoke-selector \
  --port 10094
```

另一个终端执行R2R专用两集六模式；使用新目录避免结果复用：

```bash
/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/run_smoke.py \
  --output docs/Logs/Task1/R2R_recheck \
  --task r2r \
  --max-steps 2
```

不启动模型即可重验本轮聚合；运行新增及既有SCOPE单元测试：

```bash
/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/run_smoke.py \
  --output docs/Logs/Task1/R2R \
  --task r2r \
  --aggregate-only

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope_*.py'
```

## 7. 完成边界

按Task1要求完成R2R采集/接口与两集六模式冒烟验收，不需要另一份R2R权重。
本轮没有启动训练，没有修改模型权重或外部导航动作实现，没有进行全量split评测。
learned策略训练与真实策略效果、完整episode六模式对比、稳态成本统计仍不在本轮完成范围。
本轮新增10094服务在验收后停止，原10093基线服务保留。
