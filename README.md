# SCOPE：Glance–Focus 与状态条件主动感知

这是从 AdaPercVLN 定向整理的 SCOPE 方法、Stage1 可行性与 Stage2 Task4 架构验证归档，不是 Habitat、R2R、RxR 或 SimpleNav 官方仓库的镜像。

## Stage2 Task2.2 更新
最新增量为 `Experiment/SCOPE/Stage2/Task22/`：完整 Spatial Query Actor（STOP→camera→连续latent位置）、Task21增量融合控制器和真实Habitat闭环。保留初始化Actor、30条无损压缩轨迹、4份episode边界及验证报告。67项SCOPE相关测试的历史结果不代表正式RL训练；本轮optimizer更新为0。详见 `docs/release/TASK22_RELEASE.md`。

```bash
python3.11 tools/verify_task22_release.py
```

## Stage2 Task2.1 更新


`Experiment/SCOPE/Stage2/Task21/` 实现192×192原图裁剪→256×256编码，并将Focus按真实面积覆盖写回对应相机的4×4网格。当前输出始终为 `(4,16,2560)`，不沿用Task4追加Focus token的路径；历史BATS另计。

仅校准一个门参数：96个校准样本、32个诊断样本、120步，最终gate=0.796728；6次无Focus回放的特征和动作差均为0。诊断BF16覆盖区MSE约下降79.57%，不是导航性能提升证据。最终2.2KB `artifacts/release/adapter.pt` 随代码发布；不包含采集RGB数据和预训练导航权重。

```bash
python3.11 tools/verify_task21_release.py
```

发布范围及限制见 `docs/release/TASK21_RELEASE.md`。本次不运行Task2.2、Actor/RL或完整闭环评估。

## Stage2 Task4 更新

`Experiment/SCOPE/Stage2/` 新增可复用当前视觉特征、统一语言条件 ROI Actor、独立 Critic、真实执行反馈 State GRU、顺序 Focus 记忆、感知 STOP 与 PPO/GAE。全部 Stage1 文件保持原样；具体文件范围和排除项见 `docs/release/STAGE2_RELEASE.md`。

历史实测：54 次同状态诊断，主训练 24 transitions/4 次 PPO 更新，恢复训练另 2 transitions/1 次更新。固定开发两集 SR 为 0.5→0.5，nDTW 均值 0.47005→0.58335；不宣称导航改善、训练收敛或完整评测完成。没有启动全规模 PPO 或 val_unseen 评估。

```bash
python3.11 tools/verify_stage2_release.py

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p test_scope_stage2.py
```

## 方法与范围

四相机共享 512×512 原始渲染；Glance 为每相机 128×128 / 16 tokens；Focus 从原始图裁剪 256×256 / 64 tokens，跨四相机共 16 个候选 ROI。沿用冻结的 SimpleNav RXR step-8000 检查点、Qwen3.5-4B、DiT-B、历史观测与导航动作协议。

| 实验组 | 当前观测 | 当前视觉 tokens |
|---|---|---:|
| G0 | 四张 256×256 兼容输入 | 256 |
| G1 | 四张 Glance | 64 |
| G2 | 四张 Glance + Fixed-Cyclic-2 | 192 |
| G3 | 四张 Glance + 确定性 Random-2 | 192 |

G0 来自共享 512 渲染后缩放，不宣称与原生 256 渲染像素一致。历史 tokens 单独计算。Task1 的 Fixed-2 为固定 0/3，Task3 的 Fixed-2 才使用八步循环；两者不可混用。Task1 的 learned/STOP 仅是接口冒烟，Task2 的 G4 为设计草案，Stage1 当时未训练学习策略；后续 Stage2 Task4 的受限架构与短训练验证单列，不回写 Stage1 结论。

## 已有结果（历史证据，不是发布时重跑）

R2R 隔离开发清单 64 集/组，seed=42，四组共 256 集，记录执行失败为 0。

| 组 | SR | SPL | nDTW |
|---|---:|---:|---:|
| G0 | 0.4688 | 0.3941 | 0.4103 |
| G1 | 0.5625 | 0.4396 | 0.4133 |
| G2 | 0.3750 | 0.2952 | 0.3328 |
| G3 | 0.3750 | 0.2919 | 0.3477 |

仅 Glance 在这批开发样本上表现可用；当前 Fixed/Random Focus 未显示优于 Glance 的效用。单 seed、开发场景来自原 train，不能声称统计显著、val_unseen 泛化或学习型策略有效。

## Stage2 / Task232

定向短续训与 C/C0 新增覆盖奖励探索已归档，见 [发布说明及关键训练日志](docs/release/TASK232_RELEASE.md)。A/B/C0/C各新增512条训练数据，U门禁64条，总计2112；未通过扩训/空间探索门槛，不宣称找到Task233稳定配置。OpenGL清理异常如实保留。**权重仅发布83个占位文件，完整权重和traces保留本地。**

运行 `python3.11 tools/verify_task232_release.py` 离线核验归档。

## Stage2 / Task231

失败归因与匹配反事实诊断已发布，见 [发布说明与关键日志索引](docs/release/TASK231_RELEASE.md)。包含 192 集反事实对照、128 个匹配分支、1280 次 Actor 探针及历史 4096-transition 数值审计；**未进行新训练**。forced4 的局部恢复不构成显著提升或泛化结论。权重仍留本地，仅上传占位。

使用 `python3.11 tools/verify_task231_release.py` 离线核验保留证据；完整诊断重跑仍依赖原机器权重、场景与完整 traces。

## Stage2 / Task23

Task23 导航级 PPO/GAE 与独立 Critic 的实现、文档和关键实验数据已归档，见 [发布说明](docs/release/TASK23_RELEASE.md)。完成 4096 个真实训练 transitions / 56 轮更新；最终确定性策略零 Focus 退化，不宣称有效论文基线。训练后 OpenGL 清理 exit 134 如实保留。

**Task23 checkpoint 全部保留本地，GitHub 仅保存 11 个 SHA256 占位文件。** 全量训练 traces 不上传；缩减包不支持独立完整重放。运行 `python3.11 tools/verify_task23_release.py` 校验归档与结果分母。

## 导航

- `docs/Tasks/`：三个 Stage1 原始任务及 `Stage2/` 下的 Task4。
- `Experiment/SCOPE/Stage2/`：Task4 核心代码、八份交付报告与关键实测产物。
- `docs/Logs/Task3/`：Task4 执行前独立审阅记录。
- `Experiment/SCOPE/`：Task1 入口、Stage1 协议草案、Stage1_Feasibility 已执行协议与实现。
- `patches/legacy_scope.patch`：新增采集代码与四个模型接口文件的精确补丁。
- `patches/simplenav_stage1.patch`：实际依赖的 benchmark、Habitat 接入与 legacy 服务改动。
- `docs/Logs/Task1/`、`docs/Logs/Task2/`：精简的兼容性、协议及回归证据。
- `docs/release/REVIEW.md`：发布审查、范围映射及限制。
- `docs/release/FILE_MANIFEST.json`：发布文件、原始来源及 SHA256 完整清单。
- `docs/release/EXCLUDED_FILES.json`：本轮未打包的 SCOPE 原始文件及排除原因。
- `third_party/README.md`：依赖锁、重建方式与许可证边界。

## 无 GPU 核验

Python 3.11+，归档检查仅使用标准库：

```bash
python3.11 tools/verify_release.py
```

逐步结果、GPU 样本及唯一一份结构化执行轨迹使用无损 gzip，保留解压后原始文件哈希。需要重新聚合时先解压：

```bash
python3.11 tools/verify_release.py \
  --extract-results
```

## 官方依赖重建与回归

不提交完整官方 runtime。先提供历史 legacy 源目录及锁定提交的 SimpleNav 官方 checkout；所有源文件先验 hash 校验，补丁应用后逐文件比对历史实测版本，不覆盖源仓库：

```bash
python3.11 tools/prepare_dependencies.py \
  --legacy-source /tmp/SimpleNav-RXR-official \
  --simplenav-source /home/data/czh/Third_Party/baselines/SimpleNav \
  --output .dependencies

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope_*.py'
```

原实验代码与冻结配置保留历史绝对路径，完整复现目前仍需原机器布局或显式迁移路径；它不是开箱即用的跨机器训练工程。不要直接运行 `execute.py` 覆盖归档结果。官方依赖、模型权重、Qwen、Habitat 环境、授权场景和任务数据需单独准备；本次发布不重新运行 GPU 实验。
