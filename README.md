# SCOPE：Stage1 Glance–Focus 可行性实验

这是从 AdaPercVLN 定向整理的 SCOPE 方法与第一阶段研究归档，不是 Habitat、R2R、RxR 或 SimpleNav 官方仓库的镜像。

## 方法与范围

四相机共享 512×512 原始渲染；Glance 为每相机 128×128 / 16 tokens；Focus 从原始图裁剪 256×256 / 64 tokens，跨四相机共 16 个候选 ROI。沿用冻结的 SimpleNav RXR step-8000 检查点、Qwen3.5-4B、DiT-B、历史观测与导航动作协议。

| 实验组 | 当前观测 | 当前视觉 tokens |
|---|---|---:|
| G0 | 四张 256×256 兼容输入 | 256 |
| G1 | 四张 Glance | 64 |
| G2 | 四张 Glance + Fixed-Cyclic-2 | 192 |
| G3 | 四张 Glance + 确定性 Random-2 | 192 |

G0 来自共享 512 渲染后缩放，不宣称与原生 256 渲染像素一致。历史 tokens 单独计算。Task1 的 Fixed-2 为固定 0/3，Task3 的 Fixed-2 才使用八步循环；两者不可混用。Task1 的 learned/STOP 仅是接口冒烟，Task2 的 G4 为设计草案，未训练学习策略，也没有启动第二阶段。

## 已有结果（历史证据，不是发布时重跑）

R2R 隔离开发清单 64 集/组，seed=42，四组共 256 集，记录执行失败为 0。

| 组 | SR | SPL | nDTW |
|---|---:|---:|---:|
| G0 | 0.4688 | 0.3941 | 0.4103 |
| G1 | 0.5625 | 0.4396 | 0.4133 |
| G2 | 0.3750 | 0.2952 | 0.3328 |
| G3 | 0.3750 | 0.2919 | 0.3477 |

仅 Glance 在这批开发样本上表现可用；当前 Fixed/Random Focus 未显示优于 Glance 的效用。单 seed、开发场景来自原 train，不能声称统计显著、val_unseen 泛化或学习型策略有效。

## 导航

- `docs/Tasks/`：三个原始任务。
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
