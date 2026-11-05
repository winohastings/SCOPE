# Stage2 执行入口

审阅报告：`../../../docs/Logs/Task3/STAGE2_REVIEW_REPORT.md`。
八份任务交付报告及测量结果位于本目录；最终运行只认 `artifacts/release/`，早期尝试
保留用于追踪失败/数值检查，不混入最终导航或吞吐汇总。

## 一键小规模复现

在项目根执行。使用新的输出路径，不覆盖已归档实测。入口自动启动/关闭本轮模型服务，
GPU0冻结模型、GPU1真实Habitat，限定两训练episode以内、每episode最多16训练策略步。
默认端口10096，应先确认未被其他服务使用。此命令不会全规模训练或运行val_unseen。

```bash
/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/run_pilot.py \
  --output Experiment/SCOPE/Stage2/artifacts/recheck \
  --port 10096
```

完成后只读审计：

```bash
/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/audit_results.py \
  --output Experiment/SCOPE/Stage2/artifacts/recheck
```

## 分步运行

先启动独立模型服务：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage2/server.py \
  --output Experiment/SCOPE/Stage2/artifacts/manual_runtime \
  --port 10096 \
  --regime C
```

另一个终端运行真实短训练，默认使用原train/dev隔离清单；`--regime`会明确重置对应
Actor/Critic和采样器，不是隐式保留其他场景训练权重：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/real_rollout.py \
  --output Experiment/SCOPE/Stage2/artifacts/manual_train \
  --purpose train \
  --count 2 \
  --max-steps 8 \
  --regime C \
  --update
```

从episode边界checkpoint恢复，真实下一训练episode由checkpoint cursor确定：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python Experiment/SCOPE/Stage2/real_rollout.py \
  --output Experiment/SCOPE/Stage2/artifacts/manual_resume \
  --purpose train \
  --count 1 \
  --max-steps 4 \
  --resume Experiment/SCOPE/Stage2/artifacts/manual_runtime/actor_critic_C_update_002.pt \
  --update
```

空间adapter默认关闭；仅在有经过独立train/dev验证的`SpatialAdapter.state_dict()`时使用
`--spatial-adapter`。本轮未训练该adapter，不提供虚构的adapter checkpoint。

## 文件清单

`FILES_MANIFEST.tsv`列出本轮所有新建/修改文件及SHA256（含原始日志和检查点）；
`IMPLEMENTATION_REPORT.md`给出相对旧版本的逐模块变化。第一阶段保护清单与冻结权重
未修改；所有生成的Actor/Critic权重均是新文件，不覆盖SimpleNav预训练checkpoint。
