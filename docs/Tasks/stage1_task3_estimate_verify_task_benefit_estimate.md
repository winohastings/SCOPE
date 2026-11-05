# 任务 3：执行 SCOPE 第一阶段——可行性实现、评估与结果

## 0. 任务

使用已完成的任务 1 实现和任务 2 协议包，完成新划定范围的缩减版第一阶段可行性实验。

本任务是执行任务，不是又一个研究规划任务。

目标是确定多分辨率 Glance–Focus 视觉采集在技术上是否可靠，并在受控视觉 token 预算下是否提供初步导航效用。

用户明确授权此缩减版第一阶段实验。

重要：原始任务 2 的 G0–G4 协议仍是草案，未获批准完整执行。不要执行其原始训练或完整基准范围。

创建一个单独的、带版本的可行性协议。不要覆盖或静默修改任务 1 或任务 2 的产物。

## 1. 强制源代码审查

在实现之前，阅读以下现有文件：

- Experiment/SCOPE/Stage1/STAGE1_EXPERIMENT_PROTOCOL.md
- Experiment/SCOPE/Stage1/SELECTOR_DESIGN.md
- Experiment/SCOPE/Stage1/IMPLEMENTATION_AUDIT.md
- Experiment/SCOPE/Stage1/REMAINING_IMPLEMENTATION_TASKS.md
- Experiment/SCOPE/Stage1/METRICS_AND_STATISTICS.md
- Experiment/SCOPE/Stage1/COST_MEASUREMENT_PROTOCOL.md
- 所有相关任务 1 实现报告和现有评估配置。

复用已完成的实现和现有验证证据。

不要重复已完成的工作。

如果现有产物与本任务矛盾，遵循本任务的缩减范围并记录差异。

## 2. 冻结缩减版第一阶段协议

创建：

Experiment/SCOPE/Stage1_Feasibility/STAGE1_FEASIBILITY_PROTOCOL.md

精确定义四个实验组：

### G0：原始 SimpleNav

- 四张当前视图图像。
- 每视图 256×256。
- 每视图 64 个合并后视觉 token。
- 256 个当前视图视觉 token。
- 现有历史路径不变。

### G1：仅 Glance

- 四张 128×128 Glance 图像。
- 每视图 16 个视觉 token。
- 64 个当前视图视觉 token。
- 现有历史路径不变。

### G2：Fixed-Cyclic-2

- 四张 Glance 图像。
- 每个导航步恰好两个 Focus 裁剪。
- 每个 Focus 裁剪：256×256，64 个 token。
- 总计：192 个当前视图视觉 token。

使用已验证的八步 ROI 序列：

(0,4), (8,12), (1,5), (9,13),
(2,6), (10,14), (3,7), (11,15)

在每个 episode 开始时将循环重置为相位零。

### G3：Random-2

- 四张 Glance 图像。
- 每个导航步恰好两个不同的 Focus ROI。
- 保留任务 2 协议中定义的候选相机约束。
- 使用由运行种子、episode ID 和导航步推导的确定性随机种子。
- 总计：192 个当前视图视觉 token。

所有四组必须共享：

- 相同的 512×512 Habitat 渲染配置。
- 相同的检查点。
- 相同的动作归一化。
- 相同的 waypoint 执行协议。
- 相同的历史视觉缓存协议。
- 相同的 episode 列表。

G0 使用由共享 512×512 渲染生成的 256×256 兼容输入。不要将其描述为原生 256×256 Habitat 渲染的像素级一致复现。

分别报告当前视图和历史视觉 token。

## 3. 明确排除项

不要：

- 实现或训练 Learned-2。
- 训练 Actor 或 Critic。
- 实现 GRU 或状态轨迹。
- 实现 PPO 或 GAE。
- 添加感知 STOP 或自适应 Focus 预算。
- 微调 Qwen3.5 或 DiT-B。
- 运行完整 R2R 或 RxR val_unseen 评估。
- 运行原始的 599,040 个 transition 训练协议。
- 修改预训练 SimpleNav 检查点。

这些属于后续研究阶段。

## 4. 数据集与评估

使用现有任务 2 隔离的 R2R 开发 manifest，包含 64 个固定 episode。

验证该 manifest 与训练和 val_unseen 场景分离。

在完全相同的 64 个 episode 上运行所有四组。

使用相同的 episode 顺序和确定性环境设置。

保留完整导航 episode；不要将其截断为冒烟测试。

本可行性实验使用一个固定运行种子。这是初步可行性运行，不是多种子显著性研究。

不要替换不同 episode 列表而不记录原因。

## 5. 实现

仅实现缺失的执行组件：

- Fixed-Cyclic-2 策略适配器。
- 确定性 Random-2 策略适配器。
- 配置到评估参数的编译器。
- 组特定的运行 manifest。
- 所需的日志记录和结果聚合。

复用现有任务 1 视觉编码器、裁剪路径、Qwen3.5 接口、token scatter 和 Habitat Runtime。

不要重构无关模块。

在完整运行之前，验证：

1. Fixed-Cyclic-2 在八个导航步内覆盖全部 16 个 ROI。
2. 一个 ROI 对内没有重复 ROI。
3. 满足相机对约束。
4. Random-2 可复现。
5. G0/G1/G2/G3 token 数量正确。
6. 各组历史缓存行为一致。
7. 所有四种模式通过完整 episode 冒烟测试。

## 6. 成本插桩

使用任务 2 成本测量协议。

分别测量：

- Habitat 渲染时间。
- 当前视觉编码时间。
- 历史视觉编码时间。
- 选择器/策略开销（如有）。
- Qwen 语言模型时间。
- DiT-B 动作生成时间。
- 总模型推理时间。
- 端到端导航步时间。
- 峰值 GPU 内存。
- 当前视觉 token。
- 历史视觉 token。

执行预热和稳态测量。

报告中位数和 p90 延迟。

不要从 token 数量减少推断 FLOPs 节省。使用实际模型操作核算或清晰记录的估计。

确保计时插桩不改变评估行为。

## 7. GPU 执行

启动前检查可用 GPU 内存和现有进程。

高效使用两块 RTX 4090 GPU，但不要中断无关的运行任务。

避免不必要的重复 Qwen3.5 模型实例。

如果同时执行造成内存压力或改变延迟测量，则顺序执行各组。

记录 GPU 型号、软件版本、检查点哈希和代码提交。

## 8. 执行顺序

按以下顺序执行：

1. 审查源代码产物。
2. 创建并冻结缩减版可行性协议。
3. 实现缺失的策略适配器和配置编译器。
4. 运行单元和回归测试。
5. 运行完整 episode 冒烟测试。
6. 运行简短的成本测量试点。
7. 为 G0 运行全部 64 个 episode。
8. 为 G1 运行全部 64 个 episode。
9. 为 G2 运行全部 64 个 episode。
10. 为 G3 运行全部 64 个 episode。
11. 聚合指标并生成图表。
12. 执行一致性检查并生成最终报告。

如果某组失败，保留其失败日志并诊断问题。不要静默丢弃失败 episode 或更改评估分母。

## 9. 所需交付物

创建：

Experiment/SCOPE/Stage1_Feasibility/

- STAGE1_FEASIBILITY_PROTOCOL.md
- IMPLEMENTATION_REPORT.md
- EXECUTION_MANIFEST.json
- RUN_ENVIRONMENT.json
- RESULTS_PER_EPISODE.csv
- RESULTS_PER_STEP.csv
- RESULTS_AGGREGATED.csv
- NAVIGATION_METRICS.md
- COMPUTATIONAL_COST.md
- FAILURE_ANALYSIS.md
- STAGE1_FEASIBILITY_REPORT.md
- plots/

最终报告必须回答：

1. 仅 Glance 是否保持可用的导航性能？
2. Focus 是否提供可观察的导航效用？
3. Fixed-2 与 Random-2 相比如何？
4. 相对于 G0，视觉 token 减少了多少？
5. 实际模型和端到端延迟变化是什么？
6. 视觉采集流水线是否足够稳定以支持训练学习型策略？
7. 进入第二阶段之前还有哪些实现和数据问题？

清楚区分实测结果、估计和假设。

不要声称本可行性实验证明学习型状态条件策略的有效性。

## 10. 最终完成标准

仅当以下条件满足时，第一阶段才算完成：

- 所有四组执行全部 64 个 episode，或具有记录的可复现失败。
- Episode 分母已验证。
- 所有指标和成本日志已生成。
- 组配置和检查点哈希已记录。
- 回归测试通过。
- 最终报告和图表已生成。

不要自动启动第二阶段。

完成时，提供简洁的执行摘要、实际 GPU 小时数、挂钟时长、失败计数、关键发现和建议的第二阶段起点。