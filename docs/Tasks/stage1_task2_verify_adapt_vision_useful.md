# 任务 2：SCOPE 第一阶段实验设计与协议锁定

## 目标

基于已完成的任务 1 实现，为 SCOPE 论文准备完整的第一阶段实验协议。

不要重写现有的视觉采集流水线。

## 1. 审计现有实现

验证：
- 原始检查点和输出兼容性。
- Glance 和 Focus 张量形状。
- 实际 Qwen3.5 视觉 token 数量。
- Fixed、Random 和顺序选择器行为。
- 每步计时和轨迹日志记录。
- 现有评估命令和 episode 选择。

识别任何实现缺口，但不要修改无关模块。

## 2. 最终确定 Fixed-2 基线

当前实现使用 front.R1 和 front.R4。

评估这是否足以作为固定位置基线。

实现或提出一种确定性循环 Fixed-2 策略，覆盖全部 16 个 ROI，且每个导航步恰好两个裁剪。

确保 Fixed-2、Random-2 和 Learned-2 之间的 Focus 预算相等。

## 3. 定义第一阶段实验组

G0：原始 SimpleNav
G1：仅 Glance
G2：Fixed-Cyclic-2
G3：Random-2
G4：Learned-2

保持原始检查点冻结。

第一阶段不包括 STOP 或自适应视觉预算。

## 4. 设计 Learned-2 选择器

设计一个轻量级策略，其条件为：
- 四视图 Glance 特征。
- 导航指令。
- 先前获取的 Focus 特征。
- 候选 ROI 和相机嵌入。

该策略必须依次选择两个不同的 ROI。

保持选择器与冻结的 SimpleNav 主干分离。

提供具体的训练目标、奖励设计、优化策略和可复现性协议。

不要将现有的脚本化选择器标记为学习策略。

## 5. 评估协议

主要基准：R2R-CE val_unseen。
泛化基准：RxR-CE val_unseen。

在各组之间使用相同的 episode 列表、动作协议和模拟器设置。

定义随机种子、重复运行、指标聚合和统计比较。

## 6. 成本测量

定义可复现的预热和稳态测量协议。

记录：
- 视觉 token 数量。
- 视觉编码器 FLOPs 和延迟。
- LLM 延迟。
- DiT-B 延迟。
- Habitat 渲染延迟。
- 端到端延迟。
- 峰值 GPU 内存。

将模拟器渲染成本与模型推理成本分开。

## 7. 所需交付物

产出：

1. STAGE1_EXPERIMENT_PROTOCOL.md
2. STAGE1_GROUP_CONFIGS/
3. SELECTOR_DESIGN.md
4. METRICS_AND_STATISTICS.md
5. COST_MEASUREMENT_PROTOCOL.md
6. 剩余实现任务列表及预计执行成本。

在实验协议经过审查并冻结之前，不要启动完整评估或策略训练。

保留所有现有基线代码、检查点和任务 1 产物。