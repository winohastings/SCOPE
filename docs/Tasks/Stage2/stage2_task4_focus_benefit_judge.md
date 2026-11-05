# 任务 4：SCOPE 核心架构、Focus 诊断与训练就绪性

## 目标

在已完成的stage1流水线之上，实现核心 SCOPE 状态条件主动感知架构。

使用仓库：[winohastings/SCOPE](https://github.com/winohastings/SCOPE.git)

保留所有第一阶段产物和冻结的 SimpleNav-RXR 检查点。

最终架构必须支持顺序 ROI 选择、可选的感知停止、导航状态记忆以及 Actor–Critic 训练。

本任务限于架构实现、受控诊断和小规模训练验证。不要启动全规模 PPO 训练或 val_unseen 评估。

## 第 1 部分：诊断第一阶段性能回退

阅读所有第一阶段结果和实现报告。

观察到的开发集排序为：

G1 仅 Glance > G0 原始 > G2/G3 Glance + Focus 在 SR 上。

调查 Focus 性能下降是否源于：

1. 空间和相机位置歧义。
2. 冻结 VLM 输入分布偏移。
3. 低价值或干扰性裁剪区域。
4. 与历史视觉特征的交互。

使用一小组固定的训练/开发状态。

在仅 Glance、选定 Focus 区域和带空间标注的 Focus 之间进行受控同状态比较。

绝不使用留出 val_unseen 结果进行模型设计。

交付一份诊断报告，为每个假设提供支持或反对的证据。

## 第 2 部分：实现可复用的当前视觉特征

扩展现有任务 1 视觉流水线，不重新编码先前选定的视觉观测。

要求：

- Glance 特征：[4,16,2560]。
- Focus 特征：每个裁剪 [64,2560]。
- 冻结视觉编码器。
- 带类型的当前特征缓存。
- 精确的相机和 ROI 几何。
- 正确的 image grids、M-RoPE 和原始视觉 scatter。
- 策略推理期间不得访问未选定的高分辨率 Focus 特征。

添加特征形状和缓存一致性测试。

## 第 3 部分：实现最终 SCOPE 策略模块

实现：

- 语言条件 ROI Actor。
- 独立价值头。
- 跨导航步 State GRU。
- 增量 Focus 记忆。
- 感知 STOP 动作。
- 可选轻量级空间融合适配器。

Actor 必须支持以下开关：

- use_language
- use_focus_memory
- use_navigation_state
- use_action_feedback
- enable_perception_stop

所有开关必须共享同一个核心 Actor 实现。

不要实现单独的一次性 Selector 架构。

## 第 4 部分：实现导航状态轨迹

在每个完成的导航策略步，存储：

- Glance/Focus 特征摘要。
- 选定 ROI 序列。
- 预测 waypoint 块。
- 实际执行位姿。
- 实际体坐标系位移。
- 预测–执行误差。
- 感知 token 和延迟成本。
- 导航终止信息。

仅在实际动作执行结果之后更新 State GRU。

在 episode 边界重置所有循环状态。

确保 Actor 或 Critic 观测中不包含真值导航指标。

## 第 5 部分：构建训练接口

支持：

A. 禁用历史的两 Focus 固定 Actor 预热。

B. 使用真实先前动作反馈的状态条件 Actor 训练。

C. 带 0–2 个 Focus 动作和成本感知奖励的自适应停止。

实现 PPO/GAE，正确存储旧对数概率、固定 rollout 目标、有效动作掩码以及可复现的检查点/恢复行为。

Actor 和 Critic 必须保持参数独立。

默认冻结 Qwen3.5 和 DiT-B。

将感知 STOP 与 SimpleNav 现有的导航 STOP 分开。

## 第 6 部分：小规模验证

运行：

1. 特征和空间对齐测试。
2. Actor 前向/反向测试。
3. Critic 目标和 PPO 比率测试。
4. GRU 重置和动作反馈测试。
5. 训练/验证泄漏测试。
6. 简短训练场景 rollout。
7. 小型 Actor 学习健全性测试。
8. 固定开发集导航检查。

验证策略从初始化发生变化，并且训练梯度、奖励、掩码和检查点正确。

除非实际测量，否则不要声称导航改进。

## 第 7 部分：交付物

创建 Experiment/SCOPE/Stage2/，包含：

- CORE_ARCHITECTURE.md
- FOCUS_DIAGNOSTICS.md
- STATE_TRAJECTORY_SPEC.md
- TRAINING_OBJECTIVES.md
- IMPLEMENTATION_REPORT.md
- UNIT_AND_INTEGRATION_TESTS.md
- SHORT_TRAINING_REPORT.md
- GPU_BUDGET_PILOT.md

最终报告必须回答：

1. Focus 在受控条件下是否具有可测量的潜在收益。
2. 空间对齐是否需要适配器。
3. 完整的状态条件 Actor–Critic 是否正确运行。
4. GRU 是否使用实际导航反馈且无泄漏。
5. PPO 是否能执行有效更新。
6. 测得的训练吞吐量和预计完整训练成本。

不要修改第一阶段历史结果或预训练检查点。

在架构和小规模测试通过之前，不要启动完整训练。