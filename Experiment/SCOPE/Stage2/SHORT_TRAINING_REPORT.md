# 真实小规模训练报告

## 最终运行与样本

最终主运行：`artifacts/release/`；模型/rollout/checkpoint：`artifacts/release_runtime/`。
读取既有scene隔离清单，训练episode1/2均属于训练scene7y3sRwLe3Va；开发前两集
9791/1279来自开发scene，未进入PPO。早期接线/诊断尝试保留但不混入本表。

只运行4个小rollout：A一集4步，B一集4步，C两集各8步，共24个真实导航transition、
4次PPO update，每次2epoch。训练步数上限为显式pilot截断，不把未停止训练轨迹当
完整episode性能。Qwen/视觉/TVI/DiT-B全部冻结。

## 更新证据

| 场景 | transition/update | Actor参数L2变化 | Actor State GRU最大梯度 | Critic State GRU最大梯度 |
|---|---:|---:|---:|---:|
| A | 4 / 1 | 0.32182 | 0（禁用） | 0（禁用） |
| B | 4 / 1 | 0.34212 | 0.01835 | 0.02132 |
| C | 16 / 2 | 0.34095、0.24522 | 0.02271 | 0.02410 |

每次update第一个epoch联合ratio均为1.0；后续epoch的ratio改变。Actor/Critic梯度均
有限且非零；主干全部grad=None。真实奖励范围约[-0.05915,+0.10329]，不是常数。
24个训练chunk中3个触发实际step_filter阻挡/滑移；奖励记录包含测量来源，不静默填零。

训练感知预算覆盖0、1、2块Focus：分别1、1、22个transition。A/B固定两块，C覆盖
自适应停止。82个正常导航预测（含开发检查）均提交真实反馈并执行状态更新，episode
重置清零；未执行动作的导航STOP也有明确终止记录，不把感知STOP当导航STOP。

## 学习健全性

对C第一个真实训练rollout的8个首决策，额外做30步小型过拟合检查：NLL从2.85299
降至0.002155。它只证明真实特征上的Actor可学习，不是新的on-policy PPO或导航收益
证据；不读开发状态，也不将该过拟合模型部署到导航。

## 固定开发检查

同样固定开发前两集、同导航seed、完整episode、原始导航STOP，记录执行失败均为0：

| 时点 | SR | mean nDTW | episode9791 SR/nDTW | episode1279 SR/nDTW |
|---|---:|---:|---|---|
| 初始化Actor | 0.5 | 0.47005 | 0 / 0.0000856 | 1 / 0.94001 |
| C短训练后 | 0.5 | 0.58335 | 1 / 0.90054 | 0 / 0.26616 |
| 加载同一checkpoint后 | 0.5 | 0.58335 | 与上一行相同 | 与上一行相同 |

SR没有提升，两个episode的成败交换。nDTW描述性均值上升不构成泛化或显著改善结论，
也未据此选择checkpoint、调参或更换模型。不能将初始化Actor与Stage1 G1直接等同。

## 恢复与边界

保存C update2后，通过真实服务restore，在同两集开发上完整重跑：SR、OSR、NE、
路径长、策略步数、终止原因和nDTW全部精确一致。单元测试另对固定rollout进行真正
“连续update vs 保存恢复再update”的逐参数/RNG一致性检查。

仅支持episode边界恢复，非任意Habitat中间状态快照；真实时延reward含系统噪声，
不把固定rollout恢复的逐位性扩大为每次真实重采样reward相同。

额外执行了真实训练恢复：加载C update2 checkpoint后自动取cursor指定的训练episode3，
两步真实rollout完成第3次update，Actor参数L2再变化0.21319，首次ratio仍为1.0，
主干grad=None。记录在`artifacts/resume_train_check/`和`artifacts/diagnostic_runtime/`。
这额外2个transition不混入前述24transition吞吐/开发前后对比；包括恢复检查在内，本轮
最终验证为26个真实训练transition、5次PPO update、3个独立训练episode ID。

局限：一个训练scene、两个相似路线episode、单seed、长度1截断GRU反向传播、短历史、
小rollout，不能判断训练收敛或长时记忆有效性。没有全规模PPO，没有val_unseen评估。
