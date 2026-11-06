Task 2 — AGM Actor 与完整主动注视闭环
目标
实现动作空间完整、推理时可运行、训练时可重放计算概率的 STOP → camera → continuous position Actor，连通 Task 1 的视觉融合及现有 SimpleNav 导航接口。
Codex 实施范围
1. Actor 输入 s(k,j)=(G_k, L, M_(k,j), B_(k,j))：64 个按相机身份编码的 Glance token、指令特征、当前融合特征、当前导航步内 Focus 次数/坐标/coverage。采用 2560→192 投影、带 camera/2D 位置编码的两层 Transformer 和条件 Query cross attention。
2. 分层动作分布：Bernoulli STOP；非 STOP 时 categorical(4) 相机；选中相机对应二维高斯潜变量 u，中心 x,y=96+320*sigmoid(u)；像素舍入仅发生在 crop 调用处。存储 latent u 以无损重算 Gaussian log prob。
3. 从 j=0 开始，模型可选择零次 Focus；每次继续后完成裁剪、编码、Adapter 写回、更新 coverage；最多四次。第 4 次后强制终止感知，不要对强制动作计入可训练 STOP log probability。
4. 输出 trace：每个决策的状态张量/引用、stop、camera、u、坐标、old_logprob（分层 joint action）、entropy、Focus 数、时间、对应导航 transition 标识；为任务 3/4 提供一致数据协议。对过长轨迹、缓存和环境状态重放接口不做隐式假设。
5. 推理提供 deterministic 与 training-sampling 两种明确模式。保持 Qwen 当前视觉 token 数为 64；不修改 BATS 历史组织；最终导航动作仍由既有 SimpleNav 产生。
交付物及验收
- Actor 权重初始化、端到端 glance→0..4 focus→nav action 执行路径、感知轨迹数据结构和可重复推理入口。
- 测试合法中心边界、相机区分、不同 Focus 次数、STOP 与导航终止互不混淆、旧策略 logprob 的重算一致、反向梯度可达 Actor 各 head。
- 监测 STOP、camera、position 三种熵及 N_focus 分布，避免出现结构性死分支；在相同导航状态对比不同 Focus 的融合特征及下游 SimpleNav 动作，确认感知确实能够影响决策（不预设随机 Focus 必然改善导航）；不能以未训练 Actor 的随机性能作为方法有效性的验收。