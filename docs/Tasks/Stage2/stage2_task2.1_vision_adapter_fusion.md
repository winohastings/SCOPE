Task 2.1 — 冻结导航接口上的视觉融合与 Adapter 校准
目标
在不改变 SimpleNav/Qwen 输入协议的前提下实现真实 Focus 视觉增量：Focus 可以精确地更新其所在相机、所覆盖区域的 4×4 token；无 Focus 完全恢复 Stage 1 Glance 行为。该任务的代码审计应包括现有四视角数据、BATS、视觉缓存、Qwen token 组织、Task 4 裁剪与 episode checkpoint 接口，不要另立“调研”任务。
Codex 实施范围
1. 审核原仓库的 Glance token、相机顺序、Vision Encoder 编码结果和下游输入形状；把真实张量维度与 64/64+64*N 的预期分别记录，而不是硬编码猜测。
2. 实现 512→192 裁剪、192→256 resize、Focus 编码及 Focus 特征与原相机 4×4 网格的几何映射。可以使用无新增大参数的 ROIAlign/坐标插值/格子覆盖权重映射，避免将四视角展平成一幅伪 8×8 图像。
3. 实现简洁的门控残差 Adapter：`M = G + gate * coverage_mask * (F_aligned - G)`（仅在对应相机的有效覆盖单元写回）。门初始化为接近 identity；无 Focus 时直接返回原 G；非覆盖位置保持不变；重复 Focus 可以累积，但不能凭空扩充 token 数。
4. 用 R2R train 场景的冻结高分辨率全图视觉特征进行简短校准。教师目标须对齐输出网格；只监督 Focus 实际可观测区域，并对其他区域做 Glance 保持约束，不能要求不可见区域凭空恢复高清细节。校准后冻结 Adapter。
5. 保留无 Focus、固定中心 Focus、指定 camera+position 的调用接口供后续 Actor 和基线共用。
交付物及验收
- 可复用的 Focus crop / geometry / Adapter / calibration 模块、配置、最小单测、校准 checkpoint 和报告。
- N=0 路径与 Stage 1 同输入、同输出；输出当前视觉 token 恒定为 (4,16,2560)；仅被选中的相机及几何覆盖区发生写回。
- Focus 边界裁剪合法，四相机 ID 与位置不串线；训练校准时不会更新 Qwen、视觉编码器、SimpleNav。
- 报告：校准区域特征误差、非覆盖区保真误差、不同 Focus 数量的编码时间，以及无 Focus 导航一致性抽查。
不通过则不能进入 Task 2.2：如果空间写回破坏 no-focus 或多视角语义，增加 Actor/RL 只会掩盖问题。