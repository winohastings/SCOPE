Stage 2 终态及不变量
最终运行链路：R2R/Habitat 状态 → 四视角 Glance → 指令条件 AGM Actor 决定 STOP 或 camera + continuous (x,y) → 每次从 512×512 原图裁剪 192×192 Focus，缩放为 256×256 编码 → Spatially Aligned Adapter 把 Focus 写回四视角 4×4 视觉网格 → 最多四次 Focus → 始终输出 4×16×2560（64）个“当前视觉 token”给现有 SimpleNav/Qwen（BATS 历史另计）→ 导航动作块 → 环境奖励及后继状态。

范围固定：冻结已有导航主干、已有指令编码及预训练视觉编码器；Adapter 短暂校准后冻结；RL 仅训练 AGM Actor 和独立 Critic。尽量复用 Stage 1 / Task 4 的运行与 checkpoint 契约，禁止不必要的架构重构。

配置默认：Focus patch 192²，max focus=4；Actor hidden=192、2 层空间 Transformer + cross attention；Critic hidden=128、1 层空间 Transformer；隐空间 Gaussian std 初始 0.8、限制 0.1–1.5，初始 STOP p≈0.25；PPO clip=.2、gamma=.99、GAE lambda=.95、Actor LR=1e-4、Critic LR=2e-4、2 epochs/update、gradient clip=1.0；Group K=8、anchor ratio=12.5%；初始主轨迹预算 4096 transitions，正式预算 10k–20k（依据曲线决定）。都是初始设置而非已验证最优值。

必须明确：感知 STOP 与导航任务的 STOP 是不同事件；四相机 ID 与各自几何坐标独立；归一化的坐标和裁剪像素舍入须一致；固定 Qwen 当前视觉 token 数不代表视觉编码 FLOPs 固定；训练 oracle 距离只用于 reward，不得进入 policy 观察；full-image teacher 只用于 Adapter 校准。