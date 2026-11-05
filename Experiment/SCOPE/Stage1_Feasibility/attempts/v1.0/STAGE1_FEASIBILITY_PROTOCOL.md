# SCOPE 缩减版第一阶段可行性协议

版本：feasibility-v1.0；冻结日期：2026-11-05。授权来源：
`docs/Tasks/stage1_task3_estimate_verify_task_benefit_estimate.md` 及本次用户执行指令。
仅授权以下四组开发集实验，不授权任务2草案的训练/完整基准。

| 组 | 输入 | 当前视觉token | 策略 |
|---|---|---:|---|
| G0 | 四张256兼容图 | 256 | 原始SimpleNav |
| G1 | 四张128 Glance | 64 | 无Focus |
| G2 | 四Glance、两张256 Focus | 192 | Fixed-Cyclic-2 |
| G3 | 四Glance、两张256 Focus | 192 | Random-2 |

全部使用相同512×512四相机Habitat渲染、RXR step-8000 checkpoint、BF16/SDPA、
动作归一化vln_train_train、8 waypoint执行、原导航STOP规则、BATS历史256→pool4。
历史token单列；不增加历史预算。G0为共享512渲染缩放到256的兼容路径，
不是原生256 Habitat渲染的像素级一致复现。相机顺序front/left/right/rear，
候选ID=4×camera+region，原512图直接裁剪四象限。

G2的八步序列：(0,4),(8,12),(1,5),(9,13),(2,6),(10,14),(3,7),(11,15)，
每episode从phase0开始。每对不同相机、不同ROI，八步遍历16 ROI。
G3使用SHA256(`session_seed:episode_id:navigation_step`)前8字节小端整数作为
NumPy default_rng种子，第一次从16候选均匀采样，第二次从不同相机的12候选均匀采样。
任务2Random旧实现只有无放回、未实际强制异相机；本协议明确补齐任务要求的相机约束，
不修改旧实现，不把新Random称为旧Task1逐位一致。

固定run seed42；session seed=42+原manifest全局ordinal；Habitat seed0。
使用任务2 `r2r_selector_train_dev.json` 的64个development_evaluation_episode_ids，
严格保持顺序；从原train注释物化EvalEpisode，并验证源文件SHA256、训练/开发ID和
scene隔离、R2R/RxR val_unseen scene隔离。开发集来自原train切分，不声称其与
预训练主干训练数据隔离。原R2R policy/Habitat cap200/200保留，不以冒烟上限截断。
四组完整episode冒烟使用同一首个dev episode，正式评估仍全部64集，冒烟不复用为正式结果。

执行顺序：回归→四组完整episode冒烟→成本试点→G0→G1→G2→G3→聚合审计。
单模型服务运行于GPU0，Habitat顺序运行于GPU1；保留GPU1既有10093服务。
该服务驻留造成设备显存基线污染，必须报告并持续监测，不声称双GPU完全独占。
若压力或活动干扰导致失败/不稳定，保留记录，禁止静默换数据或删除失败。

成本遵循任务2边界：当前/历史视觉编码、策略、LLM、DiT、模型总墙钟、
render observation、RPC、物理与闭环墙钟分别记录。CUDA events与墙钟分别报告；
正式模式避免新增逐模块同步，细粒度profile另行记录。完整episode首步和STOP预测均计入。
预热使用独立dev session，至少20次，前后10次中位数漂移≤5%，否则每次追加10次，
最多100次；不稳则标platform_unstable。单固定seed，不套用任务2三seed显著性声明。
短成本试点而非原任务2三进程×三历史层×200状态全基准；该缩减必须在成本报告注明。
FLOPs允许基于真实调用grid与实际模块形状的明确估计，不由token比例推算；
排除elementwise/unsupported算子并标incomplete，禁止冒充完整实测FLOPs。

分母每组固定64。保留全部失败/traceback及失败前成本；基础设施最多两次同seed重试，
各次独立保存。未解决故障禁止正式性能结论；方法失败SR/OSR/SPL/nDTW保守按0，
NE缺失不填0，另给有效轨迹均值。报告描述性效应，不做多seed显著性声称。

明确排除Learned/Actor/Critic/GRU/PPO/GAE、感知STOP、自适应预算、主干微调、
完整R2R/RxR val_unseen及599040-transition训练。不得自动启动第二阶段。
协议正文冻结后用EXECUTION_MANIFEST.json登记SHA256；任何改变须新版本及原因。
