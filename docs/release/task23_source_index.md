## 实验日志

- `../Experiment/SCOPE/Stage2/Task23/REPORT.md`：Task2.3 独立Critic与导航级PPO/GAE，完成4096真实train transitions、56轮更新、5轮固定32集val_seen验证与完整恢复审计；最佳1033步SR56.25%，最终确定性零Focus退化，非已证明有效的论文结论。入口见同目录README。

- `../Experiment/SCOPE/Stage2/Task22/REPORT.md`：Task2.2 Spatial Query Actor、Task2.1增量视觉融合、30个真实Habitat导航transition、latent概率及梯度重放、67项相关测试；未进行正式RL训练。数据契约见同目录`TRACE_SCHEMA.md`。

- `../Experiment/SCOPE/Stage2/Task21/REPORT.md`：Task2.1 固定64当前视觉token的空间融合、R2R train短校准、冻结Adapter checkpoint、48项相关测试及无Focus导航一致性抽查；不含Actor/RL或完整闭环评估。

- `../Experiment/SCOPE/Stage2/IMPLEMENTATION_REPORT.md`：Task4 状态条件SCOPE、当前特征复用、真实反馈Actor–Critic/PPO、54次受控Focus诊断及小规模训练验证；不含完整训练或val_unseen评估。

- `Logs/Task3/STAGE2_REVIEW_REPORT.md`：Task4 执行前审阅，记录第一阶段回退事实、接口与坐标/反馈风险、Stage2 验证门禁；后续实测结果以 Stage2 交付报告为准。

- `../Experiment/SCOPE/Stage1_Feasibility/IMPLEMENTATION_REPORT.md`：任务3缩减版第一阶段执行实现、64集开发清单验证、完整episode与成本重放；执行状态及最终结论以同目录实际artifacts为准。

- `Logs/Task2/execution_report.md`：第一阶段协议设计与只读实现审计；候选协议包位于 `../Experiment/SCOPE/Stage1/`，尚未审查冻结，未启动训练或完整评估。

- `Logs/Task1/R2R/implementation_report.md`：按官方共享 RXR checkpoint 协议独立执行 R2R Task1，六模式两集冒烟、真实固定观测等价性、协议审计及成本 CSV。

- `Logs/Task1/implementation_report.md`：SCOPE 多分辨率 Glance–Focus 实现、固定输入等价性、六模式两任务24次冒烟与实测成本；原始产物、CSV及文件清单同目录。

- `Logs/2026-11-05-simplenav-r2r-rxr-baseline-smoke.md`：现有 RXR checkpoint 在标准固定相机 R2R-CE / RxR-CE 上的单集闭环验证、路径及 nDTW 修复、结果和复现命令。
- `Logs/2026-10-21-rxrce-50-episode.md`：AdaPerc-Random 和标准 SimpleNav-RXRCE 各 50 集的设计、运行入口、原始日志和结果分析。

## 工程索引

- `../Experiment/Benchmark/AdaPerc_RXR_CE_VLN/README.md`：AdaPerc VLN-CE 任务注册与 Benchmark 说明。
- `../Experiment/Baselines/SimpleNav_RXR_AdaPerc_Random/README.md`：SimpleNav AdaPerc-Random baseline 的随机感知实现说明。
- `../data/logs/README.md`：两组 50 集原始 artifacts 归档索引。
