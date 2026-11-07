Task2.3.1 — Failure attribution & matched counterfactual diagnostics
定位：以已有模型/轨迹诊断为主，禁止重跑 4096 训练。
1. 核实原始验证集列表、trace、checkpoint SHA、环境版本；用原运行目录的真实权重，不通过 GitHub 占位恢复。只写新目录 Task231/，不改 Task23 训练记录。
2. 五个 checkpoint 的同一批固定 pre-Focus 状态（至少 64，尽量 128，train diagnostic + val_seen 分开）仅运行 Actor head，记录第一步 p_stop 的 P10/P50/P90、`p_stop>=0.45/0.5` 比例、预期/确定性 Focus count、Camera entropy/top-1 占比、位置 sigma/mean、各 head 与初始分布的 KL。此项不需要 Habitat 长 rollout，优先利用已存在的 detached ActorState。绝不可用含 reward/GT 的字段作 Actor 输入。
3. 从历史 5 轮固定 32 episodes 原始结果生成配对成功转移表（成功→失败、失败→成功）、per-episode SPL/nDTW/final distance/终止与截断；报告 episode-level bootstrap CI 和 exact McNemar（适用时）。单次 32 集不足以宣称泛化。
4. 在固定 8–12 集小型场景均衡诊断集，用 1033、2054、4096 三个 checkpoint 做 3×2 个对照：native deterministic STOP vs 只在诊断中强制 N=4，其余 camera/position 都使用各 checkpoint 的网络。优先对比最终 forced4 是否恢复 SR，以及 2054 forced4 能否恢复 1033 的表现。若 8–12 集趋势不明确，扩到相同的固定 32 集；不把 8–12 集作正式结果。
5. 若 2054 在 N=4 条件下仍退化：对相同一组有限状态额外执行 matched-budget learned/固定/随机 ROI 的 1-chunk 后续进展对照；恢复环境采用独立 reset + 确定性导航前缀重放，并比较姿态/指令/历史/BATS，恢复不一致则标记无效而非合并结果。记录 geodesic 进展、动作差异、无 Focus/Focus 相比、按场景分组的分布；不把一次 chunk 进展误称为最终 SR。
6. 检查 Focus 特征融合：未覆盖区保持、不同场景 teacher-MSE、ROI area/coverage、相机语义、局部 token 改变与动作改善的相关性。只在获得负面证据时考虑 Task2.1 Adapter 重校准，不提前展开模块重写。
7. 数值审计：对保留 rollout 重算 reward、GAE、return、old/new logprob、KL、裁剪比例；输出每个 Focus count 的 sample 数、advantage均值/方差、各 head 梯度范数、按序列长度的loss贡献。当前测试通过不等于这些量充分可靠。
交付：`diagnosis.json`、`paired_episodes.csv`、`checkpoint_policy_probe.csv`、`counterfactual.csv`、可重跑脚本、简短判定报告；每个假设明确支持/反证/仍不确定，并做优先级更新。门禁：至少回答“后期 forced4 是否挽救性能”“1033→2054 的退化来自哪里”“旧 reward 是否产生负面成本动机”三问，才开始正式优化试验。