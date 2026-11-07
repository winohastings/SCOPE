# Task233 Gate0 实测报告

Gate0：PASS。Gate1/2/3：NOT_RUN。

实际完成72 anchors / 1006候选分支；包括独立校准8 anchors。

held-out可信标签比例：50.00%；要求至少25%。

| held-out GT教师配对progress差（米） | 均值 | 判门98.333%CI |
| --- | --- | --- |
| teacher_minus_stop | 0.073072 | [0.018277, 0.195998] |
| teacher_minus_fixed | 0.044713 | [0.011831, 0.107319] |
| teacher_minus_random | 0.074388 | [0.021761, 0.181447] |

未通过项：

本次只做真实1-chunk反事实评价，不能报告整集SR/SPL提升。训练0步；未创建optimizer、Critic buffer或Group数据。
若GT代理未过门，即使oracle-best可获益也不授权warm-up；只能说明当前教师协议不足，不证明冻结导航无可学价值。
原始候选、动作、当前位姿专家路径、真实ROI/latent、恢复身份及成本见candidate_results.jsonl。
成本见COMPLETE.json和runtime/scope.jsonl；所有预测包含4当前Glance图的编码及实际Focus调用，历史编码另按telemetry核计。
完整learned/fixed/random/no-Focus导航验收和held-out Critic留在锁定的下游门禁，不用历史指标代填。
