# 成本测量补充（正式评估前）

完整冒烟证实，Task1既有在线路径将历史256图与当前图在一次visual.forward中重编码，
再将历史特征pool4；不能把整批时间标为当前视觉时间，更不能把历史开销填0。
这一共享历史行为四组保持不变，不引入新的跨步缓存优化。

新增独立profile阶段：相同首个dev完整episode，每步预测动作生成后，在独立重放中
用该次真实packed pixel/grid分别执行当前/历史visual.forward并用CUDA events测量。
该额外计算不反馈给动作或状态，不改变正式评估批处理，profile阶段的RPC/闭环耗时
不并入正式timed样本。分别报告分组重放成本与原始混合编码实测，不声称前者可加和。
正式日志混合批的vision_current_ms/vision_history_ms留空，vision_combined_ms保留实测，
FLOPs估计则可按实际逐图grid和来源分别核算。profile本身的资源计入总实验成本。

完整smoke验证、profile验证和短试点通过后才允许64集正式评估。
此补充不改变冻结组、ID/顺序、seed、导航/停止/历史/模型协议。
