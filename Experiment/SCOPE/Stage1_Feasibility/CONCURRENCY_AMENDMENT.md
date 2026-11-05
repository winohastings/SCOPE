# 用户授权的两worker执行补充

2026-11-05，用户允许真实评测使用2线程，双卡各约20GB显存可接受。
为避免不必要的重复Qwen模型，正式评估采用两个独立Habitat环境worker进程，
同用GPU1，连接GPU0的一个共享模型服务；不是启动两个模型副本或同时跑不同组。
服务按单个请求同步执行，session/history独立，预测RNG用fork_rng与固定session seed隔离。

每组仍为64集，原全局ordinal/order清单不变；worker0取偶数ordinal、worker1取奇数ordinal，
各自按升序执行，跨worker实际完成/到达顺序由调度决定，不能称严格串行的物理执行次序。
各组使用完全相同分片，不以分片局部ordinal重算种子；不丢弃、不重复episode。
G0→G1→G2→G3仍依次运行，每组结束后才启动下一组。

完整smoke、独立profile与固定观测试点继续单worker用于可解释的成本测量。
正式RPC/闭环耗时包含共享模型排队与并发开销，明确区别于独占单请求延迟；
不把多worker吞吐倒数称为模型延迟，不因用户允许20GB而刻意填满显存。
