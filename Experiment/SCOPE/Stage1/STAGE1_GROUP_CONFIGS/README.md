# 第一阶段配置草案

这些YAML是**协议声明，不是可直接运行的SimpleNav evaluator配置**。不得直接传给
`benchmark/vlnce/*/eval.py`。`common.yaml`中的授权全部为false，不代表现有eval.py
会自动识别这些门禁；后续必须实现有审批校验的配置编译/执行入口。

合成顺序：common → benchmark task → group → 对应seed block → 审批过的本机资源映射。
只允许表列差异，其他配置差异必须报错。编译结果必须保留manifest原始全局ordinal；
不得按worker重排后重新编号。每组输出独立run name与完整配置哈希。

G2只是提出的确定性循环方案：当前Task1 `fixed_2`仍取front.R1/R4，**不能通过改组名
把旧结果当作G2**。G4尚未实现训练与特征接口，脚本smoke选择器不得填入权重列表。
G2/G3/G4当前视觉预算均为四Glance加恰好两个Focus=192 token。
G0/G1的当前token分别为256/64。所有组沿用Task1历史契约，不声称视觉总计算只等于当前token。

评测manifest通过现有只读输入适配器生成：R2R 1839集、RxR guide English 3669集。
manifest保留原始annotation顺序及global ordinal，并对每条完整annotation做SHA256。
它们是候选锁定列表；审批时需重新校验来源文件和内容哈希并签入最终freeze manifest。
训练/开发scene划分已物化为 `manifests/r2r_selector_train_dev.json`：54/7个训练来源scene，
9837条训练episode、982条开发池episode；固定64条开发评估记录，与两任务val_unseen scene无交集。
授权、freeze manifest及G4权重仍为空，是明确的执行阻塞项，不是可执行占位值。

安全校验示例，不加载模型、不启动模拟器、不训练：

```bash
/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python Experiment/SCOPE/Stage1/validate_protocol.py \
  --output docs/Logs/Task2/protocol_validation.json
```
