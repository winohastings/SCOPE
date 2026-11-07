# Gate1 空间热启动报告

## 结论

Gate0=PASS，Gate1=INCONCLUSIVE，Gate2/3=NOT_RUN。按照预注册早停规则，未进行learner-induced追加标注、真实Gate1导航验收、Group或PPO；没有用更多预算掩盖保留集退化。

## 执行设置

- 数据仅来自R2R train的`fulltrain256_v1`教师候选；heldout轨迹未进入梯度。
- 两个初始化候选分别为Task22初始Actor和Task23 update14 Actor；camera/position输出层重置，STOP头冻结，监督loss不含STOP。
- 内部选择按轨迹隔离；最终热启动仅使用81个可信`teacher_train` anchor，12 epochs、batch=8、AdamW学习率`1e-4`，共132次更新。两个候选各做3 epochs、各27次更新。
- Qwen、SimpleNav、vision、Adapter和Critic均未训练；未产生on-policy Group数据，所有checkpoint均标记为非PPO数据。

## 结果

| 指标 | 训练前 | 热启动后 | 判定 |
|---|---:|---:|---|
| teacher-heldout可信加权NLL | 3.6031 | 4.0181 | 退化 |
| teacher-heldout可信候选top1 | 32.43% | 10.81% | 退化 |
| teacher-heldout全部加权NLL | 3.7153 | 4.3589 | 退化 |
| teacher-train可信加权NLL | 3.7932 | 2.5930 | 下降，但不能替代heldout |
| optimizer updates | 0 | 186 | 仅Actor空间监督 |

训练集损失下降而heldout指标同时变差，因此不能宣称空间策略产生泛化收益。按协议，Gate1不进入真实导航验收，也不解锁Gate2。

## 独立核验

`INDEPENDENT_VERIFICATION.json`通过全部检查：旧输入和父证据未变、教师标签重新计算一致、heldout没有梯度样本、内部选择轨迹未进入pilot拟合、最终拟合恰为81个可信train anchor、STOP参数未变、无Critic/Group、训练日志有限、CPU重放最大指标误差约`2.37e-7`。新增的`stratified_supervision.csv`按`j`、scene和state source分层；它显示heldout的j分布仍为既有不均衡（可信样本j=0/1/2/3为6/2/4/25），因此退化不能简单归因于均衡泛化测试。

运行时源码在执行后增加了“拒绝覆盖已有输出目录”保护；执行版本由`source_snapshot/spatial_warmup.py`保存并在独立核验中明确记录，不能重写本次结果。

## 停止与后续

`PROGRESS_GATE.json`保持Gate1=INCONCLUSIVE、Gate2/3=NOT_RUN、`downstream_locked=true`。Gate2所需的16–32个同状态on-policy Group与Gate3的256/512主轨迹PPO均未执行。正式K=8 GA-PPO仍归Task2.4；本次监督候选不进入Group、Critic或GAE。
