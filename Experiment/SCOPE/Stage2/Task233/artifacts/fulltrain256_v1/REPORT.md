# 全量 train 扩采至256 anchors：执行报告

本轮按用户最新要求扩大 anchor 搜索范围，完成真实采集与独立核验；不执行空间训练、Group或PPO。Gate0沿用父run结论，Gate1/2/3均未运行，数量达标不等于训练门禁通过。

## 搜索范围与停止规则

| 范围 | Episodes | Scenes | 唯一轨迹 |
| --- | ---: | ---: | ---: |
| 旧Task23候选池 | 9,837 | 54 | 3,276 |
| 旧Gate0实际选择（包含校准） | 36 | 36 | 36 |
| 本轮允许搜索的全量R2R-CE train | 10,819 | 61 | 3,603 |
| 排除旧36条轨迹后冻结的备用池 | 3,567 | 61 | 3,567 |

备用池每条轨迹只保留一个指令实例；3,567不是全train的episode总数。来源为 `/home/data/czh/Third_Party/tasks/R2R-CE/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz`，SHA256为 `f9340166cfaef49885c859f260546a2f81bb4f7d970fd66847fb213178038be1`。

旧范围并非整个train，本轮已纳入此前未开放的982 episodes/7 scenes。使用seed233256预先冻结全train备用顺序，按场景轮询，不按收益挑选状态；达到目标立即停止，并未穷举10,819 episodes。新增实际采集96条轨迹，合并旧非校准32条，共128条。原36条轨迹及其指令变体均不参与新增采样。val_seen/val_unseen未加载；新增7个原train-development场景今后不能再声称训练未见盲测。

## 实际结果

- 256个物理状态唯一的非校准anchors：旧64＋新增192；旧8个校准另计，含校准共264。
- 全61个train场景均有样本；teacher_train/teacher_heldout分别176/80，按轨迹隔离。
- expert_start/learner_induced分别128/128。后者来自冻结导航真实一步后的状态，在当前pose重新查询专家，并不保证每个状态都发生偏航。
- 可信标签118/256（46.09%）：训练侧81/176，保留侧37/80。目标是256个完整anchor，不是256个高置信样本；未用负收益、并列或低margin为由替换样本。
- 每anchor包含8个合法教师候选及匹配对照，真实执行与感知历史复用Gate0协议；微步STOP与Glance-only按实际历史区分。
- 新增2,640次模型预测/真实分支，6,048次Focus编码，含恢复重放32,448个环境原生步；新增运行4,445.64秒（74.09分钟）。合并非校准证据共3,520分支。这些不是on-policy训练transitions。
- 优化器更新0、Critic训练样本0；Actor、Qwen/SimpleNav与Adapter保持冻结，64当前视觉token及192² Focus不变。旧run文件逐项SHA256核对未变。

## 独立验证

`EXPANSION_VERIFICATION.json`全部检查通过：256唯一状态、61场景、轨迹隔离、256个ActorState文件及执行状态一致、候选/ROI/coverage与评分复核、冻结参数及旧产物完整性。评分重算最大误差0；这属于保存证据的独立核验，不是再次执行全部Habitat分支。

`ANCHOR_MANIFEST_256.json`的SHA256为 `fb44745b079e06a6b5d2a64d9975660dca72f82bb9b1eb327183fb458e42852d`。清单引用旧/新真实state与hash，不复制旧状态充当新增。

扩采相关回归测试36项通过，见Task233目录的 `expansion_regression_tests.log`。

## 必须保留的限制与后续门禁

新增episode的partition和微步j都使用ordinal%4，造成合并数据的j与split不均衡：

| Partition | j=0 | j=1 | j=2 | j=3 |
| --- | ---: | ---: | ---: | ---: |
| teacher_train | 56 | 56 | 56 | 8 |
| teacher_heldout | 8 | 8 | 8 | 56 |

虽然轨迹隔离正确，但不能据此宣称各微步均衡的泛化验证。保留既定划分，不事后将已查看的heldout标签转为训练标签；后续warm-up须按j分层报告，并另行预注册独立分层验证。此次覆盖全train不等于所有轨迹或导航时刻的代表性已得到保证。

`PROGRESS_GATE.json`中的Gate0=PASS_PARENT仅引用旧Gate0已通过证据，不表示扩大后的256样本重新通过了全部统计门禁；本轮未对扩大样本宣告新的导航收益显著性。Gate1/2/3保持NOT_RUN，未证明学习后真实收益或SR/SPL改善，不能用数量替代验收。监督候选不得混入on-policy Group、Critic或GAE。

纠正上一轮解释：原协议允许先用Gate0高置信样本做小规模warm-up，再扩至约128–256；128并非启动硬门槛。历史GATE1_PREFLIGHT产物保留，不再将其作为协议要求。本轮按用户要求完成扩采后停止，没有追加4096训练。

## 交付入口

范围证据：`TRAIN_SCOPE_AUDIT.json`；结果：`EXPANSION_AUDIT.json`、`COMPLETE.json`；复核：`EXPANSION_VERIFICATION.json`；样本入口：`ANCHOR_MANIFEST_256.json`、`candidate_quality_256.csv`。本轮源码、文档、日志与产物逐文件列表见 `DELIVERY_FILES.txt`（清单自身也列入）。
