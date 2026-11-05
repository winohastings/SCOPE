# Learned-2 选择器设计

状态：DESIGN_ONLY，未实现训练、未生成权重、未授权训练。仅用于Stage1，无STOP动作。
配置源：`STAGE1_GROUP_CONFIGS/selector_training.yaml`。

## 1. 接口与信息边界

当前Task1的 `select(glance, selected, available, allow_stop, focus_index, rng)`
只传PIL图像和候选元数据；没有instruction、编码后Glance/Focus特征、权重加载或训练状态。
因此现有 `smoke_select` 不是本设计的实现，绝不作为G4。

后续增加独立 `SelectorSession`，输入限定为：

- 四视图冻结视觉encoder的postmerge特征 `G: [4,16,2560]`，保留camera与4×4空间顺序。
- 当前导航指令token及attention mask；使用同一冻结Qwen文本embedding，最多256token。
  超长固定保留首128与尾128，忽略padding作masked mean，得到 `[2560]`。
  不使用目标坐标、GT path、环境距离、导航成功指标或评测episode难度标签。
- 第一次选择之前没有Focus特征；第二次仅可见第一次已获取的 `F1: [64,2560]`。
  禁止先编码全部16个ROI，禁止把第二个或未选Focus的特征用于第二次决策。
- 候选ID、camera ID、归一化ROI坐标及已选mask。这些是相机几何，不是GT。

训练teacher中的距离、碰撞和成功仅用于环境返回reward，不能混进上述输入字典。
推理必须支持完全不加载GT文件。导航主干仍可按既有契约使用自己的历史，selector本身无跨步RNN。

## 2. 轻量pointer结构

所有视觉输入先detach；冻结视觉/text embedding不接收梯度。仅selector与value头可训练，
初始化后统计trainable参数≤2,000,000，超过即拒绝本版本配置。

1. Glance的16个位置reshape为4×4；每个ROI对应其中一个2×2象限，mean pool得到
   16个候选的局部特征 `[16,2560]`，只利用低分辨率信息。全局四图mean得到 `[2560]`。
2. Glance、instruction和已获取Focus各用 `Linear(2560,128)+LayerNorm+GELU`。
   局部候选和全局Glance共用Glance projection；Focus对64token作mean后projection。
   无Focus时使用独立可训练null向量 `[128]`。
3. camera embedding为 `[4,16]`，ROI坐标 `[x0,y0,x1,y1]/512` 经MLP 4→32→16。
   candidate concat局部128＋camera16＋ROI16，经MLP160→128。
4. query concat全局Glance128、instruction128、已获取Focus128、选择次序embedding16，
   经MLP400→128。每个candidate与query拼成256维，再MLP256→128→1产生16个logit。
5. 对已选ID置 `-inf`；训练softmax采样，评估masked argmax，同分选最小ID。
   第一次选择a1后，先采集/编码该Focus，更新query，第二次选择a2。a2不能等于a1。

只允许恰好两次采集；策略动作空间16，然后15，不存在STOP类、预算预测头或早退分支。
crop2特征仅用于最终导航输入，不用于已经完成的选择。输出记录两个ID、logprob、熵、
selector权重哈希及feature shape；导航Prompt不添加新的ROI文本，避免改变冻结基线条件。
ROI/camera嵌入属于selector；导航仍用Task1块与TVI/M-RoPE，这一冻结主干可能不理解局部crop
全图坐标，是需在train/dev上验证的风险，不能用heldout结果反向修改Prompt。

## 3. 编码复用与主干隔离

同一导航步的目标调度：

1. 既有Qwen视觉encoder编码四Glance一次，保留原始grid及合并后特征。
2. selector第一次forward，获取并编码Focus1一次。
3. selector第二次forward，获取并编码Focus2一次。
4. 将这六块**已算过的**postmerge特征，通过既有scatter插入冻结导航上下文，
   使用对应grid计算M-RoPE，再运行一次LLM和一次DiT-B action chunk。

G2/G3也采用相同“Glance→Focus1→Focus2”编码分组，不能单次batch全部六图而只让G4承担
不可避免的顺序encoder调度成本；selector本身额外成本如实单列。四Glance可批量编码，
两个Focus各一次；历史仍遵守共同Task1合同，不能对某组独占缓存。

不把当前块重用伪装成历史BF16缓存；需要新增typed current feature bank，记录camera、ROI、
native grid、token数、dtype及encoder hash，复用既有融合入口。记录“实际encoder调用次数”，
确保没有先为selector算、再为导航重算。不得替换或重写原始采集/动作执行路径。

## 4. 优化目标与reward

训练单位为一个导航step及其两个顺序Focus选择，联合策略为
`pi(a1,a2|o) = pi(a1|G,I) * pi(a2|G,I,F1,a1)`。
得到两个Focus后，冻结SimpleNav预测8个waypoint，按现有动作协议执行，再产生一次reward。
不要把同一reward当作两个独立环境transition，也不为选择ROI瞬移机器人。

训练teacher geodesic距离d只在R2R train场景计算。定义：

`r_t = 0.2 * clip((d_t-d_next)/3, -1, 1) - 0.01 - 0.02*c_t + terminal_bonus`。

`c_t=1`当本导航chunk发生至少一次已记录碰撞，否则0，不按8个waypoint重复放大奖惩。
终止bonus为 `2*SR + 0.5*SPL - 0.5*(1-SR)`，只在episode结束计一次。
感知预算恒定，所以视觉成本惩罚系数为0；不让训练暗中学会少于两裁剪。
训练样本必须保留teacher来源和有效性；缺distance/collision时不能静默填0。
正常导航STOP与达到step cap均视为终止，无成功的cap终止得到失败bonus；基础设施中断则丢弃
未完成rollout并重试，不把中断伪装成环境终止reward。

选择PPO作为待实现优化方案：actor联合logprob为两项之和，ratio为
`exp(logpi_new_pair-logpi_old_pair)`，clip epsilon=0.2；不能分别clip后当联合动作PPO。
advantage来自只看选择前Glance与instruction的value头，gamma=0.99、GAE lambda=0.95。
loss为负clipped surrogate＋0.5×value MSE−0.01×联合熵；联合熵用第一步熵加
在采样a1条件下第二步熵的估计，不给已mask候选熵。value不读取特权GT。
参考算法：Schulman等，*Proximal Policy Optimization Algorithms*，arXiv:1707.06347。

AdamW lr=3e-4、weight_decay=0.01、grad norm≤1，优势在每个rollout batch内标准化。
4个train环境×128导航transition=512/update；4轮优化，每minibatch64。
390个update、每seed199680导航transition；训练seed101/202/303。
每20个update（10240 transition）及最终update做dev检查，无heldout early stopping。
dev按固定64集、seed909运行；每seed选择dev平均nDTW最高者，再按SR、最早step破同分。
在固定预算内选择训练checkpoint，不从三个seed里挑冠军。不启动本文件中的方案。

## 5. 复现与防泄漏

训练scene/dev划分与64集开发列表已物化，详见主协议。train reset时按固定seed RNG
打乱训练manifest，按循环无放回队列取episode；环境终止后继续，不丢弃长episode。
每seed独立初始化，保存python/numpy/torch/selector generator状态、环境episode cursor、
optimizer、value头、actor、transition计数、主干hash与所有配置，恢复时保持rollout原子性。
主干eval模式、requires_grad=false、冻结参数哈希在训练前后核验；不保存主干新权重。

缓存若用于训练，key至少含scene、精确pose/rotation、camera、ROI、resolution、encoder hash、
预处理版本；不能跨pose复用，不缓存全16个高分辨率ROI供actor提前查阅。
训练调参、结构选择仅在train/dev；官方R2R/RxR val_unseen只作冻结后的最终评估。
三个训练seed分别对应三个eval block，发布全部结果与dev选择轨迹，失败训练不能静默删掉。

## 6. 实现验收

必须新增测试：同一步两个不同ID；eval确定性；没有STOP；缺权重/脚本标记拒绝；
第二次logit确实依赖F1且第一步不能访问F1；指令改变可影响logit；未选ROI不编码；
PPO联合logprob/ratio对齐；主干梯度为None；current feature bank不重复编码；
恢复断点与连续运行一致；不带GT仍可推理；shape、grid及scatter和动作协议不变。
这些是**待实现验收项**，不与本次离线协议校验混淆。
