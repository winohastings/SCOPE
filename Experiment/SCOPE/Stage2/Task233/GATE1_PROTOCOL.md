# Gate1 v1：有限预算空间热启动

本版本在任何新梯度更新前冻结。依据Task233 v2和已通过的gate0_v3；fulltrain256_v1仅是扩采证据，不把其数量当训练成功。旧日志/模型不修改，既有heldout永不用于梯度。缺失exec-agent文档不覆盖用户明确的执行授权。

## 阶段与停止

1. 先进行最小离线监督可学习性检查：仅176个teacher_train内可信状态参加梯度（预计81个，以hash核验后的记录为准），80个teacher_heldout只评价。按原始8候选GT排序选最高分实际latent，不逆解ROI。置信权重=min(1,GT margin/0.01)，不把GT输入Actor。每个.pt来自真实递增融合状态，不伪造count/history；包含当前pose重标注的learner-induced状态。
2. 两个启动候选：Task22 final_runtime/actor_initial.pt初始化，以及Task23 update14 trunk。两者camera末层和position末层显式置零，log_std重置为log(0.8)。不强制相机均匀训练。对teacher_train按轨迹固定seed233301分出约20%内部选择集；各训练3 epochs，只依据内部加权空间NLL选择。重新从所选初始化在所有可信teacher_train上训练12 epochs，batch8，AdamW lr=1e-4，weight_decay=0，grad_clip=1，seed233301。最后epoch为固定checkpoint，不按外部heldout挑最好epoch。不训练Critic，不复用旧optimizer动量。
3. STOP头冻结且监督loss完全不含STOP；真实验收绕过感知STOP固定N=2（主）和N=4（重复诊断），保留导航STOP。每次优化检查有限loss/梯度、STOP参数与旧输入hash。若最终可信heldout加权NLL未下降，或8候选top1命中率未提高，记INCONCLUSIVE并立即停止追加预算，不进行后续环境扩采/Group/PPO。这个检查只是监督可学习性早停，绝不代替真实任务验收。
4. 只有离线检查通过，进行一次小规模learner-induced再标注：全train中排除所有旧校准/监督/保留轨迹，新选8条轨迹，每条最多2个导航状态，在训练后Actor真实递增的N=2前缀逐步重求当前pose专家和8候选标签。候选与微步STOP执行真实chunk，阈值沿用Gate0；固定2 epochs追加加权BC，缺乏可信新增标签则停止。仍不把监督候选当on-policy Group。
5. 真实验收另外预留16条未参与任何监督或调参的train轨迹、每条最多2个状态，对learned/Task22 init/update14/fixed/3seed random/no-Focus配对；固定N=2与N=4，no-Focus为零成本参考。序列每一步均真实融合；复原核验环境完整identity/context与初始状态，随机种子先在状态内均值再按episode bootstrap。主N=2对fixed/random/no-Focus三项进展均值≥0.02m且98.333%CI下界>0；N4重复率<100%且union coverage>3.52%；真实有效历史下相机条件概率须有可观测变化（max绝对差>1e-5）。不足32有效保留状态或任一统计项未过为INCONCLUSIVE，复原/冻结/数据边界错误为FAIL。只有监督与真实空间/导航指标共同通过才能Gate1 PASS。

既有j/split不均衡须分j/scene/source报告，不能事后改划分。新轨迹选择在查看新回报前固定并落盘；所有状态真实专家重标注，验证split不生成训练标签。首次训练不加载Qwen/SimpleNav/Adapter，仅读取冻结ActorState；真实评估加载这些冻结模块，64当前视觉token与192² Focus不变。禁止用离线NLL下降声称SR/SPL改善。

## 下游锁定与成本

Gate1未PASS时Gate2/3均NOT_RUN并锁定。Gate1通过后另行预注册16–32 anchors K=4 Group机制协议，通过才K=8；主分支预选、整状态复原、旁支不入Critic/GAE。Gate2通过后才预注册导航PPO成本与256/512检查、SR/SPL非劣验收及held-out Critic。正式K=8比较留给Task2.4。每阶段分别记训练更新、教师调用、真实分支/原生步和耗时，不追加4096掩盖失败。
