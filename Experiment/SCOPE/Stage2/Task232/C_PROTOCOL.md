# Task2.3.2 新C：新增覆盖奖励探索

本文件依据用户最新授权；C指空间新增覆盖奖励，不是原任务文档条件式Adapter修复。
旧C_REVIEW仅保留为原文Adapter分支审查历史，不限制本次新C。

## 冻结设计

- 从原update_0014完整状态分叉，1033 transitions，train ordinal70，保留Actor/Critic/old、optimizer及全部RNG。
- C：原奖励加 beta=0.01 × sum_t 新增像素数/192²；C0：beta=0，等预算原配置对照。
- 仅此单因素差别；cost=0.01，rollout minimum64，Actor LR1e-4、Critic LR2e-4；不叠加A/B。
- 每个导航transition从空覆盖开始，四相机独立512²像素平面；当前Focus ROI减去同相机此前ROI并集。
- 首个ROI计入奖励；相同相机完全重复奖励0，不同相机同坐标属于不同观察区域。奖励不跨导航transition累积。
- 一次Focus最大bonus0.01，一次导航最多0.04。保留原Focus成本，禁止奖励误计为导航progress。
- 该量是空间探索代理，不是真实信息增益；不计算KL/JS，不利用GT构建覆盖奖励。
- 每次Focus分量单独记录，总和进入原导航级reward/GAE；沿用序列logprob求和和同一导航advantage，不新增微步Critic或算法。

## 预算与判读

- C/C0各512新增on-policy transitions；历史轨迹仅用于探针，不能更新；固定0/256/512探针。
- 两分支先各做首更新完整恢复复验；C0第一episode复用U前向一致性审计逻辑。
- 独立32集与原历史32集都评估。独立32已使用过，称开发诊断集而非未触碰测试集；本次不按其SR调整beta。
- 主探索证据：独立集真实N4覆盖相对C0至少+25%、后续ROI重复比例至少下降0.15；固定状态探针和同初态实际Focus作一致性佐证。
- 另报采样训练空间变化，但不将其等同确定性策略改善；报零Focus、Focus分布、相机、位置KL。
- SR/SPL、一步导航进展、原奖励与shaped奖励的held-out Critic EV分别列示；探索成功不要求立即SR提升，导航退化也不能隐瞒。
- 本次明确512筛查预算，不自动追加1024、组合A+B、训练Adapter或GA-PPO。即使探索成功，Task233资格仍需独立导航与稳定性审议。
- C0用于补齐过去A/B缺少等预算原配置对照的限制，但本轮主比较为C-C0，不回溯重选A/B。
