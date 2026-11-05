# 缩减版第一阶段执行实现报告

## 角色、审查与边界

角色：用户明确指定的严谨执行者。根AGENTS/index/docs索引已读；`agents/`为空，
`agents/exec-agent.md`缺失，按用户明确任务执行，不伪造角色流程。
审查了任务要求的六份Stage1文档、Task1主/R2R报告、Task2报告、组配置与数据manifest，
以及现有scope_server、原视觉编码/scatter、legacy服务、benchmark worker、Habitat runtime。
另按用户补充要求核对成功基线日志`docs/Logs/2026-11-05-simplenav-r2r-rxr-baseline-smoke.md`。
只新增本目录与对应测试/索引，不改任务1/2实现、旧日志、预训练权重或外部SimpleNav代码。

## 新建组件

| 文件 | 相对原版本的新增能力 |
|---|---|
| policies.py | 独立八步Fixed循环；SHA256派生Random，强制两ROI不同相机 |
| compile_configs.py | 物化原任务2的64个dev ID、校验隔离/源/权重hash、生成四组配置与冻结manifest |
| validate_data.py | 对64集逐项与既有VLNCEInputAdapter输出完全比较；验证真实场景文件和train稠密GT |
| server.py | 复用ScopePolicy/Qwen/DiT/scatter；仅适配策略与非侵入CUDA events；真实encoder调用来源/grid；独立拆分profile |
| run_group.py | 复用原_run_episode/导航/终止/GT/nDTW；完整episode、固定全局ordinal、两个环境分片、失败和STOP成本保留 |
| pilot.py | 同一真实dev观测、独立session，至少20次预热与≤5%漂移检查，100次timed/组 |
| render_pilot.py | 真实dev固定状态四相机512观测接口20次预热、200次timed，渲染/读回单列 |
| verify_sessions.py | 两连接交错调用，同seed/输入逐步动作与ROI相等，验证共享模型session隔离 |
| execute.py | 单模型双GPU资源编排；四组完整smoke→独立profile→短试点→G0–G3正式评估；GPU50ms驱动采样/环境记录 |
| aggregate.py | 固定分母与逐步合同审计、三张结果CSV、独立profile/pilot汇总、SVG图表及结果报告 |
| finalize.py | 权重/历史产物完整性复核、最终源与依赖哈希、全部交付物及逐文件清单 |
| tests/test_scope_feasibility.py（项目根下） | 循环覆盖/reset、随机可复现/相机约束、预算/排除项、清单派生字段/历史兼容、FLOPs估计非token比例测试 |

## 复用与必要差异

- 继续用Task1原512图裁剪、128/256原生token、2560维、Qwen3.5接口、M-RoPE、scatter和冻结DiT。
- 不实现Learned/训练/Actor/Critic/GRU/PPO/GAE/感知STOP或自适应预算。
- Task2声明不同相机的候选约束，但Task1 Random旧代码只无放回；新适配器补齐相机约束，不修改旧函数。
- 真实smoke发现在线历史不是已编码BF16跨步缓存，而是BATS所选256历史与当前图一次混合编码后pool4。
  四组原样保留；不能把整批时间标为当前、历史填0。本轮用独立重放测分项，原始混合batch另报。
- Task2三seed/完整重放矩阵按用户缩减为一个seed的初步开发可行性；不做确认性显著性结论。
- 用户授权停止旧10093/PID1825348，已退出；本轮只启动10095一个模型实例。
- 用户允许2线程/双卡约20GB：正式阶段两独立环境worker共享单模型，保留全局ordinal与session seed；
  跨worker完成顺序可交错，不把RPC排队计时称独占延迟。详见三个AMENDMENT文档。

## 验证与失败保留

新增与既有SCOPE回归14项通过；原任务2协议4项、既有VLNCE数据路径/nDTW5项另验。
64份物化EvalEpisode与原adapter逐字段完全一致，真实场景文件及train稠密GT全部存在。
完整smoke每组一整集，不设置2步截断；正式阶段仍每组全部64集，不将smoke复用为正式数据。
分项profile另做四组完整episode并与原smoke逐预测动作allclose(1e-5)校验，结果见COST_VALIDATION.json。

前期问题均保留：v1.0缺gt_path_length；v1.1日志包装误引用不存在的write_step；
首次外部回归漏PYTHONPATH；独立成本拆分实现前的部分pilot被主动停止，避免继续无效重复。
前两项发生在首次模型调用前，修复后同一ID重跑，未更换episode、seed或导航协议。
原始日志和各次实际资源在attempts/，正式失败与工程调试失败分开报告。

## 执行与追溯

`EXECUTION_MANIFEST.json`固定协议、checkpoint、组配置和64集内容哈希。
v1.0旧冻结产物保留；v1.1为正式评估前修复，区别见协议修订节。
`RUN_ENVIRONMENT.json`记录软件、GPU和外部git提交；项目根不是git仓库，以最终源hash补充。
资源/并发/profile补充单独记录，不静默改原任务1/2协议。
自动编排只运行本轮可行性，不启动第二阶段，不提交git、不改检查点。
