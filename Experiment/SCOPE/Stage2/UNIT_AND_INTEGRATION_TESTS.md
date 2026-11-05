# 单元与集成验证

## 自动测试

第一轮 Stage2 18项通过；补充0/1/2 Focus及开关隔离后，最终 Stage2 20项通过。
全部 `test_scope_*.py` 共34项通过。原导航路径/nDTW五项回归在正确的 Habitat Python 中
通过。原始日志分别为 `artifacts/unit_tests.log`、`artifacts/stage2_tests_final.log`、
`artifacts/all_scope_tests.log`、`artifacts/external_navigation_regression_habitat.log`。

| 验证类别 | 自动覆盖 |
|---|---|
| 形状/缓存 | Glance[4,16,2560]、Focus[64,2560]、native grid、重复ROI拒绝、未选Focus拒绝、只编码一次 |
| 空间 | 原图ROI象限、候选池化顺序、相机body方向、token中心射线单位范数 |
| 可选适配器 | FP32和BF16零初始化恒等，不默认改冻结导航特征 |
| Actor | 五开关全部32组合前向反向、语言和已选Focus依赖、合法mask、STOP0/1/2预算 |
| Critic | 选择前观测、拒绝Focus后输入、参数与Actor完全不共享 |
| GRU | 真反馈依赖、禁用反馈/状态后的依赖消失、State GRU梯度、episode reset |
| PPO/GAE | 初始联合ratio=1、真正终止、截断bootstrap、episode边界、固定target、参数更新 |
| 恢复 | 同一固定rollout的连续更新/保存恢复后更新逐参数逐位一致，Python/NumPy/Torch RNG一致 |
| 泄漏 | PolicyObservation不接受GT字段、train/dev scene和episode互斥、heldout purpose拒绝 |
| 反馈/奖励 | body逆旋转、预测执行误差、实际位姿缺失拒绝、成本惩罚、缺teacher拒绝 |

## 真实集成门禁

最终测量入口为 `artifacts/release/`，服务输出为 `artifacts/release_runtime/`；以
`release/AUDIT.json`、`release/COMPLETE.json` 和各原始results/diagnostics为准。

- 正常导航 current encoder_images=4+k，reused_current=4+k；未选Focus从不编码。
- 实际模型冻结、实际八waypoint闭环、实际body反馈、训练场景teacher与GRU更新。
- A/B/C真实rollout，Actor/Critic梯度和PPO ratio/参数变化；B/C State GRU实际非零梯度。
- 固定开发前两集训练前/后/保存恢复检查，不借heldout结果调参。
- 固定像素/pose/扩散seed的train/dev诊断，原scatter缓存/原生重编码动作配对。
- 整个第一阶段和checkpoint保护hash复核，防止新代码覆盖历史。

## 测试失败与修正记录

1. 首轮真实诊断直接比较旧全batch与新顺序编码，触发feature一致性断言。固定图像的
   processor pixels最大差0，但旧BF16全batch特征relative-L2约0.025–0.045。
   同样的顺序调度原生重编码后，全部特征逐位相同，max-abs=0。
   这不是提高容差绕过错误，而是区分“缓存一致性”和“旧batch数值差异”两个问题。
   失败保存在`artifacts/final/diagnostics_train.log`及`debug_runtime/native_feature_mismatch.json`。
   随后的原生scatter对照又发现历史batch混入，首次完整诊断保留于
   `release/diagnostic_attempts/`；只替换当前特征、保持历史特征相同后重测54次，
   当前特征与最终动作最大差都为0。没有据此改变Actor/PPO或奖励参数。
2. 全部根测试发现一个既有导入缺失：`scripts.plan_perception_margin_eval` 不存在，
   `artifacts/regression_all.log`记录39项、1错误；本轮不修复无关工程缺项，也不声称全套通过。
3. 外部回归先错误地向unittest传绝对模块路径，再用模型Python时缺少`attr`；日志均保留。
   修正为Habitat Python和正确模块PYTHONPATH后5项通过，并未安装或改动外部依赖。
4. 本地端口探测可能产生 WebSocket invalid handshake 日志；有效client会话均另行验证。
   不将这些探测请求误报为真实episode失败，真实run失败数单独列出。
5. 离线审计首次在保护hash检查完成前读取COMPLETE文件，正确拒绝未完成运行；随后
   GPU CSV中的`MiB`单位触发解析错误，已只修正离线数值解析并重跑，原日志保留。

## 复现

从项目根执行：

```bash
/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope_stage2.py'

/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python -m unittest discover \
  -s tests \
  -p 'test_scope_*.py'

PYTHONPATH=/home/data/czh/Third_Party/baselines/SimpleNav/tests:/home/data/czh/Third_Party/baselines/SimpleNav /home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python -m unittest \
  test_vlnce_r2r_paths \
  test_vlnce_ndtw
```

合成测试不作为导航性能证据，真实短rollout不作为完整训练收敛证据。
