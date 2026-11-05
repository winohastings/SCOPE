# 外部依赖与重建边界

不包含 Habitat、R2R、RxR、SimpleNav 官方完整代码、模型权重、场景、环境或数据集。只保留 SCOPE 所需修改的补丁、依赖源哈希、原许可证及选择后的协议/研究结果。

SimpleNav 官方源：`https://github.com/OpenBMB/SimpleNav`。具体提交见 `dependencies.lock.json`。`simplenav_stage1.patch` 从该提交生成，包含 Stage1 所需的 R2R/RxR 数据/动作兼容、日志插桩、benchmark 适配、legacy 服务及本机基线配置；不包含无关 DiT uncertainty 实验和 AdaPerc-Random 基线。

历史 RXR legacy 副本的原始目录为 `/tmp/SimpleNav-RXR-official`，该目录没有 `.git`，因此不能诚实声明其 upstream commit。`dependencies.lock.json` 对 238 个实际 runtime 文件逐个锁定原始哈希与实测版本哈希；使用者须自行取得对应历史源。不以当前 SimpleNav HEAD 代替该历史模型接口。重建脚本遇到未知版本会拒绝继续，而不会猜测或自动下载不同版本。

本机验证从官方提交 `git archive` 建立独立 SimpleNav 副本，从 legacy 源白名单复制运行时文件，再应用补丁；不会修改现有第三方仓库。重建结果位于 `.dependencies/SimpleNav` 与 `Experiment/SCOPE/legacy_runtime`，均被 Git 忽略。

运行入口仍包含历史机器的绝对路径：重建成功不等于已经自动把入口切换到独立副本。迁移运行前须单独审核路径和依赖，不得改写冻结 manifest 来伪装同一次实验。

`SimpleNav.LICENSE` 保留上游 MIT 许可。补丁里的上游代码仍按其上游许可处理；不据此替作者替整个 SCOPE 研究工程选择新许可证。R2R/RxR/MP3D 许可与原始指令数据的再分发授权需由仓库持有者确认，发布默认建议私有仓库。
