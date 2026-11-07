# 全量 R2R train 的256-anchor扩采协议

用户本轮授权扩大全部train范围至256 anchors。本协议不改变旧Gate0结论、阈值、样本或日志，不训练任何参数，不执行Group/PPO。

旧Task23 manifest限制到9,837 episodes、54 scenes、3,276 trajectories；旧Gate0从中只选36 trajectories/36 scenes，生成72 anchors，其中8为校准、64为评价。全量R2R-CE train有10,819 episodes、61 scenes、3,603 trajectories，新增范围为原来留作train-development的982 episodes/7 scenes。用户授权后这些场景可以生成train教师标签，但不可再声称其是训练未见的盲测集。val_seen/val_unseen均不加载。

目标：256个非校准、物理导航状态唯一、有完整8候选及匹配对照的anchors；保留旧64并新增192，原8个校准状态另计，共264个含校准状态。不是256个高置信标签，不因收益小、负或并列替换状态。旧calibration与旧teacher-heldout轨迹全部从新候选池排除；其他相同轨迹的指令变体亦排除。

全train剩余唯一轨迹seed233256洗牌、场景顺序洗牌后轮询，完整备用顺序采集前落盘。每episode只取专家起点与冻结导航no-Focus真实一步后的live状态；重新查询当前pose专家。每4条新轨迹的第4条整轨分入teacher-heldout，其余teacher-train；划分不使用结果。若前缀已经终止或后状态与同轨起点pose/image完全相同，记录排除并使用下一预定轨迹，不复制计数、不改j冒充新anchor。

8候选、fixed/learned/3seed random、微步STOP及j>0额外Glance-only、真实递增前缀、64视觉token、192²裁剪、当前位姿专家、8-waypoint环境反馈、复原hash和冻结审计全部复用Gate0。完整分支全成功后才计1 anchor；restore/模型/坐标错误属于硬失败，停止而非换样本。新增目标192个anchor达到即停止，不按导航收益择优、不为可信标签数量无限搜索。最多3小时；若物理状态不足则按冻结全train备用继续，耗尽或基础设施错误如实INCOMPLETE。

合并清单引用旧证据与新证据，不复制旧.pt制造新样本；每个anchor带真实state文件SHA256。输出物理去重、全61场景覆盖、轨迹隔离、train来源、完整候选数、成本、可信标签实际比例。Gate0原PASS保留；256数量达标不等于Gate1成功。上一轮将128–256扩充建议误写为warm-up启动硬门槛，不作为新的训练批准条件；历史GATE1_PREFLIGHT保留为旧记录，本轮交付中明确纠正。
