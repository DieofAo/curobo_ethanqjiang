# 上置单臂扩展区域失败点补规划（2026-09-16，V23–V25）

## 结论与新图

原对照图右侧的 14 个失败抓取点，本轮全部找到完整抓放轨迹：V23 新增 9 点、V24 新增 4 点、V25 新增 1 点。与原 V19 的 63 个成功点合并，扩展区域为 **77/77 个离散采样点有成功规划记录**。

这不是把失败点直接涂成成功，也没有用左图邻近点代替右图原坐标；14 个点均按右图原坐标重新规划。机械臂安装位姿、TCP、grasp/place/Home 的物理目标、角度范围和轨迹验收阈值未改变。V24/V25 仅关闭独立 IK 预筛，允许完整规划器尝试其自己的关节分支。

仍需保留质量说明：**4 条补回轨迹的 J6 贴近规划限位；另有 J6 大角度转动**。这些属于当前原配置接受、但不够理想的解，不能概括为“所有关节都有充足余量、没有大转动”。

![扩展区域合并抓取角度图，77/77 个离散点有成功记录](../results_overhead/20260916/right_grasp_angle_map_merged.png)

图中沿用正常抓取角度着色，不额外区分原始成功、补规划成功或贴限位类别；背景刻度为 2 cm，实际抓取点间距仍为 **10 cm**。测试范围为旧 LINK_0 坐标下 x=[-0.61,-0.01] m、y=[-0.36,0.64] m、z=0.03 m，共 7×11 点。

## 源数据与不变条件

基线为 [V19 扩展区域结果](../results_overhead/20260915/v19_zx_expanded_trajectory/zx_expanded_trajectory_10cm/trajectory_meta.json)，即 [20260915 原对照图](../results_overhead/20260915/final_workspace_comparison.png) 的右图。源失败 index 是原网格零基编号：

```text
22, 33, 48, 52, 53, 54, 60, 61, 64, 65, 73, 74, 75, 76
```

各重试结果的 `retry_config.json` / `trajectory_meta.json.config.retry_scope` 保存了源结果路径及 SHA-256。基线 metadata 哈希为 `1e004b29081fd89a9de5e73907008aacb22354fee4090daaf608c29d7afbbf28`，`plan_skipped.json` 哈希为 `8f32553a2c330f19fb4176adad94acf5218e38e05d03d8497345bdfea5ed0b29`。

本轮保持如下配置：

- 单臂 `xtrainer.yml`，无 mimic / 第二臂；安装位置 (-0.51,0.10,0.50) m，RPY=(90,0,90)°，新 LINK_0 +Z 沿旧 +X。
- TCP_joint 沿 LINK_6 局部 +Z 偏移 0.19 m；旧坐标系及安装补偿变换保留。
- place=(-0.16,-0.23,0.10) m；Home 位置=(-0.31,0.11,0.03) m、姿态 RPY=(-180,0,0)°，均为旧坐标。
- grasp 绕工具 X 轴 [-30,0]°、步长 2°，place 与 grasp 同角度；第二阶段未开启。没有扩大角度范围。
- Home IK 使用源结果的已解关节角 `[56.3451,5.4028,138.6529,35.9441,56.3452,89.9997]` 度作为种子，**并非把 Home 物理目标改为另一处**。本轮未使用额外自定义 `--home-seed-deg`。
- 每段 J1–J5 的插值轨迹极差上限 170°；抓取/放置直线段横向偏移上限 3 mm、姿态偏差上限 5°；原最小限位余量门槛为 0°。
- 原自碰撞和六面硬墙保留。自碰撞使用全部 24 个碰撞球（沿用相邻 link 排除表）；世界硬墙按原配置只约束 LINK_6/LINK_3 的 12 个球，未扩展成所有 link 的墙碰撞。

配置逐项对比：除了输出目录、重试选择审计信息，V23 相对 V19 仅设置上述 Home IK 种子；V24/V25 另外将 `criterion.prescreen_by_ik` 从 true 改为 false。物理目标、场景、碰撞参数与实际轨迹后验阈值未放宽。

## V23–V25 过程

| 版本 | 选取的原失败 index | 策略 | 新增成功 | 累计覆盖 |
|---|---|---|---:|---:|
| V19 基线 | 全部 77 点 | 原批量顺序和预筛 | 63 | 63/77 |
| V23 | 原 14 个失败点 | 新建一批，从原 Home 初始化；保留 IK 预筛 | 9/14 | 72/77 |
| V24 | 22,33,52,53,64 | 新建一批，关闭独立 IK 预筛，其余验收不变 | 4/5 | 76/77 |
| V25 | 33 | 仅该点从原 Home 独立重试，继续关闭预筛 | 1/1 | 77/77 |

这里“从 Home 初始化”指每一批开始时初始化一次，**不是每一个 case 都自动回 Home**。同批中下一点接续上一条成功轨迹的末端关节状态，继续复用上次成功角度；失败 case 不被拼入最终成功轨迹。改变重试批次会改变进入目标点时的状态和求解历史。

- [V23 metadata](../results_overhead/20260916/v23_failed_retry_from_home/trajectory_meta.json) / [日志](../results_overhead/20260916/v23_failed_retry_from_home/plan.log)：成功 index 为 48,54,60,61,65,73,74,75,76；剩余 22,33,52,53,64。
- [V24 metadata](../results_overhead/20260916/v24_failed_retry_full_planner/trajectory_meta.json) / [日志](../results_overhead/20260916/v24_failed_retry_full_planner/plan.log)：成功 22,52,53,64，均采用 -6°；33 仍失败。日志中仍能看到 `JOINT_DELTA_EXCEED`、`FINETUNE_TRAJOPT_FAIL`、`IK_FAIL`，说明关闭预筛不等于无条件接收。
- [V25 metadata](../results_overhead/20260916/v25_last_point_from_home/trajectory_meta.json) / [日志](../results_overhead/20260916/v25_last_point_from_home/plan.log)：33 在首个 0° 候选成功，六段均第 1 次规划通过，保存 421 个采样点。最大受检关节段内极差 154.682°，最小限位余量 38.314°。

## 为什么右图失败不能解释成位置几何不可达

### 48：确实紧邻左图已成功位置

右图 index 48 为 (-0.21,0.04,0.03) m。左图最近成功点 index 388 为 (-0.20,0.0526316,0.03) m，距离仅 **1.611 cm**，左图采用 -16°。

V19 原日志中，48 的候选主要被独立 IK 预筛得到的关节分支挡住，例如 -16° 进入抓取抬升点时 J3 变化约 202°，-8° 约 208°。这描述的是当次选中分支相对进入状态的变化，不是该空间位置完全没有 IK。

V23 将失败点单独组成新批后，48 用 -8° 完成抓放，J1–J5 最大段内极差 162.938°，直线偏移 0.060 mm。这是对**右图原坐标**的直接成功证据；不需要从左图的邻近成功点推断可达性。

### 33：批处理中失败，独立重试首个 0° 即成功

index 33 为 (-0.31,-0.36,0.03) m。V24 中它接在 index 22 成功之后，0° 候选的放置抬升段出现 J4 极差 241.3°，超过 170°；其他角度还出现轨迹优化和 IK 失败。V25 把它单独放在新批首位，从原 Home 开始，0° 即得到符合原验收条件的完整六段轨迹。

这说明该目标不是几何上必然无解。更精确地说，结果受**进入状态、关节分支、候选角顺序和求解器搜索历史**影响，不能仅凭一次批处理的“均无可行解”判为整个位置无 IK。V23 中 33 在前一点失败后也未接续成功轨迹，却曾失败，因此也不把全部差异简单归因于“是否从 Home 开始”这一单独因素。

独立 IK 预筛逐个路点选择少量 IK 解再检查相邻关节差，可能选中不利分支，进而把该角度整体过滤掉；完整 MotionGen 有自己的多种子 IK/轨迹优化搜索，可能找到另一条分支。V24/V25 的 `--skip-ik-prescreen` 只是避免这种提前过滤，**MotionGen 自身的 IK、碰撞约束以及完成后的段内关节极差、直线度检查仍在**。

## 14 个补回点的角度与质量

下表坐标是旧 LINK_0 的 x/y，z 均为 0.03 m；index 为零基。角度列同时适用于 grasp/place。最大变化只统计原判据中的 **J1–J5 单段极差**，不是全部关节、不是瞬时速度，也不是整个抓放循环的总运动量。限位余量则覆盖该成功 case 轨迹的全部六个关节，包含继承的段起点。

| index | x (m) | y (m) | 成功版本 | 角度 (°) | J1–J5 最大段内极差 (°) | 最小限位余量 (°) | 最大直线偏移 (mm) |
|---:|---:|---:|---|---:|---:|---:|---:|
| 22 | -0.41 | -0.36 | V24 | -6 | 139.142 | 0.000000 | 0.287 |
| 33 | -0.31 | -0.36 | V25 | 0 | 154.682 | 38.314219 | 0.277 |
| 48 | -0.21 | 0.04 | V23 | -8 | 162.938 | 3.350221 | 0.060 |
| 52 | -0.21 | 0.44 | V24 | -6 | 147.933 | 0.000000 | 0.303 |
| 53 | -0.21 | 0.54 | V24 | -6 | 164.032 | 0.000000 | 0.310 |
| 54 | -0.21 | 0.64 | V23 | -18 | 167.271 | 3.350221 | 0.011 |
| 60 | -0.11 | 0.14 | V23 | -16 | 125.806 | 43.201493 | 0.084 |
| 61 | -0.11 | 0.24 | V23 | -16 | 142.881 | 31.490970 | 0.011 |
| 64 | -0.11 | 0.54 | V24 | -6 | 79.279 | 0.000014 | 0.285 |
| 65 | -0.11 | 0.64 | V23 | -16 | 79.907 | 46.024123 | 0.004 |
| 73 | -0.01 | 0.34 | V23 | -16 | 135.747 | 44.263628 | 0.012 |
| 74 | -0.01 | 0.44 | V23 | -16 | 113.571 | 46.023495 | 0.010 |
| 75 | -0.01 | 0.54 | V23 | -16 | 90.253 | 46.023495 | 0.002 |
| 76 | -0.01 | 0.64 | V23 | -20 | 79.736 | 40.490359 | 0.417 |

最大直线姿态偏差：V23 为 0.685°、V24 为 1.040°、V25 为 0.143°，均低于原 5° 上限。

### 必须单独保留的 J6 风险提示

- 当前 `criterion.joints=[1,2,3,4,5]`，**170° 门槛不检查 J6**；J6 仍接受位置限位检查，并没有被设成无限旋转关节。
- V24 的 22/52/53 最小余量为 0°，64 为约 0.000014°，均是 J6 几乎或恰好触及当前规划限位。独立复核使用的限位约为 ±3.09000015 rad（±177.044°），这是 `position_limit_clip=0.05 rad` 后的模型规划限位，不应等同于已验证的真实机械硬止挡。
- V24 共 251 个采样点的 J6 余量不足 1°，但独立位置限位检查没有发现超限。原配置 `min_limit_margin_deg=0` 不额外要求正的安全余量，因此这些轨迹被当前规则接受。保留它们用于用户允许的“不那么合适的解”覆盖图，不代表建议直接在实机执行。
- V23 的 index 54 第一段 J6 极差约 **263.6942°**；index 48 第一段 J6 极差约 **179.9977°**。这是连续轨迹中的大幅转动，不能因 J1–J5 通过 170° 就宣称“没有大关节转动”；它本身也不等于采样帧间发生瞬时不连续跳变。
- NPZ 的相邻 20 ms 保存帧另行检查：V23/V24/V25 全关节最大相邻差分别为 3.444041°、3.185136°、3.106022°。因此上述约 264° 是一段运动的总跨度，不是两个相邻保存帧瞬跳 264°；该帧差检查不代替速度/加速度等动力学验收。
- V23 的最小余量 3.350221°来自 index 48 结束时 J6 约 -173.693746°；index 54 继承这一段起始状态，所以也报告相同最小余量，不能据此说 index 54 的抓取目标姿态本身必然贴限位。

## 独立复核与软范围说明

三个补规划结果均已通过独立复核，合计 **5173 个保存采样点**：

| 批次 | 保存采样数 | 独立位置限位超限 | 自碰撞采样 | 配置硬墙碰撞采样 | 独立 FK 与保存 FK |
|---|---:|---:|---:|---:|---|
| [V23 验证](../results_overhead/20260916/v23_failed_retry_from_home/independent_verification.json) | 2918 | 0 | 0 | 0 | 一致 |
| [V24 验证](../results_overhead/20260916/v24_failed_retry_full_planner/independent_verification.json) | 1834 | 0 | 0 | 0 | 一致 |
| [V25 验证](../results_overhead/20260916/v25_last_point_from_home/independent_verification.json) | 421 | 0 | 0 | 0 | 一致 |

复核另确认 M×C=单位变换、新 LINK_0 +Z 沿旧 +X、TCP=0.19 m、URDF 与记录哈希一致，全部 24 个自碰撞球保留。它检查保存的离散轨迹采样，不验证采样间扫掠碰撞、速度/加速度/jerk 限制或真实安装结构安全；直线度由规划器原有后验另行检查。

**软 `workspace.bounds` 不等于实际参与碰撞规划的 `wall.bounds_override`。** 原配置 `fail_on_violation=false` 保留，所以有成功轨迹超出软报告范围，但仍无配置硬墙碰撞：

| 批次 | TCP 超出带容差软范围的采样数 | TCP 最大报告越界 (mm) | 夹爪球包络超软 bounds 的采样数 | 夹爪包络最大越界 (mm) |
|---|---:|---:|---:|---:|
| V23 | 107/2918 | 16.644 | 1018/2918 | 85.830 |
| V24 | 17/1834 | 26.580 | 116/1834 | 88.097 |
| V25 | 0/421 | 0 | 0/421 | 0 |

TCP 报告的 `check_margin=0.005 m` 是**允许向外越界 5 mm 的容差**：先把 bounds 向外扩 5 mm，再统计剩余越界。因此 16.644/26.580 mm 不是相对原始 bounds 的全部越界；不能把该 margin 理解成向内缩进 5 mm 的安全带。夹爪球包络统计则直接相对原 `bounds`，没有该 TCP 容差。

本例机器人基座坐标下软 z 上界为 0.51 m，硬墙内侧上界为 0.61 m；两者相差 10 cm，而且新基座 +Z 映射到旧 +X。不能把这里的 z 越界直接理解为旧世界坐标的“高度越界”，也不能将软越界误报成硬墙碰撞。

## 复现与查看

从仓库根目录运行。输出目录必须不存在，下面使用 `reproduce_*` 避免覆盖本轮证据。重新数值求解可能因种子/求解历史得到不同角度或分支；原始结果与哈希用于精确核对本轮记录。

```bash
cd /home/ethanqjiang/workspace/curobo
unset PYTHONPATH
source /home/ethanqjiang/miniconda3/etc/profile.d/conda.sh
conda activate curobo

retry_source=xtrainer_plan/results_overhead/20260915/v19_zx_expanded_trajectory/zx_expanded_trajectory_10cm

# V23：原 14 个失败点，保留预筛，从原 Home 初始化一批。
python xtrainer_plan/scripts/retry_failed_pick_place_single.py \
  "$retry_source" xtrainer_plan/results_overhead/20260916/reproduce_v23

# V24：本轮 V23 剩余 5 点，仅取消独立 IK 预筛。
python xtrainer_plan/scripts/retry_failed_pick_place_single.py \
  "$retry_source" xtrainer_plan/results_overhead/20260916/reproduce_v24 \
  --indices 22 33 52 53 64 --skip-ik-prescreen --no-incremental-save

# V25：最后一点单独从原 Home 开始。
python xtrainer_plan/scripts/retry_failed_pick_place_single.py \
  "$retry_source" xtrainer_plan/results_overhead/20260916/reproduce_v25 \
  --indices 33 --skip-ik-prescreen --no-incremental-save
```

任一命令追加 `--validate-only` 可先检查源失败点选择及配置策略，不创建结果目录、不导入 GPU 规划模块。该模式尚不重建网格核对原始坐标，坐标一致性检查在实际规划入口选点时执行。`--indices` 是**原零基 index**，不是重试列表中的序号；拒绝重复、成功点或不存在的 index，保留给定顺序。`--no-incremental-save` 仅改变落盘频率，不改变轨迹约束。

重画本轮合并图（只合并成功点记录，不拼接跨批轨迹）：

```bash
python xtrainer_plan/scripts/plot_overhead_results.py \
  "$retry_source" \
  --replan-result-dir xtrainer_plan/results_overhead/20260916/v23_failed_retry_from_home \
  --replan-result-dir xtrainer_plan/results_overhead/20260916/v24_failed_retry_full_planner \
  --replan-result-dir xtrainer_plan/results_overhead/20260916/v25_last_point_from_home \
  --out xtrainer_plan/results_overhead/20260916/right_grasp_angle_map_merged_reproduced.png
```

RViz 可分别播放每批，模型保持本轮上置安装；例如最后一点：

```bash
bash xtrainer_plan/run_overhead_rviz.sh \
  --traj xtrainer_plan/results_overhead/20260916/v25_last_point_from_home \
  --speed 1 --loop
```

辅助脚本新增的失败子集/种子/预筛选项经过 12 项 CPU 单测；安装配置 9 项 CPU 回归通过。原 `pick_place_default.yaml` 和独立上置默认配置未改。

## 结论边界

77/77 表示原 63 点与三批补规划的成功记录并集，**不等于已生成一条按原 77 点顺序全部连续执行的轨迹**。如需这种连续执行，应按指定顺序重新规划，必要时设计并验证返回 Home 或切换分支的过渡段，不能直接拼接不同批次 NPZ。

同样，它不证明 10 cm 格子内部任意位置都可达，不证明所有抓取角可行，不是全局最大工作空间证明，也不表示这 77 个点的所有轨迹都远离限位或都适合实机。当前证据足以说明：原右图的 14 个失败标记并非这些位置必然几何无解，改变重试上下文并避免独立预筛的过早过滤后，全部找到了当前配置接受的完整抓放轨迹。
