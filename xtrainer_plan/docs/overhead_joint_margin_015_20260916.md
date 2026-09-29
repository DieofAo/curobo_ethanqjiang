# V48–V49：双侧 0.15 rad 关节余量与最高成功率安装全量复核

承接[竖直倒装测试](overhead_downward_20260916.md)和[Y 正向外移测试](overhead_yedge_link3_20260916.md)。本轮选择当前竖直倒装组中完整冒烟成功率最高的安装：基座 `(-0.36, 0.15, 0.65)`、RPY `(180, 0, 90)`°，旧冒烟 7/9、旧 20×20 全量 265/400。Y=0.40 与 0.45 的冒烟分别为 6/9、4/9，未选中。这里的“最高”限于已测这组安装，不代表全局最优，也不代表 LINK3 避让合格。

## 本轮变化和保留项

- `robot.joint_limit_clip` 从 0.05 改为 **0.15 rad**。相对 URDF 原始下限加 0.15、上限减 0.15，各只收缩一次。J1–J6 原始均 `[-3.14, 3.14] rad`，有效范围均 `[-2.99, 2.99] rad`；不是按 ±π 计算，也不是把 0.15 再加到旧 0.05 上。
- Home IK、规划内部 IK、轨迹优化/约束及独立验证使用同一收缩后的模型。`criterion.min_limit_margin_deg=0` 保持不变，避免在模型已收缩后重复要求另一份 0.15 rad。
- 旧报告的“限位余量”是相对当时**有效限位**。本轮同时审计相对原始 URDF 的余量，要求所有保存采样至少 0.15 rad（约 8.594°，允许浮点容差 1e-6 rad）。
- 保留 TCP=0.19m、单臂无 mimic、原始任务坐标、grasp x=[-0.62,-0.20] / y=[-0.10,0.40] / z=0.03、place=(-0.16,-0.23,0.10)、Home=(-0.31,0,0.03)。
- 保留 0,-2,…,-30° 的抓放同角搜索、关闭独立 IK 分支预筛、J1–J5 单段跨度 ≤170°、直线 3mm/5°、全部 24 球自碰撞和 LINK3/LINK6 对六面世界墙约束。J6 不在既有 170° 单段跨度判据内；本轮另行报告它的实际跨度，没有悄悄更改判据。
- 未新增 grasp 区物体或 LINK3 禁入区。LINK3 相对 grasp 参考平面的球包络间隙仅作独立诊断，不把既有世界墙无碰撞说成对物体安全。

当前可复用的 `pick_place_default.yaml`、`pick_place_overhead_default.json`、`pick_place_overhead_downward_h065.json`、`pick_place_overhead_ymax_smoke.json` 已同步 clip=0.15；共享机器人 YAML、URDF、无关 task 默认配置和历史结果配置未改动。没有改动当前 RViz 播放内容。

## V48：3×3 冒烟

[配置](../results_overhead/20260916/configs/v48_h065_clip015_smoke.json) / [角度图](../results_overhead/20260916/v48_h065_clip015_smoke/grasp_angle_map.png) / [统计](../results_overhead/20260916/v48_h065_clip015_smoke/analysis_summary.json) / [LINK3 诊断](../results_overhead/20260916/v48_h065_clip015_smoke/link3_grasp_clearance.json)。

重新规划仍为 **7/9**，失败 index 4、5；2330 个采样、46.58s。成功采样距收缩后的限位最小 1.29174°，对应原始 URDF 余量约 9.88611°。Home 在新限位下重新 IK 成功。

[独立采样复核](../results_overhead/20260916/v48_h065_clip015_smoke/v48_h065_clip015_smoke/independent_verification.json)和[实际求解器限位审计](../results_overhead/20260916/v48_h065_clip015_smoke/v48_h065_clip015_smoke/joint_limit_clip_audit.json)均通过。新建 IK 的 4 个 rollout、MotionGen 的 24 个 rollout 的运动学与 bound_constraint 均为 ±2.99 rad（float32 误差约 9.54e-9 rad），不是只检查配置文本。所有 2330 个保存采样相对 URDF 的最小余量 0.17254509 rad，违规数 0。

LINK3 球包络仍有进入 grasp 参考平面下方的问题：放置阶段最小高度差约 **−94.56mm**，发生在 index6 放置阶段。收紧关节限位本身不能替代抓取区的显式避障约束。

## V49：20×20 全量

[配置](../results_overhead/20260916/configs/v49_h065_clip015_full.json) / [完整结果](../results_overhead/20260916/v49_h065_clip015_full/v49_h065_clip015_full/trajectory_meta.json) / [单独角度图](../results_overhead/20260916/v49_h065_clip015_full/grasp_angle_map.png) / [新旧对照图](../results_overhead/20260916/v49_h065_clip015_full/clip005_vs_clip015.png) / [统计及逐 case 失败详情](../results_overhead/20260916/v49_h065_clip015_full/analysis_summary.json)。

**400 点全部完成：268 成功（67%）、132 失败。** 99,967 个轨迹采样，轨迹时长 1999.32s；批处理实际计算耗时 752.40s，正常退出。旧成功轨迹没有混入本轮结果。

| 指标 | V45：clip=0.05 | V49：clip=0.15 |
|---|---:|---:|
| 全量规划成功 | 265/400（66.25%） | 268/400（67%） |
| 全量规划失败 | 135 | 132 |
| 距各自有效限位不足 1° 的成功 case | 39 | 16 |
| J1–J5 最大单段跨度 | 168.987° | 169.996° |
| J6 最大单段跨度 | 292.712° | **336.600°** |
| 放置阶段 LINK3 球包络与有限 grasp 平面相交采样 | 3473/44854 | **9369/45406** |

新旧共同成功 258 点、新成功 10 点、旧成功转失败 7 点、共同失败 125 点。新旧从头重规划，改变了 IK 分支、选中角度和后续 case 的连续起始状态；成功多 3 点不代表物理可达空间因收紧限位而增大。表中 LINK3 指标也来自不同轨迹/成功集合，不构成仅改变一个关节的受控安全性比较。

![左：旧 0.05 rad；右：新 0.15 rad](../results_overhead/20260916/v49_h065_clip015_full/clip005_vs_clip015.png)

两图均为原始 task/world 坐标、2cm 刻度，颜色只代表成功选中的 grasp 角度，红叉代表本次规划失败；没有另外区分“较差解”等类别。“近有效限位”是相对各自已经收缩的限位，不是原始 URDF 余量不足 0.15 rad。绘图刻度间距不代表规划点阵采样间距。

### 全采样与实际模型独立复核

[独立 FK/碰撞/限位复核](../results_overhead/20260916/v49_h065_clip015_full/v49_h065_clip015_full/independent_verification.json)与[IK、MotionGen、全部 rollout 限位来源审计](../results_overhead/20260916/v49_h065_clip015_full/v49_h065_clip015_full/joint_limit_clip_audit.json)均 `passed=true`。

- 99,967/99,967 采样均满足新的限位；原始 URDF 最小余量为 **0.14999999046 rad（8.59436638°）**，与 0.15 的差约 9.54e-9 rad，来自 float32 的 ±2.99 表示，远小于检查容差 1e-6 rad。违规数为 0。
- 六轴相对原始 URDF 的最小余量依次约 `[8.594,32.351,34.325,43.145,56.233,8.594]°`。J1/J6 有触及有效边界的采样，不是触及原始 URDF 边界。16 条成功 case 距有效边界不足 1°，其中 14 条存在有效余量为 0 的采样。
- 实际独立 IK 的 4 个 rollout、MotionGen 的 24 个 rollout 的模型限位和启用的 bound_constraint 都匹配 URDF 双侧各收 0.15 rad，没有二次收缩。
- 新模型 FK 与保存数据一致；自碰撞 0/99,967、LINK3/LINK6 对既有世界墙碰撞 0/99,967。保持单臂、24 球自碰撞、12 球世界墙约束和 TCP=0.19m。
- 记录的最大直线横向偏移 **2.80285mm**、姿态偏差 **1.99985°**，满足原有 3mm/5° 判据。独立 GPU 复核不重新计算直线判据，这两项来自规划分段验收记录。

### 仍需注意的运动与 LINK3 问题

1. **关节范围合规不等于没有大幅绕转。** J1–J5 最大单段跨度 169.99576°，仍低于 170°。但 index107 的 `i107_p_lift_in` 中 J6 单段峰峰值达到 **336.59971°**；J6 不在原有跨度判据中，所以被保留。整条轨迹最大相邻采样差为 J6 的 **5.31721°/20ms**，不可把 336.6°称为瞬时跳变。本轮没有擅自把 J6 加入 170° 判据。
2. [LINK3 完整诊断](../results_overhead/20260916/v49_h065_clip015_full/link3_grasp_clearance.json)：268 条成功 case 中，**89 条**在放置相关阶段有 LINK3 球包络与有限 grasp 参考平面相交；全程计为 **103 条**。放置阶段相交 9369/45406 采样，全程相交 11661/99967。最小球面高度相对 grasp z=0.03 平面为 **−94.56794mm**，发生在 index341 的 `i341_place`，sample83598、t=1671.96s。球中心原始坐标 `(-0.395008,0.072475,0.005432)`、半径0.07m。此处仅是球包络对零厚度参考平面的诊断，不等价于真实 CAD/物体碰撞。
3. 当前任务并未给 grasp 区域建物体或 LINK3 禁入约束，因此上项与“世界墙复核通过”不矛盾。另有夹爪包络相对信息性 `workspace.bounds` 的 592 个越界采样、最大85.68mm；该边界不同于世界墙硬障碍，原配置 `fail_on_violation=false`，本轮只记录、不改变判据。

因此，**本轮完成了两侧 0.15 rad 余量要求和 400 点全量检查，但不能把 268 个成功视为都已解决 LINK3 避让或 J6 大绕转。** 没有轨迹的失败 case 也不能记为安全。要进一步把这些筛掉，需要另行确定 J6 跨度判据，以及 grasp 区真实物体最高 Z 和所需间隙，再明确添加约束。

### 失败分类

132 个失败 case 的每个都试完 16 组抓放同角候选，共 2112 次失败尝试：IK_FAIL 1587 次、JOINT_DELTA_EXCEED 483 次、FINETUNE_TRAJOPT_FAIL 42 次。

按每个失败 case 的**最后一次尝试**统计为 IK_FAIL 101、JOINT_DELTA_EXCEED 26、FINETUNE_TRAJOPT_FAIL 5；这不是说每个 case 的所有角度都只因该原因失败。按“是否曾出现该类失败”统计分别是121、63、19条，会重叠。IK_FAIL 仅表示本次约束和求解设置未找到可行端点，不是数学上不存在 IK 解的证明。

## 复现

以下输出根目录需选未使用的新目录；批处理拒绝覆盖历史运行。

```bash
bash xtrainer_plan/run_overhead_batch.sh --mode plan \
  --configs xtrainer_plan/results_overhead/20260916/configs/v49_h065_clip015_full.json \
  --out-root xtrainer_plan/results_overhead/20260916/v49_h065_clip015_full_rerun
```

本轮配置由保存的 V45 配置通过 `derive_overhead_experiment.py --joint-limit-clip 0.15` 派生，记录来源，不依赖今天默认配置的其它值。安装、目标、角度搜索和碰撞模型均未改变。

## 实现与验证范围

新增 `derive_overhead_experiment.py --joint-limit-clip` 支持可追溯替换余量，拒绝负数或非有限值；新增 `verify_joint_limit_clip.py` 从原始 URDF 解析限位，构造独立 IK 与 MotionGen，核对所有 rollout 运动学和启用的 bound_constraint，再按原始限位检查轨迹采样。它不运行求解，也不控制实体机械臂。

44 项相关 CPU 回归通过：派生配置 6 项、限位审计 9 项、LINK3 几何 6 项、独立验证输入 5 项、悬挂坐标/碰撞配置 9 项、单臂任务变换 9 项。几何测试包含“XY 重叠但三维不相交”和“整球低于平面但并不穿过平面”，避免把参考平面指标误写为真实物体碰撞。

审计对象是保存的离散采样；没有验证采样间极值/扫掠、动态限值、实际物体高度或真实机械安装安全。计算中的 URDF 限位也不能替代对实体机械硬止挡的确认。
