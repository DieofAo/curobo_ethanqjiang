# V29–V30：悬挂基座移到 grasp 的 Y 正向边缘

保留 V28 的 CAD 装配体显示配置：原左基座不再带额外 Z 偏转或 Y 平移。此次只按用户要求调整抓取区 Y 下限和上方单臂的 Y 安装位置，使用 3×3 完整抓放冒烟测试。

## 配置

以下位置均在保留的旧 LINK_0 / task_world 中，单位 m。

| 项目 | 本次值 |
|---|---|
| grasp 范围 | x=[-0.62,-0.20]，y=[-0.10,0.40]，z=0.03 |
| 悬挂基座位置 | (-0.51,0.40,0.50)，Y 位于 grasp 正向最远端 |
| 悬挂朝向 | RPY=(90,0,90)°，新 LINK_0 +Z 沿旧 +X |
| place | (-0.16,-0.23,0.10)，不变 |
| Home | (-0.31,0.11,0.03)，姿态不变 |
| TCP | LINK_6 局部 +Z=0.19m，保留减短3cm配置 |
| 搜索 | 工具X轴 -30…0°、步长2°，grasp/place 同角度 |

基座 X 沿用 -0.51m，没有额外移到网格 X 中心 -0.41m。配置由原单臂 YAML 重新构造 M/C 和墙体变换，避免对已经上置的配置重复变换。先复建旧安装并与既有上置默认配置逐项比较 robot/pick_place/planner/workspace，确认一致；新配置 M×C=I，原六面墙的物理位置不变。

[可复用冒烟配置](../config/pick_place_overhead_ymax_smoke.json)已单独保存，输出使用新目录和时间戳；原上置默认配置未覆盖。装配体仍只用于显示，不计入规划碰撞。

## V29：第一轮完整抓放

[结果](../results_overhead/20260916/v29_ymax_smoke/v29_grasp_ym010_ymax_mount/trajectory_meta.json)为 **8/9 成功**，保存2689个轨迹点，运动时长53.76s。

| x | y | 结果 / grasp角度 |
|---:|---:|---|
| -0.62 | -0.10 | 失败 |
| -0.62 | 0.15 | 0° |
| -0.62 | 0.40 | -14° |
| -0.41 | -0.10 | 0° |
| -0.41 | 0.15 | 0° |
| -0.41 | 0.40 | -18° |
| -0.20 | -0.10 | -4° |
| -0.20 | 0.15 | -4° |
| -0.20 | 0.40 | -4° |

成功轨迹最小关节限位余量32.074°；受检 J1–J5 单段最大摆幅158.370°（上限170°）；抓放直线段最大横向偏移约2.38mm（上限3mm）、姿态漂移4.256°（上限5°）。因此未贴限位，但部分直线段的姿态误差接近现有阈值。J6 单段最大摆幅181.212°，原170°门槛不覆盖J6，不能称所有关节都小幅运动。

[独立复核](../results_overhead/20260916/v29_ymax_smoke/v29_grasp_ym010_ymax_mount/independent_verification.json)通过全部2689个保存采样：限位有效、FK一致、自碰撞0、配置硬墙碰撞0。沿用原墙体仅约束LINK_6/LINK_3、自碰撞覆盖全部碰撞球的范围。TCP软范围越界0；夹爪球包络相对软bounds仍有82点越界、最大约77.4mm，原规则仅报告，不等于硬墙碰撞。

## V30：对唯一失败角点补测

失败点是原 index=0，位置 `(-0.62,-0.10,0.03)`。V29中16个角度都在抓取抬升 `i0_g_lift_in` 的独立IK预筛未找到解，尚未到放置阶段。

[V30重试日志](../results_overhead/20260916/v30_ymax_smoke_retry/plan.log)：仅对该点关闭独立IK预筛，从同一Home重新完整规划；16个角度仍全部为 `MotionGenStatus.IK_FAIL @ i0_g_lift_in`，没有补回，未生成成功轨迹。最终仍是8/9，不是9/9，也不是单纯被170°关节摆幅门槛拦下。两套求解都未找到该抬升姿态的解，但这不构成全局数学无解证明。

## 图与播放

[抓取角度图](../results_overhead/20260916/v29_ymax_smoke/grasp_angle_map.png)采用正常角度配色、2cm刻度；实际仅测试3×3点，不能推断格子内部连续覆盖。

[RViz效果](../results_overhead/20260916/v29_ymax_smoke/rviz_smoke_initial.png)显示新安装和V28原装配体。播放的是V29的8条成功抓放，失败点没有执行轨迹。新预览使用独立 ROS master 11333，避免旧会话退出时的同名节点冲突。

```bash
# 重跑同样3×3冒烟，输出新时间戳目录
bash xtrainer_plan/run_pick_place.sh \
  --config xtrainer_plan/config/pick_place_overhead_ymax_smoke.json

# 查看本次成功轨迹
ROS_MASTER_URI=http://127.0.0.1:11333 ROS_IP=127.0.0.1 \
  bash xtrainer_plan/run_overhead_rviz.sh \
  --traj xtrainer_plan/results_overhead/20260916/v29_ymax_smoke/v29_grasp_ym010_ymax_mount \
  --speed 1 --loop
```

此次为规划/显示冒烟，不是实机执行验收，也未检查真实CAD装配体与机械臂碰撞、采样间扫掠碰撞或倒装负载。
