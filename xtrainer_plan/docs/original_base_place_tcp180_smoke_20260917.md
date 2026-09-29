# V59：原始 LINK0 安装 + 当前区域 + place TCP Z 旋转 180°

## 授权范围与状态

用户确认：base 回到原始 LINK0 的 `(0,0,0)`、原始安装朝向；grasp/place 使用当前版本的位置与区域。
本轮仅进行 **5×5 完整抓放冒烟与 RViz 预览**；20×20 全量必须等待用户看过 RViz 后确认。默认 YAML、URDF、现有实验结果均未修改。

结果根目录：`../results_overhead/20260917/v59_original_base_place_tcp180_smoke/`。
虽然沿用历史实验目录和坐标审计脚本名称，本版本不是悬挂安装。

## 配置与可追溯性

- 当前任务来源：`v57_grasp30_asc_full/configs/v57_00.json`。
- 原始安装 Home IK 种子来源：`results_pick_place/link0_z_0_tcp_0p19_current_yaml_full_20260914_132747/trajectory_meta.json`。
- 基座 `M=I`，目标坐标变换 `C=I`，RPY=`[0,0,0]`。无额外平移、yaw 修正或朝下翻转。
- grasp：X=`[-0.62,-0.20]`，Y=`[-0.10,0.40]`，Z=`0.03 m`，本轮5×5。
- place：`[-0.36,-0.12,0.10] m`。
- Home：保持当前位置 `[-0.31,0,0.03]` 和 RPY=`[-180,0,0]`；只把 IK 种子换回原始安装分支，不固定关节解。
- grasp 工具 X 角按 `-30,-28,...,+30°` 搜索，每点都从 -30° 开始，不复用上次成功角，stage2 关闭。
- place 候选角仍与 grasp 耦合；最终姿态再右乘自身 TCP Z 的180°旋转：`R_new_place(g) = R_old_place(g) @ Rz_local(180°)`。位置和 TCP Z 轴方向不变，不能等同于增加原始坐标系 yaw 180°。
- TCP 保持 LINK6 局部 Z=`0.19 m`，即已缩短3 cm。
- 单臂6关节，无 mimic/双臂碰撞。自碰撞覆盖24球；世界边界仍只约束 LINK3/LINK6 的12球。
- IK 和规划均内缩0.14 rad：URDF `[-3.14,3.14]` → 生效 `[-3.00,3.00] rad`。
- 保持当前 `prescreen_by_ik=false`、每段 J1–J5 最大跨度170°、J6不纳入该跨度判据；抓/放下降段直线偏移≤3 mm、姿态偏差≤5°。
- 工作空间恢复 V57 的 `overhead.task_workspace`，从而保持墙体在原始任务坐标系中的位置，未错误沿用悬挂基座下的数字边界。

派生脚本：`../scripts/prepare_original_mount_smoke.py`。
结果根目录内 `manifest.json` 记录源文件、配置、模型与规划器哈希，以及全部31组候选的186个阶段目标旋转预检查；`configs/v59_00_original_base_place_tcp180.json` 为实际输入。

## 冒烟结果

实际运行：`runs/v59_00_original_base_place_tcp180`，退出码0。成功 **22/25（88%）**；保存7109个样本，轨迹142.16 s，20 ms采样。

- 21个成功点在 -30° 首次尝试成功。
- `(-0.62,-0.10,0.03)` 在第15组、-2°成功。
- 以下3点遍历31组候选仍未成功；日志均为 `MotionGenStatus.IK_FAIL`，不是关节跨度判据拒绝：

| 0-based index / 1-based case | grasp XYZ (m) | 失败阶段 |
|---|---|---|
| 4 / 5 | (-0.62, 0.40, 0.03) | 全部在抓取前抬高点 g_lift_in |
| 21 / 22 | (-0.20, 0.025, 0.03) | 抓取前抬高点或 grasp |
| 22 / 23 | (-0.20, 0.15, 0.03) | 部分候选抓取端失败；+8°至+30°在放置前抬高点 p_lift_in 失败 |

上述仅说明当前搜索与约束下未找到可行解，不能视为无约束几何 IK 数学无解的证明。

![5×5冒烟抓取角图，2 cm附加刻度](../results_overhead/20260917/v59_original_base_place_tcp180_smoke/grasp_angle_map.png)

## 保存轨迹复核

- `joint_limit_clip_audit.json`：通过。独立构造的 IK、MotionGen 及其关节约束均为双侧0.14 rad内缩；7109点无越限。
- `independent_verification.json`：通过。全部7109个保存样本的坐标变换、TCP配置、FK、关节限位、自碰撞及设定世界边界碰撞复核通过。
- J6全程最近余量：相对内缩后限位 **20.51°**；相对原URDF限位 **28.54°**。这是本轨迹所有关节的最小余量。
- 最大单段 J1–J5 跨度162.295°；最大单段J6跨度171.300°。最大相邻20 ms采样差3.105°，不是瞬时转动171°。
- 规划器记录最大直线横向偏移2.903 mm，最大姿态偏差1.071°。
- LINK3球体与有限grasp参考平面相交样本为0。球体在区域投影内的最低表面距该平面：全程123.63 mm，放置相关阶段131.40 mm。

注意：CAD装配体是显示模型，不是完整CAD碰撞约束；检查不代表连续扫掠、动力学、实际物体高度或实机安装安全认证。

## RViz 预览

使用原始单臂 `build_scene_urdf.py` / `display_xtrainer_traj.launch`，不使用会添加悬挂支架的 overhead 场景构造器。
装配体显式使用 `config/cad_mounts_overhead_display.yaml`，保留此前确认的无额外yaw/Y平移版本；不显示第二臂，不镜像驱动。

预览入口、局部RViz设置和任务区域标记位于结果根目录；ROS master 使用独立端口11340，避免影响已有11311会话。播放的是22个成功case的完整六阶段轨迹，失败case仅在区域中标记、不伪造运动。

RViz 已在本机 `DISPLAY=:0` 打开并1倍速循环，绿色点成功、红色点失败。`xtrainer_arm1_place` 坐标轴只表示固定目标位置；逐case实际TCP姿态看 `TCP_LINK`。

在独立 master 已启动时可重新打开：

```bash
source /opt/ros/noetic/setup.bash
ROS_MASTER_URI=http://127.0.0.1:11340 ROS_IP=127.0.0.1 DISPLAY=:0 \
  roslaunch xtrainer_plan/results_overhead/20260917/v59_original_base_place_tcp180_smoke/preview.launch
```

末尾可加 `speed:=2.0` 加快显示，或 `player_extra_args:="--start-item 13"` 从第13个case开始（失败case不可播放）。

![原始安装单臂、原装配体与当前区域](../results_overhead/20260917/v59_original_base_place_tcp180_smoke/rviz_preview_confirmed.png)

显示验证：局部 XML/YAML、marker坐标、原始基座身份变换检查通过；RViz 全局状态正常。初版空的未记录点集导致显示告警，已改为删除空marker，未改规划结果或重启播放器。

日志注意：ROS 自动将进程环境写入日志；本次两份 roslaunch 日志的环境记录行已脱敏。排查过程中环境凭据字段曾进入工具输出，已提示用户轮换相关凭据。不要把未经检查的 ROS 启动日志对外分享。

**全量尚未启动，等待用户确认。**

后续：用户已确认并授权三组全量；另立 [V60实验记录](original_base_threeway_full_20260917.md)，本V59冒烟结果保持不变。
