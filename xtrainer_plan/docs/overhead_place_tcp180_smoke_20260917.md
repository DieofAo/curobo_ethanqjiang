# V58：place 绕最终 TCP 局部 Z 轴自转 180° 的配对冒烟测试

本轮仅做 5×5、25 点冒烟，不自动切换默认配置或启动 400 点全量。基线为最新 V57（grasp 从 −30° 优先搜索）。结果：原姿态 **23/25（92%）**，新姿态 **20/25（80%）**；当前未显示覆盖改善，J6 最大单段跨度反而增加。

## 先看效果

- [同一个中心抓取点的左右对比 MP4](../results_overhead/20260917/v58_place_tcp180_smoke/place_tcp180_comparison.mp4)：左为原 place，右为 place 额外局部 TCP-Z +180°；完整六阶段，1 倍速。
- [放置瞬间左右截图](../results_overhead/20260917/v58_place_tcp180_smoke/place_pose_comparison.png)。
- [5×5 抓取角/成功率对比图](../results_overhead/20260917/v58_place_tcp180_smoke/grasp_angle_comparison.png)：左原姿态，右 +180°。颜色仍代表 grasp 搜索角，不包含 place 的固定自转。2 cm 刻度不代表采样密度；实际 x/y 点距为 10.5/12.5 cm。
- [新姿态完整冒烟轨迹目录](../results_overhead/20260917/v58_place_tcp180_smoke/runs/v58_01_place_tcp180) / [数值对比](../results_overhead/20260917/v58_place_tcp180_smoke/comparison.json)。

视频使用相同原始抓取点 `(−0.41, 0.15, 0.03)` m，case 编号 13 / index 12，两侧选中的 grasp/place 第一阶段搜索角均为 −30°。左原轨迹 5.40 s，右 6.30 s；各自首尾停留 1 s，左侧在结束后额外保持 0.9 s 以并排展示完整右侧轨迹。合成 MP4 为 2560×936、50 fps、8.32 s，非阶段强制同步，也未人为加速某一侧。

这条中心 case 的完整周期 J6 跨度为 **37.684° → 142.383°**；六关节中最大跨度为 **84.676°（J4）→142.383°（J6）**。视频直接使用保存样本，不重规划、不平滑。起始姿态仍是各自连续规划中的上一个成功 case 终点，而不是独立 Home。

## 精确旋转定义与保持项

对每个候选角 g，要求：

```text
R_place_new(g) = R_place_old(g) @ Rz_local(180°)
p_place_new = p_place_old
```

因此 TCP +Z 指向不变，局部 X/Y 反向。不能简单把原任务系 yaw 加 180°；也不能只把 `base_rpy.place` 右乘 Rz 再照常做 Rx(g)，因为那会改变旋转顺序，g=±30° 时与所需姿态相差 60°。

新增可选配置：

```yaml
pick_place:
  place:
    tool_z_rotation_deg: 180.0
```

固定自转在第一阶段候选及可选第二阶段之后，仅追加到 place（包括放置前接近和放后抬起）。缺省或为 0 时不追加 quaternion 运算，原行为不变。CLI 对应 `--place-tool-z-rotation-deg 180`；本次没有修改默认 YAML。

两组除这个固定自转及输出/记录字段外完全一致：

- base XYZ `(−.31,.40,.65)` m，RPY `(180,0,90)°`，原始 LINK0/task_world 任务坐标系保留。
- place XYZ `(−.36,−.12,.10)` m；grasp x `[−.62,−.20]`、y `[−.10,.40]`、z `.03` m；只将采样数量从 20×20 减为 5×5。
- TCP `.19` m，即原缩短 3 cm；单臂，无 mimic；Home 位姿和 IK 种子不变。
- 六关节上下限均内缩 `.14 rad`，IK 与 MotionGen 一致；J1–J5 单段跨度 ≤170°，不额外限制 J6 跨度。
- 每点严格从 −30、−28…+30° 搜索；p=g 耦合，禁止复用上一成功角度；stage2 与独立 IK 预筛仍关闭，实际规划 IK 保留。
- 自碰撞、世界墙、线性要求及软工作区域报告规则不变。

`angle_place_deg` 仍记录第一阶段耦合角 g，不把 180 加进这个字段；最终姿态见 `place_rpy_deg` 和 `pose_sequence`，固定自转另在 resolved config 中保存。

## 冒烟结果

| 指标 | 原 place | place 局部 TCP-Z +180° |
|---|---:|---:|
| 完整成功数 | 23/25（92%） | 20/25（80%） |
| 规划耗时 | 26.4 s | 41.8 s |
| 保存采样 | 7155 | 6682 |
| J6 最大单段跨度 | 144.047° | 187.400° |
| J6 最大相邻采样变化 | 2.647°/20 ms | 3.753°/20 ms |
| J6 最小原始限位余量 | .296551 rad（16.991°） | .296551 rad（16.991°） |
| J6 最小内缩后有效余量 | .156551 rad（8.970°） | .156551 rad（8.970°） |
| J6 距有效限位不足 1° 的 case/采样 | 0/0 | 0/0 |
| LINK3 放置相关参考平面相交 case | 2/23 | 2/20 |
| LINK3 放置相关相交采样 | 149 | 107 |
| LINK3 最低参考平面相对高度 | −73.839 mm | −70.795 mm |

新姿态无新增成功，少成功 3 个点（0 基 index）：0=`(−.62,−.10,.03)`、18=`(−.305,.275,.03)`、19=`(−.305,.40,.03)`。原来共同失败的 14、24 仍失败。这里只能说本次候选/分支/连续轨迹规划失败，不能说目标已经被证明没有 IK 解。

新增失败的 31 候选完整分类（不是仅看最后一次尝试）：

| index | IK 失败次数 | 单段跨度失败次数 | 阶段 |
|---|---:|---:|---|
| 0 | 11 | 20（J5 约 219.6–226.2°） | IK 为 10 次抓取入口、1 次抓取；跨度全在放置入口 |
| 18 | 22 | 9（J4） | 全在抓取入口 |
| 19 | 29 | 2（J4） | 全在抓取入口 |

18/19 的失败发生在串行任务中上一放置终态改变后的下一次抓取入口，不可归因为“place IK 都解不出”或“grasp 点必然不可达”。新组 5 个失败的全部 155 次尝试合计 105 次 IK、50 次跨度失败，无优化/直线类别失败。

LINK3 的上述总数基于不同成功集合，不可据其样本减少就断言避让改善；参考平面是有限零厚度平面，数值来自球包络，不是实体桌面/物体穿透深度。CAD 装配体仍为显示几何。保存轨迹通过既有限位、硬自碰撞和世界墙审计，不等于完整实机安全认证。

对齐 **20 个共同成功点** 后，LINK3 放置相关平面相交为 **1 点 → 2 点**：原 index 5 消除，但新增 index 10、20；相交样本 **53/3174 → 107/3355**，同集合最低相对高度 **−22.870 → −70.795 mm**。所以本次并不支持“整体 LINK3 避让改善”。共同点全程相交数为 4→4，但涉及点位发生变化（消除 1/5/9，新增 4/10/11）。

两组本次软 `workspace` 越界与 LINK6 球包络软边界越界均为 0；J1–J5 最大单段跨度为 147.729°→164.010°，仍低于 170°。这些是本轮采样结果，不能外推到未测试的全量网格。

## 验证与溯源

- [manifest](../results_overhead/20260917/v58_place_tcp180_smoke/manifest.json) 记录源/新配置、URDF、robot/spheres YAML 与规划器脚本哈希。
- [旋转预检查](../results_overhead/20260917/v58_place_tcp180_smoke/target_rotation_precheck.json)：31 个候选的位移、抓取姿态、TCP +Z 均不变，仅 place 按指定局部轴旋转。另经独立检查 25×31×6×2（原/规划坐标系）共 9300 个姿态，最大矩阵误差 `6.67e−16`。
- [搜索顺序审计](../results_overhead/20260917/v58_place_tcp180_smoke/angle_order_audit.json) 两组全部通过；25 点各自完整执行，不用成功结果混合补图。
- [运行状态](../results_overhead/20260917/v58_place_tcp180_smoke/sweep_status.json)：两个 planner 均退出 0，四项后处理分别通过，独立确认所有保存采样满足原始限位双侧 `.14 rad` 内缩。
- 对新组 20 个成功 case 的 120 个保存目标路点，再以独立矩阵公式核验最终 place 确实局部右乘 180°、位置和 TCP Z 不变，误差小于 `8e−16`，且 `item.place_rpy_deg` 一致。
- 新增局部轴旋转 CPU 测试 8 项全部通过。完整 suite 303 项中 302 通过；唯一失败为修改前就存在的双臂测试 `test_default_first_twelve_pairs_cover_every_grasp_angle`（预期 12，现有配置 16），没有新增失败，本轮不改双臂逻辑。
- 首次启动因 CuRobo 环境导入失败，未进入 IK/运动规划；失败日志保留在 `startup_import_failure/`，正确激活 conda 后在上述正式路径重跑。它不是可达性失败结果。
- 两个片段及逐帧渲染记录位于 [video_comparison](../results_overhead/20260917/v58_place_tcp180_smoke/video_comparison)。独立 RViz/ROS 录制窗口已关闭，用户原会话及默认配置未改动。

## 复现入口

```bash
unset PYTHONPATH
source /home/ethanqjiang/miniconda3/etc/profile.d/conda.sh
conda activate curobo
python xtrainer_plan/scripts/prepare_place_tcp_rotation_smoke.py \
  --source xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/configs/v57_00.json \
  --out-root <新的输出目录> --size 5
python xtrainer_plan/scripts/run_recorded_overhead_sweep.py \
  --manifest <新的输出目录>/manifest.json
```

准备脚本拒绝覆盖已有实验目录。尚未运行新姿态 400 点全量；先查看这组对照效果再决定后续方案。
