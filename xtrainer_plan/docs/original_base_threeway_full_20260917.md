# V60：原始基座三组400点全量对照

用户在 V59 冒烟与 RViz 预览后授权全量，并要求保留三组供后续对比。

## 实验版本

结果根目录：`../results_overhead/20260917/v60_original_base_threeway_full/`。

| 顺序 | run | place最终TCP局部Z额外旋转 | grasp候选搜索顺序 |
|---|---|---|---|
| A | v60_00_place180_asc | 180° | -30,-28,...,+30° |
| B | v60_01_place0_desc | 0°（取消额外180°） | +30,+28,...,-30° |
| C | v60_02_place0_asc | 0°（取消额外180°） | -30,-28,...,+30° |

基于 V59 已确认的原始 LINK0 基座 `(0,0,0)`、原始朝向，`M=C=I`。三组均为20×20完整网格，grasp X=[-0.62,-0.20]、Y=[-0.10,0.40]、Z=0.03 m；place=[-0.36,-0.12,0.10] m；Home=[-0.31,0,0.03] m。TCP仍为0.19 m，单臂、无mimic。

所有点各自从指定首角开始，不复用上次成功角，stage2关闭，保持place候选角与grasp耦合。B/C取消的是最终局部Z180°，不是取消原始place姿态或原有工具X候选旋转。

三组IK/规划器均双侧内缩0.14 rad，原URDF±3.14 rad → 求解器±3.00 rad。碰撞球、物理工作空间边界、Home种子、规划参数、抓放直线约束、失败跳过策略与V59保持一致。顺序运行并在各组后执行独立复核，不影响当前 RViz 冒烟播放，不改默认 YAML/URDF。

来源配置与模型/源码SHA256见根目录 `manifest.json`；三组完整配置分别在 `configs/`，结果在 `runs/`。
派生工具为 `../scripts/prepare_original_base_threeway.py`：除网格5→20、输出路径与两个实验因素外，检查其它配置完全一致，并验证31个候选的局部轴旋转关系。

## 对比口径

- A与C：相同搜索顺序，对比额外180°的影响。
- B与C：相同place姿态，对比搜索顺序的影响。
- A与B同时改变两个因素，不能把差异全部归因于其中一个。
- 成功率分母固定400；关节运动量分布以该组成功case为分母，不把失败点当成小运动量。
- 同时报告三组共同成功的同坐标case；不过每轮沿用上一成功case的末态，三组可能选择不同IK分支，故共同目标不等于相同进入起点。
- 单case关节跨度按完整六阶段保存轨迹计算 `max(q)-min(q)`，含从上一成功case到本case的进入运动，六关节不做2π取模；它不同于单段跨度或相邻采样跳变。
- 累计关节行程为采样间绝对关节变化之和。既保留各关节，也保留六关节合计。
- 限位余量同时区分原URDF限位和内缩后有效限位；不能把0.14 rad本身当成新增的实机保证。
- 同一模型与求解参数下各运行一次，结论针对本次保存结果，不代表随机重复统计或连续工作区域的可达性证明。

## 检查与边界

每组依次生成完整规划日志、逐case结果、轨迹NPZ/CSV及：

1. `analysis_summary.json`：轨迹分段、失败日志分类和运动量。
2. `link3_grasp_clearance.json`：LINK3球体与有限grasp参考区域关系。
3. `independent_verification.json`：保存采样的独立FK/坐标/自碰撞/世界边界/限位复核。
4. `joint_limit_clip_audit.json`：IK、MotionGen及其rollout约束内缩验证。

本轮没有修改核心规划逻辑。局部旋转8项CPU回归测试、新对比脚本14项CPU测试和配置派生预检查均通过。三组规划和每组四项后处理全部正常退出，另有各组 `angle_order_audit.json` 验证400点候选搜索次序；模型、配置与核心源码哈希复核一致。绘图公共标签改用“实际基座 / 单臂”，避免原始基座结果被误标成“上方基座 / 倒装”；不影响规划数据。

CAD装配体仍仅用于显示，未当作完整CAD硬碰撞障碍物；以上复核不涵盖采样间连续扫掠、真实物体高度、动力学或实机结构安全。

## 结果

三组400点全量及全部审计已完成。下表余量为所有保存成功轨迹采样的最小值；近限位比例分母为该组成功case。

| 组 | 成功/400 | 成功率 | J6最小原始/有效余量 | J6有效余量<1°的case | 规划时间 |
|---|---:|---:|---:|---:|---:|
| A：place额外180°，−30°起搜 | 368 | 92.00% | 20.43° / 12.41° | 0/368（0%） | 442.9 s |
| B：place无额外旋转，+30°起搜 | 364 | 91.00% | 8.02° / 0° | 345/364（94.78%） | 2640.1 s |
| C：place无额外旋转，−30°起搜 | 363 | 90.75% | 8.02° / 0° | 4/363（1.10%） | 458.6 s |

A全量仍明显远离限位，所有关节最小有效余量也是J6的12.41°；不是沿用V59冒烟的20.51°有效余量。B/C中有效余量0°表示触及内缩后边界，并非突破原URDF限位，原限位仍有0.14 rad（8.02°）。C的4个近限位case为0基index 1、2、307、308。

三组共同成功363点，至少一组成功368点。完整case任意关节跨度≥180°的比例分别为A 0/368、B 6/364（1.65%）、C 21/363（5.79%）；这是运动范围，不是相邻采样瞬时跳变。

按本次配置的离散模型检查，三组保存轨迹均未发现限位、自碰撞或硬墙违规。LINK3与有限grasp参考平面的诊断：A、C全程无相交；B有29个case相交，其中place相关阶段涉及2个case（0基231、252）。该参考平面不是硬碰撞对象，不可因此把B的“审计通过”理解为实际装配安全。

已按同一400点索引对齐数据，并分别保留自身成功总体、三组共同成功总体统计：

- [三组工作区域对比图](../results_overhead/20260917/v60_original_base_threeway_full/grasp_angle_comparison.png)，额外刻度间隔2 cm，成功点统一按角度着色。
- [对比摘要](../results_overhead/20260917/v60_original_base_threeway_full/summary.md)。
- [逐点对比CSV](../results_overhead/20260917/v60_original_base_threeway_full/case_comparison.csv)。
- [完整统计JSON](../results_overhead/20260917/v60_original_base_threeway_full/summary.json)，含采样范围、各关节余量/跨度/累计行程及来源哈希。
- 完整轨迹与审计均独立保存在结果根目录的 `runs/` 三个子目录中。

默认配置未改动，当前RViz仍为V59冒烟预览，未自动切换到V60全量。

## 后续：第一组 J6 位置分布图

按用户要求从第一组保存轨迹补充实际J6角度分布，不重跑规划。[主图（grasp到位）](../results_overhead/20260917/v60_original_base_threeway_full/runs/v60_00_place180_asc/joint_angle_maps/j6_grasp_angle_map.png)、[grasp/place/全程最小最大四图](../results_overhead/20260917/v60_original_base_threeway_full/runs/v60_00_place180_asc/joint_angle_maps/j6_angle_comparison.png) 和 [口径、数据与验证记录](../results_overhead/20260917/v60_original_base_threeway_full/runs/v60_00_place180_asc/joint_angle_maps/README.md) 已保存。grasp末端J6为−159.48°～61.83°，place末端为−1.32°～19.93°。失败32点不填角度；图中颜色不是工具姿态搜索角，保持2 cm刻度。

## 后续：三组 J6 统一色标横向对比

B/C单组J6图及三组横排图已补齐。[抓取角对比](../results_overhead/20260917/v60_original_base_threeway_full/joint_angle_comparison/j6_grasp_threeway.png)、[放置角对比](../results_overhead/20260917/v60_original_base_threeway_full/joint_angle_comparison/j6_place_threeway.png)、[有效限位余量对比](../results_overhead/20260917/v60_original_base_threeway_full/joint_angle_comparison/j6_min_limit_margin_threeway.png)、[四行三列完整对照](../results_overhead/20260917/v60_original_base_threeway_full/joint_angle_comparison/j6_overview_threeway.png)。左A/中B/右C顺序固定，角度统一±180°，余量统一0°～160°。分组与共同成功363点的统计、对齐CSV、详细验证记录见 [对比说明](../results_overhead/20260917/v60_original_base_threeway_full/joint_angle_comparison/README.md)。仅从保存轨迹作图，原配置、轨迹和RViz未改动。

## 后续：各关节最大运动跨度

新增 [各关节完整case最大跨度柱状图](../results_overhead/20260917/v60_original_base_threeway_full/joint_motion_comparison/joint_max_cycle_span_threeway.png)、[每位置最大变化关节及跨度分布图](../results_overhead/20260917/v60_original_base_threeway_full/joint_motion_comparison/max_joint_span_map_threeway.png)，并单独保留共同成功363点的结果及整条run角度范围图。主口径是每case六阶段的max(q)−min(q)再跨case取最大值，包含Home/上一case入场段，非瞬时跳变。三组最大均为J6：179.41°、239.61°、239.61°，后两组来自首个成功case的Home入场。各关节数值、对应case和口径区别见 [详细说明与数据](../results_overhead/20260917/v60_original_base_threeway_full/joint_motion_comparison/README.md)。只做CPU离线分析，未重跑规划。
