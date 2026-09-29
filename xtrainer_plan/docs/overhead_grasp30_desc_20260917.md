# V55/V56：grasp 扩展到 +30°、降序优先的安装位置扫描

**已完成：48 组冒烟选出的两组 20×20 全量分别为 359/400（89.75%）与 380/400（95%）。本轮覆盖率较高的是基座 (−.31,.40,.65)、place (−.36,−.12,.10)。全部 247,766 个保存轨迹采样通过内缩 .14 rad 及已配置碰撞审计，但 LINK3/grasp 平面相交仍明显，不是实机安全结论。**

[最终双方案角度图](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/grasp_angle_comparison.png) / [最终数值对比](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/comparison.json) / [800 点角度顺序审计](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/angle_order_audit.json)。第二组中断后在新目录完整重跑，最终有效结果名为 `v56r_00`，详见恢复记录。

## 请求与固定条件

承接 [V54 全量对照](overhead_cartesian_full_clip014_20260917.md)，本轮按用户要求将 grasp 范围扩到 **−30°～+30°，步长 2°**，每个 case 依次尝试 **+30,+28,…,−30°**。先重复相同 48 组安装/放置笛卡尔积的 3×3 完整抓放冒烟，再选择前两名做 20×20 全量。

关闭本轮配置的 `reuse_last_success`，确保下一 case 不会优先沿用上一次成功角度。仍保持连续轨迹的上一成功终点作为下一次规划起点，不重置关节构型；失败不更新起点。原有 grasp/place 耦合规则不改：本轮最小 grasp 为负，因此 place 与 grasp 同角，包含正角度。

所有坐标均为保留的原始 LINK0/task_world，单位 m：

| 参数 | 扫描值/固定值 |
|---|---|
| 基座 X | −.51、−.41、−.31 |
| 基座 Y | .25、.35、.40、.45 |
| 基座 Z / RPY | .65 / (180,0,90)° |
| place X | −.46、−.36、−.26、−.16 |
| place Y / Z | −.12 / .10 |
| grasp XYZ 范围 | x=[−.62,−.20]、y=[−.10,.40]、z=.03 |
| TCP | .19 m，保留减少 3 cm 的配置 |
| Home | (−.31,0,.03)，原姿态与 IK 种子不变 |

单臂无 mimic，原始 URDF 上下限各内收 **.14 rad**，六轴实际有效限位 ±3.00 rad。IK、MotionGen 及最终保存轨迹均审计同一内缩范围。J1–J5 单段跨度 ≤170°，J6 不新增跨度限制；直线误差 3 mm/5°；全 24 球自碰撞与原世界墙规则不变。装配 CAD/真实物体未加入硬碰撞模型，LINK3/grasp 有限平面仍只作诊断。

本轮与旧轮同时改变角度范围、角度顺序与角度复用策略，结果变化不能单独归因于“多了正角度”。不修改旧结果、默认配置、URDF、当前 RViz 或真实机械臂。

## 版本与验证

- `plan_pick_place.py` 新增 `desc` 候选排序，保留旧 `abs`/`asc`、耦合及阶段二规则。
- `derive_overhead_experiment.py` 增加显式角度范围/步长/排序/关闭角度复用的派生参数，验证有限数值与合法范围，继续拒绝覆盖。
- 派生、安装平移与网格细化现有/新增测试分别 11、4、5 项通过；排序专项 11 项通过。
- 整套 274 项 CPU 测试为 273 通过、1 项旧双臂测试失败：`test_default_first_twelve_pairs_cover_every_grasp_angle` 假定默认 12 角，现有默认 −30～0/2 实际为 16 角。将候选函数在内存恢复旧实现后该测试仍以相同断言失败，确认不是本轮降序改动引入；没有改动无关双臂测试或默认配置。
- 逐一深比较 48 组 V55 与对应 V52 配置：除角度最大值、排序、复用策略、溯源和输出路径外完全一致，`M*C=I` 与 `.14 rad` 均验证通过。
- 新增独立 CPU 角度日志审计 `audit_overhead_angle_order.py`，专项 9 项测试通过。它逐 case 核对实际候选顺序、place=grasp、成功角度与尝试次数、配置/元数据一致性及源文件哈希；运行中与 Home 失败不会被误记为通过。
- 最终再次运行整套 CPU 测试（含角度审计新增测试）：**283 项中 282 通过**，唯一失败仍是上述已证实与本轮修改无关的双臂默认角度数断言；本轮新增/相关 31 项专项测试全部通过。

## 冒烟与选择规则

[V55 manifest](../results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/manifest.json) 保存源配置与 48 组配置的哈希、坐标及生成命令。

每组遍历整个 3×3 矩形，按**完整抓放成功率**降序选择前两名，仅独立验证通过的结果进入排名。同成功率时依次比较 LINK3 放置阶段平面相交占比（低优先）、J6 最小有效限位余量（大优先）、J6 最大单段跨度（小优先），最后以路径稳定排序。这些只用于同分排序，不新增规划拒绝条件；3×3 成功不保证 20×20 成功。

每组成功结果执行规划统计、LINK3 几何诊断、独立轨迹 FK/限位/自碰撞/墙验证、新建 IK/MotionGen 及 rollout 的限位审计。仅保存轨迹采样通过不代表连续扫掠或实机安全证明。

## V55 冒烟结果：48 组已完成

结果见 [逐方案汇总与排序](../results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/comparison.json)、[运行与后处理状态](../results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/sweep_status.json)、[独立角度顺序审计](../results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/angle_order_audit.json)。48 组均完整执行，均有成功轨迹；四项后处理与独立角度顺序审计全部通过，无 Home 失败、零成功或未完成方案混入排名。

- 48×9 共 **432 次“方案×抓取点”试验，成功 349 次、失败 83 次**，合计 80.79%。这是相同 9 个抓取点在 48 种安装/放置配置下重复测试的总数，**不是一个固定安装方案覆盖了 349 个位置，也不是抓取区域的连续面积覆盖率**。
- 共保存并独立复核 **131,653 个轨迹采样点**；编排总耗时 **1,766.06 s（约 29.43 min）**，包含各组规划与后处理，不是机械臂播放时长。
- 角度审计确认全部 432 个 case 都从 `g=+30 p=+30` 开始；共尝试 **5,057** 组角度。83 个失败 case 各完整尝试 31 组；成功 case 在 1～31 组内停止，保存角度与最后成功日志一致。
- 仅这 83 个最终失败 case 的 **2,573 次失败尝试**中：IK 失败 1,480 次、关节单段跨度超限 1,046 次、优化失败 47 次。这里统计的是全部候选尝试，不是将每个 case 仅按最后一次失败归类；同一 case 可以遇到多种失败原因。
- 共有 **7 组达到 9/9**；这 7 组的粗网格成功率相同，下面排序来自预先声明的诊断指标，仅用于选出两组细查，不证明前两名一定在完整网格最优。

保存样本的独立限位与已配置自碰撞/世界墙检查均无违规，原始上下限各保留 .14 rad（限位比较容差 1e−6 rad）。跨 48 组的 J1–J5 最大单段跨度为 169.95357°；直线段最大横向偏差/旋转偏差为 2.37611 mm/3.12949°，均在原约束内。J6 最大单段跨度为 332.84431°，仍作为大绕转诊断报告，不偷偷加入原先只约束 J1–J5 的 170° 拒绝判据。

![V55 基座 XY × place X 冒烟覆盖率](../results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/cartesian_coverage_heatmap.png)

图的三个面板分别固定基座 X；横轴是 place X，纵轴是基座 Y。每格 `N/9` 是该固定安装/放置方案完整抓放成功数，绿色表示独立验证通过的成功比例，橙框标出全部 7 个并列最高覆盖方案，而不只标出最终选中的两组。

### 七组 9/9 与全量选择

基座 Z 均为 .65 m；place Y/Z 均为 −.12/.10 m。LINK3 列统计成功 case 中“放置相关阶段至少一个球体采样与有限 grasp 平面相交”的数量；它不是实际物体碰撞次数。J6 余量列是距**内缩后的有效限位**的最小距离。

| 同分排序 | 冒烟方案 | 基座 (X,Y) m | place X m | 成功 | LINK3 放置相交 case | J6 最小有效余量 | 后续 |
|---|---|---|---|---|---|---|---|
| 1 | `v55_25` | (−.41,.40) | −.36 | 9/9 | 3/9 | 16.75° | `v56_00`，20×20 |
| 2 | `v55_41` | (−.31,.40) | −.36 | 9/9 | 6/9 | 8.97° | `v56_01` 中断，改 `v56r_00` 重跑 20×20 |
| 3 | `v55_29` | (−.41,.45) | −.36 | 9/9 | 6/9 | 7.85° | 本轮不加跑全量 |
| 4 | `v55_42` | (−.31,.40) | −.26 | 9/9 | 7/9 | .29° | 本轮不加跑全量 |
| 5 | `v55_15` | (−.51,.45) | −.16 | 9/9 | 7/9 | .00° | 本轮不加跑全量 |
| 6 | `v55_28` | (−.41,.45) | −.46 | 9/9 | 8/9 | 13.04° | 本轮不加跑全量 |
| 7 | `v55_11` | (−.51,.40) | −.16 | 9/9 | 8/9 | 2.45° | 本轮不加跑全量 |

前两名在本次 3×3 下的 J6 最小**原始 URDF 限位余量**分别为 24.77°、16.99°；J6 最大单段全程角度跨度分别为 274.09°、200.08°，不是相邻采样的瞬时跳变。两者仍有 LINK3/grasp 平面相交诊断，不能理解成“所有碰撞场景都安全”。`v55_15` 的有效余量 .00° 表示碰到本轮收紧后的界限，并不是越过原始机械限位；原始限位仍内留 .14 rad。

### 与旧 V52 的同几何逐点对照

独立复核 [旧 V52 源结果](../results_overhead/20260917/v52_cartesian_clip014_smoke/manifest.json) 与 [V55 源结果](../results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/manifest.json)，按相同基座位置、place 位置及原始 grasp 点配对，48 组各 9 点：

- 总成功由 **298/432 → 349/432**；新增成功 74 次、丢失原成功 23 次，净增 51 次。
- 逐安装/放置方案看，30 组提高、12 组持平、6 组下降；不是所有方案都单调改善。
- V55 的 349 次成功中，采用正角 285 次、负角 63 次、零角 1 次；其中 +30° 被采用 173 次。
- 新增 74 次成功中，62 次采用正角、12 次仍采用负角。因为本轮也改变搜索顺序、关闭成功角度复用，而且各 case 连续衔接，不能把这些增益全部归因于扩大正角范围；失败也不等价于目标绝对不存在 IK 解。

## V56 第一组全量：`v56_00` 已完成

基座 **(−.41,.40,.65)**、place **(−.36,−.12,.10)** 的 `v56_00` 完成全部 400 个 case：**359/400 成功（89.75%），41 个失败**。已保存 **126,925 个采样点、2,538.48 s 轨迹**；规划进程耗时 **2,388.53 s**。四项后处理完成，独立轨迹检查与限位模型审计均通过，且 [400 点角度顺序审计](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/v56_00_angle_order_audit.json) 通过；它不是只完成了 359 个 case、其余尚未测试。

![V56 第一组 20×20 抓放结果](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/v56_00_grasp_angle_map.png)

详细证据：[规划统计](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/runs/v56_00/analysis_summary.json)、[独立保存轨迹验证](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/runs/v56_00/independent_verification.json)、[IK/MotionGen 限位审计](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/runs/v56_00/joint_limit_clip_audit.json)、[LINK3/grasp 平面诊断](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/runs/v56_00/link3_grasp_clearance.json)。

### 关节、角度与几何诊断

| 指标 | `v56_00` 全量结果 |
|---|---|
| 六轴限位 / 已配置碰撞 | 保存样本无越界、自碰撞或所配置世界墙碰撞；原始上下限各内缩 .14 rad |
| J6 最小原始 / 有效限位余量 | .14 / .00 rad，即约 8.0214° / .00° |
| J6 距有效限位 <1° | 29 个成功 case，1,791 个采样点 |
| J1–J5 最大单段跨度 | 169.89251°，未超过 170° |
| J6 最大单段跨度 | 338.84108°，`i136_p_lift_in`；J6 不在 170° 拒绝判据中 |
| 最大相邻采样变化 | J6 为 4.78734° / 20 ms，与整段跨度不是同一指标 |
| 最大直线横向 / 旋转偏差 | 2.71809 mm / 3.47687°，在 3 mm / 5° 约束内 |
| 成功采用角度 | 正角 288、负角 70、零角 1；其中 +30° 为 211 个 case |
| LINK3 全阶段与有限 grasp 平面相交 | 291/359 个成功 case，29,256 个采样点 |
| LINK3 放置相关阶段与有限平面相交 | 279/359 个成功 case，25,287/64,173 个放置阶段采样点 |

LINK3 球体在 grasp 矩形上方投影范围内的最低表面，相对 grasp 平面最低为 **−101.02873 mm**，发生于 `i132_p_lift_in`、样本 44,890、轨迹时刻 897.8 s。这是“最低表面相对平面的高度”，不要与球体到有限平面的有符号间隙最小值 **−69.98639 mm** 混为一谈。真实物体/CAD 碰撞未启用，因此已配置碰撞检查通过与这项平面相交诊断可以同时成立；本结果仍不是实机抓取安全证明。

### 41 个失败 case

41 个失败 case 均尝试完整 31 角，共 1,271 次失败尝试：**881 次 IK 失败、208 次优化失败、181 次关节跨度超限、1 次直线约束失败**。按失败轨迹阶段统计，1,261 次在 `g_lift_in`，9 次在 `p_lift_in`，1 次在 grasp。仅按每个 case 的最后一次尝试统计，则为优化 23、跨度 15、IK 3；不能用这 41 个末次标签替代前面完整 1,271 次原因。

失败点主要位于 x≈−.465～−.377 的部分中间列、偏 +Y 一侧；另有 y=−.10 的少数点，以及 x≈−.288、y≈.242/.268 两点。普通角度图给出完整采样分布。多数失败尝试发生在接近下一抓取点的 `g_lift_in`，但这不证明目标不存在其它 IK 分支；顺序规划的上一成功终点、所选角度及单段约束均影响结果。

## 第二组中断与恢复记录

[原 V56 manifest](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/manifest.json) 记录原计划 `v55_25 → v56_00`、`v55_41 → v56_01`。第一组完成后，第二组原 `v56_01` 在第 **137/400** 个 case 处理中，整体编排进程以 **143** 退出；日志未说明终止原因，不能把这次中断归因为 IK 无解或规划算法失败。随后进程检查未见残留的 planner/batch 子进程。

原目录 [runs/v56_01/plan.log](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/runs/v56_01/plan.log) 原样保留，只有日志，没有 `run_status.json`、最终 `trajectory_meta.json` 或 `trajectory.npz`。日志中有 135 个已报 OK、1 个已报失败，第 137 个尚未终结；这些 **135 个未保存的 OK 不是已经独立验证的轨迹，也不拼接进最终成功数**。

恢复采用新的独立目录 [retry01/manifest.json](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/manifest.json)，从同一个 `v55_41` 再派生 20×20，前缀 `v56r`，**从 Home 开始整组重跑**。新 `v56r_00` 与原 `v56_01` 已逐字段深比较：除输出路径/溯源外配置完全一致，基座仍为 **(−.31,.40,.65)**、place 仍为 **(−.36,−.12,.10)**，并未改变角度策略、限位或碰撞约束。

最终第二组有效结果路径为：

```text
xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/runs/v56r_00
```

原 `runs/v56_01` 作为此次中断证据保留，不覆盖、不当作完成结果，也不要求原双方案 manifest 的全部结果通过审计。后续比较使用已完成的 `runs/v56_00` 与上述重跑结果。

## 第二组完整重跑：`v56r_00` 已完成

第二组独立从 Home 重跑全部 400 点，最终 **380/400（95%）成功、20 点失败**；规划耗时 785.8 s，保存 120,841 个采样、2416.80 s 轨迹。四项后处理通过，800 点合并角度政策审计也通过。原中断日志中的 135 个 OK 未计入或拼接，原 `runs/v56_01` 仍是未完成记录，不能用于播放或当作完整结果。

[有效结果目录](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/runs/v56r_00) / [完整重跑状态](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/sweep_status.json) / [第二组单独角度图](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/v56_01_grasp_angle_map.png)。单图文件沿用第二组编号 `v56_01`，图内标题 `v56r_00` 对应真实完整重跑目录。

J6 最小原始余量 .14 rad（8.0214°），距有效限位最小余量 0；3 个成功 case、77 个采样距有效限位不足 1°。图标题的近限位 6 个包括所有关节，不是 J6 单独的 3 个。J6 最大单段跨度 265.82624°，最大相邻采样变化 4.27264°/20 ms；J1–J5 最大单段 169.68354°。直线段最大横向偏差 1.95362 mm、旋转偏差 2.04738°，符合原判据。

LINK3 全程 308/380 case（35,954 采样）、放置相关 284/380 case（28,661/61,615 采样）与有限 grasp 平面相交；最低相对平面高度均为 −115.46080 mm。这里是球模型相对零厚度参考平面的诊断，不是已纳入规划的实体碰撞检测；高度值也不能直接视为球和平面的穿透深度。

另一个信息性检查 `gripper_extent_check` 记录 82 个采样的 LINK6 球包络越过 `workspace.bounds`，最大 18.666 mm；该软范围与硬碰撞墙使用的 `wall.bounds_override` 不同，原配置并不据此拒绝轨迹。TCP 工作范围违规数为 0，独立自碰撞/世界墙违规数为 0；不能把这些通过概括为“所有几何包络均未越界”。

20 个失败点全部位于 y≥.3210526 m，各完整尝试 31 角：620 次失败尝试共 362 次 IK、256 次跨度、2 次优化失败。按每个 case 的最后一次失败计为 19 个跨度超限、1 个 IK_FAIL，因此不能说剩余 20 点全部 IK 无解。逐点位置与全部失败尝试见有效结果目录的 `analysis_summary.json`。

## 最终双方案比较与同几何旧策略对照

![左：第一组；右：第二组完整重跑](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/grasp_angle_comparison.png)

两组 place 均为 (−.36,−.12,.10)，安装 Z=.65、Y=.40，只在基座 X 上不同。

| 指标 | 第一组 `v56_00` | 第二组 `v56r_00` |
|---|---:|---:|
| 基座 X | −.41 m | −.31 m |
| 完整成功 | 359/400（89.75%） | **380/400（95%）** |
| J6 最小原始余量 | .14 rad | .14 rad |
| J6 距有效限位<1°的 case / 采样 | 29 / 1791 | 3 / 77 |
| J6 最大单段跨度 | 338.84108° | 265.82624° |
| J6 最大相邻采样变化 | 4.78734°/20 ms | 4.27264°/20 ms |
| LINK3 全程平面相交 case | 291/359 | 308/380 |
| LINK3 放置相关平面相交 case | 279/359 | 284/380 |
| LINK3 放置阶段最低相对高度 | −101.02873 mm | −115.46080 mm |

相对第一组，第二组新增成功 30 点、丢失 9 点、共同成功 350 点，净多 21 点。两组成功并集为 389 点，但安装不同，**不能将其拼成一个固定安装的 389/400 结果**。冒烟前两名的顺序在全量中反转；本轮按覆盖率选择第二组，但没有把它自动写成默认配置。

对齐两组共同成功的 350 点，LINK3 放置平面相交 277 → 268（新增 39、消除 48、双方 229），但逐点最低高度仅 78 点改善、272 点恶化，中位变化 −41.16 mm；全程相交均为 289 点（新增/消除各 37）。因此即使相交 case 数略少，也不能概括第二组 LINK3 整体更安全。

### 第二组与旧 `v54_01`：同几何、同点序

[同几何策略对照报告](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/comparison_same_geometry_v54_01.json)。深比较确认除 grasp 上限、搜索顺序、角度复用与输出/溯源外，两个配置完全相同；400 点的编号、顺序、原始及变换后坐标一致。

- 旧 360/400 → 新 380/400，新增 27、丢失 7、共同成功 353、共同失败 13。低 Y 的 80 点和中间 Y 的 120 点均维持全成功；y≥.15 的 200 点由 160 → 180，全部增损都在此区域。
- J6 距有效限位<1°的成功 case 175 → 3，采样 9272 → 77；最大单段跨度 334.71496° → 265.82624°。最小原始余量仍为 .14 rad，并未消除大绕转。
- LINK3 放置平面相交由 113/360 → 284/380；本轮不能称为“覆盖和抓取区域避让全面改善”，实际物体与桌面仍未加入该硬碰撞模型。

对齐 **353 个共同成功点** 后，LINK3 放置平面相交仍由 **113 → 279**：新增相交 175、消除 9、双方均相交 104；最低表面相对 grasp 平面高度改善 99 点、恶化 248 点、近似不变 6 点（1e−6 m），成对高度变化中位数 −66.26 mm。因此它不是仅因新结果多成功了 20 点才使相交总量变大。

同样对齐 353 点，J6 近有效限位<1°从 173 case/9223 采样降到 2 case/43 采样；每 case 的最大 J6 单段跨度有 214 点变小、139 点变大，中位数 91.08° → 68.14°。新增 27 个成功点中只有 4 个采用正角、23 个采用负角，再次说明搜索顺序和前序终点分支也影响覆盖，不能将净增 20 全部归功于新增正角度。

旧全量另一安装 `v54_00` 曾达到 386/400，仍高于本轮最佳 380/400，但其基座 Y 和 place X 不同；本轮没有为它增加第三组新策略全量，也不能把跨安装的差异都归因于角度策略。

## 完成状态

48 组冒烟、选定两种安装的全部 800 个 full-grid case、成功轨迹的四项独立审计、实际角度顺序审计及两组普通角度图均已完成。800 点实际尝试 7,216 组角度，逐点从 +30° 递减的日志与元数据一致。全部 247,766 个保存采样通过原始限位双侧内收 .14 rad 和现有碰撞模型复核；没有声称连续扫掠、真实物体/CAD 或实机跟踪安全。

保留了原 `v56_01` 中断日志，第二组唯一有效完整轨迹为 `retry01/runs/v56r_00`。原双组编排未生成最终 sweep_status，完成依据是第一组 `run_status.json` 与其四份审计、第二组独立重跑状态、最终 `comparison.json` 和 `angle_order_audit.json`，不伪造原编排成功状态。

默认 YAML、URDF、当前 RViz 和真实机械臂均未切换；没有额外运行第三种安装方案。

## 生成、运行与验证命令记录

以下路径相对仓库根目录。生成器、运行器与 JSON 审计均拒绝覆盖现有实验；这些是本轮命令记录，**不要原地重复启动 V55/V56**。复现实验请使用新的输出目录与报告名。

```bash
unset PYTHONPATH
source /home/ethanqjiang/miniconda3/etc/profile.d/conda.sh
conda activate curobo

# 派生本轮角度策略源配置。
python xtrainer_plan/scripts/derive_overhead_experiment.py \
  --source xtrainer_plan/results_overhead/20260916/configs/v49_h065_clip015_full.json \
  --output xtrainer_plan/results_overhead/20260917/configs/v55_grasp30_desc_source.json \
  --joint-limit-clip 0.14 --grasp-angle-range -30 30 --grasp-angle-step 2 \
  --search-order desc --no-angle-reuse

# 48 组基座 X × 基座 Y × place X；随后逐组规划并执行四项后处理。
python xtrainer_plan/scripts/generate_overhead_cartesian.py \
  --source xtrainer_plan/results_overhead/20260917/configs/v55_grasp30_desc_source.json \
  --out-root xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke \
  --prefix v55 --base-x -0.51 -0.41 -0.31 --base-y 0.25 0.35 0.40 0.45 \
  --base-z 0.65 --place-x -0.46 -0.36 -0.26 -0.16 --place-y -0.12 --clip 0.14 --size 3
python xtrainer_plan/scripts/run_recorded_overhead_sweep.py \
  --manifest xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/manifest.json

# 全部完成后独立核对每个 case 的实际角度顺序，汇总审计通过的方案并绘制热图。
python xtrainer_plan/scripts/audit_overhead_angle_order.py \
  --manifest xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/manifest.json \
  --out xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/angle_order_audit.json
python xtrainer_plan/scripts/summarize_overhead_cartesian.py \
  --results xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/runs/v55_* \
  --out xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/comparison.json \
  --plot xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/cartesian_coverage_heatmap.png

# 明确选择前两组，直接细化到 20×20；本轮不经过额外 5×5。
python xtrainer_plan/scripts/refine_overhead_cartesian.py \
  --manifest xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/manifest.json \
  --names v55_25 v55_41 \
  --out-root xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full \
  --prefix v56 --size 20
# 原双方案编排记录；第二组中断后，不在此 manifest 上原地重启。
python xtrainer_plan/scripts/run_recorded_overhead_sweep.py \
  --manifest xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/manifest.json
```

第二组中断后的恢复命令记录（新目录，不覆盖原 `v56_01`）：

```bash
python xtrainer_plan/scripts/refine_overhead_cartesian.py \
  --manifest xtrainer_plan/results_overhead/20260917/v55_cartesian_grasp30_desc_smoke/manifest.json \
  --names v55_41 \
  --out-root xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01 \
  --prefix v56r --size 20
python xtrainer_plan/scripts/run_recorded_overhead_sweep.py \
  --manifest xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/manifest.json
```

运行器对有成功轨迹并正常结束的每组保存 `analysis_summary.json`、`link3_grasp_clearance.json`、`independent_verification.json`、`joint_limit_clip_audit.json`，并分别记录退出状态；运行器退出 0 本身不能替代逐组成功数和审计报告。独立角度审计支持零成功场景的完整日志校验，但其“政策通过”不等价于规划成功。

第二组重跑结束且两组审计通过后，已执行以下汇总；显式指定第一组与重跑第二组，排除未完成的原 `v56_01`。全量不使用固定写有 smoke 标签的 Cartesian 汇总图：

```bash
python xtrainer_plan/scripts/audit_overhead_angle_order.py \
  --results xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/runs/v56_00 \
            xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/runs/v56r_00 \
  --out xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/angle_order_audit.json
python xtrainer_plan/scripts/compare_overhead_yshift.py \
  --results xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/runs/v56_00 \
            xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/runs/v56r_00 \
  --out xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/comparison.json
python xtrainer_plan/scripts/plot_overhead_results.py \
  xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/runs/v56_00 \
  xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/runs/v56r_00 \
  --annotate none \
  --out xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/grasp_angle_comparison.png
```

普通角度图保留 2 cm 坐标刻度；颜色表达实际采用的正/负 grasp 角度，不额外划分三种成功类型。2 cm 刻度不是规划采样间距；20×20 对当前矩形的实际 x/y 采样间距分别约 2.21/2.63 cm。
