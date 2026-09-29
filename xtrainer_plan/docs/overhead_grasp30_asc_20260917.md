# V57：同一 95% 安装方案改为 −30° 优先搜索

**已完成 400 点：V57 为 365/400（91.25%），原 +30° 优先版为 380/400（95%）。覆盖净少 15 点，但 LINK3 放置阶段与 grasp 有限参考平面相交从 284/380 降到 15/365；对齐共同成功点仍为 282→15，无新增。** 原限位双侧 .14 rad 内缩保持生效，但仍有参考平面相交和软工作区域越界，不是实体碰撞或实机安全保证。

[左右对比图](../results_overhead/20260917/v57_grasp30_asc_full/grasp_angle_comparison.png) / [新版本单图](../results_overhead/20260917/v57_grasp30_asc_full/grasp_angle_map.png) / [对比数值](../results_overhead/20260917/v57_grasp30_asc_full/comparison.json) / [完整结果与轨迹](../results_overhead/20260917/v57_grasp30_asc_full/runs/v57_00)。

## 请求与唯一规划参数变化

用户在确认 [V55/V56 最终对比图](../results_overhead/20260917/v56_cartesian_grasp30_desc_full/grasp_angle_comparison.png) 后要求从 −30° 开始再测一版。本轮以右图 **380/400（95%）** 的完整重跑 `v56r_00` 为基线，只改变 `pick_place.angle_search.order: desc → asc`，不重跑 48 组笛卡尔积，也不改变左图方案。

每个 case 按 **−30,−28,…,+30°** 搜索，共 31 组；`reuse_last_success=false` 仍保留，grasp/place 耦合且 p=g，stage2 关闭。上一成功轨迹终点仍是下一 case 起点；禁止复用成功角度不等于重置关节构型。

## 保持不变的条件

- 基座 XYZ=(−.31,.40,.65) m、RPY=(180,0,90)°，坐标属于保留的原始 LINK0/task_world。
- place XYZ=(−.36,−.12,.10) m；grasp x=[−.62,−.20]、y=[−.10,.40]、z=.03 m，完整 20×20、400 点，顺序不变。
- TCP=.19 m（原缩短 3 cm）、单臂无 mimic、Home 位置/姿态/种子均不变。
- 原始 URDF 限位 ±3.14 rad，双侧内收 .14 rad，IK/规划有效限位 ±3.00 rad；J1–J5 单段跨度 ≤170°，不额外限制 J6 跨度。
- 独立 IK 预筛关闭，实际规划 IK 与碰撞检查保留；直线误差 3 mm/5°。
- 自碰撞覆盖全部 24 球，原世界墙规则不变。grasp 参考平面与装配 CAD 仍未作为实体障碍加入硬碰撞模型。

[V57 manifest](../results_overhead/20260917/v57_grasp30_asc_full/manifest.json) 保存源配置/新配置哈希、派生命令及准确结果路径。深比较确认除搜索顺序与输出/溯源外配置完全一致；URDF 哈希与旧完整轨迹审计一致。真实候选函数确认从 −30 到 +30 的 31 对；现有排序与配置派生专项各 11 项测试通过。未修改核心规划器、默认 YAML、URDF 或 RViz。

独立角度审计新增显式 `--expected-order asc`，默认仍为 desc，不根据待测配置自动接受其顺序。16 项审计测试与 11 项排序测试通过；旧降序两组的 800 点/7216 次尝试重审通过，报告除生成时间外完全一致。

### 模型输入溯源边界

旧结果保存了 URDF 历史哈希，当前仍为 `7a1b8511e64e6d7c5e43069775c5e7659a8939bf63669e04b3a7ec4fcc2d1ebc`；旧 metadata/NPZ 哈希也匹配，15 项只读输入检查通过。

旧审计未记录 robot/spheres YAML 的历史哈希，因此不能声称其历史哈希也已逐一匹配。当前文件及 SHA256 在本轮两次读取一致，mtime 均早于旧实验（2026-06-08）；这里只记录当前值与这一溯源限制，不将 mtime 当成内容哈希证明：

- `src/curobo/content/configs/robot/xtrainer.yml`：`e203080e848d8a2c76d661d3c9c96961b3a6b941593619cd678064fdbfc3c581`。
- `src/curobo/content/configs/robot/spheres/xtrainer.yml`：`ab7500a2445edc4ac455a90e4210810476112833cc5fe0d31e18b4f5791b2ea1`。

## 完成结果

400 点全部完成，**365 成功、35 失败**；规划耗时 385.1 s（原降序 785.8 s），保存 111,670 个采样、2233.38 s 轨迹。四项后处理及 [实际升序搜索审计](../results_overhead/20260917/v57_grasp30_asc_full/angle_order_audit.json) 均通过；[运行状态](../results_overhead/20260917/v57_grasp30_asc_full/sweep_status.json) 记录退出码和每项验证结果。

全部 400 个 case 均从 −30° 开始，共尝试 1614 对角度；35 个失败各试足 31 对，未通过预筛跳过候选，未复用前一成功角度。365 个成功全部使用负角，其中 −30° 为 317 个，−28/−26/−24° 分别 10/11/10 个，其余分布在 −22～−8°。正角也在失败 case 中完整尝试，不能因为成功图全蓝就说正角没有被搜索。

![左：原 +30° 优先；右：新 −30° 优先](../results_overhead/20260917/v57_grasp30_asc_full/grasp_angle_comparison.png)

图保留普通 grasp 角度配色、失败红框和 2 cm 刻度，不额外区分成功类别。2 cm 是刻度间隔，20×20 实际 x/y 采样间距约为 2.21/2.63 cm。

### 同几何逐点增损

新旧 meta 中的实际配置与各自源配置一致，除 order/output/provenance 外全等；400 点的编号、顺序、原始及变换后坐标均一致，报告与轨迹哈希也经复核。

- 新增成功 8、丢失 23、共同成功 357、共同失败 12，净减少 15 点。
- y<0 的 80 点与 0≤y<.15 的 120 点均保持全成功；y≥.15 的 200 点由 180→165，全部增损都在高 Y 区域。
- 新增 8 点均采用 −30°；丢失的 23 点在原降序版中也全部采用负角。结果差异不来自角度范围不足，而与候选优先级、选中角度和连续轨迹前序分支有关；未做额外分支重置实验，不能断言某个具体分支变化是唯一原因。

以下 index 为 0-based：

```text
新增：119,137,138,139,156,157,158,159
丢失：195,196,197,198,199,338,339,354,355,356,357,358,359,
      375,376,377,378,379,395,396,397,398,399
共同失败：176,177,178,179,219,238,239,258,259,278,279,298
```

35 个失败的全部 1085 次尝试中，832 次 IK、248 次单段跨度、5 次 FINETUNE 优化失败；阶段为 1069 次 `g_lift_in`、15 次 `p_lift_in`、1 次 `g_lift_out`。按每个 case 最后一次尝试计为 18 个跨度、17 个 IK；不是 35 点全部已证明几何不可达。

### 限位与 LINK3 对比

| 指标 | +30° 优先 `v56r_00` | −30° 优先 `v57_00` |
|---|---:|---:|
| 成功 | 380/400（95%） | 365/400（91.25%） |
| J6 最小原始限位余量 | .14 rad | .14 rad |
| J6 距有效限位<1°的 case / 采样 | 3 / 77 | 4 / 343 |
| J6 最大单段跨度 | 265.826° | 237.427° |
| J6 最大相邻采样变化 | 4.273°/20 ms | 3.672°/20 ms |
| LINK3 全程平面相交 case | 308/380 | 21/365 |
| LINK3 放置相关平面相交 case | 284/380 | 15/365 |
| LINK3 放置相关相交采样 | 28661 | 820 |
| LINK3 最低相对平面高度 | −115.461 mm | −73.836 mm |

对齐 **357 个共同成功点**，放置阶段相交由 **282→15**，消除 267 个、无新增；最低相对平面高度改善 320 点、恶化 37 点，中位提高 126.252 mm。全程相交由 303→21，消除 283 个、新增 1 个。放置相交大幅减少的结论并非只是因为少成功了 15 点。

新轨迹六关节距原限位最小余量(rad)为 `[.1471094,1.0583605,.6049746,.7111474,.8317405,.14]`，所有保存采样均满足有效 ±3.00 rad，原始上下限保留 .14 rad。J6 有效最小余量仍为 0；图标题的近限位 6 case 包含全部关节，不是 J6 单独的 4 case。单段最大跨度不是瞬时跳变。

J1–J5 最大单段跨度 169.997806°、直线最大横向偏差 2.979485 mm、旋转偏差 1.999482°，满足 170°/3 mm/5°，但前两项已接近阈值。限位、IK/MotionGen rollout 及现有自碰撞/世界墙独立检查通过，不代表所有几何包络均处于软工作范围内。

### 仍需保留的几何限制

- V57 `workspace_check` 记录 12 个越界采样，最大 10.365 mm。
- LINK6 球包络相对软 `workspace.bounds` 有 49 个越界采样，最大 85.584 mm；原降序为 82 个、18.666 mm，不能只因样本数减少便称此项改善。
- 原配置将这些软范围检查设为信息报告，硬碰撞墙使用不同的 `wall.bounds_override`；没有为让本轮通过而修改约束或忽略新增错误。
- LINK3 仍有 15 个放置相关相交 case，最低高度出现在 `i0_place`、t=5.54 s。球包络相对零厚度平面的高度不是实体穿透深度，CAD/桌面/真实物体尚未成为该硬碰撞模型。

本轮结论是**覆盖率降低，但 LINK3 对 grasp 参考平面的干涉明显减少**，并非所有指标均改善。没有自动切换默认配置、RViz 或真实机械臂，也没有额外补点或混合两种顺序的轨迹。

## 命令记录

以下为本轮命令记录，输出目录拒绝覆盖；复现必须另选新目录。

```bash
unset PYTHONPATH
source /home/ethanqjiang/miniconda3/etc/profile.d/conda.sh
conda activate curobo
python xtrainer_plan/scripts/derive_overhead_experiment.py \
  --source xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/configs/v56r_00.json \
  --output xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/configs/v57_00.json \
  --search-order asc \
  --run-output-dir /home/ethanqjiang/workspace/curobo/xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/runs/v57_00
python xtrainer_plan/scripts/run_recorded_overhead_sweep.py \
  --manifest xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/manifest.json

python xtrainer_plan/scripts/audit_overhead_angle_order.py \
  --manifest xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/manifest.json \
  --expected-order asc \
  --out xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/angle_order_audit.json
python xtrainer_plan/scripts/compare_overhead_yshift.py \
  --results xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/runs/v56r_00 \
            xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/runs/v57_00 \
  --out xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/comparison.json
python xtrainer_plan/scripts/plot_overhead_results.py \
  xtrainer_plan/results_overhead/20260917/v56_cartesian_grasp30_desc_full/retry01/runs/v56r_00 \
  xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/runs/v57_00 \
  --annotate none \
  --out xtrainer_plan/results_overhead/20260917/v57_grasp30_asc_full/grasp_angle_comparison.png
```
