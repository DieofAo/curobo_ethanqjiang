# V46–V47：基座移向Y正边缘，检查LINK3对抓取区的避让

承接[倒装全量测试V31–V45](overhead_downward_20260916.md)。用户观察到抓取区中间偏原始+X的case在放置时LINK3靠近抓取区，提出将基座移至原始Y正边缘或再向外一些。

## 实验不变项

基座X=-0.36、Z=0.65，安装RPY=(180,0,90)°（新+Z沿旧−Z）；grasp范围x=[-0.62,-0.20]、y=[-0.10,0.40]、z=0.03；place=(-0.16,-0.23,0.10)；Home=(-0.31,0,0.03)、姿态仍(-180,0,0)°。TCP仍0.19m，原装配体对齐不变。

仅比较安装Y=0.40（边缘）、0.45（外移5cm）、0.50（外移10cm），独立预筛关闭，所有最终轨迹/限位/直线/碰撞验收不变。每个安装从独立IK得到的Home关节只用作分支种子，仍重新求解和验收Home。没有覆盖默认配置或替换当前Y=0.15的RViz基线播放。

## 端点与完整冒烟

| 基座Y | Home | 同角度四端点交集 | place可行角数 | 完整3×3规划 |
|---:|---|---:|---:|---:|
| 0.15（V44基线） | 可行 | 9/9 | 16/16 | 7/9 |
| 0.40 | 可行 | 9/9 | 16/16 | 6/9 |
| 0.45 | 可行 | 9/9 | 14/16 | 4/9 |
| 0.50 | 可行 | 0/9（抓取侧8/9） | 0/16 | 未进行：place端点未找到解 |

[端点对照](../results_overhead/20260916/v46_yedge_endpoint_smoke/endpoint_comparison.png) / [实际轨迹角度对照](../results_overhead/20260916/v47_yedge_trajectory_smoke/grasp_angle_comparison.png)。端点解不保证连续抓放成功，未对新候选跑20×20全量。

Y=0.40失败index0、5、8；成功2135采样、42.68s，最小关节余量约3.645°。Y=0.45失败index0、3、5、6、8；成功1299采样、25.96s，最小余量约18.426°。两组成功轨迹全部采样都通过[0.40独立复核](../results_overhead/20260916/v47_yedge_trajectory_smoke/v47_y040_smoke_seeded/independent_verification.json)、[0.45独立复核](../results_overhead/20260916/v47_yedge_trajectory_smoke/v47_y045_smoke_seeded/independent_verification.json)。该复核只覆盖既有约束，**抓取区没有被建成LINK3禁入障碍**。

## LINK3与抓取区的量化对照

使用当前碰撞模型的LINK3三个半径70mm球，对保存关节做CPU URDF FK，转换回原始task_world；与保存TCP位置交叉核验为亚微米级误差。[分析脚本](../scripts/analyze_link3_grasp_clearance.py)未改动规划约束。解析几何小测试覆盖矩形内外、有限平面相交、球冠边缘高度。

“放置阶段”包括移向place抬升点、下降放置及抬升退出三段；表中间隙为**球包络在grasp矩形XY范围内的最低表面，相对z=0.03平面的高度**，不是距真实物体的安全距离。负数表示包络伸到此参考平面以下，不能直接等同真实CAD网格碰撞。

| 基座Y | 已成功轨迹放置阶段最小高度差 | 与有限grasp平面相交采样 |
|---:|---:|---:|
| 0.15 | −84.30mm | 495/1104 |
| 0.40 | −25.71mm | 348/976 |
| 0.45 | +28.97mm | 0/652 |

由于成功case集合不同，上表不是严格一一匹配的统计比较。对三组都成功、且符合用户关注方向的**同一点index7=(-0.20,0.15,0.03)**，最小高度差依次为 **−52.14 → −23.78 → +29.24mm**。说明这批重规划结果确有改善，不只是少测了风险case。

需要保留的限制：

- Y=0.40不仅放置仍有参考平面相交，抓取转移`i6_g_lift_in`最低高度差达到−115.03mm，不能仅凭放置改善认定问题解决。
- Y=0.45的4条成功轨迹在全程保存采样中未与该平面相交，但放置阶段652帧**全部仍有XY投影重叠**；它是抬高经过，不是绕开抓取区上方。实际最低表面Z约0.059m，物体若更高，仍可能冲突。
- Y=0.45的index6失败，两个新安装的index8都失败；没有轨迹的case不能记为避让成功。
- 密网格V45原方案也存在这个问题：放置相关44,854帧中3,473帧与参考平面相交。该事实与“原配置硬墙无碰撞”并不矛盾，说明当前环境模型并未保护这块区域。
- 全部结果仅针对现球包络和保存采样；不包含实际物体高度、真实CAD网格、其它link或采样间扫掠碰撞。

详细报告：[Y=0.15冒烟](../results_overhead/20260916/v44_h065_home_y0_smoke_direct/link3_grasp_clearance.json)、[Y=0.40](../results_overhead/20260916/v47_yedge_trajectory_smoke/v47_y040_smoke_seeded/link3_grasp_clearance.json)、[Y=0.45](../results_overhead/20260916/v47_yedge_trajectory_smoke/v47_y045_smoke_seeded/link3_grasp_clearance.json)、[V45密网格](../results_overhead/20260916/v45_h065_home_y0_full_direct/link3_grasp_clearance_with_cases.json)。

## 结论与预览

向Y正向外移可以改善LINK3放置避让；**Y=0.40尚未解决，Y=0.45值得继续优化，但当前只有4/9成功、距参考平面最低约2.9cm；Y=0.50又使现place端点求解失败**。不能简单“越远越好”。

[Y=0.45放置阶段最低间隙构型截图](../results_overhead/20260916/v47_yedge_trajectory_smoke/rviz_y045_place_min_clearance.png)取自真实轨迹采样285、t=5.70s、index1的放置抬升段；仅用于候选模型展示，截图后关闭独立预览窗口，保留原基线RViz。模型与候选安装一致，不使用旧位置模型播放新轨迹。

下一步需要确认抓取区物体的最高原始Z及所需余量，再讨论给LINK3增加针对该区域的显式避让要求；本轮尚未加入此约束，也未擅自修改place或物体模型。

```bash
# 查看Y=0.45成功轨迹（当前仅4条成功抓放）
bash xtrainer_plan/run_overhead_rviz.sh \
  --traj xtrainer_plan/results_overhead/20260916/v47_yedge_trajectory_smoke/v47_y045_smoke_seeded \
  --speed 1 --loop
```
