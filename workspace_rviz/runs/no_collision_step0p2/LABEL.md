## 数据归档标签

| 字段 | 值 |
|---|---|
| 标签 | **no_collision_step0p2** |
| 采样时间 | 2026-05-27 13:23（采样完成）|
| 关节步长 | 0.2 rad（笛卡尔积枚举）|
| 总样本量 | 7.23 × 10¹⁰ |
| 体素分辨率 | 0.01 m |
| HEALPix nside | 2（48 方向桶）|
| 末端 link | tool |
| 机器人 | jaka 7-DoF (left_jaka.urdf)|

### ⚠️ 重要：本批数据**未做任何碰撞检测**

口径：
- ✅ 仅做了关节限位 + FK
- ✅ 末端落点纳入 bbox `[-1.5, 1.5]³` 即计入可达
- ❌ **未检查自碰撞**（self-collision）
- ❌ **未检查环境碰撞**（worldcoll，地面/工装等）
- ❌ **未检查关节速度/加速度可行性**

因此 `reachable_voxels = 3,599,197` 是**几何可达上界**，
真实可用工作空间会比这个**显著小**。

### 文件清单

- `workspace_data.npz`     体素中心 + SO(3) 命中桶
- `workspace_meta.json`    采样参数
- `summary.json`           覆盖率/包络统计
- `sample.log`             采样过程日志
- `figs/scatter3d.png`     3D 散点
- `figs/z_slices.png`      6 层 XY 切片
- `figs/max_projections.png` 三正交最大投影

### 后续对比

当跑出含碰撞检测的版本（标签建议 `with_collision_step0p2`）后，
可用差集对比："碰撞剔除掉了多少体素 / 灵巧度下降多少"。
