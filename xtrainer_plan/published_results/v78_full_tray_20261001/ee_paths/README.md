# 双臂末端路径分布

近臂蓝色，远臂黄色，坐标系为 `task_world`，单位 m。路径来自完整关节轨迹的正向运动学，取工具中心点 `TCP_LINK` / `far_TCP_LINK` 的位置。

- [三维与 XY / XZ / YZ 投影](ee_paths_3d_projections.png)
- [三维图](ee_paths_3d.png)：聚焦末端高度，基座 z=0.65 m 高于该图范围；完整布置见组合图。
- [俯视图](ee_paths_top.png)

各臂第一条完整抓放已分别从路径显示中删除：近臂 case381 保留从样本716开始；远臂 case1 保留从样本488开始。对应14.32 s / 9.76 s，下一段共享起点保留。过滤后近臂188件、远臂210件；全盘视频和机器人播放仍包含400件。

[精确数据](ee_paths_excluding_first_pieces.npz) 保存 TCP 坐标、关节角、原全局样本号、时间及400点 grasp 网格；键名前缀 `near_` / `far_` 表示近臂 / 远臂。`q_rad` 为关节角，单位弧度；`times_s` 为原时间，单位秒；`source_global_sample` 为原轨迹中从0开始的样本号。`moving_source_global_sample` 仅合并连续静止重复点，运动点未抽样。

[source_and_exclusion.json](source_and_exclusion.json) 保留生成时的来源、截断边界和 SHA256（文件内容校验值）；发布副本校验值见上级 [manifest.json](../manifest.json)。运行方式见 [结果 README](../README.md#rviz-播放)。
