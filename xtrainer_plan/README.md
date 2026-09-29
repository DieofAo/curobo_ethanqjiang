# XTrainer 轨迹规划（curobo + ROS1 rviz）

给定**起始位姿**与**目标位姿**，用 curobo MotionGen 规划一条带自碰撞检测、
且末端受工作空间约束的轨迹，并在 ROS1 rviz 中播放，同时把起始/目标/路点/运动中的
末端位姿全部发布到 tf。

```
起始/目标位姿(可调)  ->  自动插入抬升路点  ->  分段 MotionGen 规划
                                              (自碰撞 + 工作空间墙)
                                                    |
                                              trajectory.npz
                                                    |
                                        ROS1: joint_states + tf + marker  ->  rviz
```

---

## 快速开始

```bash
cd ~/workspace/curobo/xtrainer_plan

# 1) 规划（自动切到 conda curobo 环境）
./run_plan.sh

# 2) 在 rviz 播放（自动切到 ROS noetic，自动挑最新一次结果）
./run_rviz.sh
```

## 独立双臂交错 pick & place（12 DOF）

新的双臂任务使用一份 combined URDF 和一个 12-DOF `MotionGen`，不再是
`--both-arms` 把一号臂的 6 个关节角复制给二号臂。

```bash
# 默认共享世界网格；2 表示两臂合计规划 2 件物料
./run_dual_pick_place.sh --max-items 2
# 诊断：仅忽略联合 IK 预筛的关节跳变，最终轨迹仍保留 170° 验收
./run_dual_pick_place.sh --max-items 2 --no-ik-jump-check
# 等价入口
./run_pick_place.sh --dual --max-items 2

# 播放 12 轴联合轨迹；不会再复制 second_J_*
./run_rviz.sh --traj results_dual_pick_place/<时间戳> --both-arms \
  --mounts config/cad_mounts_same_side.yaml
```

同侧挂载位姿在 `config/cad_mounts_same_side.yaml`。以一号臂 `LINK_0`
为联合根坐标系，二号臂基座是 `xyz=[-0.8,0,0], yaw=180°`；这样两臂
都在传送带同一侧，并在同一个 12-DOF 规划空间中使用相同的姿态搜索、
工作空间和关节判据。

若 XTrainer 的任务 `LINK_0` 还需要一层标定修正，可在
`config/pick_place_default.yaml` 的 `pick_place.link0_target_transform` 配置
平移和轴角旋转，例如：

```yaml
pick_place:
  link0_target_transform:
    position: [0.01, -0.02, 0.0]  # 米
    rotation:
      axis: [0.0, 0.0, 1.0]
      angle_deg: 5.0              # 度，遵循右手定则
```

`position` 和 `axis` 都在规划 root/LINK_0 中表达；旋转轴不要求预先归一化，
但不能为零向量。单臂和双臂入口都会把两部分组装成刚体变换 `C`，对当前已生成的
目标完整左乘：`T_effective = C @ T_current`。单臂和双臂一号臂使用 `C`，双臂
二号臂使用 `C @ T12`。基于位姿配置的 Home、抓取、抬升、放置和 IK 预筛都会
应用同一修正；显式 `home.joint_deg` 是关节目标，因此保持不变。双臂静态 place
预览也会应用修正。设为零平移和零角度即可保持旧行为，已有的 4×4 矩阵写法仍可
兼容读取。`T12` 物理挂载、机器人/料台/墙和共享网格的分区顺序不随之改变。
启用 Cartesian 直线约束时，`C` 必须把自由轴映射到规划 root 的 x/y/z 之一，
否则规划会在启动时拒绝该配置。

默认 `dual_arm.task_layout=shared_global_outside_in` 只生成一份以联合根坐标
表达的物料网格，并以两基座的 x 中线分区（当前中线为 `x=-0.4`）：一号臂
从全局 x 大端、即靠近自身的 `x=-0.2` 一侧开始，二号臂从全局 x 小端、
即靠近自身的 `x=-0.6` 一侧开始，二者都向中间推进。同一个 source 只会
分配给一条手臂；恰好位于中线的物料归一号臂。`--max-items N` 表示两臂
合计最多处理 N 件，一号臂取 `ceil(N/2)`、二号臂取 `floor(N/2)`，因此
奇数时一号臂多一件，并可在二号臂结束后继续。

旧的镜像双任务仍可通过设置 `dual_arm.task_layout=same_local_grid` 使用：
该兼容模式给两臂复制同一份本地网格，经各自基坐标变换后成为两组世界坐标
镜像物料，并保留原来的成对任务及每臂计数语义。

二号臂放置点按一号臂基坐标表达。默认不是另存一套完整坐标，而是每次从
一号臂 place 派生，严格保留其 y/z，只把 x 改成
`dual_arm.second_place_x`。这两个配置位置都是应用 `C` 前的原始坐标；规划和
预览使用的最终位置会再统一左乘 `pick_place.link0_target_transform`。

默认固定相位为 3 个工艺路点：一号臂先执行 `grasp_lift → grasp →
retract`，二号臂随后开始。每件有 6 个路点；一号臂完成第 6 步后，下一个
tick 立即去下一件的抓取上方，二号臂完成当前 6 步后也立即循环。物料之间
没有逐轮同步屏障。二号臂完成自己的最后一件后，下一 tick 的回 Home 也作为
同一条 12-DOF 联合轨迹中的碰撞规划事件；到达后保持 Home，一号臂若还有物料
则继续，不需要等待。例如 `--max-items 3` 时一号臂 2 件、二号臂 1 件；二号臂
回 Home 会与一号臂后半程交叠，而不是在整条轨迹末尾另行追加。

共享布局中的实际 Home 是两臂各自“最终获配首件”的 `g_lift_in` 预抓取
位姿。配置里的 `pick_place.home` 只用作物料 IK 预筛和首件联合 Home IK 的
bootstrap seed；先一次性完成不可达 source 的跳过/补位，再把最终首件的
“双臂抓取姿态角对 × 联合 IK 关节分支”作为整条流水的根节点。根 0
无法连续到 place（含规划、关节判据或后验碰撞/工作空间失败）时，其所有
chunk 都会被丢弃，并从下一根的 12-DOF `q_home` 重新规划事件 0；
不拿动态 Home 反过来重复改变 source。根节点按 IK branch rank 轮转，
先覆盖所有首件抓取角，再覆盖各姿态的第二条 IK 分支。二号臂末尾返回
的也是选中根的首件预抓取 Home，并以 `home_branch_tolerance_deg`
检查仍处于同一关节分支；若二号臂没有获配任务，它保留配置 Home 作为兜底。

`dual_arm.start_delay_stages` 可调整固定相位。规划器把时间线拆成 prime 和
rolling blocks，并使用有界 DFS 回溯跨物料的姿态角；下游 block 无解时会回退
前一 block，而不会提交半条轨迹。等待中的手臂会被 TCP 全位姿约束在本段起点，
并逐点检查关节漂移；二号臂到达 Home 后则固定到选中根的 Home 关节参考，
跨后续所有 block 检查累计漂移。默认 `on_fail=skip` 的所有剔除都发生在轨迹发布/执行前：
两臂分别对各自获配物料做 primary 抓取 IK 检查；某件无解时只从对应手臂队列
剔除并沿其扫描方向补位。该独立跳过只用于运动前能明确证明的抓取 prefix IK
无解；完整流水中的联合 IK、轨迹优化、判据、异常、碰撞、跨物料冲突或二号臂
回 Home 失败仍整条 fail closed，不能再归因成某一侧物料并继续跳过。旧的
`same_local_grid` 兼容模式仍保留“成对 source 的流水联合 IK 可唯一归因后，
从 home 整线重规划”的逻辑。`--max-items N` 表示尽量保留 N 个最终物料（两臂
合计），跳过记录写入 `plan_skipped.json`；`on_fail=stop` 则禁用上述跳过。
连续模式始终拒绝 `--sequential-fallback`，
避免产出“已抓未放”或破坏固定相位的轨迹。

姿态搜索沿用单臂的两阶段语义：先为两臂找到可行的一阶段角，再固定这些角度
联合搜索非零 `stage2` 增量；二阶段全部失败才复用已缓存的一阶段轨迹。
`dual_arm.max_pair_angle_trials` 限制给定上游状态时一次 prime/rolling block
搜索的角度对数；回溯到不同上游状态会重新计数。跨分支及两阶段的总成本由
`dual_arm.max_pipeline_search_nodes` 在每个 Home 根内限制。Home 层另由
`max_home_angle_pair_trials`、`max_home_ik_branches_per_pair` 和
`max_home_root_trials` 限制；失败 JSON 会区分“已穷尽枚举”与“预算耗尽”。

每个角度对/block 会先跑一次带两臂和碰撞模型的联合 IK 路点预筛，再进入较慢的
MotionGen。每个 Home 根产生整线候选后都会立即分块检查臂内/臂间碰撞；
发现根相关的碰撞或工作空间越界时会继续回退下一根，检查器本身不可用则立即
fail closed。最终失败写入 `plan_failed.json` 并以非零状态退出，RViz 自动选轨迹时也会跳过该目录。
显式 `--traj` 指到带失败 marker 的诊断轨迹时同样拒绝播放。

相关模型/配置：

- `xtrainer_dual_independent.urdf`：两条独立运动链，`J_1..J_6 + second_J_1..second_J_6`，无 mimic。
- `xtrainer_dual_independent.yml`：12 维 cspace，两臂 48 个碰撞球全部进入自碰撞，未忽略任何跨臂球对。
- `scripts/plan_dual_pick_place.py`：主 TCP 用 `goal_pose`，二号臂 TCP 用 `link_poses`，每段一次联合优化。
- `config/dual_pick_place_default.yaml`：只定义双臂差异项，其余参数从单臂默认配置继承。

仓库内 CuRobo 官方范式是 `examples/isaac_sim/multi_arm_reacher.py` 与
`configs/robot/dual_ur10e.yml`：它们演示了 combined robot + `link_poses` 的多末端联合规划。
CuRobo 没有现成的“双臂 pick/place 工艺交错”调度器，所以本目录在该联合规划
原语之上增加了任务级节拍。

> 碰撞边界：臂内和臂间碰撞、原有 workspace 墙会进入规划；RViz 里的
> `table.stl` 目前仍是纯 visual，工件/夹持负载也尚未作为 attached object 加入。
> 所以当前保证的是“双机器人模型 + 配置中的墙”无碰撞，不包含手中工件彼此碰撞。

### 为什么分两步

`curobo` 只在 conda `curobo`(py3.11) 环境可用，而 `rospy` 属于 ROS noetic 的
py3.8，两者无法在同一进程共存。因此规划进程导出 `trajectory.npz`，
播放进程再读取它。两个 `.sh` 已各自处理好环境切换，无需手动 activate。

> npz 中不含任何 object dtype 数组（`joint_names` 用定长 ASCII `S64` 存储），
> 因此 numpy>=2 写出的文件能被 numpy<2 直接读取。

---

## 修改起始 / 目标位姿

**约定**：所有位姿都相对机器人 base `LINK_0`；`rpy` 单位为**度**，
采用 ROS/URDF 标准的**固定轴 XYZ 外旋**（`R = Rz(yaw)·Ry(pitch)·Rx(roll)`，
等价于 scipy 的 `'xyz'` 外旋）。位姿描述的是 **`LINK_6`（法兰面）**，不是夹爪 TCP。

### 方式一：改配置文件（推荐）

编辑 `config/task_default.yaml`：

```yaml
task:
  start:
    position: [-0.31, -0.05, 0.20]
    rpy_deg: [-135.56, -0.93, -8.90]
  goal:
    position: [-0.34, -0.09, 0.22]
    rpy_deg: [177.0, 53.0, -84.0]
```

### 方式二：命令行覆盖（优先级最高）

```bash
./run_plan.sh --start-position -0.31 -0.05 0.2 --start-rpy -135.56 -0.93 -8.9 \
              --goal-position  -0.34 -0.09 0.22 --goal-rpy   177 53 -84
```

### 方式三：自定义配置文件（只写差异项）

```yaml
# config/my_task.yaml —— 会自动与 task_default.yaml 深层合并
task:
  goal:
    position: [-0.40, 0.05, 0.30]
```

```bash
./run_plan.sh --config config/my_task.yaml
```

### 从真机当前关节角出发

不指定时起点由 IK 求解（可能与真机构型不同）。已知真机关节角时：

```yaml
task:
  start_joint_state: [-2.4624, -0.2403, -2.1528, 0.1868, 1.091, 2.5592]
```

---

## 中间路点（进入抓取 / 进入放置）

默认会自动插入两个抬升路点，形成 `start → start_lift → goal_lift → goal` 四点三段：

```yaml
task:
  waypoints:
    auto_lift:
      enable: true
      start_lift_z: 0.08      # 起点正上方抬升高度 (m)
      goal_lift_z:  0.08      # 目标正上方抬升高度 (m)
      keep_orientation: true  # 抬升点沿用端点姿态
      lift_axis: base_z       # base_z=沿 base +Z；tool_z_neg=沿工具 -Z 后退
```

```bash
./run_plan.sh --lift-z 0.10 0.12   # 起点抬 10cm，目标抬 12cm
./run_plan.sh --no-lift            # 不要抬升，直接 start -> goal
```

手工追加任意路点（插在两个抬升点之间，按顺序经过）：

```yaml
task:
  waypoints:
    extra:
      - position: [-0.40, 0.00, 0.35]
        rpy_deg: [180.0, 0.0, 0.0]
```

各段独立规划、段间速度归零，然后拼接成一条连续轨迹（去掉重复衔接点）。

---

## 工作空间约束

```yaml
workspace:
  bounds:                     # 你声明的真实工作空间，用于规划后逐点软校验
    x: [-0.50, 0.00]
    y: [-0.20, 0.20]
    z: [0.10, 0.70]
  wall:
    collision_link_names: ['LINK_6']   # 只有末端受墙约束
    bounds_override:                   # 墙实际使用的盒子
      x: [-0.50, 0.00]
      y: [-0.20, 0.20]
      z: [-0.05, 0.70]
```

```bash
./run_plan.sh --bounds-x -0.45 -0.05 --bounds-z 0.15 0.6
./run_plan.sh --no-wall     # 去掉硬墙，只做规划后软校验
```

实现上有三个关键点，都是踩坑后确定的：

### 1. 墙只约束末端，但自碰撞覆盖全部 link

curobo 的**世界碰撞**与**自碰撞**是两个彼此独立的 cost，各自接收同一份碰撞球张量。
而 `sphere_obb_kernel.cu` 对负半径球会直接 early-return 写 0 代价：

```cpp
if (sphere_cache.w < 0.0) { out_distance[bn_sph_idx] = 0; return; }
```

所以只需在**世界碰撞**的 `forward` 入口处 clone 张量、把非末端 link 的球半径置负，
墙就会跳过它们；自碰撞 cost 收到的仍是原始张量，完全不受影响
（见 `plan_trajectory.py:restrict_world_collision_to_links`）。

> 不能用官方的 `disable_link_spheres()` —— 它直接修改 kinematics 里的球半径，
> 会让自碰撞一起失效：`self_collision_kernel.cu` 里 `if (sph1.w <= 0) continue`
> 那段被注释掉了，负半径会使 `r_diff = r1+r2` 变成大负数、`distance` 恒为负。

如果把 `collision_link_names` 留空（墙约束整臂），实测 `LINK_2/3/5` 必然贴墙，
规划无解。

### 2. 墙盒的 z 下界要低于声明的工作空间

`LINK_6` 的碰撞球覆盖**整个夹爪**，沿 `LINK_6` 局部 `+z` 一直伸到 0.292m。
在朝下的抓取姿态下，法兰位于 `z=0.20` 时夹爪尖端已降到 `z≈-0.02`，
**「法兰在盒内」与「夹爪在盒内」无法同时满足**。

因此 `wall.bounds_override` 把 z 下界放到 `-0.05`，而 `bounds` 保持你声明的
真实范围用于软校验。夹爪越界量会作为参考信息写入 meta，不判失败。

用 `scripts/probe_ee_extent.py` 可针对新位姿重新算出所需的 `bounds_override`。

### 3. base 让位孔（仅当墙约束 `LINK_0/LINK_1` 时才需要）

`LINK_0/LINK_1` 的碰撞球球心全部位于 `J_1` 旋转轴上，**与关节角无关**，
实测立柱包围盒 `x=[-0.08,0.08] y=[-0.08,0.08] z=[-0.047,0.303]`，必然穿透
`x=0` 与 `z=0.1` 两个墙面。`wall.base_clearance` 可在这两面上开孔让位
（默认 `enable: false`，因为默认墙只约束 `LINK_6`）。
数值由 `scripts/probe_base_spheres.py` 探测得出。

---

## rviz 中看到什么

```bash
./run_rviz.sh                       # 播一遍
./run_rviz.sh --speed 0.3 --loop    # 0.3 倍速循环
./run_rviz.sh --traj results/20260829_114225
./run_rviz.sh --no-rviz             # 只发 topic，不起 rviz
./run_rviz.sh --no-walls            # 不画墙体方块
./run_rviz.sh --no-scene            # 不加载料台装配体，只看单臂（旧行为）
./run_rviz.sh --arm right           # 换成右侧挂载位为「被规划的臂」
./run_rviz.sh --no-static-arm       # 只显示料台，不显示另一条静态臂
```

**Topic**

| Topic | 内容 |
|---|---|
| `/joint_states` | 6 个关节角，驱动 `robot_state_publisher` 让整臂动起来 |
| `/tf` | 见下表 |
| `/xtrainer_plan/markers` | `workspace` 线框盒、`walls` 半透明墙、`poses` 位姿点+名称、`ee_path` 末端轨迹线 |

**tf frame**（父坐标系均为 `LINK_0`）

| frame | 含义 |
|---|---|
| `xtrainer_start` | 起始位姿 |
| `xtrainer_goal` | 目标位姿 |
| `xtrainer_wp_start_lift` / `xtrainer_wp_goal_lift` | 抬升路点 |
| `xtrainer_ee_cmd` | **运动过程中**末端的实时位姿 |
| `xtrainer_ee_planned` | 同上（同源），预留用于日后与真机反馈对比 |

> 这些 frame 走普通 `/tf` 并持续刷新，而**不用** `StaticTransformBroadcaster`：
> ROS1 的 `/tf_static` 是 latch topic，`robot_state_publisher` 也会往里发
> （`gripper_link` 等 fixed joint），后发布者会顶掉先前的内容，导致位姿 frame
> 在 rviz 里消失。

---

## 料台装配体场景（table.STEP）

`run_rviz.sh` 默认把料台 CAD 装配体和另一条臂一起显示出来。

**做法**：`table.STEP` 里原本装了两台 DOBOT Nova 2。先把这两台机器人从 STEP 里
剔除、只把料台/立柱/安装板导出成 `table.stl`，再按 CAD 里的挂载位姿把我们自己的
可动 XTrainer 装回那两个位置——一条由 `/joint_states` 驱动，另一条纯静态展示。

**坐标系**：URDF 的根仍然是 `LINK_0`（被规划那条臂的基座），料台通过挂载位姿的
**逆变换**挂到 `LINK_0` 下：

```text
LINK_0 --inv(M_active)--> scene_root --+--> workbench          (table.stl，纯 visual)
                                       +--> static_LINK_0 ...  (静态展示臂，关节全 fixed)
```

因此 rviz 的 Fixed Frame 依旧是 `LINK_0`，`play_trajectory_ros.py` 发布的
`/joint_states`、工作空间线框、墙体、start/goal tf 全部不需要任何改动；静态臂也
不需要往 `/joint_states` 里补关节角。

**相关文件**

| 文件 | 作用 |
|---|---|
| `config/cad_mounts.yaml` | 两个挂载位姿（CAD 装配体根系下，米/弧度） |
| `scripts/extract_cad_mounts.py` | 解析 STEP 装配结构，提取上面那两个位姿 |
| `scripts/export_table_mesh.py` | 剔除 STEP 里的 DOBOT 后导出 `table.stl` |
| `scripts/build_scene_urdf.py` | 把 `xtrainer.urdf` + 料台 + 静态臂拼成一份 URDF |
| `.../meshes/xtrainer/table.stl` | 已过滤的料台网格（毫米单位，URDF 里 `scale=0.001`） |

**重新生成 `table.stl`**（换了新版 STEP 时才需要，依赖 gmsh）

```bash
conda activate curobo && pip install gmsh
python3 scripts/export_table_mesh.py \
  ../src/curobo/content/assets/robot/ur_description/meshes/xtrainer/table.STEP \
  ../src/curobo/content/assets/robot/ur_description/meshes/xtrainer/table.stl
```

**重新提取挂载位姿**（换了新版 STEP 时才需要）

```bash
python3 scripts/extract_cad_mounts.py \
  ../src/curobo/content/assets/robot/ur_description/meshes/xtrainer/table.STEP \
  --emit-yaml
```

脚本输出可直接粘进 `config/cad_mounts.yaml`。DOBOT 子装配原点与 XTrainer `LINK_0`
原点之间的固定差量（局部平移 `[0, -0.002, -0.00541]` m + 绕局部 Z 转 180°）已由
`--extra-xyz` / `--extra-rpy` 默认值并入结果。

> 当前这份 STEP（2026-08-25 版）相对旧版已把两臂沿料台长边（X）**各向外移 40 mm**，
> `x` 从 `{-0.43877, 0.28123}` 变为 `{-0.47877, 0.32123}`，无需再额外平移。

---

## 输出文件

每次规划生成 `results/<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `trajectory.npz` | `joint_names, positions[N,6], velocities, accelerations, times[N], ee_positions[N,3], ee_quats_wxyz[N,4]` |
| `trajectory_meta.json` | 位姿序列、逐段统计与误差、工作空间/自碰撞/夹爪校验结果、墙 cuboid、完整配置快照 |
| `trajectory.csv` | 人类可读：`t, q_J_1..q_J_6, ee_xyz, ee_rpy(度)` |
| `plan_failed.json` | 仅规划失败时生成，含失败段与配置，便于排查 |

四元数顺序：npz 内为 **wxyz**（curobo 约定），发布到 ROS 时转成 **xyzw**。

---

## 规划失败时怎么办

脚本会打印失败段与 `status`。按以下顺序排查：

```bash
# 1) 先定位是「不可达 / 自碰撞 / 撞墙」中的哪一种
python scripts/diagnose_ik.py
```

`diagnose_ik.py` 对每个位姿分三档求 IK 并给出逐 link 碰撞报告：

| 现象 | 结论 |
|---|---|
| A(纯运动学) FAIL | 位姿运动学不可达，需改位置或姿态 |
| A OK、B(自碰撞) FAIL | 该位姿必然自碰撞，需改姿态 |
| B OK、C(加墙) FAIL | 撞墙。看逐 link 报告决定放宽 `bounds_override` 还是调 `collision_link_names` |
| C OK | 位姿可用，问题在段与段之间的路径上 |

```bash
# 2) 探测末端(含夹爪)在各目标位姿下的占据范围，得到建议的 bounds_override
python scripts/probe_ee_extent.py

# 3) 若墙需要约束 LINK_0/LINK_1，探测底座立柱包围盒以确定让位孔
python scripts/probe_base_spheres.py

# 4) 常用兜底手段
./run_plan.sh --enable-graph        # 启用图搜索(PRM)寻找绕行种子
./run_plan.sh --lift-z 0.05         # 减小抬升高度
./run_plan.sh --no-wall             # 暂时去掉硬墙，确认是否墙的问题
./run_plan.sh --max-attempts 120
```

> 诊断脚本也需在 conda curobo 环境下跑：
> `source ~/miniconda3/etc/profile.d/conda.sh && conda activate curobo`

---

## 姿态变体组合（start × goal 笛卡尔积）

`start` 与 `goal` 各自定义若干旋转增量，两侧做**笛卡尔积**，
每个组合 = 一个独立的 `(start, goal)` 任务 = 一次独立规划。

```yaml
task:
  start_variants:
    enable: true
    mode: right          # right=绕「工具自身轴」；left=绕「base 固定轴」
    rotations:
      - [-30, 0, 0]
      - [30, 0, 0]
      - [0, 0, 0]
    position_delta: null

  goal_variants:
    enable: true
    mode: right
    rotations:
      - [-30, 0, 0]
      - [30, 0, 0]
      - [0, 0, 0]
    position_delta: null   # 可选，位置也能一起偏移
```

上例 = 3 × 3 = **9 次规划**。

> `start` 变体启用后，该组合会强制用 IK 重解起始关节角
> （原先的 `start_joint_state` 已不对应新的起始位姿）。

### 右乘 / 左乘

| mode | 公式 | 含义 |
|---|---|---|
| `right`（默认） | \(R_{new} = R_{base} \cdot R_{delta}\) | 绕**工具自身**轴转（内旋） |
| `left` | \(R_{new} = R_{delta} \cdot R_{base}\) | 绕 **base 固定**轴转（外旋） |

增量自身仍按项目统一约定解释（固定轴 XYZ，`R = Rz·Ry·Rx`），只是作用在工具坐标系下。

### 运行

```bash
./run_plan.sh                                  # 跑全部组合
./run_plan.sh --variant 3                      # 只跑 3 号组合
./run_plan.sh --variant s0_rpy_m30_p0_p0       # 按 start 侧筛选（该 start 的全部 goal）
./run_plan.sh --variant g1_rpy_p30_p0_p0       # 按 goal 侧筛选
./run_plan.sh --goal-rotations '0,20,0'        # 命令行覆盖 goal 变体
./run_plan.sh --no-variants                    # 忽略变体，只按 start/goal 规划一次
```

### 输出结构

```
results/<时间戳>/
├── variants_summary.json                        # 全部组合的对比汇总
├── s0_rpy_m30_p0_p0__g0_rpy_m30_p0_p0/          # 组合名 = start名__goal名
│   ├── trajectory.npz
│   ├── trajectory_meta.json                     # meta.variant 含 start/goal 两侧信息
│   └── trajectory.csv
├── s0_rpy_m30_p0_p0__g1_rpy_p30_p0_p0/
└── ...
```

`MotionGen` 只构建一次、全部组合复用，因此第 2 个组合起 solve 时间显著下降。

### 实测结果（base start `(-180,0,0)`、base goal `(-90,0,-90)`）

```
#   start          goal           结果   点数  时长s  关节行程  往返  误差mm  越界 自碰  备注
0   [-30,0,0]      [-30,0,0]      OK     273   5.48    418.3  1.30  1.299    0   0  误差偏大, 有往返绕行, J_6贴限位x21
1   [-30,0,0]      [30,0,0]       OK     171   3.44    549.7  1.00  0.010    0   0
2   [-30,0,0]      [0,0,0]        OK     170   3.42    589.5  1.00  0.007    0   0
3   [30,0,0]       [-30,0,0]      OK     272   5.46    394.0  1.33  1.270    0   0  误差偏大, 有往返绕行, J_6贴限位x21
4   [30,0,0]       [30,0,0]       OK     171   3.44    598.2  1.05  0.007    0   0
5   [30,0,0]       [0,0,0]        OK     174   3.50    617.2  1.02  0.005    0   0
6   [0,0,0]        [-30,0,0]      OK     287   5.76    386.0  1.34  1.266    0   0  误差偏大, 有往返绕行, J_6贴限位x21
7   [0,0,0]        [30,0,0]       OK     186   3.74    562.0  1.00  0.009    0   0
8   [0,0,0]        [0,0,0]        OK     191   3.84    601.6  1.00  0.013    0   0
```

**关键规律：决定成败的是 `goal` 侧，`start` 侧几乎无影响。**

- **`goal = [-30,0,0]`（#0/#3/#6）全部有问题**：J6 贴限位 21 个点、误差 ~1.3mm、往返比 1.3。
  这三个组合的"关节行程最小"（386~418°）是假象 —— 省下的度数换来的是在限位边缘反复挣扎。
- **`goal = [30,0,0]` 或 `[0,0,0]`（其余 6 个）都很干净**：往返比 1.00~1.05、误差 <0.02mm、零贴限位。
- `start` 三个变体只影响关节行程 ±5%，不改变可达性。

因此 `goal` 下俯 30° 这个姿态处于 J6 工作范围边界，真机上不建议使用。

### 关节限位余量检测

每次规划都会检查各关节离限位的余量，贴限位会明确告警：

```
[LIMIT] 以下关节贴到限位(余量 < 1.0deg), 会导致末端精度下降:
         J_6: 21/273 点, 最小余量 -0.04deg, 限位 [-177.0, 177.0]deg
```

正常情况：

```
[LIMIT] 全部关节离限位余量 >= 1.0deg (最小 25.6deg @ J_1)
```

**贴限位是「规划成功但位姿误差偏大」的最常见原因** —— 关节转不动了，
优化器只能牺牲末端精度去凑。数据写入 meta 的 `joint_limit_margin`，
并在汇总表的「备注」列标出。

### 播放某个组合

```bash
./run_rviz.sh                                    # 自动选最新，并列出同批全部组合
./run_rviz.sh --traj results/<时间戳>/s1_rpy_p30_p0_p0__g2_rpy_p0_p0_p0
```

---

## 常用规划参数

```yaml
planner:
  self_collision_check: true          # 自碰撞检测（勿关，除非调试）
  collision_activation_distance: 0.015  # 碰撞代价激活距离(m)，越大越保守
  interpolation_dt: 0.02              # 轨迹点时间间隔(s)
  velocity_scale: 0.25                # 速度上限缩放，播放太快就调小
  num_trajopt_seeds: 48               # 种子数，越多越可能挑到关节位移更小的解
  trajopt_tsteps: 40
  max_attempts: 60
  enable_graph: false                 # 直线优化失败时启用 PRM 绕行
  position_threshold: 0.003           # 位置收敛阈值(m)
  rotation_threshold: 0.03            # 姿态收敛阈值(四元数误差)
  check_ik_branch: true               # 规划前预检 IK 分支跳变
  ik_branch_warn_deg: 90.0
```

机器人侧：

```yaml
robot:
  ee_link: LINK_6                     # 规划目标 link
  collision_sphere_buffer: 0.0        # 碰撞球统一膨胀量(m)，给真机留安全余量
  joint_limit_clip: 0.05              # 关节限位内缩(rad)，避免贴硬限位
  cspace_distance_weight: [5,5,5,5,5,5]  # 关节空间距离权重，见下节
```

单臂 pick/place 默认及悬挂预设现在使用 `joint_limit_clip: 0.15`：相对 URDF
原始下限加 0.15 rad、上限减 0.15 rad，IK 和规划共用，且只收缩一次。
图表中的“有效限位余量”是在这份内缩后再剩余的空间；详见
[0.15 rad 全量复核记录](docs/overhead_joint_margin_015_20260916.md)。上例为通用任务参数，历史运行仍以各自保存的配置为准。

---

## 关于「关节角变化最小」

**当前不以关节位移最小为主目标。** trajopt 的代价权重优先级为：

| 代价项 | 权重 |
|---|---|
| 世界碰撞 | 100000 |
| 位置 / 姿态误差 | 50000 / 2000 |
| 关节限位 | 50000 |
| 终点 cspace | 20000 |
| 自碰撞 | 5000 |
| 平滑度（抑制加速度） | `smooth_weight=[0,10000,5,0]` |
| **关节路径长度** | **仅用于多种子筛选** |

即目标是「无碰撞 + 精确到位 + 时间最优 + 加速度平滑」，关节位移最小只是**弱偏好**：
多个种子各自收敛后，用 `path_length = Σ(|关节速度| × cspace_distance_weight)`
挑一个最短的（`evaluator.compute_path_length_cost`）。

### 两个可调旋钮

```bash
./run_plan.sh --num-trajopt-seeds 48                      # 更多候选解
./run_plan.sh --cspace-weight 5 5 5 5 5 5                 # 调整各关节权重
```

`cspace_distance_weight` 覆盖在本任务配置里，**不改**共享的
`src/curobo/content/configs/robot/xtrainer.yml`（`marvin_arm/sample_xtrainer.py` 与
`relative_ee_analysis/analyze_relative_ee.py` 也在用它）。

### 重要局限（实测结论）

这两个旋钮**只能影响「从已收敛的种子里挑哪一个」，无法改变各路点的终点关节角** ——
终点由 IK 在 trajopt 之前就确定了，所有种子共享同一个终点。

默认位姿下的实测数据（总行程 = 沿轨迹累计，含往返）：

| 配置 | 总行程 | J2 | J3 | J4 | J6 |
|---|---|---|---|---|---|
| 基线 权重`[2.5,1.75,1.5,1.25,.7,.4]` 12 种子 | 411.2° | 24.7 | 34.5 | 52.5 | **275.7** |
| 同权重 24 种子 | 409.0° | 24.7 | 34.0 | 52.0 | **275.7** |
| 同权重 48 种子 | 405.5° | 24.4 | 32.7 | 49.6 | **275.7** |
| 权重`[5,5,5,5,5,5]` 48 种子 | **400.2°** | 24.4 | 32.7 | 48.3 | **275.7** |
| 权重把 J6 提到 3.0 或 8.0 | 409.5° / 405.0° | — | — | — | **275.7** |

两个关键结论：

1. **改善幅度有限（411°→400°，约 2.7%）**，因为可压缩的只是 J2/J3/J4 的往返绕行。
2. **J6 的 275.7° 完全无法优化** —— 无论权重怎么调都恒定不变。原因是它源自 IK 解本身：
   `start` 的 `J6=+146.6°`、`goal` 的 `J6=-129.0°`，走反向短边需要终点变成 `+231°`，
   而 **J6 限位仅 ±177°**，`231°` 超限，因此只能走长边穿过 0°。这是**运动学必需**。

规划时会自动识别并提示这种情况：

```
[IK-CHK] 警告: 段 [2] 的关节跳变超过 90deg。
         其中以下关节的大幅转动是运动学必需的, 无法通过调参优化:
           - J_6(转 276deg, 限位 [-177,177]deg 内无 ±360 等价短边 -> 运动学必需)
```

每次规划都会输出关节运动量统计（同时写入 meta 的 `joint_motion`）：

```
[MOTION] 关节运动量统计:
         joint      净变化(deg)   总行程(deg)
         J_4               5.9         48.3   <== 行程远大于净变化(有往返绕行)
         J_6             275.7        275.7
         合计              311.7        407.9
```

`总行程 ≫ 净变化` 说明该关节有往返绕行，属于可优化部分；两者接近则已是最短路径。

### 若确实需要更强的约束

- **改目标姿态**：J6 大转动的根源是起终点姿态的 IK 分支不同，调整 `goal.rpy_deg`
  可能找到同分支解（参考上文「规划失败时怎么办」的 yaw 敏感性分析）。
- **指定 `start_joint_state`**：直接给出与目标同分支的起始关节角，绕开 IK 选解。
- **关节空间目标**：若已知目标关节角，`plan_single_js` 走关节空间插值，天然关节位移最小。
- **约束笛卡尔路径**：curobo 的 `PoseCostMetric` 可强制末端走直线或锁定某姿态轴，
  间接减少关节绕行（当前未启用）。

---

## 目录结构

```
xtrainer_plan/
├── config/
│   ├── task_default.yaml          # 全部可调参数（位姿/路点/工作空间/规划器）
│   └── xtrainer_traj.rviz         # rviz 配置
├── launch/
│   └── display_xtrainer_traj.launch
├── scripts/
│   ├── xtrainer_common.py         # rpy↔四元数、墙体构造、轨迹 IO（两环境通用）
│   ├── plan_trajectory.py         # curobo 分段规划（conda curobo）
│   ├── play_trajectory_ros.py     # ROS1 播放（ROS py3.8）
│   ├── diagnose_ik.py             # IK 失败原因诊断
│   ├── probe_ee_extent.py         # 探测夹爪占据范围 -> bounds_override
│   └── probe_base_spheres.py      # 探测底座立柱包围盒 -> base_clearance
├── results/<时间戳>/               # 规划输出
├── run_plan.sh
└── run_rviz.sh
```

---

## 当前基线结果

用默认配置（起始 `(-0.31,-0.05,0.20)` rpy `(-135.56,-0.93,-8.9)`，
目标 `(-0.34,-0.09,0.22)` rpy `(177,53,-84)`）：

```
段 1/3 start      -> start_lift   39 点  0.78s  误差 0.01mm / 0.00deg
段 2/3 start_lift -> goal_lift   114 点  2.28s  误差 0.00mm / 0.00deg
段 3/3 goal_lift  -> goal         39 点  0.78s  误差 0.00mm / 0.00deg
总计 190 点，3.82s
工作空间: 190/190 合法
自碰撞  : 190/190 无碰撞
夹爪    : 相对 bounds 最大越界 119.9mm（朝下抓取姿态的正常现象，仅供参考）
```

链路一致性已验证：`rosrun tf tf_echo LINK_0 xtrainer_goal` 返回
`[-0.340, -0.090, 0.220]` / `RPY(degree) [177.000, 53.000, -84.000]`，与输入逐位一致；
`tf_echo xtrainer_ee_cmd LINK_6` 为零平移单位四元数，说明 curobo FK 与 URDF FK 完全吻合。

---

## 已知说明

- 使用**标称** URDF `xtrainer.urdf`（6 关节限位均为 ±3.14）。若要换成标定版
  `xtrainer_cali.urdf`（J2~J5 收紧至 ±1.92/2.53/2.46/2.51），需新建一份 robot yml
  指向它，再改 `robot.robot_yml`。
- URDF 里 mesh 路径写的是 `package://robotics/drivers/dobot/...`，ROS 无法解析，
  launch 文件用 `sed` 替换成本地绝对路径。
- 默认起始与目标相距仅 5.4cm。若实际是"抓取后搬运到较远处放置"，请复核目标位置。
