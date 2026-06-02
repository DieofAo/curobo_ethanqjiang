"""临时诊断: 看 marvin 左臂在 retract / 随机位姿下的 d_self 分布"""
import torch
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

cfg = load_yaml(join_path(get_robot_configs_path(), "marvin_left_arm.yml"))
cfg["robot_cfg"]["kinematics"]["ee_link"] = "Link7_L"
cfg["robot_cfg"]["kinematics"]["link_names"] = ["Link7_L"]
rwc = RobotWorldConfig.load_from_config(robot_config=cfg, world_model=None)
rw = RobotWorld(rwc)
km = rw.kinematics
print("n_spheres =", km.kinematics_config.total_spheres)
print("self_coll_matrix shape =", km.kinematics_config.self_collision_data.collision_matrix.shape if hasattr(km.kinematics_config, "self_collision_data") else "n/a")

# 1) retract (q=0)
q0 = torch.zeros(1, 7, device="cuda:0")
st = km.get_state(q0)
sph = st.link_spheres_tensor.unsqueeze(1)
print("sph shape:", sph.shape, "  (B, H, n_sph, 4)")
d = rw.get_self_collision_distance(sph)
print(f"d_self @ q=0 : {d.detach().cpu().numpy()}  (>0=穿透, <=0=无碰撞)")

# 2) 10 组随机
torch.manual_seed(0)
lim = km.get_joint_limits().position
low, high = lim[0], lim[1]
q = low + (high - low) * torch.rand(10, 7, device="cuda:0")
st = km.get_state(q)
sph = st.link_spheres_tensor.unsqueeze(1)
d = rw.get_self_collision_distance(sph).detach().cpu().numpy().reshape(-1)
print(f"d_self 10 rand: {d}")
print(f"  min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}")
print(f"  d>0:{(d>0).sum()}, d==0:{(d==0).sum()}, d<0:{(d<0).sum()}")
