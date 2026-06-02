"""遍历每对球, 找 retract 时穿透量最大的几对 (用球->link 名称做归因)"""
import torch
import numpy as np
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

cfg = load_yaml(join_path(get_robot_configs_path(), "marvin_left_arm.yml"))
cfg["robot_cfg"]["kinematics"]["ee_link"] = "Link7_L"
cfg["robot_cfg"]["kinematics"]["link_names"] = ["Link7_L"]
rwc = RobotWorldConfig.load_from_config(robot_config=cfg, world_model=None)
rw = RobotWorld(rwc)
km = rw.kinematics

# 球索引->link 映射先跳过, 直接看几何重叠


q0 = torch.zeros(1, 7, device="cuda:0")
st = km.get_state(q0)
sph = st.link_spheres_tensor[0].detach().cpu().numpy()  # (n_sph, 4) xyz r
print("spheres at q=0:")
for i, s in enumerate(sph):
    print(f"  [{i:2d}] xyz=({s[0]:+.3f},{s[1]:+.3f},{s[2]:+.3f})  r={s[3]:.3f}")

n = sph.shape[0]
pairs = []
for i in range(n):
    for j in range(i+1, n):
        d = np.linalg.norm(sph[i, :3] - sph[j, :3])
        pen = (sph[i, 3] + sph[j, 3]) - d  # >0 = 穿透
        if pen > 0 and sph[i, 3] > 0 and sph[j, 3] > 0:
            pairs.append((pen, i, j, d, sph[i, 3], sph[j, 3]))
pairs.sort(reverse=True)
print(f"\n穿透对总数 (r>0 的有效球之间): {len(pairs)}")
print("Top-15 穿透:")
for pen, i, j, d, ri, rj in pairs[:15]:
    print(f"  i={i:2d} j={j:2d}  pen={pen:.3f}  d={d:.3f}  ri={ri:.3f}  rj={rj:.3f}")
