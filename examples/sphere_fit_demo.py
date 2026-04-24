import os
import numpy as np
import trimesh

# ===== 关键调参区 =====
# 沿主轴的球数量（细长连杆 3~8 个即可严格包络）
N_SPHERES = 6
# 半径膨胀因子：在截面外接圆半径基础上再乘该系数，>=1.0 保证覆盖
# 1.0 = 刚好外接，1.05~1.15 = 略微放大避免数值误差造成漏检
RADIUS_INFLATE = 1.05
# 端部额外留白：沿主轴在两端各延伸的比例（相对主轴长度），0 = 不延伸
END_PADDING_RATIO = 0.02
# =====================

MESH_PATH = os.path.join(
    os.path.dirname(__file__),
    "../src/curobo/content/assets/robot/jaka_description/meshes/LINK_6.STL",
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sphere_fit_skeleton.glb")


def compute_principal_axis(vertices: np.ndarray) -> tuple:
    """用 PCA 求点云的主轴方向（最长的主成分方向）。

    Returns:
        (axis_unit, centroid): 主轴单位向量 (3,) 和点云质心 (3,)。
    """
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    # SVD 分解，V 的第一列就是最大方差方向
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis_unit = vt[0] / np.linalg.norm(vt[0])
    return axis_unit, centroid


def fit_spheres_along_skeleton(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    radius_inflate: float = 1.05,
    end_padding_ratio: float = 0.02,
) -> tuple:
    """沿 mesh 主轴（骨架）等距放球，每个球半径 = 截面外接圆半径。

    Args:
        mesh: 输入 mesh
        n_spheres: 沿主轴切多少片 = 放多少球
        radius_inflate: 半径膨胀系数（>=1.0 保证包络）
        end_padding_ratio: 两端延伸比例（相对主轴长度）

    Returns:
        (centers, radii): 球心 (N,3)，半径 (N,)
    """
    vertices = mesh.vertices  # (V, 3)
    axis, centroid = compute_principal_axis(vertices)

    # 把所有顶点投影到主轴上，取投影值的 min/max 作为主轴长度范围
    proj = (vertices - centroid) @ axis  # (V,)
    t_min, t_max = proj.min(), proj.max()
    length = t_max - t_min
    # 两端各延伸一点，避免端部切片无顶点
    t_min -= length * end_padding_ratio
    t_max += length * end_padding_ratio

    # 沿主轴等距取 n_spheres 个切片中心
    t_samples = np.linspace(t_min, t_max, n_spheres)

    centers = np.zeros((n_spheres, 3))
    radii = np.zeros(n_spheres)

    # 每片的"厚度"，用于决定哪些顶点属于这一片
    slab_half = (t_max - t_min) / (n_spheres - 1) / 2 if n_spheres > 1 else length

    for i, t in enumerate(t_samples):
        # 找出在该主轴位置附近（slab）内的顶点
        mask = np.abs(proj - t) <= slab_half
        if not mask.any():
            # 该切片没顶点，退化到用主轴上的点 + 全局最小半径
            centers[i] = centroid + t * axis
            radii[i] = 0.01
            continue

        slab_pts = vertices[mask]
        # 球心：该切片所有顶点在主轴上的投影中点，加上主轴方向偏移
        # 这里用"顶点到主轴的垂足均值"作为更稳定的球心
        center_on_axis = centroid + t * axis
        # 半径：切片内所有顶点到 center_on_axis 的最大距离 = 严格包络该切片
        dists = np.linalg.norm(slab_pts - center_on_axis, axis=1)
        radii[i] = dists.max() * radius_inflate
        centers[i] = center_on_axis

    return centers, radii


# ========= 主流程 =========
mesh = trimesh.load(MESH_PATH, force="mesh")
bbox_size = mesh.bounds[1] - mesh.bounds[0]
print(f"mesh bbox size (xyz): {bbox_size}")
print(f"longest axis: {bbox_size.max():.4f} m, "
      f"shortest axis: {bbox_size.min():.4f} m")

centers, radii = fit_spheres_along_skeleton(
    mesh,
    n_spheres=N_SPHERES,
    radius_inflate=RADIUS_INFLATE,
    end_padding_ratio=END_PADDING_RATIO,
)

print(f"\nfit {len(centers)} spheres along principal axis:")
for i, (c, r) in enumerate(zip(centers, radii)):
    print(f"  sphere {i}: center=[{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}], "
          f"radius={r:.4f}")

# 覆盖率检查：统计有多少 mesh 顶点被球覆盖
covered = np.zeros(len(mesh.vertices), dtype=bool)
for c, r in zip(centers, radii):
    d = np.linalg.norm(mesh.vertices - c, axis=1)
    covered |= (d <= r)
coverage = covered.mean() * 100
print(f"\ncoverage: {coverage:.2f}% of mesh vertices inside spheres")
if coverage < 99.9:
    print(f"[warn] {len(mesh.vertices) - covered.sum()} vertices NOT covered. "
          f"Try increasing N_SPHERES or RADIUS_INFLATE.")

# ========= 可视化 =========
scene = trimesh.Scene()
mesh.visual.face_colors = [200, 200, 200, 180]
scene.add_geometry(mesh)
for c, r in zip(centers, radii):
    s = trimesh.creation.icosphere(radius=float(r), subdivisions=2)
    s.apply_translation(c)
    s.visual.face_colors = [255, 0, 0, 100]
    scene.add_geometry(s)

try:
    scene.show()
except (ModuleNotFoundError, ImportError, RuntimeError) as e:
    print(f"\n[warn] interactive viewer unavailable ({type(e).__name__}: {e})")
    scene.export(OUTPUT_PATH)
    print(f"[info] scene exported to: {OUTPUT_PATH}")