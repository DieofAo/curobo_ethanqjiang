#!/usr/bin/env python3
"""Audit and plot all 126 V64/V67 local-Y smoke mount layouts.

Only existing completed results are read; no planner run is performed. The
0-degree controls occur once, in V64. The output heatmap shares a 0..9 scale.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from plot_grasp_angle_map import pick_cjk_font


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "xtrainer_plan/results_overhead/20260928"
SOURCES = (
    ("V64", BASE / "v64_near_zero_x_local_ytilt_joint_home_smoke", 72),
    ("V67", BASE / "v67_near_zero_x_local_y_negative_smoke_retry03", 54),
)
OUT = BASE / "signed_local_y_full_comparison/smoke"
ANGLES = (-60.0, -45.0, -30.0, 0.0, 30.0, 45.0, 60.0)
XS = (-0.2, -0.1)
YS = (0.15, 0.35, 0.45)
ZS = (0.45, 0.55, 0.65)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> tuple[list[dict], dict]:
    rows = []
    origins = {}
    source_sha = None
    runtime_inputs = None
    for label, root, expected_count in SOURCES:
        manifest_path = root / "manifest.json"
        status_path = root / "sweep_status.json"
        manifest, status = read(manifest_path), read(status_path)
        require(status["manifest_sha256"] == sha(manifest_path),
                f"{label}: manifest changed after sweep")
        require(manifest["n_configs"] == len(manifest["candidates"]) == expected_count,
                f"{label}: unexpected candidate count")
        require(len(status["outcomes"]) == expected_count,
                f"{label}: missing outcome")
        require(manifest["parameters"]["tilt_axis"] == "original_base_local_y"
                and manifest["parameters"]["size"] == 3,
                f"{label}: incompatible axis or smoke grid")
        require(sha(Path(manifest["source"])) == manifest["source_sha256"],
                f"{label}: source configuration changed")
        if source_sha is None:
            source_sha, runtime_inputs = manifest["source_sha256"], manifest["runtime_inputs"]
        else:
            require(manifest["source_sha256"] == source_sha
                    and manifest["runtime_inputs"] == runtime_inputs,
                    "V64/V67 planner inputs differ")
        for item in manifest["candidates"]:
            name = item["name"]
            outcome = status["outcomes"][name]
            config = Path(item["config"])
            require(sha(config) == item["config_sha256"] == outcome["config_sha256"],
                    f"{name}: config hash mismatch")
            require(outcome["name"] == name and outcome["result"] == item["result"]
                    and outcome["n_total"] == 9,
                    f"{name}: outcome metadata mismatch")
            count = int(outcome["n_success"])
            state = outcome["state"]
            require(0 <= count <= 9 and
                    state == ("completed_zero_success" if count == 0 else "completed_verified"),
                    f"{name}: not a completed smoke result")
            if count:
                result = Path(item["result"])
                for audit_name in ("independent_verification.json", "joint_limit_clip_audit.json"):
                    audit = read(result / audit_name)
                    require(audit.get("passed") is True and
                            audit.get("verification_completed") is True,
                            f"{name}: {audit_name} not verified")
            xyz = tuple(float(v) for v in item["base_xyz_m"])
            angle = float(item["local_y_tilt_deg"])
            require(item["tilt_axis"] == "original_base_local_y" and
                    xyz in {(x, y, z) for x in XS for y in YS for z in ZS} and
                    angle in ANGLES and (label == "V64") == (angle >= 0),
                    f"{name}: unexpected mount or signed angle")
            rows.append({"source": label, "name": name, "base_x_m": xyz[0],
                         "base_y_m": xyz[1], "base_z_m": xyz[2],
                         "local_y_tilt_deg": angle, "n_success": count,
                         "n_total": 9, "state": state, "config_sha256": sha(config),
                         "result": item["result"]})
        origins[label] = {"manifest": str(manifest_path),
                          "manifest_sha256": sha(manifest_path),
                          "status": str(status_path),
                          "status_sha256": sha(status_path)}
    lookup = {(r["base_x_m"], r["base_y_m"], r["base_z_m"], r["local_y_tilt_deg"]): r
              for r in rows}
    require(len(rows) == len(lookup) == 126, "Duplicate signed mount")
    require(set(lookup) == {(x, y, z, a) for x in XS for y in YS for z in ZS for a in ANGLES},
            "Missing mount in the complete 2×3×3×7 grid")
    rows.sort(key=lambda r: (r["base_x_m"], r["base_z_m"], r["base_y_m"],
                             r["local_y_tilt_deg"]))
    by_angle = {str(int(a)): sum(r["n_success"] for r in rows
                                 if r["local_y_tilt_deg"] == a) for a in ANGLES}
    summary = {"schema_version": 1, "n_layouts": len(rows),
               "n_xyz_positions": 18, "n_targets_per_layout": 9,
               "by_angle_success": by_angle,
               "negative_success": sum(by_angle[str(int(a))] for a in ANGLES if a < 0),
               "zero_success": by_angle["0"],
               "positive_success": sum(by_angle[str(int(a))] for a in ANGLES if a > 0),
               "n_completed_targets": sum(r["n_success"] for r in rows),
               "n_attempted_targets": 126 * 9,
               "sources": origins,
               "note": "Success counts are completed 3×3 pick/place trajectories, not full 20×20 coverage."}
    require((summary["negative_success"], summary["zero_success"],
             summary["positive_success"], summary["n_completed_targets"])
            == (317, 102, 214, 633), "Totals differ from source signed comparison")
    return rows, summary


def plot(rows: list[dict], output: Path) -> None:
    font = pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    lookup = {(r["base_x_m"], r["base_y_m"], r["base_z_m"], r["local_y_tilt_deg"]):
              r["n_success"] for r in rows}
    colors = ["#e5e7eb", "#d8e7f4", "#a9d1e8", "#79bdd4", "#4ba9b2",
              "#70bf83", "#a4cf63", "#d4db4f", "#f0c74b", "#e88742"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([n - .5 for n in range(11)], cmap.N)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for i, x in enumerate(XS):
        for j, z in enumerate(ZS):
            ax = axes[i, j]
            matrix = [[lookup[(x, y, z, a)] for a in ANGLES] for y in YS]
            im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for k, y in enumerate(YS):
                best = max(matrix[k])
                for l, angle in enumerate(ANGLES):
                    value = matrix[k][l]
                    ax.text(l, k, str(value), ha="center", va="center", fontsize=12,
                            color="#17212b", fontweight="bold" if value == best else "normal")
                    if value == best:
                        ax.add_patch(Rectangle((l-.48, k-.48), .96, .96,
                                               fill=False, edgecolor="#7e3f16", lw=2.3))
            ax.axvline(2.5, color="#52647a", lw=1.8)
            ax.axvline(3.5, color="#52647a", lw=1.8)
            ax.set_xticks(range(7), [f"{a:+g}°" if a else "0°" for a in ANGLES])
            ax.set_yticks(range(3), [f"{y:.2f}" for y in YS])
            ax.set_xlabel("原始基座局部 +Y 倾角")
            ax.set_ylabel("基座 Y (m)")
            ax.set_title(f"基座 X={x:+.2f} m, Z={z:.2f} m", fontsize=12)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=range(10), shrink=.85,
                      fraction=.025, pad=.02)
    cb.set_label("3×3 完整抓放成功数 / 9")
    fig.suptitle("V64/V67 共 126 组冒烟：同一 XYZ 下比较 −60° 到 +60°；棕框为该位置并列最高",
                 fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing overwrite: {OUT}")
    rows, summary = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".smoke_126_", dir=OUT.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        with (stage / "smoke_126_layouts.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (stage / "smoke_126_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        plot(rows, stage / "smoke_126_heatmap.png")
        (stage / "README.md").write_text(
            "# 126 组局部 +Y 冒烟总图\n\n"
            "每格是相同基座 XYZ、指定带符号倾角下的 3×3 完整抓放成功数，0° 只计一次。"
            "六个面板按 X/Z 分开，面板内三行是基座 Y，七列是倾角。"
            "所有格子共享 0–9 色标；棕框标注该 XYZ 七种倾角中的最高成功数（并列均标）。"
            "0/9 表示九点均已尝试但没有完整轨迹。九点冒烟不等于 20×20 全量覆盖。\n\n"
            "复现：`python3 xtrainer_plan/scripts/report_signed_local_y_smoke.py`。"
            "脚本要求 V64/V67 126 个配置状态均完成，逐组配置哈希和有轨迹组的两项审计通过，"
            "并拒绝覆盖已有输出。\n", encoding="utf-8")
        (stage / "artifact_sha256.json").write_text(json.dumps(
            {p.name: sha(p) for p in stage.iterdir() if p.is_file()},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.rename(OUT)
    print(f"[ok] {OUT}: {summary['n_completed_targets']}/"
          f"{summary['n_attempted_targets']} over {summary['n_layouts']} layouts")


if __name__ == "__main__":
    main()
