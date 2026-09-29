#!/usr/bin/env python3
"""Render and audit the V67 best-negative-angle, 18-position walkthrough.

Reuses the V64 walkthrough's world drawing, ffmpeg encoding and frame audit.
V67-specific labels, negative angles, gray incomplete-grid points and source
provenance are supplied here. Run only after postprocess_v67_negative_smoke.py.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from postprocess_v67_negative_smoke import DATE_DIR, validate_complete_smoke


DEFAULT_ROOT = DATE_DIR / "v67_near_zero_x_local_y_negative_smoke"
LEGACY_RENDERER = (DATE_DIR / "v64_near_zero_x_local_ytilt_joint_home_smoke"
                   / "best_angle_3d/video/render_walkthrough.py")
VIDEO_NAME = "v67_negative_best_angle_18_positions_walkthrough.mp4"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(root):
    validate_complete_smoke(root)
    source = root / "best_angle_3d/best_angle_by_position.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if (data["n_positions"] != 18 or data["n_smoke_configs"] != 54 or
            data["rotation_axis"] != "original base local +Y" or
            data["coordinate_frame"] != "task_world" or len(data["points"]) != 18):
        raise ValueError("Expected audited V67 best-angle data for 18 XYZ × 3 negative tilts")
    for name, digest in data["source_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Best-angle source changed after report generation: {name}")
    points = []
    for item in data["points"]:
        x, y, z = item["base_xyz_m"]
        choice = item["choice"]
        if choice is not None:
            if (choice["n_total"] != 9 or choice["n_success"] < 0 or
                    choice["n_success"] > 9 or
                    choice["local_y_tilt_deg"] not in (-30.0, -45.0, -60.0)):
                raise ValueError(f"Invalid selected V67 smoke result: {choice}")
            choice = dict(choice, success_rate=choice["n_success"] / 9)
        elif item["status"] != "no_grid_result":
            raise ValueError(f"Missing selected case at tested point: {item}")
        points.append({"base_x_m": x, "base_y_m": y, "base_z_m": z,
                       "status": item["status"], "choice": choice,
                       "tied_best_angles_deg": item["same_success_angles_deg"],
                       "n_grid_results": item["n_grid_results"]})
    if (len({(p["base_x_m"], p["base_y_m"], p["base_z_m"]) for p in points}) != 18 or
            sorted({p["base_x_m"] for p in points}) != [-.2, -.1] or
            any(sum(p["base_x_m"] == x for p in points) != 9 for x in (-.2, -.1))):
        raise ValueError("Expected 18 unique, 9-per-X V67 base positions")
    if sum(p["choice"] is None for p in points) != data["n_no_grid_result_positions"]:
        raise ValueError("Gray point count differs from V67 report")
    data["n_untested_positions"] = data["n_no_grid_result_positions"]
    data["n_tied_positions"] = sum(len(p["tied_best_angles_deg"]) > 1 for p in points)
    data["n_tested_positions"] = sum(p["choice"] is not None for p in points)
    return source, data, points


def import_legacy(path):
    spec = importlib.util.spec_from_file_location("v64_walkthrough_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V64 renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_v67_views(old):
    def draw_matrix(ax, points, active_x, selected_index=None):
        ax.set_facecolor("#f2f7fa")
        group = [(index, point) for index, point in enumerate(points)
                 if point["base_x_m"] == active_x]
        if len(group) != 9:
            raise ValueError("Expected nine positions in each X cross-section")
        ax.set_xlim(.09, .51)
        ax.set_ylim(.405, .695)
        ax.set_xticks([.15, .35, .45])
        ax.set_yticks([.45, .55, .65])
        ax.grid(color="#c9d8df", linestyle=":", linewidth=1)
        ax.set_xlabel("Y / m", fontsize=10.5)
        ax.set_ylabel("Z / m", fontsize=10.5)
        ax.tick_params(labelsize=10)
        ax.set_title(f"侧视定位：X = {active_x:+.2f} m 的 9 个基座点",
                     fontsize=12.5, color=old.INK, pad=9)
        for index, point in group:
            y, z = point["base_y_m"], point["base_z_m"]
            chosen = point["choice"]
            if index == selected_index:
                ax.scatter([y], [z], s=1510, facecolors="none",
                           edgecolors=old.PINK, linewidths=3.2, zorder=4)
            ax.scatter([y], [z], s=950,
                       c=[old.PALETTE(chosen["success_rate"]) if chosen else old.GRAY],
                       edgecolors="white", linewidths=2.0, zorder=5)
            label = f"{chosen['n_success']}/9" if chosen else "无网格"
            ax.text(y, z, label, ha="center", va="center", fontsize=11.2,
                    color="white" if chosen and chosen["n_success"] >= 7 else old.NAVY,
                    fontweight="bold", zorder=6)
        for spine in ax.spines.values():
            spine.set_color("#abc2cc")

    def draw_legend(fig, data):
        fig.text(.035, .119, "点色 = 每个 XYZ 所选负角的 3×3 冒烟成功数",
                 fontsize=11.3, fontweight="bold", color=old.INK)
        labels = (("0/9", 0), ("3/9", 3), ("6/9", 6), ("9/9", 9))
        for j, (label, n) in enumerate(labels):
            x = .35 + j * .105
            fig.patches.append(old.Circle((x, .121), .009, transform=fig.transFigure,
                                          facecolor=old.PALETTE(n / 9), edgecolor=old.NAVY, lw=.6))
            fig.text(x + .014, .119, label, fontsize=10.5, color=old.INK, va="center")
        fig.patches.append(old.Circle((.79, .121), .009, transform=fig.transFigure,
                                      facecolor=old.GRAY, edgecolor=old.NAVY, lw=.6))
        fig.text(.806, .119, f"无完整九点网格 {data['n_no_grid_result_positions']} 个",
                 fontsize=10.1, color=old.INK, va="center")
        fig.text(.035, .069,
                 "浅青 = grasp 抓取区域 · 粉色 X = 固定 place · 箭头 = 基座局部 +Z 的实际方向",
                 fontsize=10.2, color=old.MUTED)
        fig.text(.035, .039, "倾角绕原始基座局部 +Y 轴；坐标为 task_world，单位米。",
                 fontsize=9.2, color=old.MUTED)

    def render_detail(data, points, index, out):
        point = points[index]
        chosen = point["choice"]
        x, y, z = point["base_x_m"], point["base_y_m"], point["base_z_m"]
        fig = old.fig_base("V67  |  负角基座逐点巡览",
                           "18 个 XYZ × -30° / -45° / -60°；每个位置展示冒烟成功数最高的版本")
        ax = fig.add_axes([.018, .151, .625, .735], projection="3d", facecolor=old.BG)
        old.draw_world(ax, data, points, selected_index=index,
                       elev=24, azim=-63 if x < -.15 else -113)
        if chosen is None:
            ax.scatter([x], [y], [z], s=1150, facecolors="none",
                       edgecolors=old.PINK, linewidths=2.8, depthshade=False)
        fig.text(.044, .858, f"截面 X = {x:+.2f} m  ·  本截面 {index % 9 + 1}/9",
                 fontsize=13, color=old.TEAL, fontweight="bold")
        old.right_panel(fig)
        fig.text(.67, .835, f"位置 {index + 1:02d} / 18", fontsize=18.5,
                 color=old.INK, fontweight="bold")
        fig.text(.67, .783, f"XYZ  ({x:+.2f}, {y:+.2f}, {z:+.2f}) m",
                 fontsize=16.5, color=old.NAVY)
        if chosen:
            fig.text(.67, .727, "所选局部 +Y 负倾角", fontsize=14, color=old.MUTED)
            fig.text(.67, .672, f"{chosen['local_y_tilt_deg']:g}°",
                     fontsize=34, fontweight="bold", color=old.TEAL)
            fig.text(.797, .684, f"case {chosen['name']}", fontsize=12, color=old.MUTED)
            fig.text(.67, .620, "完整抓放成功", fontsize=14, color=old.MUTED)
            fig.text(.67, .563, f"{chosen['n_success']} / 9", fontsize=34,
                     fontweight="bold", color=old.PALETTE(chosen["success_rate"]))
            ties = "、".join(f"{angle:g}°" for angle in point["tied_best_angles_deg"])
            fig.text(.67, .511, f"同成功数倾角：{ties}", fontsize=12.5,
                     color="#8d4772" if len(point["tied_best_angles_deg"]) > 1 else old.MUTED)
        else:
            fig.text(.67, .700, "无完整九点抓取网格结果", fontsize=17,
                     color=old.GRAY, fontweight="bold")
            fig.text(.67, .623, "本位置三个负角均无法计算成功率", fontsize=12.5,
                     color=old.MUTED)
            fig.text(.67, .549, "不等同于已完成的 0/9", fontsize=12.5,
                     color=old.MUTED)
        ax2 = fig.add_axes([.676, .21, .268, .27])
        old.draw_matrix(ax2, points, x, selected_index=index)
        draw_legend(fig, data)
        fig.savefig(out, dpi=120, facecolor=old.BG)
        old.plt.close(fig)

    def render_overview(data, points, azim, out, mode="orbit"):
        if mode == "orbit":
            fig = old.fig_base("V67  |  18 个负角基座位置的三维巡览",
                               "先旋转观察整体，再按 X 截面逐点展示全部 18 个安装位置")
        else:
            fig = old.fig_base("V67  |  18 / 18 个位置均已展示",
                               "点色为各位置最高冒烟成功数；灰色为没有完整九点网格结果的位置")
        ax = fig.add_axes([.015, .15, .70, .73], projection="3d", facecolor=old.BG)
        old.draw_world(ax, data, points, elev=24 if mode == "orbit" else 28, azim=azim)
        old.right_panel(fig)
        if mode == "orbit":
            lines = [("空间参照", 20, .822), ("18 个基座位置", 24, .755),
                     ("每处测试 -30° / -45° / -60°", 14, .699),
                     ("点色：最高成功数 / 9", 14, .633),
                     (f"灰色：无完整网格 {data['n_no_grid_result_positions']} 个", 13.5, .578),
                     ("箭头：基座局部 +Z 方向", 13, .509),
                     ("grasp：浅青色平面", 13, .447),
                     ("place：粉色 X", 13, .385),
                     ("下段按两个 X 截面依次查看", 12.7, .270)]
        else:
            success = [p["choice"]["n_success"] for p in points if p["choice"]]
            span = f"最高 {max(success)}/9，最低 {min(success)}/9" if success else "无完整网格"
            lines = [("逐点巡览完成", 20, .821), ("18 / 18 个位置", 24, .756),
                     (f"54 / 54 组冒烟已有终态；{data['n_grids_attempted']} 组完成九点网格", 13, .685),
                     (f"{data['n_tied_positions']} 个位置成功数并列", 14, .625),
                     (span, 14, .566),
                     ("九点冒烟仅用于布置筛选；", 14, .472),
                     ("不能当作 20×20 全量覆盖率。", 14, .426),
                     ("数据：V67 独立审核结果", 12.5, .300)]
        for label, size, y in lines:
            fig.text(.67, y, label, fontsize=size,
                     color=old.TEAL if size >= 24 else old.INK,
                     fontweight="bold" if size >= 20 else "normal")
        draw_legend(fig, data)
        fig.savefig(out, dpi=120, facecolor=old.BG)
        old.plt.close(fig)

    old.draw_matrix = draw_matrix
    old.render_detail = render_detail
    old.render_overview = render_overview


def make_contact_sheet(old, points):
    thumbs = []
    for index, point in enumerate(points):
        image = old.Image.open(old.STILLS / f"point_{index + 1:02d}.png").convert("RGB")
        image.thumbnail((576, 324))
        tile = old.Image.new("RGB", (590, 362), "#e5edf2")
        tile.paste(image, ((590 - image.width) // 2, 4))
        chosen = point["choice"]
        label = (f"{chosen['name']}  {chosen['n_success']}/9  "
                 f"{chosen['local_y_tilt_deg']:g} deg" if chosen else "无完整九点网格")
        old.ImageDraw.Draw(tile).text((12, 331), f"{index + 1:02d}  {label}", fill=old.NAVY)
        thumbs.append(tile)
    sheet = old.Image.new("RGB", (590 * 3, 362 * 6), old.BG)
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % 3) * 590, (index // 3) * 362))
    output = old.HERE / "all_18_detail_frames_contact_sheet.jpg"
    sheet.save(output, quality=90, optimize=True)
    return output


def frame_links(old, source, data, points, renderer_path):
    frames = old.HERE / "_encode_frames"
    frames.mkdir()
    segments = []
    frame_index = 0

    def add(still_name, duration_frames, kind, point=None, point_index=None):
        nonlocal frame_index
        still = old.STILLS / still_name
        if not still.is_file():
            raise FileNotFoundError(still)
        start = frame_index
        for _ in range(duration_frames):
            (frames / f"{frame_index:05d}.png").symlink_to(Path("..") / "stills" / still_name)
            frame_index += 1
        entry = {"kind": kind, "source_still": still_name,
                 "start_frame": start, "end_frame_exclusive": frame_index,
                 "start_s": start / old.FPS, "end_s": frame_index / old.FPS}
        if point is not None:
            chosen = point["choice"]
            entry.update(position_number=point_index + 1,
                         case=chosen["name"] if chosen else None,
                         xyz_m=[point[k] for k in ("base_x_m", "base_y_m", "base_z_m")],
                         tilt_deg=chosen["local_y_tilt_deg"] if chosen else None,
                         n_success=chosen["n_success"] if chosen else None,
                         tied_best_angles_deg=point["tied_best_angles_deg"],
                         status=point["status"])
        segments.append(entry)

    for index in range(old.ORBIT_COUNT):
        add(f"overview_{index:03d}.png", old.ORBIT_FRAMES_EACH, "orbit")
    for index, point in enumerate(points):
        add(f"point_{index + 1:02d}.png", old.DETAIL_FRAMES_EACH,
            "detail", point, index)
    add("outro.png", old.OUTRO_FRAMES, "outro")
    manifest = {"data_file": str(source), "data_sha256": sha256(source),
                "v67_renderer_sha256": sha256(Path(__file__)),
                "v64_renderer_base_sha256": sha256(renderer_path),
                "video_file": old.VIDEO.name,
                "width": old.FRAME_SIZE[0], "height": old.FRAME_SIZE[1],
                "fps": old.FPS, "expected_frames": frame_index,
                "expected_duration_s": frame_index / old.FPS,
                "n_individual_detail_segments": len(points), "segments": segments}
    expected = (old.ORBIT_COUNT * old.ORBIT_FRAMES_EACH +
                len(points) * old.DETAIL_FRAMES_EACH + old.OUTRO_FRAMES)
    if frame_index != expected or len(points) != 18:
        raise ValueError("Walkthrough frame/position count mismatch")
    (old.HERE / "frame_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return frames, manifest


def write_readme(old, data, frame_manifest, video_audit):
    details = [part for part in frame_manifest["segments"] if part["kind"] == "detail"]
    lines = ["# V67 负角最佳布置：18 个基座位置三维逐点视频", "",
             f"- [完整视频]({old.VIDEO.name})：1920×1080、{old.FPS} fps、"
             f"{frame_manifest['expected_duration_s']:.1f} 秒、无声。",
             "- 开头旋转观察 18 个位置；随后按 X=-0.20 m、X=-0.10 m 两个截面，逐点展示 XYZ、选中的负倾角、成功数、case 和同成功数的角度。",
             "- 每处从 -30°/-45°/-60° 三组中按完整轨迹成功数选最佳；同分按 J1–J6 全轨迹最小有效关节余量、原候选编号决胜。",
             f"- 点色按成功数/9，已完成 0/9 为红色；{data['n_no_grid_result_positions']} 个无完整九点网格结果的位置为灰色。箭头为实际基座局部 +Z 朝向。",
             "- 浅青色为 grasp（抓取）区域，粉色 X 为固定 place（放置）点。task_world 是任务世界坐标系，坐标单位米；九点冒烟只用于布置筛选，不能代表 20×20 全量覆盖率。",
             "- J1–J6 是机械臂六个运动关节；局部 +Y 指原始零倾角基座的 Y 正方向，局部 +Z 是随安装角度转动的基座 Z 正方向。",
             "",
             f"审计：最佳角度输入 SHA256 `{frame_manifest['data_sha256']}`；视频 SHA256 `{video_audit['video_sha256']}`；"
             f"{frame_manifest['expected_frames']} 帧完整解码通过，18 个逐点画面中点均与对应源图一致。"
             "详见 [逐段帧清单](frame_manifest.json)、[视频审计](video_audit.json)和"
             "[18 点联系表](all_18_detail_frames_contact_sheet.jpg)。", "",
             "复现：`python3 xtrainer_plan/scripts/render_v67_negative_best_angle_walkthrough.py --smoke-root <V67冒烟根目录>`。",
             "", "| 序号 | 时间段 | 基座 XYZ (m) | 所选 case | 倾角 | 成功数 | 同成功数倾角 |",
             "| ---: | ---: | --- | --- | ---: | ---: | --- |"]
    for part in details:
        xyz = ", ".join(f"{value:+.2f}" for value in part["xyz_m"])
        choice = part["case"] or "无完整网格"
        angle = f"{part['tilt_deg']:g}°" if part["tilt_deg"] is not None else "—"
        score = f"{part['n_success']}/9" if part["n_success"] is not None else "—"
        ties = ", ".join(f"{value:g}°" for value in part["tied_best_angles_deg"]) or "—"
        lines.append(f"| {part['position_number']:02d} | {part['start_s']:.1f}–{part['end_s']:.1f}s | "
                     f"({xyz}) | {choice} | {angle} | {score} | {ties} |")
    (old.HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--legacy-renderer", type=Path, default=LEGACY_RENDERER)
    args = parser.parse_args()
    root = args.smoke_root.resolve()
    renderer = args.legacy_renderer.resolve()
    if not renderer.is_file():
        raise FileNotFoundError(renderer)
    source, data, points = load_data(root)
    out = root / "best_angle_3d/video"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite walkthrough directory: {out}")
    old = import_legacy(renderer)
    old.HERE, old.SOURCE = out, source
    old.STILLS = out / "stills"
    old.VIDEO = out / VIDEO_NAME
    install_v67_views(old)
    out.mkdir(parents=True)
    old.set_font()
    old.render_all(data, points)
    make_contact_sheet(old, points)
    frames, frame_manifest = frame_links(old, source, data, points, renderer)
    try:
        old.encode(frames)
    finally:
        shutil.rmtree(frames)
    video_audit = old.audit(data, points, frame_manifest)
    video_audit["no_grid_result_positions"] = data["n_no_grid_result_positions"]
    video_audit.pop("untested_positions", None)
    (out / "video_audit.json").write_text(
        json.dumps(video_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(old, data, frame_manifest, video_audit)
    print(f"[VIDEO] {old.VIDEO}: {frame_manifest['expected_frames']} frames, "
          "18/18 detail frames independently checked")


if __name__ == "__main__":
    main()
