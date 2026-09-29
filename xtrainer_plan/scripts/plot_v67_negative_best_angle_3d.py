#!/usr/bin/env python3
"""Build an offline, rotatable 3D map of each XYZ's best V67 negative tilt.

Only complete audited smoke reports are accepted. Success count decides first;
ties use the smallest effective margin over J1–J6, then manifest index.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil

from postprocess_v67_negative_smoke import DATE_DIR, validate_complete_smoke


DEFAULT_ROOT = DATE_DIR / "v67_near_zero_x_local_y_negative_smoke"
DEFAULT_PLOTLY_JS = (DATE_DIR / "v64_near_zero_x_local_ytilt_joint_home_smoke"
                     / "best_angle_3d/plotly.min.js")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_by_name(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate candidate in {path}")
    return {row["name"]: row for row in rows}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maybe_float(value):
    if value in (None, "", "None"):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite reported metric: {value}")
    return result


def collect(root):
    manifest_path = validate_complete_smoke(root)
    manifest = read_json(manifest_path)
    comparison_path = root / "final_comparison.csv"
    margin_path = root / "final_joint_margin.csv"
    posture_path = root / "final_observed_posture_cases.csv"
    comparison = read_csv_by_name(comparison_path)
    margins = read_csv_by_name(margin_path)
    postures = read_csv_by_name(posture_path)
    candidates = manifest["candidates"]
    names = {row["name"] for row in candidates}
    if any(set(report) != names for report in (comparison, margins, postures)):
        raise ValueError("Comparison, margin and posture reports must each cover all 54 candidates")
    by_xyz = defaultdict(list)
    base_grid, place = None, None
    n_grids_attempted = 0
    for item in candidates:
        name = item["name"]
        config_path = Path(item["config"])
        if sha256(config_path) != item["config_sha256"]:
            raise ValueError(f"Smoke configuration hash changed: {name}")
        config = read_json(config_path)
        grid = config["pick_place"]["grasp_grid"]
        current_place = config["pick_place"]["place"]["position"]
        if base_grid is None:
            base_grid, place = grid, current_place
        elif grid != base_grid or current_place != place:
            raise ValueError(f"Task grasp/place geometry differs: {name}")
        xyz = tuple(item["base_xyz_m"])
        tilt = float(item["local_y_tilt_deg"])
        transform = config["robot"]["mount_transform"]
        direction = [float(transform[i][2]) for i in range(3)]
        if any(abs(float(transform[i][3]) - xyz[i]) > 1e-9 for i in range(3)) or not math.isclose(
                math.sqrt(sum(value * value for value in direction)), 1, abs_tol=1e-9):
            raise ValueError(f"Mount transform differs from manifest: {name}")
        c, m, p = comparison[name], margins[name], postures[name]
        if (c["status"] != m["status"] or
                any(abs(float(c[key]) - xyz[index]) > 1e-9 for index, key in enumerate(
                    ("base_x_m", "base_y_m", "base_z_m"))) or
                abs(float(c["local_y_tilt_deg"]) - tilt) > 1e-9):
            raise ValueError(f"Summary coordinates/status disagree: {name}")
        status = c["status"]
        expected_posture_status = {"completed_verified": "observed",
                                   "completed_zero_success": "zero_success",
                                   "home_failed": "home_failed"}.get(status)
        if (p["status"] != expected_posture_status or
                p["tilt_axis"] != "original_base_local_y" or
                abs(float(p["local_y_tilt_deg"]) - tilt) > 1e-9):
            raise ValueError(f"Posture status/axis differs from smoke report: {name}")
        n_success = int(c["n_success"]) if c["n_success"] else None
        n_total = int(c["n_total"]) if c["n_total"] else None
        margin = maybe_float(m["minimum_effective_margin_deg"])
        posture = maybe_float(p["median_deg"])
        if status == "completed_verified":
            if (not c["ranking_eligible"] == "True" or n_total != 9 or
                    n_success is None or n_success < 1 or margin is None or margin < 0 or
                    posture is None or p["audit"] != "verified" or
                    int(p["n_success"]) != n_success or int(m["n_success"]) != n_success):
                raise ValueError(f"Verified metrics missing or inconsistent: {name}")
            n_grids_attempted += 1
        elif status == "completed_zero_success":
            if n_success != 0 or n_total != 9 or margin is not None or posture is not None:
                raise ValueError(f"Zero-success case has inconsistent metrics: {name}")
            n_grids_attempted += 1
        elif status == "home_failed":
            if n_success is not None or margin is not None or posture is not None:
                raise ValueError(f"Home-failed case has saved metrics: {name}")
        else:
            raise ValueError(f"Incomplete smoke status: {name}: {status}")
        by_xyz[xyz].append({"name": name, "index": item["index"],
                            "base_xyz_m": list(xyz), "local_y_tilt_deg": tilt,
                            "status": status, "n_success": n_success,
                            "n_total": n_total,
                            "minimum_effective_margin_deg": margin,
                            "observed_shoulder_wrist_angle_deg": posture,
                            "local_plus_z_task_world": direction})
    if len(by_xyz) != 18 or any(len(rows) != 3 for rows in by_xyz.values()):
        raise ValueError("Expected 18 XYZ positions with three negative angles each")
    points = []
    for xyz, tested in sorted(by_xyz.items()):
        valid = [row for row in tested if row["n_total"] == 9]
        valid.sort(key=lambda row: (-row["n_success"],
                                    -(row["minimum_effective_margin_deg"]
                                      if row["minimum_effective_margin_deg"] is not None
                                      else -math.inf), row["index"]))
        choice = valid[0] if valid else None
        top = choice["n_success"] if choice else None
        points.append({"base_xyz_m": list(xyz), "status": "tested" if choice else "no_grid_result",
                       "n_grid_results": len(valid),
                       "same_success_angles_deg": [row["local_y_tilt_deg"] for row in valid
                                                   if row["n_success"] == top],
                       "choice": choice})
    report = {"schema_version": 1, "coordinate_frame": "task_world",
              "rotation_axis": "original base local +Y",
              "ranking_rule": ["more successful complete 3×3 trajectories",
                               "larger audited minimum effective margin over J1–J6 and all saved samples",
                               "smaller original manifest index"],
              "source_sha256": {path.name: sha256(path) for path in
                                (manifest_path, comparison_path, margin_path, posture_path)},
              "n_positions": 18, "n_smoke_configs": 54,
              "n_grids_attempted": n_grids_attempted,
              "n_no_grid_result_positions": sum(point["choice"] is None for point in points),
              "best_success_counts": dict(Counter(
                  point["choice"]["n_success"] for point in points if point["choice"])),
              "grasp_grid": base_grid, "place_xyz_m": place, "points": points}
    return report


def write_csv(report, path):
    fields = ("base_x_m", "base_y_m", "base_z_m", "status", "n_grid_results",
              "selected_case", "selected_tilt_deg", "n_success", "n_total",
              "minimum_effective_margin_deg", "observed_shoulder_wrist_angle_deg",
              "same_success_angles_deg")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in report["points"]:
            choice = point["choice"] or {}
            xyz = point["base_xyz_m"]
            writer.writerow({"base_x_m": xyz[0], "base_y_m": xyz[1],
                             "base_z_m": xyz[2], "status": point["status"],
                             "n_grid_results": point["n_grid_results"],
                             "selected_case": choice.get("name"),
                             "selected_tilt_deg": choice.get("local_y_tilt_deg"),
                             "n_success": choice.get("n_success"),
                             "n_total": choice.get("n_total"),
                             "minimum_effective_margin_deg": choice.get("minimum_effective_margin_deg"),
                             "observed_shoulder_wrist_angle_deg":
                                 choice.get("observed_shoulder_wrist_angle_deg"),
                             "same_success_angles_deg": ",".join(
                                 f"{angle:g}" for angle in point["same_success_angles_deg"])})


def html_plot(report):
    traces, groups = [], []

    def add(trace, group="fixed"):
        traces.append(trace)
        groups.append(group)

    grid = report["grasp_grid"]
    place = report["place_xyz_m"]
    x0, x1 = grid["x_range"]
    y0, y1 = grid["y_range"]
    z = grid["z"]
    add({"type": "surface", "x": [[x0, x1], [x0, x1]],
         "y": [[y0, y0], [y1, y1]], "z": [[z, z], [z, z]],
         "surfacecolor": [[0, 0], [0, 0]],
         "colorscale": [[0, "#a9e2dd"], [1, "#a9e2dd"]],
         "opacity": .34, "showscale": False, "hoverinfo": "skip"})
    for x_slice in (-.2, -.1):
        key = f"x{x_slice:+.1f}"
        points = [point for point in report["points"] if point["base_xyz_m"][0] == x_slice]
        good = [point for point in points if point["choice"]]
        gray = [point for point in points if not point["choice"]]
        for point in good:
            xyz = point["base_xyz_m"]
            direction = point["choice"]["local_plus_z_task_world"]
            end = [xyz[i] + .085 * direction[i] for i in range(3)]
            add({"type": "scatter3d", "mode": "lines", "showlegend": False,
                 "x": [xyz[0], end[0]], "y": [xyz[1], end[1]],
                 "z": [xyz[2], end[2]],
                 "line": {"color": "#28465b", "width": 4}, "hoverinfo": "skip"}, key)
        hover = []
        for point in good:
            chosen = point["choice"]
            xyz = point["base_xyz_m"]
            margin = chosen["minimum_effective_margin_deg"]
            posture = chosen["observed_shoulder_wrist_angle_deg"]
            hover.append(
                f"<b>{html.escape(chosen['name'])}</b><br>基座 XYZ = "
                f"({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:+.2f}) m"
                f"<br>局部 +Y 倾角 {chosen['local_y_tilt_deg']:g}°"
                f"<br>完整抓放成功 {chosen['n_success']}/9"
                f"<br>同成功数倾角 {', '.join(f'{a:g}°' for a in point['same_success_angles_deg'])}"
                + (f"<br>最小有效关节余量 {margin:.2f}°" if margin is not None else "")
                + (f"<br>肩到腕距竖直角 {posture:.2f}°" if posture is not None else ""))
        add({"type": "scatter3d", "name": f"基座 X={x_slice:+.2f} m",
             "mode": "markers+text", "x": [p["base_xyz_m"][0] for p in good],
             "y": [p["base_xyz_m"][1] for p in good],
             "z": [p["base_xyz_m"][2] for p in good],
             "text": [f"{p['choice']['local_y_tilt_deg']:g}°" for p in good],
             "textposition": "top center", "textfont": {"size": 12, "color": "#16364a"},
             "marker": {"size": 9, "color": [p["choice"]["n_success"] / 9 for p in good],
                        "cmin": 0, "cmax": 1,
                        "colorscale": [[0, "#b2182b"], [.5, "#efaa52"], [1, "#149277"]],
                        "line": {"color": "#183549", "width": 1},
                        "showscale": x_slice == -.2,
                        "colorbar": {"title": "成功数", "tickvals": [0, 3/9, 6/9, 1],
                                     "ticktext": ["0/9", "3/9", "6/9", "9/9"],
                                     "len": .65}},
             "hovertext": hover, "hovertemplate": "%{hovertext}<extra></extra>"}, key)
        if gray:
            add({"type": "scatter3d", "name": f"未完成抓取网格 X={x_slice:+.2f} m",
                 "mode": "markers", "x": [p["base_xyz_m"][0] for p in gray],
                 "y": [p["base_xyz_m"][1] for p in gray],
                 "z": [p["base_xyz_m"][2] for p in gray],
                 "marker": {"size": 9, "color": "#9aa3ad"},
                 "hovertext": [f"基座 {p['base_xyz_m']} m：三个角度均无完整 3×3 抓取网格结果"
                               for p in gray],
                 "hovertemplate": "%{hovertext}<extra></extra>"}, key)
    add({"type": "scatter3d", "name": "固定 place", "mode": "markers+text",
         "x": [place[0]], "y": [place[1]], "z": [place[2]],
         "text": ["place"], "textposition": "top center",
         "marker": {"size": 11, "color": "#d42e7c", "symbol": "diamond"},
         "hovertemplate": "固定 place<extra></extra>"})
    add({"type": "scatter3d", "name": "task_world 原点", "mode": "markers",
         "x": [0], "y": [0], "z": [0],
         "marker": {"size": 7, "color": "#d1aa2d", "symbol": "diamond"},
         "hovertemplate": "task_world 原点<extra></extra>"})
    layout = {"title": {"text": "V67 负角布置：18 个基座位置的最佳冒烟角度", "x": .5},
              "paper_bgcolor": "#f8fafc", "font": {"family": "Noto Sans CJK SC, sans-serif"},
              "margin": {"l": 0, "r": 0, "t": 65, "b": 0},
              "scene": {"xaxis": {"title": "task_world X / m", "range": [-.69, .08]},
                        "yaxis": {"title": "task_world Y / m", "range": [-.18, .55]},
                        "zaxis": {"title": "task_world Z / m", "range": [-.025, .75]},
                        "aspectmode": "manual", "aspectratio": {"x": 1, "y": .95, "z": 1},
                        "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.1}},
                        "dragmode": "orbit"}}
    script = '''const traces=DATA; const groups=GROUPS; const layout=LAYOUT;
Plotly.newPlot('plot',traces,layout,{responsive:true,displaylogo:false,scrollZoom:true});
let timer=null,phase=0;
function stop(){if(timer){clearInterval(timer);timer=null;}}
function show(which){stop();const visible=groups.map(g=>g==='fixed'||which==='all'||g===which);
  Plotly.restyle('plot',{visible:visible}); const cameras={all:{eye:{x:1.5,y:-1.5,z:1.1}},
  'x-0.2':{eye:{x:2.3,y:0.01,z:0.15},up:{x:0,y:0,z:1}},
  'x-0.1':{eye:{x:2.3,y:0.01,z:0.15},up:{x:0,y:0,z:1}}};
  Plotly.relayout('plot',{'scene.camera':cameras[which]});}
function orbit(){stop();show('all');timer=setInterval(()=>{phase+=0.032;
  Plotly.relayout('plot',{'scene.camera.eye':{x:1.85*Math.cos(phase),
  y:1.85*Math.sin(phase),z:1.05}});},65);}
'''
    script = (script.replace("DATA", json.dumps(traces, ensure_ascii=False, allow_nan=False))
             .replace("GROUPS", json.dumps(groups))
             .replace("LAYOUT", json.dumps(layout, ensure_ascii=False)))
    note = ("点色表示每个 XYZ 中最高的 3×3 冒烟成功数；数字为负向局部 +Y 倾角。"
            "细线是原始基座局部 +Z 的真实朝向；灰色表示该位置三个角度均未完成抓取网格。"
            "0/9 是已完成测试，以红色显示。拖动旋转、悬停查看 case 和关节余量。")
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V67 负角基座最佳角度三维图</title>
<style>body{{margin:0;background:#f8fafc;color:#203548;font-family:"Noto Sans CJK SC",sans-serif}}
#bar{{padding:10px 16px;background:#edf4f7;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
button{{padding:8px 12px;background:white;border:1px solid #b5c8d1;border-radius:6px;cursor:pointer}}
#plot{{height:calc(100vh - 140px);min-height:560px}}#note{{padding:10px 18px;font-size:13px}}</style>
<script src="plotly.min.js"></script></head><body>
<div id="bar"><strong>V67 基座布置</strong><button onclick="show('all')">全部位置</button>
<button onclick="show('x-0.2')">X=-0.20</button><button onclick="show('x-0.1')">X=-0.10</button>
<button onclick="orbit()">自动旋转</button><button onclick="stop()">停止旋转</button></div>
<div id="plot"></div><div id="note">{note}</div><script>{script}</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plotly-js", type=Path, default=DEFAULT_PLOTLY_JS)
    args = parser.parse_args()
    root = args.smoke_root.resolve()
    bundle = args.plotly_js.resolve()
    if not bundle.is_file():
        raise FileNotFoundError(f"Offline 3D renderer bundle missing: {bundle}")
    out = root / "best_angle_3d"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    report = collect(root)
    report["plotly_js_sha256"] = sha256(bundle)
    out.mkdir()
    (out / "best_angle_by_position.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    write_csv(report, out / "best_angle_by_position.csv")
    (out / "best_angle_3d.html").write_text(html_plot(report), encoding="utf-8")
    shutil.copy2(bundle, out / "plotly.min.js")
    print(f"[3D] {out / 'best_angle_3d.html'}: {report['n_positions']} positions, "
          f"{report['n_grids_attempted']}/54 full smoke grids")


if __name__ == "__main__":
    main()
