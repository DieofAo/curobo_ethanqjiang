#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 pick&place 结果里「每个抓取位的实际采用抓取角」画成二维分布图。

  横轴 = LINK_0 系下的 x (m)
  纵轴 = LINK_0 系下的 y (m)
  每个采样点标注该物料最终采用的抓取角 = base_rpy.grasp 右乘「绕工具 X 轴」的度数
  (对应 trajectory_meta.json 里的 items[].angle_grasp_deg)

用法:
    python3 plot_grasp_angle_map.py results_pick_place/20260831_113106
    python3 plot_grasp_angle_map.py <dir> --out /tmp/a.png --annotate tried
    # 完成但全失败的结果也可直接传目录，脚本会读取 plan_skipped.json
    python3 plot_grasp_angle_map.py <原结果目录> \
        --replan-result-dir <失败点重规划目录>
    python3 plot_grasp_angle_map.py <原结果目录> \
        --replan-result-dir <第一轮目录> \
        --replan-result-dir <第二轮目录>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


def pick_cjk_font() -> str | None:
    """挑一个能显示中文的字体, 找不到就退回英文标签。"""
    prefer = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
              "WenQuanYi Micro Hei", "Source Han Sans CN", "SimHei",
              "Microsoft YaHei", "AR PL UMing CN"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in have:
            return name
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_dir", help="results_pick_place/<时间戳> 目录")
    ap.add_argument("--out", default="", help="输出图片路径, 默认写到结果目录内")
    ap.add_argument(
        "--replan-result-dir",
        action="append",
        default=[],
        help=("可选的失败点重规划结果目录，可重复传入；按 index 依次合并成功项，"
              "绿色边框表示余量正常，橙色边框表示限位余量小于 1 度"),
    )
    ap.add_argument(
        "--uniform-success-style",
        action="store_true",
        help=("合并重规划结果时不区分原成功、重规划成功和近限位成功；"
              "所有成功点使用普通样式，标题/图例/脚注也不显示来源分类"),
    )
    ap.add_argument("--annotate", default="angle",
                    choices=["angle", "tried", "both"],
                    help="点上标什么: 抓取角 / 搜索次数 / 两者")
    ap.add_argument("--place-angle", action="store_true",
                    help="额外再画一张放置角的图")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--figsize", type=float, nargs=2, default=[9.5, 7.5])
    ap.add_argument("--cell-scale", type=float, default=0.9,
                    help="方格占相邻采样间距的比例, 默认 0.9")
    ap.add_argument("--cell-grid", action="store_true",
                    help="为每个方格绘制更清晰的边框网格")
    ap.add_argument("--cell-grid-linewidth", type=float, default=0.8,
                    help="--cell-grid 的边框线宽, 默认 0.8")
    ap.add_argument("--axis-grid-step", type=float, default=0.0,
                    help="X/Y 坐标轴背景网格间距(m), 0 表示自动")
    ap.add_argument("--left-rotate-z-deg", type=float, default=0.0,
                    help="绘图前对抓取点和放置点左乘纯 Rz 旋转(度), 不加平移")
    args = ap.parse_args()

    if not 0.0 < args.cell_scale <= 1.0:
        ap.error("--cell-scale 必须在 (0, 1] 范围内")
    if args.cell_grid_linewidth <= 0.0:
        ap.error("--cell-grid-linewidth 必须大于 0")
    if args.axis_grid_step < 0.0:
        ap.error("--axis-grid-step 不能小于 0")

    rd = Path(args.result_dir)
    meta_path = rd / "trajectory_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        # 所有点都失败时，规划器不会写 trajectory_meta.json，但会把
        # 完整网格留在 plan_skipped.json，配置则留在 plan_failed.json。
        # 在内存中补出与 trajectory_meta 相同的最小结构，不改写原结果。
        skipped_path = rd / "plan_skipped.json"
        failed_path = rd / "plan_failed.json"
        if not skipped_path.is_file() or not failed_path.is_file():
            raise SystemExit(
                f"[ERR] 找不到 {meta_path}，且缺少可用的 "
                "plan_skipped.json / plan_failed.json"
            )

        skipped_doc = json.loads(skipped_path.read_text(encoding="utf-8"))
        failed_doc = json.loads(failed_path.read_text(encoding="utf-8"))
        config = failed_doc.get("config") or {}
        pick_place = config.get("pick_place") or {}
        grid = pick_place.get("grasp_grid") or {}
        skipped = list(skipped_doc.get("skipped") or [])
        expected_total = int(
            skipped_doc.get("n_items_total")
            or grid.get("n_items")
            or (int(grid.get("rows", 0)) * int(grid.get("cols", 0)))
            or len(skipped)
        )
        n_items_done = int(skipped_doc.get("n_items_done", len(skipped)))
        if (
            failed_doc.get("stage") != "no_item_success"
            or not skipped
            or len(skipped) != expected_total
            or n_items_done != expected_total
        ):
            raise SystemExit(
                "[ERR] 结果没有 trajectory_meta.json，且不是已完成的全失败网格"
            )

        items = []
        for skipped_item in sorted(skipped, key=lambda it: int(it["index"])):
            item = dict(skipped_item)
            item["success"] = False
            item.setdefault("angle_grasp_deg", None)
            item.setdefault("angle_place_deg", None)
            item.setdefault("n_angles_tried", item.get("n_combos_tried", 0))
            items.append(item)

        def apply_target_transform(position: list[float]) -> list[float]:
            transform = pick_place.get("link0_target_transform") or {}
            translation = np.asarray(
                transform.get("position", [0.0, 0.0, 0.0]), dtype=float
            )
            rotation = transform.get("rotation") or {}
            axis = np.asarray(rotation.get("axis", [0.0, 0.0, 1.0]), dtype=float)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= np.finfo(float).eps:
                raise SystemExit("[ERR] link0_target_transform.rotation.axis 不能为零")
            axis /= axis_norm
            angle = np.deg2rad(float(rotation.get("angle_deg", 0.0)))
            cross = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ])
            rotation_matrix = (
                np.eye(3) * np.cos(angle)
                + (1.0 - np.cos(angle)) * np.outer(axis, axis)
                + np.sin(angle) * cross
            )
            return (
                rotation_matrix @ np.asarray(position, dtype=float) + translation
            ).tolist()

        place_raw = (pick_place.get("place") or {}).get("position")
        meta = {
            "task_type": "pick_place_cycle",
            "robot": config.get("robot") or {},
            "home_joint_deg": None,
            "grid": grid,
            "place_position_raw": place_raw,
            "place_position": (
                apply_target_transform(place_raw) if place_raw is not None else None
            ),
            "angle_search": pick_place.get("angle_search") or {},
            "linear_move": pick_place.get("linear_move") or {},
            "criterion": pick_place.get("criterion") or {},
            "n_items_total": expected_total,
            "n_items_success": 0,
            "items": items,
            "skipped": skipped,
            "n_items_skipped": len(skipped),
            "config": config,
        }
        print(
            f"[info] {meta_path.name} 不存在；已从完整的全失败记录"
            f"加载 {len(items)} 个点（未改写原结果）"
        )

    items = [it for it in meta.get("items", [])]
    if not items:
        raise SystemExit("[ERR] meta 里没有 items")

    replan_summary: dict[str, object] | None = None
    if args.replan_result_dir:
        index_to_offset: dict[int, int] = {}
        for offset, item in enumerate(items):
            index = int(item["index"])
            if index in index_to_offset:
                raise SystemExit(f"[ERR] 原结果存在重复 index={index}")
            index_to_offset[index] = offset

        original_items = list(items)
        original_success = sum(bool(item.get("success")) for item in items)
        recovered = 0
        recovered_near_limit = 0
        replan_total = 0
        replan_layers: list[dict[str, object]] = []
        grasp_searches = [((meta.get("angle_search") or {}).get("grasp") or {})]
        for layer, replan_dir_arg in enumerate(args.replan_result_dir, start=1):
            replan_dir = Path(replan_dir_arg)
            replan_meta_path = replan_dir / "trajectory_meta.json"
            if not replan_meta_path.is_file():
                raise SystemExit(f"[ERR] 找不到 {replan_meta_path}")
            replan_meta = json.loads(replan_meta_path.read_text(encoding="utf-8"))
            replan_items = [it for it in replan_meta.get("items", [])]
            if not replan_items:
                raise SystemExit(f"[ERR] {replan_meta_path} 里没有 items")

            seen_replan: set[int] = set()
            layer_recovered = 0
            layer_near_limit = 0
            for replan_item in replan_items:
                index = int(replan_item["index"])
                if index in seen_replan:
                    raise SystemExit(
                        f"[ERR] 第 {layer} 个重规划结果存在重复 index={index}"
                    )
                seen_replan.add(index)
                if index not in index_to_offset:
                    raise SystemExit(f"[ERR] 重规划 index={index} 不在原结果中")

                offset = index_to_offset[index]
                reference_item = original_items[offset]
                for field in ("row", "col"):
                    if reference_item.get(field) != replan_item.get(field):
                        raise SystemExit(
                            f"[ERR] index={index} 的 {field} 不一致: "
                            f"{reference_item.get(field)} != {replan_item.get(field)}"
                        )
                if not np.allclose(
                    np.asarray(reference_item["position"], dtype=float),
                    np.asarray(replan_item["position"], dtype=float),
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise SystemExit(f"[ERR] index={index} 的 position 不一致")

                if not replan_item.get("success"):
                    continue
                if items[offset].get("success"):
                    raise SystemExit(
                        f"[ERR] 第 {layer} 个重规划成功项 index={index} "
                        "在此前结果中已经成功"
                    )
                merged_item = dict(replan_item)
                near_limit = (
                    merged_item.get("min_limit_margin_deg") is not None
                    and float(merged_item["min_limit_margin_deg"]) < 1.0
                )
                merged_item["map_replanned_success"] = True
                merged_item["map_replan_near_limit"] = near_limit
                merged_item["map_replan_layer"] = layer
                items[offset] = merged_item
                recovered += 1
                layer_recovered += 1
                if near_limit:
                    recovered_near_limit += 1
                    layer_near_limit += 1

            replan_total += len(replan_items)
            replan_layers.append({
                "dir": str(replan_dir),
                "n_items": len(replan_items),
                "recovered": layer_recovered,
                "near_limit": layer_near_limit,
            })
            grasp_searches.append(
                ((replan_meta.get("angle_search") or {}).get("grasp") or {})
            )

        meta = dict(meta)
        meta["items"] = items
        meta["n_items_total"] = len(items)
        meta["n_items_success"] = original_success + recovered
        meta["n_items_skipped"] = len(items) - meta["n_items_success"]
        search_axes = {search.get("axis") for search in grasp_searches}
        search_steps = {search.get("step_deg") for search in grasp_searches}
        if len(search_axes) != 1 or len(search_steps) != 1:
            raise SystemExit(
                "[ERR] 多轮重规划的 grasp axis/step_deg 不一致，无法合并搜索范围"
            )
        combined_angle_search = dict(meta.get("angle_search") or {})
        combined_grasp_search = dict(combined_angle_search.get("grasp") or {})
        combined_grasp_search["min_deg"] = min(
            float(search["min_deg"]) for search in grasp_searches
        )
        combined_grasp_search["max_deg"] = max(
            float(search["max_deg"]) for search in grasp_searches
        )
        combined_angle_search["grasp"] = combined_grasp_search
        meta["angle_search"] = combined_angle_search
        replan_summary = {
            "original_success": original_success,
            "recovered": recovered,
            "near_limit": recovered_near_limit,
            "replan_total": replan_total,
            "layers": replan_layers,
        }

    cjk = pick_cjk_font()
    if cjk:
        plt.rcParams["font.family"] = cjk
    plt.rcParams["axes.unicode_minus"] = False
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    grid = meta.get("grid") or {}
    base = (meta.get("robot") or {}).get("base_link", "LINK_0")
    place = meta.get("place_position")
    home_deg = meta.get("home_joint_deg")
    asearch = ((meta.get("angle_search") or {}).get("grasp") or {})
    theta = np.deg2rad(args.left_rotate_z_deg)
    point_rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    transform_zh = (
        f"   点位左乘 Rz({args.left_rotate_z_deg:+g}°)"
        if abs(args.left_rotate_z_deg) > 1e-12 else ""
    )
    transform_en = (
        f"   points left-multiplied by Rz({args.left_rotate_z_deg:+g} deg)"
        if abs(args.left_rotate_z_deg) > 1e-12 else ""
    )

    def grid_spacing(values: np.ndarray) -> float:
        """返回相邻采样坐标的典型间距。"""
        unique = np.unique(values)
        delta = np.diff(unique)
        delta = delta[delta > np.finfo(float).eps * 10]
        if len(delta):
            return float(np.median(delta))
        return 0.02

    def grid_basis(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 row/col 相邻点在绘图平面内的典型位移向量。"""
        rc_to_xy: dict[tuple[int, int], np.ndarray] = {}
        for it, xi, yi in zip(items, x, y):
            if it.get("row") is None or it.get("col") is None:
                continue
            rc_to_xy[(int(it["row"]), int(it["col"]))] = np.array(
                [xi, yi], dtype=float
            )

        row_steps: list[np.ndarray] = []
        col_steps: list[np.ndarray] = []
        for (row, col), xy in rc_to_xy.items():
            if (row + 1, col) in rc_to_xy:
                row_steps.append(rc_to_xy[(row + 1, col)] - xy)
            if (row, col + 1) in rc_to_xy:
                col_steps.append(rc_to_xy[(row, col + 1)] - xy)

        if row_steps and col_steps:
            row_step = np.median(np.stack(row_steps), axis=0)
            col_step = np.median(np.stack(col_steps), axis=0)
            eps = np.finfo(float).eps * 10
            if np.linalg.norm(row_step) > eps and np.linalg.norm(col_step) > eps:
                return row_step, col_step

        # 兼容缺少 row/col 的旧结果；这时退回轴对齐方格。
        return (
            np.array([grid_spacing(x), 0.0]),
            np.array([0.0, grid_spacing(y)]),
        )

    def draw(key: str, title_zh: str, title_en: str, out: Path) -> None:
        xy = np.array([it["position"][:2] for it in items], dtype=float)
        # 点按列向量解释，等价于逐点执行 xy' = Rz @ xy；变换无平移项。
        xy = (point_rotation @ xy.T).T
        x = xy[:, 0]
        y = xy[:, 1]
        a = np.array([
            float(it[key]) if it.get(key) is not None else np.nan
            for it in items
        ])
        ok = np.array([bool(it.get("success")) for it in items])
        replanned = np.array([
            bool(it.get("map_replanned_success")) for it in items
        ])
        replan_near_limit = np.array([
            bool(it.get("map_replan_near_limit")) for it in items
        ])
        show_replan_classes = replan_summary is not None and not args.uniform_success_style
        tried = np.array([int(it.get("n_angles_tried", 0)) for it in items])

        fig, ax = plt.subplots(figsize=tuple(args.figsize))

        # 用对称色标: 0 度居中, 正负分别向两侧发散
        known = np.isfinite(a)
        vmax = max(1.0, float(np.abs(a[known]).max())) if known.any() else 1.0
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.get_cmap("coolwarm")

        # 方格尺寸使用数据坐标而不是固定的 points^2。这样不论网格密度和
        # figsize 如何变化，方格都按相邻采样点的真实间距排列且不会重叠。
        row_step, col_step = grid_basis(x, y)
        cell_row = row_step * args.cell_scale
        cell_col = col_step * args.cell_scale
        for xi, yi, ai, good, newly_replanned, near_limit in zip(
            x, y, a, ok, replanned, replan_near_limit
        ):
            face = cmap(norm(ai)) if np.isfinite(ai) else "0.88"
            if show_replan_classes and newly_replanned and near_limit:
                edgecolor = "#e67e22"
                linewidth = max(1.35, args.cell_grid_linewidth)
            elif show_replan_classes and newly_replanned:
                edgecolor = "#008837"
                linewidth = max(1.25, args.cell_grid_linewidth)
            else:
                edgecolor = "0.12" if args.cell_grid and good else (
                    "0.25" if good else "red"
                )
                linewidth = args.cell_grid_linewidth if args.cell_grid else (
                    0.35 if good else 0.8
                )
            center = np.array([xi, yi])
            half_row = cell_row / 2
            half_col = cell_col / 2
            vertices = np.array([
                center - half_row - half_col,
                center + half_row - half_col,
                center + half_row + half_col,
                center - half_row + half_col,
            ])
            ax.add_patch(Polygon(
                vertices,
                closed=True,
                facecolor=face,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=2 if good else 3,
            ))

        row_ids = {int(it["row"]) for it in items if it.get("row") is not None}
        col_ids = {int(it["col"]) for it in items if it.get("col") is not None}
        n_x = len(row_ids) if row_ids else len(np.unique(x))
        n_y = len(col_ids) if col_ids else len(np.unique(y))
        text_size = max(3.5, min(10.5, 120.0 / max(n_x, n_y)))
        if args.annotate == "both":
            text_size *= 0.85
        for xi, yi, ai, ti in zip(x, y, a, tried):
            if not np.isfinite(ai):
                if args.annotate == "tried":
                    txt = f"{ti}"
                elif args.annotate == "both":
                    txt = f"×\n({ti})"
                else:
                    txt = "×"
            elif args.annotate == "angle":
                txt = f"{ai:+.0f}"
            elif args.annotate == "tried":
                txt = f"{ti}"
            else:
                txt = f"{ai:+.0f}\n({ti})"
            # 深色底用白字, 浅色底用黑字
            shade = abs(ai) / vmax if np.isfinite(ai) else 0.0
            ax.text(xi, yi, txt, ha="center", va="center", zorder=4,
                    fontsize=text_size, fontweight="bold",
                    color="white" if shade > 0.55 else
                    ("#b2182b" if not np.isfinite(ai) else "black"))

        # 抓取顺序路径, 看得出 snake 遍历。用箭头标出前进方向
        ax.plot(x, y, "-", color="0.42", lw=1.0, alpha=0.55, zorder=1,
                label=L(f"抓取顺序 ({grid.get('order','?')})",
                        f"pick order ({grid.get('order','?')})"))
        for i in range(len(x) - 1):
            dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            ax.annotate("", xy=(x[i] + dx * 0.62, y[i] + dy * 0.62),
                        xytext=(x[i] + dx * 0.38, y[i] + dy * 0.38),
                        arrowprops=dict(arrowstyle="-|>", color="0.3",
                                        lw=1.0, alpha=0.75), zorder=1)
        # 首/末物料标记挪到格子角上, 不遮挡中间的角度数字
        endpoint_off = (cell_row + cell_col) * 0.36 if len(x) > 1 else np.zeros(2)
        endpoint_size = min(80.0, max(30.0, 1600.0 / max(n_x, n_y)))
        ax.scatter(x[:1] - endpoint_off[0], y[:1] - endpoint_off[1],
                   s=endpoint_size, marker="o",
                   color="lime", edgecolors="black", linewidths=1.0, zorder=6,
                   label=L("第 1 个物料", "first item"))
        ax.scatter(x[-1:] - endpoint_off[0], y[-1:] - endpoint_off[1],
                   s=endpoint_size, marker="o",
                   color="black", edgecolors="white", linewidths=1.0, zorder=6,
                   label=L("最后 1 个物料", "last item"))

        # 机器人基座原点
        ax.scatter([0], [0], s=200, marker="*", color="gold",
                   edgecolors="black", linewidths=1.0, zorder=5,
                   label=L(f"{base} 原点", f"{base} origin"))
        if place:
            place_xy = point_rotation @ np.asarray(place[:2], dtype=float)
            ax.scatter([place_xy[0]], [place_xy[1]], s=180, marker="X",
                       color="magenta", edgecolors="black", zorder=5,
                       label=L(f"放置点 z={place[2]:.3f}",
                               f"place z={place[2]:.3f}"))

        if known.any():
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array(a[known])
            cb = fig.colorbar(sm, ax=ax, pad=0.02)
            cb.set_label(L("绕工具 X 轴转角 (度)", "rotation about tool X (deg)"))

        rng = f"[{asearch.get('min_deg')}, {asearch.get('max_deg')}]" \
              f" step {asearch.get('step_deg')}"
        if show_replan_classes:
            source_summary_zh = (
                f"   原成功 {replan_summary['original_success']}"
                f" + 单臂重规划新增 {replan_summary['recovered']}"
            )
            source_summary_en = (
                f"   original {replan_summary['original_success']}"
                f" + single-arm replan {replan_summary['recovered']}"
            )
        else:
            source_summary_zh = ""
            source_summary_en = ""
        if show_replan_classes:
            sub = L(
                f"网格 {grid.get('rows')}x{grid.get('cols')} @ z={grid.get('z')}m"
                f"   成功 {meta.get('n_items_success')}/{meta.get('n_items_total')}"
                f"{source_summary_zh}\n"
                f"搜索范围 {rng}{transform_zh}",
                f"grid {grid.get('rows')}x{grid.get('cols')} @ z={grid.get('z')}m"
                f"   ok {meta.get('n_items_success')}/{meta.get('n_items_total')}"
                f"{source_summary_en}\n"
                f"search {rng}{transform_en}",
            )
        else:
            sub = L(
                f"网格 {grid.get('rows')}x{grid.get('cols')} @ z={grid.get('z')}m"
                f"   成功 {meta.get('n_items_success')}/{meta.get('n_items_total')}"
                f"   搜索范围 {rng}"
                f"{transform_zh}",
                f"grid {grid.get('rows')}x{grid.get('cols')} @ z={grid.get('z')}m"
                f"   ok {meta.get('n_items_success')}/{meta.get('n_items_total')}"
                f"   search {rng}"
                f"{transform_en}",
            )
        ax.set_title(f"{L(title_zh, title_en)}\n{sub}", fontsize=12.5, pad=12)
        ax.set_xlabel(L(f"{base} x (m)", f"{base} x (m)"))
        ax.set_ylabel(L(f"{base} y (m)", f"{base} y (m)"))
        if args.axis_grid_step > 0.0:
            locator = MultipleLocator(args.axis_grid_step)
            ax.xaxis.set_major_locator(locator)
            ax.yaxis.set_major_locator(MultipleLocator(args.axis_grid_step))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, which="major", ls=":", linewidth=0.55, alpha=0.55)
        else:
            ax.grid(True, ls=":", alpha=0.45)
        ax.set_axisbelow(True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.13)
        handles, labels = ax.get_legend_handles_labels()
        if (~ok).any():
            failed = Patch(facecolor="0.88", edgecolor="red", linewidth=0.8,
                           label=L("规划失败", "failed"))
            handles = [failed] + [h for h, label in zip(handles, labels)
                                  if label != L("规划失败", "failed")]
            labels = [L("规划失败", "failed")] + [
                label for label in labels if label != L("规划失败", "failed")
            ]
        replanned_healthy = replanned & ~replan_near_limit
        if show_replan_classes and replanned_healthy.any():
            recovered_patch = Patch(
                facecolor="0.88", edgecolor="#008837", linewidth=1.25,
                label=L(
                    f"单臂重规划新增成功 ({int(replanned_healthy.sum())})",
                    f"recovered by single-arm replan ({int(replanned_healthy.sum())})",
                ),
            )
            insert_at = 1 if (~ok).any() else 0
            handles.insert(insert_at, recovered_patch)
            labels.insert(insert_at, recovered_patch.get_label())
        if show_replan_classes and replan_near_limit.any():
            near_limit_patch = Patch(
                facecolor="0.88", edgecolor="#e67e22", linewidth=1.35,
                label=L(
                    f"重规划成功但限位余量 < 1° ({int(replan_near_limit.sum())})",
                    f"replanned, limit margin < 1 deg ({int(replan_near_limit.sum())})",
                ),
            )
            insert_at = 1 + int(replanned_healthy.any()) if (~ok).any() else int(
                replanned_healthy.any()
            )
            handles.insert(insert_at, near_limit_patch)
            labels.insert(insert_at, near_limit_patch.get_label())
        ax.legend(handles, labels, loc="upper right", fontsize=9,
                  framealpha=0.9)

        replan_note_zh = ""
        replan_note_en = ""
        if show_replan_classes and replanned_healthy.any():
            replan_note_zh += "; 绿色边框 = 单臂重规划新增成功"
            replan_note_en += "; green border = recovered by single-arm replan"
        if show_replan_classes and replan_near_limit.any():
            replan_note_zh += "; 橙色边框 = 成功但限位余量 < 1°"
            replan_note_en += "; orange border = success with limit margin < 1 deg"
        if known.any():
            note = L(
                f"标注 = 实际采用的抓取角; 基准姿态 rpy="
                f"{(meta.get('config') or {}).get('pick_place', {}).get('base_rpy', {}).get('grasp')} 右乘该角"
                f"{replan_note_zh}",
                "label = adopted grasp angle (right-multiplied onto base rpy)"
                f"{replan_note_en}",
            )
        else:
            note = L(
                "× = 规划失败；本轮无成功解，因此没有实际采用的抓取角",
                "x = planning failed; no adopted grasp angle because all items failed",
            )
        fig.text(0.5, 0.015, note, ha="center", fontsize=8.5, color="0.35")

        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"[ok] {out}")
        print(f"[info] 方格边长 = {np.linalg.norm(cell_row):.6f}m x "
              f"{np.linalg.norm(cell_col):.6f}m "
              f"(scale={args.cell_scale:g})")

    if args.out:
        out = Path(args.out)
    elif args.replan_result_dir:
        out = Path(args.replan_result_dir[-1]) / "grasp_angle_map_combined.png"
    else:
        out = rd / "grasp_angle_map.png"
    draw("angle_grasp_deg",
         ("XTrainer 抓取位姿角分布（原结果 + 单臂重规划）"
          if replan_summary and not args.uniform_success_style
          else "XTrainer 抓取位姿角分布 (grasp)"),
         ("XTrainer grasp angle map (original + single-arm replan)"
          if replan_summary and not args.uniform_success_style
          else "XTrainer grasp angle map"), out)

    if args.place_angle:
        draw("angle_place_deg",
             "XTrainer 放置位姿角分布 (place)",
             "XTrainer place angle map",
             out.with_name(out.stem.replace("grasp", "place") + out.suffix))

    if home_deg:
        print(f"[info] home joint(deg) = {np.round(home_deg, 2).tolist()}")
    grasp_angles = sorted({
        it["angle_grasp_deg"] for it in items
        if it.get("angle_grasp_deg") is not None
    })
    print(f"[info] 点位左乘纯 Rz({args.left_rotate_z_deg:+g}deg), 平移=[0, 0, 0]")
    print(f"[info] 抓取角取值集合 = {grasp_angles}")
    if replan_summary:
        print(
            f"[info] 合并结果 = 原成功 {replan_summary['original_success']} + "
            f"单臂重规划新增 {replan_summary['recovered']} = "
            f"{meta['n_items_success']}/{meta['n_items_total']}"
        )
        if replan_summary["near_limit"]:
            print(
                f"[warn] 重规划新增成功中限位余量 < 1deg: "
                f"{replan_summary['near_limit']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
