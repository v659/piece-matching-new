import argparse
import json
import math
import os
import time
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Polygon
from getsides import getsides
from side import Side


def rot2d(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def apply_transform(points, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if len(pts) == 0:
        return pts
    return (pts @ R.T) + t


def resample_curve(points, n=64) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.insert(np.cumsum(seg), 0, 0.0)
    total = s[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=float)
    idx = 0
    for i, td in enumerate(targets):
        while idx < len(s) - 2 and s[idx + 1] < td:
            idx += 1
        a, b = s[idx], s[idx + 1]
        if b - a < 1e-9:
            out[i] = pts[idx]
            continue
        r = (td - a) / (b - a)
        out[i] = pts[idx] + r * (pts[idx + 1] - pts[idx])
    return out


def rigid_fit_2d(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean

    H = src0.T @ dst0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - (src_mean @ R.T)
    return R, t


def symmetric_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    fwd = directed_hausdorff(a, b)[0]
    bwd = directed_hausdorff(b, a)[0]
    return max(fwd, bwd)


def side_profile(side, n=80):
    pts = resample_curve(side.side_points, n=n)
    p0 = pts[0]
    p1 = pts[-1]
    chord = p1 - p0
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-8:
        x = np.linspace(0.0, 1.0, n)
        y = np.zeros(n, dtype=float)
        return x, y

    u = chord / chord_len
    v = np.array([-u[1], u[0]], dtype=float)
    rel = pts - p0
    x = (rel @ u) / chord_len
    y = (rel @ v) / chord_len
    return x, y


def side_shape_compat_score(out_side, in_side):
    xo, yo = side_profile(out_side, n=80)
    _, yi = side_profile(in_side, n=80)

    yi_cmp = -yi[::-1]
    yi_aligned = yi_cmp

    profile_rmse = float(np.sqrt(np.mean((yo - yi_aligned) ** 2)))
    slope_rmse = float(np.sqrt(np.mean((np.gradient(yo) - np.gradient(yi_aligned)) ** 2)))
    path_rel = abs(out_side.path_length - in_side.path_length) / max(
        1e-6, (out_side.path_length + in_side.path_length) * 0.5
    )
    chord_out = np.linalg.norm(np.asarray(out_side.p2) - np.asarray(out_side.p1))
    chord_in = np.linalg.norm(np.asarray(in_side.p2) - np.asarray(in_side.p1))
    chord_rel = abs(chord_out - chord_in) / max(1e-6, (chord_out + chord_in) * 0.5)

    # Disambiguate similarly shaped edges with peak geometry features.
    peak_out_idx = int(np.argmax(yo))
    peak_in_idx = int(np.argmax(yi_aligned))
    peak_x_delta = abs(xo[peak_out_idx] - xo[peak_in_idx])
    peak_h_delta = abs(yo[peak_out_idx] - yi_aligned[peak_in_idx])

    half_out = 0.5 * yo[peak_out_idx]
    half_in = 0.5 * yi_aligned[peak_in_idx]
    width_out = float(np.mean(yo >= half_out))
    width_in = float(np.mean(yi_aligned >= half_in))
    width_delta = abs(width_out - width_in)

    end_slope_delta = abs((yo[1] - yo[0]) - (yi_aligned[1] - yi_aligned[0]))
    end_slope_delta += abs((yo[-1] - yo[-2]) - (yi_aligned[-1] - yi_aligned[-2]))

    return (
        120.0 * profile_rmse
        + 45.0 * slope_rmse
        + 55.0 * path_rel
        + 35.0 * chord_rel
        + 28.0 * peak_x_delta
        + 45.0 * peak_h_delta
        + 22.0 * width_delta
        + 160.0 * end_slope_delta
    )


def estimate_piece_pose_from_side_match(target_side_world: np.ndarray, source_side_local: np.ndarray):
    dst = resample_curve(target_side_world, n=80)
    src = resample_curve(source_side_local, n=80)
    src_rev = src[::-1]

    candidates = []
    for variant in (src, src_rev):
        R, t = rigid_fit_2d(variant, dst)
        aligned = apply_transform(variant, R, t)
        err = symmetric_hausdorff(aligned, dst)
        candidates.append((err, R, t))

    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def build_piece_contour(sides):
    ordered = sorted(sides, key=lambda s: s.side_index)
    if not ordered:
        return np.zeros((0, 2), dtype=float)

    contour = []
    last = None
    for side in ordered:
        pts = np.asarray(side.side_points, dtype=float)
        if len(pts) == 0:
            continue
        if last is not None:
            d_start = np.linalg.norm(last - pts[0])
            d_end = np.linalg.norm(last - pts[-1])
            if d_end < d_start:
                pts = pts[::-1]

        if not contour:
            contour.extend(pts.tolist())
        else:
            if np.linalg.norm(np.asarray(contour[-1]) - pts[0]) > 1e-6:
                contour.append(pts[0].tolist())
            contour.extend(pts[1:].tolist())
        last = np.asarray(contour[-1], dtype=float)

    if contour and np.linalg.norm(np.asarray(contour[0]) - np.asarray(contour[-1])) > 1e-6:
        contour.append(contour[0])
    return np.asarray(contour, dtype=float)


def transformed_polygon(contour: np.ndarray, R: np.ndarray, t: np.ndarray):
    if len(contour) < 4:
        return None
    pts = apply_transform(contour, R, t)
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def overlap_is_bad(new_poly, placed_polys, area_ratio_limit=0.015, abs_area_limit=80.0):
    for other in placed_polys.values():
        inter = new_poly.intersection(other)
        if inter.is_empty:
            continue
        inter_area = inter.area
        if inter_area < abs_area_limit:
            continue
        ratio = inter_area / max(1e-9, min(new_poly.area, other.area))
        if ratio > area_ratio_limit:
            return True
    return False


def side_key(pid, side_idx):
    return pid, side_idx


def build_piece_motion_plans(placements, pieces, piece_ids):
    plans = []
    for pid in piece_ids:
        if pid not in placements:
            continue

        R, t = placements[pid]
        contour = np.asarray(pieces[pid]["contour"], dtype=float)
        if len(contour) == 0:
            center = np.zeros(2, dtype=float)
        else:
            center = contour.mean(axis=0)

        theta_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        final_center = (center @ R.T) + t
        move_delta = final_center - center

        plans.append({
            "piece_id": int(pid),
            "center": center,
            "rotate_degrees": float(theta_deg),
            "move": {"x": float(move_delta[0]), "y": float(move_delta[1])},
        })
    return plans


def write_assembly_steps_json(placements, pieces, piece_ids, out_path="assembly_steps.json"):
    steps = []
    piece_plans = []

    for plan in build_piece_motion_plans(placements, pieces, piece_ids):
        pid = plan["piece_id"]
        theta_deg = plan["rotate_degrees"]
        move = plan["move"]

        piece_plans.append({
            "piece_id": pid,
            "rotate_degrees": theta_deg,
            "move": {"x": move["x"], "y": move["y"]},
        })

        steps.append({"command": "pickup piece", "piece_id": pid})
        steps.append({
            "command": "rotate",
            "piece_id": pid,
            "degrees": theta_deg,
        })
        steps.append({
            "command": "move",
            "piece_id": pid,
            "x": move["x"],
            "y": move["y"],
        })
        steps.append({"command": "drop piece", "piece_id": pid})

    payload = {
        "coordinate_frame": "image_xy",
        "rotation_convention": "positive_degrees_counterclockwise",
        "assumption": "rotate each picked piece around its current centroid, then apply move delta",
        "piece_plans": piece_plans,
        "steps": steps,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def animate_assembly(placements, pieces, piece_ids, fps=14, show=True):
    plans = build_piece_motion_plans(placements, pieces, piece_ids)
    if not plans:
        return False

    if show:
        backend = str(plt.get_backend()).lower()
        if "inline" in backend or "agg" in backend:
            for candidate in ("MacOSX", "TkAgg", "Qt5Agg"):
                try:
                    plt.switch_backend(candidate)
                    break
                except Exception:
                    continue

    plan_by_pid = {p["piece_id"]: p for p in plans}
    order = [p["piece_id"] for p in plans]
    order_idx = {pid: i for i, pid in enumerate(order)}

    # Sequence: initial hold, then pickup -> rotate -> move -> drop for each piece.
    initial_hold = 10
    pickup_hold = 3
    rotate_frames = 14
    move_frames = 20
    drop_hold = 3
    end_hold = 16

    frames = [(-1, "initial", 0.0)] * initial_hold
    for pid in order:
        frames.extend([(pid, "pickup", 1.0)] * pickup_hold)
        frames.extend([(pid, "rotate", (i + 1) / rotate_frames) for i in range(rotate_frames)])
        frames.extend([(pid, "move", (i + 1) / move_frames) for i in range(move_frames)])
        frames.extend([(pid, "drop", 1.0)] * drop_hold)
    frames.extend([(-1, "final", 1.0)] * end_hold)

    local_contours = {}
    for pid in order:
        local_contours[pid] = np.asarray(pieces[pid]["contour"], dtype=float)

    # Stable bounds from initial + final positions to avoid axis jumps.
    all_pts = []
    for pid in order:
        local = local_contours[pid]
        if len(local) >= 3:
            all_pts.append(local)
            R, t = placements[pid]
            all_pts.append(apply_transform(local, R, t))
    if not all_pts:
        return False
    stacked = np.vstack(all_pts)
    minx, miny = np.min(stacked[:, 0]), np.min(stacked[:, 1])
    maxx, maxy = np.max(stacked[:, 0]), np.max(stacked[:, 1])
    pad = 80.0

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")

    def piece_points_at(pid, active_pid, phase, prog):
        pts = local_contours[pid]
        if len(pts) < 3:
            return pts

        plan = plan_by_pid[pid]
        theta_full = math.radians(plan["rotate_degrees"])
        mv = np.array([plan["move"]["x"], plan["move"]["y"]], dtype=float)
        center = np.asarray(plan["center"], dtype=float)

        if active_pid == -1 and phase == "final":
            theta = theta_full
            delta = mv
        elif pid == active_pid:
            if phase in ("pickup", "rotate"):
                theta = theta_full * (prog if phase == "rotate" else 0.0)
                delta = np.zeros(2, dtype=float)
            elif phase in ("move", "drop"):
                theta = theta_full
                delta = mv * (prog if phase == "move" else 1.0)
            else:
                theta = 0.0
                delta = np.zeros(2, dtype=float)
        else:
            if active_pid == -1:
                theta = 0.0
                delta = np.zeros(2, dtype=float)
            elif order_idx[pid] < order_idx[active_pid]:
                theta = theta_full
                delta = mv
            else:
                theta = 0.0
                delta = np.zeros(2, dtype=float)

        R_step = rot2d(theta)
        return (pts - center) @ R_step.T + center + delta

    def update(frame_idx):
        active_pid, phase, prog = frames[frame_idx]
        ax.clear()

        for i, pid in enumerate(order):
            contour = piece_points_at(pid, active_pid, phase, prog)
            if len(contour) < 3:
                continue
            alpha = 0.65 if pid == active_pid and phase != "drop" else 0.35
            edge_w = 1.6 if pid == active_pid and phase != "drop" else 1.0
            ax.fill(contour[:, 0], contour[:, 1], color=cmap(i % 20), alpha=alpha)
            ax.plot(contour[:, 0], contour[:, 1], color="black", linewidth=edge_w)
            c = contour.mean(axis=0)
            ax.text(c[0], c[1], str(pid), ha="center", va="center", fontsize=9)

        if active_pid == -1:
            title = "Assembly Animation"
        elif phase == "pickup":
            title = f"Piece {active_pid}: pickup piece"
        elif phase == "rotate":
            title = f"Piece {active_pid}: rotate"
        elif phase == "move":
            title = f"Piece {active_pid}: move"
        else:
            title = f"Piece {active_pid}: drop piece"

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(maxy + pad, miny - pad)  # Keep the y-axis orientation used by the static plot.
        ax.axis("off")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    # Keep a hard reference on the figure so the animation object is not garbage-collected early.
    fig._assembly_anim = anim
    if show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return True


def load_pieces():
    with open("pieces_data.json", "r") as f:
        data = json.load(f)

    pieces = {}
    for piece in data:
        pid = piece["piece_number"]
        sides = []
        for s in piece["sides"]:
            side = Side(s["side_points"], s["side_index"], piece_number=pid)
            sides.append(side)
        contour = build_piece_contour(sides)
        pieces[pid] = {"sides": sides, "contour": contour}
    return pieces


def collect_candidates(pieces):
    outward = []
    inward = []
    flats = []

    for pid, pdata in pieces.items():
        for side in pdata["sides"]:
            if side.type == "outward":
                outward.append((pid, side))
            elif side.type == "inward":
                inward.append((pid, side))
            else:
                flats.append((pid, side))

    print(f"Side types -> outward: {len(outward)}, inward: {len(inward)}, flat: {len(flats)}")

    candidates = []
    for pid_a, side_a in outward:
        for pid_b, side_b in inward:
            if pid_a == pid_b:
                continue
            score = side_shape_compat_score(side_a, side_b)
            candidates.append({
                "score": float(score),
                "a_pid": pid_a,
                "a_idx": side_a.side_index,
                "b_pid": pid_b,
                "b_idx": side_b.side_index,
            })

    candidates.sort(key=lambda x: x["score"])
    return candidates, outward, inward


def build_match_graph(candidates):
    out_keys = sorted({side_key(c["a_pid"], c["a_idx"]) for c in candidates})
    in_keys = sorted({side_key(c["b_pid"], c["b_idx"]) for c in candidates})
    out_idx = {k: i for i, k in enumerate(out_keys)}
    in_idx = {k: i for i, k in enumerate(in_keys)}

    out_scores = defaultdict(list)
    in_scores = defaultdict(list)
    for c in candidates:
        out_scores[side_key(c["a_pid"], c["a_idx"])].append(c["score"])
        in_scores[side_key(c["b_pid"], c["b_idx"])].append(c["score"])

    out_margin = {}
    for k, vals in out_scores.items():
        vals = sorted(vals)
        if len(vals) >= 2:
            out_margin[k] = vals[1] - vals[0]
        else:
            out_margin[k] = 1e6

    in_margin = {}
    for k, vals in in_scores.items():
        vals = sorted(vals)
        if len(vals) >= 2:
            in_margin[k] = vals[1] - vals[0]
        else:
            in_margin[k] = 1e6

    BIG = 1e6
    cost = np.full((len(out_keys), len(in_keys)), BIG, dtype=float)
    edge_lookup = {}
    for c in candidates:
        ko = side_key(c["a_pid"], c["a_idx"])
        ki = side_key(c["b_pid"], c["b_idx"])
        i, j = out_idx[ko], in_idx[ki]
        uncertainty = (1.0 / max(0.5, out_margin[ko])) + (1.0 / max(0.5, in_margin[ki]))
        adjusted = c["score"] + 8.0 * uncertainty
        if adjusted < cost[i, j]:
            cost[i, j] = adjusted
            edge_lookup[(i, j)] = c

    rows, cols = linear_sum_assignment(cost)
    selected = []
    for r, c in zip(rows, cols):
        if cost[r, c] >= BIG * 0.5:
            continue
        edge = dict(edge_lookup[(r, c)])
        edge["assigned"] = True
        selected.append(edge)

    if selected:
        sel_scores = np.array([e["score"] for e in selected], dtype=float)
        cutoff = float(np.median(sel_scores) + 0.8 * np.std(sel_scores))
        pruned = [e for e in selected if e["score"] <= cutoff]
        if len(pruned) >= max(4, int(0.6 * len(selected))):
            selected = pruned

    adjacency = defaultdict(list)
    for edge in selected:
        adjacency[edge["a_pid"]].append(edge)
        adjacency[edge["b_pid"]].append(edge)

    print(f"Selected {len(selected)} side-pairs from one-to-one global assignment")
    return selected, adjacency


def refine_with_pose_averaging(placements, adjacency, pieces, anchor, iterations=8, alpha=0.35):
    for _ in range(iterations):
        current_polys = {}
        for pid, (R, t) in placements.items():
            poly = transformed_polygon(pieces[pid]["contour"], R, t)
            if poly is not None:
                current_polys[pid] = poly

        updates = {}
        for pid in placements:
            if pid == anchor:
                continue

            proposals = []
            for edge in adjacency.get(pid, []):
                if edge["a_pid"] == pid:
                    other = edge["b_pid"]
                    pid_side_idx = edge["a_idx"]
                    other_side_idx = edge["b_idx"]
                else:
                    other = edge["a_pid"]
                    pid_side_idx = edge["b_idx"]
                    other_side_idx = edge["a_idx"]

                if other not in placements:
                    continue

                R_other, t_other = placements[other]
                other_side_world = apply_transform(
                    pieces[other]["sides"][other_side_idx].side_points, R_other, t_other
                )
                err, R_p, t_p = estimate_piece_pose_from_side_match(
                    other_side_world,
                    pieces[pid]["sides"][pid_side_idx].side_points,
                )
                if err < 60:
                    proposals.append((R_p, t_p))

            if not proposals:
                continue

            thetas = [math.atan2(R[1, 0], R[0, 0]) for R, _ in proposals]
            theta_avg = math.atan2(np.mean(np.sin(thetas)), np.mean(np.cos(thetas)))
            R_avg = rot2d(theta_avg)
            t_avg = np.mean([t for _, t in proposals], axis=0)

            R_old, t_old = placements[pid]
            theta_old = math.atan2(R_old[1, 0], R_old[0, 0])
            theta_new = (1.0 - alpha) * theta_old + alpha * theta_avg
            R_new = rot2d(theta_new)
            t_new = (1.0 - alpha) * t_old + alpha * t_avg
            new_poly = transformed_polygon(pieces[pid]["contour"], R_new, t_new)
            if new_poly is None:
                continue
            others = {k: v for k, v in current_polys.items() if k != pid}
            if overlap_is_bad(new_poly, others):
                continue
            updates[pid] = (R_new, t_new)

        placements.update(updates)


def attach_remaining_pieces(
    placements,
    pieces,
    candidates,
    used_edges,
    max_err=62.0,
    area_ratio_limit=0.05,
    abs_area_limit=140.0,
):
    piece_ids = sorted(pieces.keys())
    unplaced = {pid for pid in piece_ids if pid not in placements}
    if not unplaced:
        return

    placed_polys = {}
    for pid, (R, t) in placements.items():
        poly = transformed_polygon(pieces[pid]["contour"], R, t)
        if poly is not None:
            placed_polys[pid] = poly

    progress = True
    while progress and unplaced:
        progress = False
        best = None

        for edge in candidates:
            a, b = edge["a_pid"], edge["b_pid"]
            if (a in placements) == (b in placements):
                continue

            if a in placements:
                placed_pid, placed_idx = a, edge["a_idx"]
                free_pid, free_idx = b, edge["b_idx"]
            else:
                placed_pid, placed_idx = b, edge["b_idx"]
                free_pid, free_idx = a, edge["a_idx"]

            if free_pid not in unplaced:
                continue

            R_p, t_p = placements[placed_pid]
            placed_side_world = apply_transform(
                pieces[placed_pid]["sides"][placed_idx].side_points, R_p, t_p
            )
            err, R_f, t_f = estimate_piece_pose_from_side_match(
                placed_side_world,
                pieces[free_pid]["sides"][free_idx].side_points,
            )
            if err > max_err:
                continue

            free_poly = transformed_polygon(pieces[free_pid]["contour"], R_f, t_f)
            if free_poly is None or overlap_is_bad(
                free_poly, placed_polys, area_ratio_limit=area_ratio_limit, abs_area_limit=abs_area_limit
            ):
                continue

            objective = edge["score"] + 1.8 * err
            if best is None or objective < best[0]:
                best = (objective, free_pid, R_f, t_f, placed_pid, placed_idx, free_idx, edge["score"])

        if best is None:
            break

        _, free_pid, R_f, t_f, placed_pid, placed_idx, free_idx, score = best
        placements[free_pid] = (R_f, t_f)
        poly = transformed_polygon(pieces[free_pid]["contour"], R_f, t_f)
        if poly is not None:
            placed_polys[free_pid] = poly
        unplaced.remove(free_pid)
        used_edges.append((placed_pid, placed_idx, free_pid, free_idx, score))
        progress = True


def solve(refresh_sides=False):
    start = time.time()
    if refresh_sides or not os.path.exists("pieces_data.json"):
        print("Refreshing sides from image...")
        getsides(show=False)
    else:
        print("Using existing pieces_data.json")

    pieces = load_pieces()
    piece_ids = sorted(pieces.keys())
    print(f"Loaded {len(piece_ids)} pieces: {piece_ids}")

    candidates, outward, inward = collect_candidates(pieces)
    if not candidates:
        print("No candidates found. Exiting.")
        return

    selected_edges, adjacency = build_match_graph(candidates)
    if not selected_edges:
        print("No selected edges. Exiting.")
        return

    placement_edges = selected_edges
    if len(selected_edges) < len(piece_ids) - 1:
        print("Selected graph is sparse; allowing fallback edges during placement.")
        placement_edges = candidates

    anchor = max(piece_ids, key=lambda pid: len(adjacency.get(pid, [])))
    placements = {anchor: (np.eye(2), np.zeros(2))}
    placed_polys = {}
    anchor_poly = transformed_polygon(pieces[anchor]["contour"], np.eye(2), np.zeros(2))
    if anchor_poly is not None:
        placed_polys[anchor] = anchor_poly

    queue = deque([anchor])
    used_edges = []
    print(f"Anchor piece: {anchor}")

    while queue:
        curr = queue.popleft()
        R_curr, t_curr = placements[curr]

        for edge in adjacency.get(curr, []):
            if edge["a_pid"] == curr:
                nbr = edge["b_pid"]
                curr_side_idx = edge["a_idx"]
                nbr_side_idx = edge["b_idx"]
            else:
                nbr = edge["a_pid"]
                curr_side_idx = edge["b_idx"]
                nbr_side_idx = edge["a_idx"]

            if nbr in placements:
                continue

            curr_side_world = apply_transform(
                pieces[curr]["sides"][curr_side_idx].side_points, R_curr, t_curr
            )
            err, R_nbr, t_nbr = estimate_piece_pose_from_side_match(
                curr_side_world,
                pieces[nbr]["sides"][nbr_side_idx].side_points,
            )

            if err > 42:
                continue

            new_poly = transformed_polygon(pieces[nbr]["contour"], R_nbr, t_nbr)
            if new_poly is None:
                continue
            if overlap_is_bad(new_poly, placed_polys):
                continue

            placements[nbr] = (R_nbr, t_nbr)
            placed_polys[nbr] = new_poly
            queue.append(nbr)
            used_edges.append((curr, curr_side_idx, nbr, nbr_side_idx, edge["score"]))

    progress = True
    while progress:
        progress = False
        for edge in placement_edges:
            a = edge["a_pid"]
            b = edge["b_pid"]
            if (a in placements) == (b in placements):
                continue

            if a in placements:
                placed_pid, placed_idx = a, edge["a_idx"]
                free_pid, free_idx = b, edge["b_idx"]
            else:
                placed_pid, placed_idx = b, edge["b_idx"]
                free_pid, free_idx = a, edge["a_idx"]

            R_p, t_p = placements[placed_pid]
            placed_side_world = apply_transform(
                pieces[placed_pid]["sides"][placed_idx].side_points, R_p, t_p
            )
            err, R_f, t_f = estimate_piece_pose_from_side_match(
                placed_side_world,
                pieces[free_pid]["sides"][free_idx].side_points,
            )
            if err > 42:
                continue

            free_poly = transformed_polygon(pieces[free_pid]["contour"], R_f, t_f)
            if free_poly is None or overlap_is_bad(free_poly, placed_polys):
                continue

            placements[free_pid] = (R_f, t_f)
            placed_polys[free_pid] = free_poly
            used_edges.append((placed_pid, placed_idx, free_pid, free_idx, edge["score"]))
            progress = True

    attach_remaining_pieces(placements, pieces, candidates, used_edges, max_err=62.0)

    used_adjacency = defaultdict(list)
    for a_pid, a_idx, b_pid, b_idx, score in used_edges:
        edge = {"a_pid": a_pid, "a_idx": a_idx, "b_pid": b_pid, "b_idx": b_idx, "score": score}
        used_adjacency[a_pid].append(edge)
        used_adjacency[b_pid].append(edge)

    refine_with_pose_averaging(placements, used_adjacency, pieces, anchor, iterations=10, alpha=0.32)
    placed_polys = {}
    for pid, (R, t) in placements.items():
        poly = transformed_polygon(pieces[pid]["contour"], R, t)
        if poly is not None:
            placed_polys[pid] = poly

    unplaced = [pid for pid in piece_ids if pid not in placements]
    if unplaced:
        print(f"Could not place {len(unplaced)} pieces with confident matches: {unplaced}")
        if placed_polys:
            union = list(placed_polys.values())[0]
            for poly in list(placed_polys.values())[1:]:
                union = union.union(poly)
            minx, miny, maxx, maxy = union.bounds
            cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
            radius = max(maxx - minx, maxy - miny) * 0.8 + 150
        else:
            cx, cy, radius = 0.0, 0.0, 600.0

        for i, pid in enumerate(unplaced):
            ang = 2 * math.pi * i / max(1, len(unplaced))
            tx = cx + radius * math.cos(ang)
            ty = cy + radius * math.sin(ang)
            placements[pid] = (np.eye(2), np.array([tx, ty], dtype=float))

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")
    for i, pid in enumerate(piece_ids):
        R, t = placements[pid]
        contour = apply_transform(pieces[pid]["contour"], R, t)
        if len(contour) < 3:
            continue
        ax.fill(contour[:, 0], contour[:, 1], color=cmap(i % 20), alpha=0.38)
        ax.plot(contour[:, 0], contour[:, 1], color="black", linewidth=1.1)
        center = contour.mean(axis=0)
        ax.text(center[0], center[1], str(pid), ha="center", va="center", fontsize=9)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Solved Piece Layout")
    fig.tight_layout()
    fig.savefig("results/solved_assembly.png", dpi=180)
    plt.close(fig)
    write_assembly_steps_json(placements, pieces, piece_ids, out_path="assembly_steps.json")
    animation_shown = False
    try:
        animation_shown = animate_assembly(placements, pieces, piece_ids, fps=14, show=True)
    except Exception as exc:
        print(f"Animation display skipped: {exc}")

    elapsed = time.time() - start
    print("\n=== SOLVER DONE ===")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Placed: {len([pid for pid in piece_ids if pid in placements])}/{len(piece_ids)}")
    print(f"Accepted side connections: {len(used_edges)}")
    for c in used_edges:
        print(f"  Piece {c[0]} side {c[1]} -> Piece {c[2]} side {c[3]} (score {c[4]:.2f})")
    print("Wrote: solved_assembly.png")
    print("Wrote: assembly_steps.json")
    if animation_shown:
        print("Displayed: assembly animation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble puzzle pieces into solved formation.")
    parser.add_argument(
        "--refresh-sides",
        action="store_true",
        help="Recompute piece sides from source image before solving.",
    )
    args = parser.parse_args()
    solve(refresh_sides=args.refresh_sides)
