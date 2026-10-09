import numpy as np
import cv2


def _find_segments(binary_row):
    binary_row = binary_row.astype(np.uint8)
    padded = np.pad(binary_row, (1, 1), mode="constant")
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))


def _moving_average_1d(arr, k=5):
    if len(arr) < k:
        return arr.copy()
    kernel = np.ones(k, dtype=np.float32) / k
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _polyfit_safe(y, x, deg=2, w=None):
    if len(y) < deg + 1:
        return None
    try:
        if w is None:
            return np.poly1d(np.polyfit(y, x, deg))
        return np.poly1d(np.polyfit(y, x, deg, w=w))
    except Exception:
        return None


def _clip_int(x, low, high):
    return int(np.clip(int(round(x)), low, high))


def _robust_polyfit_iter(y, x, deg=2, base_w=None, iters=3, sigma=20.0):
    if len(y) < deg + 1:
        return None
    if base_w is None:
        base_w = np.ones_like(x, dtype=np.float32)

    poly = _polyfit_safe(y, x, deg=deg, w=base_w)
    if poly is None:
        return None

    for _ in range(iters):
        pred = poly(y)
        res = x - pred
        robust_w = 1.0 / (1.0 + (res / max(sigma, 1e-6)) ** 2)
        final_w = base_w * robust_w
        if np.sum(final_w) < 1e-6:
            break
        poly = _polyfit_safe(y, x, deg=deg, w=final_w)
        if poly is None:
            break
    return poly


def extract_corridor_lines_robust(
    pred_mask,
    path_class=2,
    roi_top_ratio=0.20,
    bottom_band_ratio=0.22,
    n_strips=20,
    min_component_area=60,
    min_seg_width=2,
    max_seg_width_ratio=0.90,
    max_center_jump=80,
    max_width_jump=70,
    smooth_k=5,
    fit_degree=1,
    morph_kernel_open=0,
    morph_kernel_close=5,
    max_tracks=6,
    min_track_points=4,
    robust_sigma=24.0
):
    H, W = pred_mask.shape[:2]

    left_line_map = np.zeros((H, W), dtype=np.float32)
    right_line_map = np.zeros((H, W), dtype=np.float32)
    center_line_map = np.zeros((H, W), dtype=np.float32)

    path_mask = (pred_mask == path_class).astype(np.uint8)

    roi_top = int(H * roi_top_ratio)
    roi_mask = np.zeros_like(path_mask, dtype=np.uint8)
    roi_mask[roi_top:, :] = 1
    path_mask = path_mask * roi_mask

    if morph_kernel_open > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_open, morph_kernel_open))
        path_mask = cv2.morphologyEx(path_mask, cv2.MORPH_OPEN, k_open)

    if morph_kernel_close > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_close, morph_kernel_close))
        path_mask = cv2.morphologyEx(path_mask, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(path_mask, connectivity=8)
    filtered = np.zeros_like(path_mask)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered[labels == lbl] = 1
    path_mask = filtered

    bottom_h = max(5, int(H * bottom_band_ratio))
    bottom_start = H - bottom_h
    bottom_band = path_mask[bottom_start:, :]
    bottom_score = bottom_band.sum(axis=0).astype(np.float32)

    if bottom_score.max() <= 0:
        return left_line_map, right_line_map, center_line_map, {
            "status": "no_path_in_bottom_band"
        }

    roi_h = H - roi_top
    strip_h = max(1, roi_h // n_strips)

    strips = []
    max_seg_width = int(W * max_seg_width_ratio)

    for s in range(n_strips):
        ys = H - (s + 1) * strip_h
        ye = H - s * strip_h
        ys = max(roi_top, ys)
        ye = min(H, ye)
        if ye <= ys:
            continue

        strip = path_mask[ys:ye, :]
        profile = strip.sum(axis=0).astype(np.float32)
        binary = (profile > 0).astype(np.uint8)

        segs = _find_segments(binary)
        candidates = []

        for x0, x1 in segs:
            width = x1 - x0 + 1
            if width < min_seg_width or width > max_seg_width:
                continue

            local_score = float(profile[x0:x1 + 1].sum())
            center = 0.5 * (x0 + x1)

            candidates.append({
                "x0": int(x0),
                "x1": int(x1),
                "width": float(width),
                "center": float(center),
                "score": local_score,
            })

        strips.append({
            "ys": int(ys),
            "ye": int(ye),
            "y": float(0.5 * (ys + ye)),
            "candidates": candidates
        })

    if len(strips) < 3:
        return left_line_map, right_line_map, center_line_map, {
            "status": "not_enough_strips"
        }

    bottom_candidates = strips[0]["candidates"]
    if len(bottom_candidates) == 0:
        return left_line_map, right_line_map, center_line_map, {
            "status": "no_bottom_candidates"
        }

    def seed_cost(c):
        return (
            center_bias_weight * abs(c["center"] - W / 2.0)
            - score_weight * c["score"]
        )

    bottom_candidates_sorted = sorted(bottom_candidates, key=seed_cost)
    seeds = bottom_candidates_sorted[:max_tracks]

    tracks = []
    for c in seeds:
        tracks.append({
            "points": [{
                "y": strips[0]["y"],
                "x0": c["x0"],
                "x1": c["x1"],
                "xc": c["center"],
                "width": c["width"],
                "score": c["score"],
                "mode": "two_edges"
            }],
            "alive": True,
            "misses": 0,
            "total_score": c["score"]
        })

    for i in range(1, len(strips)):
        y = strips[i]["y"]
        cands = strips[i]["candidates"]

        for tr in tracks:
            if not tr["alive"]:
                continue

            last = tr["points"][-1]
            prev_center = last["xc"]
            prev_width = last["width"]

            if len(cands) == 0:
                tr["misses"] += 1
                tr["total_score"] -= gap_penalty
                if tr["misses"] > 2:
                    tr["alive"] = False
                continue

            scored = []
            for c in cands:
                jump = abs(c["center"] - prev_center)
                width_jump = abs(c["width"] - prev_width)

                if jump > max_center_jump:
                    continue
                if width_jump > max_width_jump:
                    continue

                cost = (
                    1.25 * jump
                    + 0.45 * width_jump
                    + center_bias_weight * abs(c["center"] - W / 2.0)
                    - score_weight * c["score"]
                )
                scored.append((cost, c))

            if len(scored) == 0:
                tr["misses"] += 1
                tr["total_score"] -= gap_penalty
                if tr["misses"] > 2:
                    tr["alive"] = False
                continue

            scored.sort(key=lambda x: x[0])
            _, best = scored[0]

            smoothed_center = 0.65 * prev_center + 0.35 * best["center"]
            smoothed_width = 0.70 * prev_width + 0.30 * best["width"]

            x0_pred = smoothed_center - 0.5 * smoothed_width
            x1_pred = smoothed_center + 0.5 * smoothed_width

            tr["points"].append({
                "y": y,
                "x0": max(0.0, x0_pred),
                "x1": min(W - 1.0, x1_pred),
                "xc": smoothed_center,
                "width": smoothed_width,
                "score": best["score"],
                "mode": "two_edges"
            })
            tr["misses"] = 0
            tr["total_score"] += best["score"]

    valid_tracks = []
    for tr in tracks:
        pts = tr["points"]
        if len(pts) < min_track_points:
            continue

        ys = np.array([p["y"] for p in pts], dtype=np.float32)
        xcs = np.array([p["xc"] for p in pts], dtype=np.float32)
        ws = np.array([p["width"] for p in pts], dtype=np.float32)

        coverage = len(pts) / max(1, len(strips))
        width_std = float(np.std(ws))
        center_dev = float(np.mean(np.abs(xcs - W / 2.0)))

        global_score = (
            2.0 * coverage
            + 0.003 * tr["total_score"]
            - 0.015 * width_std
            - 0.004 * center_dev
        )

        valid_tracks.append({
            "track": tr,
            "coverage": coverage,
            "mean_width": float(np.mean(ws)),
            "width_std": width_std,
            "center_dev": center_dev,
            "global_score": global_score
        })

    if len(valid_tracks) == 0:
        return left_line_map, right_line_map, center_line_map, {
            "status": "no_valid_track",
            "num_tracks": len(tracks)
        }

    valid_tracks.sort(key=lambda d: d["global_score"], reverse=True)
    best_track_info = valid_tracks[0]
    pts = best_track_info["track"]["points"]

    ys = np.array([p["y"] for p in pts], dtype=np.float32)
    x0s = np.array([p["x0"] for p in pts], dtype=np.float32)
    x1s = np.array([p["x1"] for p in pts], dtype=np.float32)
    xcs = np.array([p["xc"] for p in pts], dtype=np.float32)
    ws = np.array([max(1.0, p["width"]) for p in pts], dtype=np.float32)
    base_w = ws / (np.max(ws) + 1e-6)

    x0s_s = _moving_average_1d(x0s, k=smooth_k)
    x1s_s = _moving_average_1d(x1s, k=smooth_k)
    xcs_s = _moving_average_1d(xcs, k=smooth_k)

    deg = min(fit_degree, len(ys) - 1)
    left_fit = _robust_polyfit_iter(ys, x0s_s, deg=deg, base_w=base_w, iters=3, sigma=robust_sigma)
    right_fit = _robust_polyfit_iter(ys, x1s_s, deg=deg, base_w=base_w, iters=3, sigma=robust_sigma)
    center_fit = _robust_polyfit_iter(ys, xcs_s, deg=deg, base_w=base_w, iters=3, sigma=robust_sigma)

    corridor_widths = []
    for y in range(roi_top, H):
        xl = float(left_fit(y)) if left_fit is not None else None
        xr = float(right_fit(y)) if right_fit is not None else None
        xc = float(center_fit(y)) if center_fit is not None else None

        if xl is not None:
            xl = np.clip(xl, 0, W - 1)
        if xr is not None:
            xr = np.clip(xr, 0, W - 1)
        if xc is not None:
            xc = np.clip(xc, 0, W - 1)

        xc_edges = None
        if xl is not None and xr is not None and xr > xl:
            xc_edges = 0.5 * (xl + xr)
            corridor_widths.append(xr - xl)

        if xc_edges is not None and xc is not None:
            width_here = max(1.0, xr - xl)
            alpha = 0.78 if width_here >= min_seg_width else 0.55
            x_final = alpha * xc_edges + (1.0 - alpha) * xc
        elif xc_edges is not None:
            x_final = xc_edges
        elif xc is not None:
            x_final = xc
        else:
            continue

        if xl is not None:
            left_line_map[y, _clip_int(xl, 0, W - 1)] = 1.0
        if xr is not None:
            right_line_map[y, _clip_int(xr, 0, W - 1)] = 1.0

        center_line_map[y, _clip_int(x_final, 0, W - 1)] = 1.0

    left_line_map = cv2.dilate(left_line_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    right_line_map = cv2.dilate(right_line_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    center_line_map = cv2.dilate(center_line_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    if center_fit is not None:
        pred_xc = center_fit(ys)
        residual = np.abs(xcs - pred_xc)
        mean_residual = float(np.mean(residual))
    else:
        mean_residual = 1e6

    mean_width = float(np.mean(corridor_widths)) if len(corridor_widths) > 0 else None
    confidence = float(
        0.35 * np.clip(best_track_info["coverage"], 0.0, 1.0) +
        0.25 * np.clip((best_track_info["mean_width"] or 0.0) / (0.20 * W + 1e-6), 0.0, 1.0) +
        0.25 * np.clip(1.0 - mean_residual / max(robust_sigma, 1e-6), 0.0, 1.0) +
        0.15 * np.clip(1.0 - best_track_info["width_std"] / max(0.15 * W, 1e-6), 0.0, 1.0)
    )

    debug = {
        "status": "ok",
        "roi_top": int(roi_top),
        "num_strips": int(len(strips)),
        "num_tracks_seeded": int(len(tracks)),
        "num_valid_tracks": int(len(valid_tracks)),
        "best_track_global_score": float(best_track_info["global_score"]),
        "coverage": float(best_track_info["coverage"]),
        "mean_corridor_width": mean_width,
        "width_std": float(best_track_info["width_std"]),
        "center_dev": float(best_track_info["center_dev"]),
        "mean_residual": float(mean_residual),
        "confidence": confidence,
        "tracked_points": pts,
        "all_track_scores": [float(v["global_score"]) for v in valid_tracks],
        "left_fit": left_fit,
        "right_fit": right_fit,
        "center_fit": center_fit,
    }

    return left_line_map, right_line_map, center_line_map, debug