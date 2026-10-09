import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Fit lineare pesato con reiezione robusta degli outlier (usato per i bordi).
# ---------------------------------------------------------------------------


def _line_x_from_model(model, y):
    return model["m"] * y + model["q"]


def _fit_line_weighted(ys, xs, ws):
    ys = np.asarray(ys, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)
    ws = np.asarray(ws, dtype=np.float32)

    if len(ys) < 2 or np.sum(ws) < 1e-6:
        return None

    coeff = np.polyfit(ys, xs, deg=1, w=ws)
    return {"m": float(coeff[0]), "q": float(coeff[1])}


def _fit_line_weighted_robust(ys, xs, base_w, robust_iters=4, robust_sigma=18.0):
    model = _fit_line_weighted(ys, xs, base_w)
    if model is None:
        return None, None

    ys = np.asarray(ys, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)
    base_w = np.asarray(base_w, dtype=np.float32)

    for _ in range(robust_iters):
        residuals = xs - _line_x_from_model(model, ys)
        robust_w = 1.0 / (1.0 + (residuals / max(robust_sigma, 1e-6)) ** 2)
        final_w = base_w * robust_w

        if np.sum(final_w) < 1e-6:
            break

        new_model = _fit_line_weighted(ys, xs, final_w)
        if new_model is None:
            break
        model = new_model

    residuals = np.abs(xs - _line_x_from_model(model, ys))
    return model, residuals


def path_finder(
    pred_mask,
    path_class=1,
    min_row_ratio=0.20,
    n_strips=24,
    min_component_area=500,
    min_interval_width=8,
    max_interval_gap=30,
    border_margin=2,
    min_edge_jump=42.0,
    robust_iters=4,
    robust_sigma=18.0,
    band_half_width=3,
    local_band_half_width=5,
    seed_bottom_strips=6,
    local_support_th=0.27,
    ok_conf_threshold=0.60,
    degraded_conf_threshold=0.38,
):
    H, W = pred_mask.shape[:2]
    path_matrix = np.zeros((H, W), dtype=np.float32)
    image_center = (W - 1) / 2.0

    # 1. ROI: solo la parte inferiore dell'immagine.
    y0 = int(H * min_row_ratio)
    roi_mask = pred_mask[y0:, :]
    roi_path = (roi_mask == path_class).astype(np.uint8)

    # 2. Pulizia morfologica.
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    roi_path = cv2.morphologyEx(roi_path, cv2.MORPH_OPEN, k1)
    roi_path = cv2.morphologyEx(roi_path, cv2.MORPH_CLOSE, k2)

    # 3. Rimozione componenti troppo piccole per essere il corridoio.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_path, connectivity=8)
    filtered = np.zeros_like(roi_path)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_component_area:
            filtered[labels == lbl] = 1
    roi_path = filtered

    # 4. Per ogni strip orizzontale, trova gli intervalli path (solo bordi start/end).
    roi_h = roi_path.shape[0]
    strip_h = max(1, roi_h // n_strips)

    strips = []
    for s in range(n_strips):
        ys = max(0, roi_h - (s + 1) * strip_h)
        ye = min(roi_h, roi_h - s * strip_h)
        if ye <= ys:
            continue

        strip_path = roi_path[ys:ye, :]
        profile = strip_path.sum(axis=0)
        binary = (profile > 0).astype(np.int8)
        padded = np.pad(binary, (1, 1), mode="constant")
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        # Richiude i buchi piccoli
        raw_spans = list(zip(starts.tolist(), ends.tolist()))
        merged_spans = []
        for start, end in raw_spans:
            if merged_spans and (start - merged_spans[-1][1] - 1) <= max_interval_gap:
                merged_spans[-1] = (merged_spans[-1][0], end)
            else:
                merged_spans.append((start, end))

        intervals = []
        for start, end in merged_spans:
            width = end - start + 1
            if width < min_interval_width:
                continue

            touches_both = (start <= border_margin) and (end >= (W - 1 - border_margin))

            if touches_both:
                full_strip_mean = float(np.mean(strip_path))
                if full_strip_mean < 0.7:
                    continue

            # Calcola local_path_support PRIMA di appendere
            cx = int(round(0.5 * (start + end)))
            xl = max(0, cx - local_band_half_width)
            xr = min(W, cx + local_band_half_width + 1)
            local_path_support = float(np.mean(strip_path[:, xl:xr] == 1))

            intervals.append({
                "start": float(start),
                "end": float(end),
                "width": float(width),
                "local_path_support": local_path_support,
                "touches_left": bool(start <= border_margin),
                "touches_right": bool(end >= (W - 1 - border_margin)),
            })

        strips.append({
            "strip_idx": s,
            "ys": ys,
            "ye": ye,
            "y_center_img": y0 + 0.5 * (ys + ye),
            "intervals": intervals,
        })

    if len(strips) == 0:
        return path_matrix, {"status": "lost", "reason": "no_strips"}

    # 5. Seed: la strip valida più bassa.
    bottom_strips = strips[:min(seed_bottom_strips, len(strips))]

    valid_bottom_strips = []
    for s in bottom_strips:
        valid_cands = [c for c in s["intervals"] if c["local_path_support"] >= local_support_th]
        if len(valid_cands) > 0:
            valid_bottom_strips.append({"strip": s, "cands": valid_cands})

    if len(valid_bottom_strips) == 0:
        return path_matrix, {"status": "lost", "reason": "no_seed_candidate"}

    seed_entry = min(valid_bottom_strips, key=lambda item: item["strip"]["strip_idx"])
    seed_strip = seed_entry["strip"]
    seed_cands = seed_entry["cands"]

    current = min(
        seed_cands,
        key=lambda c: (abs(0.5 * (c["start"] + c["end"]) - image_center), -c["width"])
    )

    # 6. Tracking bottom-up
    tracked_points = [{
        "x0": current["start"],
        "x1": current["end"],
        "y": seed_strip["y_center_img"],
        "width": current["width"],
        "strip_idx": seed_strip["strip_idx"],
        "touches_left": current["touches_left"],
        "touches_right": current["touches_right"],
    }]

    prev_x0 = current["start"]
    prev_x1 = current["end"]
    prev_width = current["width"]

    for i in range(seed_strip["strip_idx"] + 1, len(strips)):
        candidates = strips[i]["intervals"]
        if len(candidates) == 0:
            continue

        max_edge_jump = max(min_edge_jump, 0.5 * prev_width)

        valid = []
        for cand in candidates:
            if cand["local_path_support"] < local_support_th:
                continue
            edge_jump = max(abs(cand["start"] - prev_x0), abs(cand["end"] - prev_x1))
            if edge_jump <= max_edge_jump:
                valid.append((cand, edge_jump))

        if len(valid) == 0:
            continue

        best, best_jump = min(valid, key=lambda item: item[1] + 0.1 * abs(item[0]["width"] - prev_width))

        tracked_points.append({
            "x0": best["start"],
            "x1": best["end"],
            "y": strips[i]["y_center_img"],
            "width": best["width"],
            "strip_idx": strips[i]["strip_idx"],
            "touches_left": best["touches_left"],
            "touches_right": best["touches_right"],
        })

        prev_x0 = best["start"]
        prev_x1 = best["end"]
        prev_width = best["width"]

    if len(tracked_points) < 3:
        return path_matrix, {"status": "lost", "reason": "tracking_failed", "points": tracked_points}

    # 7. Fit robusto dei bordi
    ys = np.array([p["y"] for p in tracked_points], dtype=np.float32)
    left_xs = np.array([p["x0"] for p in tracked_points], dtype=np.float32)
    right_xs = np.array([p["x1"] for p in tracked_points], dtype=np.float32)
    left_not_clipped = np.array([not p["touches_left"] for p in tracked_points])
    right_not_clipped = np.array([not p["touches_right"] for p in tracked_points])

    left_fit_mask = left_not_clipped
    right_fit_mask = right_not_clipped

    if np.sum(left_fit_mask) < 2 or np.sum(right_fit_mask) < 2:
        return path_matrix, {
            "status": "lost",
            "reason": "insufficient_unclipped_edge_points",
            "left_unclipped_points": int(np.sum(left_fit_mask)),
            "right_unclipped_points": int(np.sum(right_fit_mask)),
        }

    uniform_weights = np.ones_like(ys)

    left_model, left_residuals = _fit_line_weighted_robust(
        ys[left_fit_mask], left_xs[left_fit_mask], uniform_weights[left_fit_mask],
        robust_iters=robust_iters, robust_sigma=robust_sigma,
    )
    right_model, right_residuals = _fit_line_weighted_robust(
        ys[right_fit_mask], right_xs[right_fit_mask], uniform_weights[right_fit_mask],
        robust_iters=robust_iters, robust_sigma=robust_sigma,
    )
    if left_model is None or right_model is None:
        return path_matrix, {"status": "lost", "reason": "fit_failed"}

    model = {
        "m": 0.5 * (left_model["m"] + right_model["m"]),
        "q": 0.5 * (left_model["q"] + right_model["q"]),
    }

    # 8. Rasterizzazione della centerline
    # Per ogni y, calcola la centerline SOLO se ENTRAMBI i bordi sono affidabili in quella zona.
    # Se un bordo è clippato, usa solo il bordo affidabile per stimare la centerline.
    
    first_tracked_strip = strips[tracked_points[0]["strip_idx"]]
    last_tracked_strip = strips[tracked_points[-1]["strip_idx"]]
    line_start_y = y0 + last_tracked_strip["ys"]
    line_end_y = min(H, y0 + first_tracked_strip["ye"])

    band_hits = 0
    band_total = 0
    rasterized_rows = 0
    
    # Proietta la media delle due rette, stimate esclusivamente da bordi non clippati.
    for y in range(line_start_y, line_end_y):
        x = int(np.clip(_line_x_from_model(model, y), 0, W - 1))
        
        path_matrix[y, x] = 1.0
        rasterized_rows += 1

        xl = max(0, x - band_half_width)
        xr = min(W, x + band_half_width + 1)
        band = pred_mask[y, xl:xr]
        band_hits += np.sum(band == path_class)
        band_total += band.size

    path_matrix = cv2.dilate(
        path_matrix, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    # 9. Qualitàıı del fit e stato finale.
    strip_tracking_ratio = float(len(tracked_points) / max(1, len(strips)))
    band_on_path_ratio = float(band_hits / max(1, band_total))
    residuals = np.concatenate((left_residuals, right_residuals))
    mean_residual = float(np.mean(residuals)) if residuals.size else 999.0
    residual_score = float(np.clip(1.0 - (mean_residual / max(robust_sigma, 1e-6)), 0.0, 1.0))

    confidence = float(
        0.50 * np.clip(strip_tracking_ratio, 0.0, 1.0)
        + 0.30 * np.clip(band_on_path_ratio, 0.0, 1.0)
        + 0.20 * residual_score
    )

    if confidence < degraded_conf_threshold or strip_tracking_ratio < 0.25:
        status = "lost"
    elif confidence < ok_conf_threshold:
        status = "degraded"
    else:
        status = "ok"

    # 10. Payload di debug
    roi_binary_full = np.zeros((H, W), dtype=np.uint8)
    roi_binary_full[y0:, :] = roi_path * 255

    debug = {
        "status": status,
        "confidence": confidence,
        "band_on_path_ratio": band_on_path_ratio,
        "strip_tracking_ratio": strip_tracking_ratio,
        "mean_residual": mean_residual,
        "residual_score": residual_score,
        "roi_binary": roi_binary_full,
        "left_edge_model": left_model,
        "right_edge_model": right_model,
        "centerline_model": model,
        "line_start_y": line_start_y,
        "line_end_y": line_end_y,
        "left_clipped_points": int(np.sum(~left_not_clipped)),
        "right_clipped_points": int(np.sum(~right_not_clipped)),
        "rasterized_rows": rasterized_rows,
    }
    return path_matrix, debug