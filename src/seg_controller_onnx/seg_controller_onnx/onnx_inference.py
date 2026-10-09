from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort


CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (139, 69, 19),
    3: (0, 0, 255),
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def build_onnx_session(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Modello ONNX non trovato: {model_path}")

    available_providers = ort.get_available_providers()
    #requested_providers = ["CPUExecutionProvider"]
    requested_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=requested_providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    active_providers = session.get_providers()
    ort_device = ort.get_device()

    return (
        session,
        input_name,
        output_name,
        available_providers,
        requested_providers,
        active_providers,
        ort_device,
    )


def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = frame_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = (img - MEAN) / STD
    return img.astype(np.float32)


def run_inference_on_frame(session, input_name, output_name, frame_bgr):
    input_tensor = preprocess_frame(frame_bgr)
    output = session.run([output_name], {input_name: input_tensor})[0]
    pred_mask = output.argmax(axis=1)[0].astype(np.uint8)
    return pred_mask


def path_finder_filari(pred_mask):
    path_matrix = np.zeros_like(pred_mask, dtype=np.float32)

    valid_rows = []
    raw_centers = []
    lengths = []

    for row in range(pred_mask.shape[0]):
        ground_cols = np.where(pred_mask[row] == 2)[0]
        if len(ground_cols) > 0:
            start = ground_cols[0]
            end = ground_cols[-1]
            valid_rows.append(row)
            raw_centers.append((start + end) / 2.0)
            lengths.append(end - start + 1)

    if len(raw_centers) < 2:
        return path_matrix

    coeff = np.polyfit(valid_rows, raw_centers, 1)
    line_func = np.poly1d(coeff)

    for i in range(len(valid_rows)):
        row_idx = valid_rows[i]
        perfect_center = int(line_func(row_idx))
        perfect_center = np.clip(perfect_center, 0, pred_mask.shape[1] - 1)
        path_matrix[row_idx, perfect_center] = lengths[i]

    return path_matrix

import numpy as np
import cv2


def path_finder_filari_robust(
    pred_mask,
    path_class=2,
    min_row_ratio=0.45,
    n_strips=24,
    min_component_area=300,
    min_interval_width=8,
    max_interval_width_ratio=0.85,
    max_center_jump=40,
    smooth_center_weight=0.65,
    fit_degree=1,
    robust_iters=3,
    robust_sigma=20.0
):
    """
    Estrae una centerline robusta da una mask di segmentation.

    Input:
        pred_mask: HxW mask intera con class IDs
        path_class: classe del terreno/path
    Output:
        path_matrix: HxW float32, con un pixel acceso per riga/strip
        debug: dict con punti, score e info utili
    """
    # palette = np.array([
    #     [0, 0, 0],      # 0 background
    #     [0, 255, 0],    # 1
    #     [0, 0, 255],    # 2
    #     [255, 0, 0],    # 3
    # ], dtype=np.uint8)

    # vis = palette[pred_mask.astype(np.int32)]
    # cv2.imwrite("/workspaces/workspace/ros2_ws/debug_path_finder/predmask_color.png", vis)

    H, W = pred_mask.shape[:2]
    path_matrix = np.zeros((H, W), dtype=np.float32)

    # ----------------------------
    # 1) ROI: usa solo zona utile
    # ----------------------------
    y0 = int(H * min_row_ratio)
    roi = (pred_mask[y0:, :] == path_class).astype(np.uint8)

    # Pulizia morfologica: riduce frammenti, piccoli buchi, rumore
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k2)

    # Rimuovi componenti troppo piccoli
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8) #Con 4 contano solo i vicini sopra/sotto/destra/sinistra; con 8 contano anche le diagonali.
    filtered = np.zeros_like(roi)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered[labels == lbl] = 1
    roi = filtered
    # img = (roi.astype(np.uint8) * 255)
    # cv2.imwrite("/workspaces/workspace/ros2_ws/debug_path_finder/roi.png", img)

    # ----------------------------------------------------
    # 2) Dividi in strips orizzontali e trova intervalli
    # ----------------------------------------------------
    roi_h = roi.shape[0]
    strip_h = max(1, roi_h // n_strips)

    strips = []
    for s in range(n_strips):
        ys = roi_h - (s + 1) * strip_h
        ye = roi_h - s * strip_h
        ys = max(0, ys)
        ye = min(roi_h, ye)
        if ye <= ys:
            continue

        strip = roi[ys:ye, :]
        profile = strip.sum(axis=0)  # densità path per colonna nella strip
        binary = (profile > 0).astype(np.uint8)

        intervals = []
        in_run = False
        start = 0

        for x in range(W):
            if binary[x] == 1 and not in_run:
                start = x
                in_run = True
            elif binary[x] == 0 and in_run:
                end = x - 1
                width = end - start + 1
                if min_interval_width <= width <= int(W * max_interval_width_ratio):
                    score = profile[start:end + 1].sum()
                    center = 0.5 * (start + end)
                    intervals.append({
                        "start": start,
                        "end": end,
                        "width": width,
                        "center": center,
                        "score": float(score)
                    })
                in_run = False

        if in_run:
            end = W - 1
            width = end - start + 1
            if min_interval_width <= width <= int(W * max_interval_width_ratio):
                score = profile[start:end + 1].sum()
                center = 0.5 * (start + end)
                intervals.append({
                    "start": start,
                    "end": end,
                    "width": width,
                    "center": center,
                    "score": float(score)
                })

        strips.append({
            "ys": ys,
            "ye": ye,
            "y_center_img": y0 + 0.5 * (ys + ye),
            "intervals": intervals
        })

    if len(strips) < 2:
        return path_matrix, {"status": "not_enough_strips"}

    # ----------------------------------------------------
    # 3) Seed dal basso: candidato plausibile iniziale
    # ----------------------------------------------------
    bottom_idx = 0
    bottom_candidates = strips[bottom_idx]["intervals"]

    if len(bottom_candidates) == 0:
        return path_matrix, {"status": "no_bottom_candidate"}

    image_center = W / 2.0

    def seed_cost(cand):
        center_cost = abs(cand["center"] - image_center)
        width_bonus = -0.15 * cand["width"]
        score_bonus = -0.002 * cand["score"]
        return center_cost + width_bonus + score_bonus

    current = min(bottom_candidates, key=seed_cost)
    tracked_points = []
    tracked_points.append({
        "x": current["center"],
        "y": strips[bottom_idx]["y_center_img"],
        "width": current["width"],
        "score": current["score"]
    })

    # ----------------------------------------------------
    # 4) Tracking bottom-up con vincolo di continuità
    # ----------------------------------------------------
    prev_center = current["center"]
    prev_width = current["width"]

    for i in range(1, len(strips)):
        cands = strips[i]["intervals"]
        if len(cands) == 0:
            continue

        valid = []
        for cand in cands:
            jump = abs(cand["center"] - prev_center)
            if jump <= max_center_jump:
                valid.append(cand)

        if len(valid) == 0:
            continue

        def track_cost(cand):
            jump_cost = abs(cand["center"] - prev_center)
            width_cost = 0.35 * abs(cand["width"] - prev_width)
            center_bias = 0.08 * abs(cand["center"] - image_center)
            score_bonus = -0.002 * cand["score"]
            return jump_cost + width_cost + center_bias + score_bonus

        best = min(valid, key=track_cost)

        # smoothing causale del centro
        smoothed_center = (
            smooth_center_weight * prev_center +
            (1.0 - smooth_center_weight) * best["center"]
        )

        tracked_points.append({
            "x": smoothed_center,
            "y": strips[i]["y_center_img"],
            "width": best["width"],
            "score": best["score"]
        })

        prev_center = smoothed_center
        prev_width = best["width"]

    if len(tracked_points) < 3:
        return path_matrix, {"status": "tracking_failed", "points": tracked_points}

    # ----------------------------------------------------
    # 5) Fit robusto iterativo
    # ----------------------------------------------------
    ys = np.array([p["y"] for p in tracked_points], dtype=np.float32)
    xs = np.array([p["x"] for p in tracked_points], dtype=np.float32)
    ws = np.array([max(1.0, p["width"]) for p in tracked_points], dtype=np.float32)

    weights = ws / (ws.max() + 1e-6)

    coeff = np.polyfit(ys, xs, deg=fit_degree, w=weights)
    poly = np.poly1d(coeff)

    for _ in range(robust_iters):
        pred_xs = poly(ys)
        residuals = xs - pred_xs

        robust_w = 1.0 / (1.0 + (residuals / robust_sigma) ** 2)
        final_w = weights * robust_w

        if np.sum(final_w) < 1e-6:
            break

        coeff = np.polyfit(ys, xs, deg=fit_degree, w=final_w)
        poly = np.poly1d(coeff)

    # ----------------------------------------------------
    # 6) Disegna centerline riga per riga solo nella ROI
    # ----------------------------------------------------
    for y in range(y0, H):
        x = int(np.clip(poly(y), 0, W - 1))
        path_matrix[y, x] = 1.0

    # opzionale: allarga leggermente la linea per visualizzarla meglio
    path_matrix = cv2.dilate(
        path_matrix,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    # ----------------------------------------------------
    # 7) Metriche di affidabilità
    # ----------------------------------------------------
    pred_xs = poly(ys)
    residuals = np.abs(xs - pred_xs)

    confidence = float(
        0.4 * min(1.0, len(tracked_points) / max(6, len(strips))) +
        0.3 * np.clip(np.mean(ws) / (0.15 * W + 1e-6), 0, 1) +
        0.3 * np.clip(1.0 - (np.mean(residuals) / max(robust_sigma, 1e-6)), 0, 1)
    )

    debug = {
        "status": "ok",
        "points": tracked_points,
        "poly_coeff": coeff.tolist(),
        "fit_degree": fit_degree,
        "mean_residual": float(np.mean(residuals)),
        "confidence": confidence,
        "roi_start_y": y0
    }

    return path_matrix, debug




def path_finder_filari_robust_monitored(
    pred_mask,
    path_class=2,
    prev_state=None,
    min_row_ratio=0.45,
    n_strips=24,
    min_component_area=500,
    min_interval_width=8,
    max_interval_width_ratio=0.85,
    max_center_jump=40.0,
    smooth_center_weight=0.65,
    fit_degree=1,
    robust_iters=3,
    robust_sigma=20.0,
    band_half_width=3,
    bottom_band_half_width=6,
    ok_conf_threshold=0.60,
    degraded_conf_threshold=0.35,
    max_temporal_center_jump=60.0,
    max_heading_jump_deg=12.0,
):
    H, W = pred_mask.shape[:2]
    path_matrix = np.zeros((H, W), dtype=np.float32)

    # 1) ROI inferiore
    y0 = int(H * min_row_ratio)
    roi_mask = pred_mask[y0:, :]
    roi = (roi_mask == path_class).astype(np.uint8)

    # 2) Pulizia morfologica
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k2)

    # 3) Filtraggio componenti piccole
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    filtered = np.zeros_like(roi)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered[labels == lbl] = 1
    roi = filtered

    # 4) Strips orizzontali
    roi_h = roi.shape[0]
    strip_h = max(1, roi_h // n_strips)

    strips = []
    for s in range(n_strips):
        ys = roi_h - (s + 1) * strip_h
        ye = roi_h - s * strip_h
        ys = max(0, ys)
        ye = min(roi_h, ye)
        if ye <= ys:
            continue

        strip = roi[ys:ye, :]
        profile = strip.sum(axis=0)
        binary = (profile > 0).astype(np.uint8)

        intervals = []
        in_run = False
        start = 0

        for x in range(W):
            if binary[x] == 1 and not in_run:
                start = x
                in_run = True
            elif binary[x] == 0 and in_run:
                end = x - 1
                width = end - start + 1
                if min_interval_width <= width <= int(W * max_interval_width_ratio):
                    score = float(profile[start:end + 1].sum())
                    center = 0.5 * (start + end)
                    intervals.append({
                        "start": start,
                        "end": end,
                        "width": width,
                        "center": center,
                        "score": score,
                    })
                in_run = False

        if in_run:
            end = W - 1
            width = end - start + 1
            if min_interval_width <= width <= int(W * max_interval_width_ratio):
                score = float(profile[start:end + 1].sum())
                center = 0.5 * (start + end)
                intervals.append({
                    "start": start,
                    "end": end,
                    "width": width,
                    "center": center,
                    "score": score,
                })

        strips.append({
            "ys": ys,
            "ye": ye,
            "y_center_img": y0 + 0.5 * (ys + ye),
            "intervals": intervals,
        })

    if len(strips) < 2:
        return path_matrix, {
            "status": "lost",
            "reason": "not_enough_strips",
        }

    image_center = W / 2.0

    # 5) Seed basso
    bottom_candidates = strips[0]["intervals"]
    if len(bottom_candidates) == 0:
        return path_matrix, {
            "status": "lost",
            "reason": "no_bottom_candidate",
        }

    def seed_cost(cand):
        center_cost = abs(cand["center"] - image_center)
        width_bonus = -0.15 * cand["width"]
        score_bonus = -0.002 * cand["score"]

        temporal_cost = 0.0
        if prev_state is not None and prev_state.get("status", "lost") != "lost":
            prev_center = prev_state.get("bottom_center", None)
            if prev_center is not None:
                temporal_cost = 0.35 * abs(cand["center"] - prev_center)

        return center_cost + temporal_cost + width_bonus + score_bonus

    current = min(bottom_candidates, key=seed_cost)

    # 6) Tracking bottom-up
    tracked_points = [{
        "x": current["center"],
        "y": strips[0]["y_center_img"],
        "width": current["width"],
        "score": current["score"],
    }]

    prev_center = current["center"]
    prev_width = current["width"]

    for i in range(1, len(strips)):
        cands = strips[i]["intervals"]
        if len(cands) == 0:
            continue

        valid = []
        for cand in cands:
            if abs(cand["center"] - prev_center) <= max_center_jump:
                valid.append(cand)

        if len(valid) == 0:
            continue

        def track_cost(cand):
            jump_cost = abs(cand["center"] - prev_center)
            width_cost = 0.35 * abs(cand["width"] - prev_width)
            center_bias = 0.08 * abs(cand["center"] - image_center)
            score_bonus = -0.002 * cand["score"]
            return jump_cost + width_cost + center_bias + score_bonus

        best = min(valid, key=track_cost)

        smoothed_center = (
            smooth_center_weight * prev_center +
            (1.0 - smooth_center_weight) * best["center"]
        )

        tracked_points.append({
            "x": smoothed_center,
            "y": strips[i]["y_center_img"],
            "width": best["width"],
            "score": best["score"],
        })

        prev_center = smoothed_center
        prev_width = best["width"]

    if len(tracked_points) < 3:
        return path_matrix, {
            "status": "lost",
            "reason": "tracking_failed",
            "points": tracked_points,
        }

    # 7) Fit lineare robusto
    ys = np.array([p["y"] for p in tracked_points], dtype=np.float32)
    xs = np.array([p["x"] for p in tracked_points], dtype=np.float32)
    ws = np.array([max(1.0, p["width"]) for p in tracked_points], dtype=np.float32)

    weights = ws / (ws.max() + 1e-6)

    coeff = np.polyfit(ys, xs, deg=fit_degree, w=weights)
    poly = np.poly1d(coeff)

    for _ in range(robust_iters):
        pred_xs = poly(ys)
        residuals = xs - pred_xs
        robust_w = 1.0 / (1.0 + (residuals / robust_sigma) ** 2)
        final_w = weights * robust_w

        if np.sum(final_w) < 1e-6:
            break

        coeff = np.polyfit(ys, xs, deg=fit_degree, w=final_w)
        poly = np.poly1d(coeff)

    # 8) Disegno centerline + metriche globali
    line_hits = 0
    band_hits = 0
    band_total = 0

    for y in range(y0, H):
        x = int(np.clip(poly(y), 0, W - 1))
        path_matrix[y, x] = 1.0

        if pred_mask[y, x] == path_class:
            line_hits += 1

        xl = max(0, x - band_half_width)
        xr = min(W, x + band_half_width + 1)
        band = pred_mask[y, xl:xr]
        band_hits += np.sum(band == path_class)
        band_total += band.shape[0]

    path_matrix = cv2.dilate(
        path_matrix,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    # 9) Metriche fit
    pred_xs = poly(ys)
    residuals = np.abs(xs - pred_xs)

    support_ratio = float(len(tracked_points) / max(1, len(strips)))
    line_on_path_ratio = float(line_hits / max(1, H - y0))
    band_on_path_ratio = float(band_hits / max(1, band_total))

    # 10) Bottom center corretto: dalla retta finale
    bottom_center = float(np.clip(poly(H - 1), 0, W - 1))
    lateral_error_px = float(bottom_center - image_center)

    bottom_y = H - 1
    bottom_x = int(round(bottom_center))

    bottom_center_is_path = bool(pred_mask[bottom_y, bottom_x] == path_class)

    bottom_xl = max(0, bottom_x - bottom_band_half_width)
    bottom_xr = min(W, bottom_x + bottom_band_half_width + 1)
    bottom_band = pred_mask[bottom_y, bottom_xl:bottom_xr]
    bottom_band_path_ratio = float(
        np.sum(bottom_band == path_class) / max(1, bottom_band.shape[0])
    )

    # 11) Heading e coerenza temporale
    heading_deg = None
    if len(coeff) >= 2:
        heading_deg = float(np.degrees(np.arctan(coeff[0])))

    temporal_center_jump = 0.0
    heading_jump_deg = 0.0

    if prev_state is not None:
        prev_center_ref = prev_state.get("bottom_center", None)
        prev_heading_ref = prev_state.get("heading_deg", None)

        if prev_center_ref is not None:
            temporal_center_jump = float(abs(bottom_center - prev_center_ref))

        if prev_heading_ref is not None and heading_deg is not None:
            heading_jump_deg = float(abs(heading_deg - prev_heading_ref))

    # 12) Confidence semplice ma robusta
    confidence = float(
        0.28 * np.clip(support_ratio, 0.0, 1.0) +
        0.22 * np.clip(line_on_path_ratio, 0.0, 1.0) +
        0.28 * np.clip(band_on_path_ratio, 0.0, 1.0) +
        0.12 * np.clip(bottom_band_path_ratio, 0.0, 1.0) +
        0.10 * np.clip(1.0 - (np.mean(residuals) / max(robust_sigma, 1e-6)), 0.0, 1.0)
    )

    # 13) Classificazione stato
    degraded_flags = 0
    if confidence < ok_conf_threshold:
        degraded_flags += 1
    if band_on_path_ratio < 0.65:
        degraded_flags += 1
    if support_ratio < 0.45:
        degraded_flags += 1
    if heading_jump_deg > max_heading_jump_deg:
        degraded_flags += 1
    if temporal_center_jump > max_temporal_center_jump:
        degraded_flags += 1
    if bottom_band_path_ratio < 0.35:
        degraded_flags += 1

    if confidence < degraded_conf_threshold or support_ratio < 0.25:
        status = "lost"
    elif degraded_flags >= 2:
        status = "degraded"
    else:
        status = "ok"

    debug = {
        "status": status,
        "points": tracked_points,
        "poly_coeff": coeff.tolist(),
        "fit_degree": fit_degree,
        "mean_residual": float(np.mean(residuals)),
        "confidence": confidence,
        "support_ratio": support_ratio,
        "line_on_path_ratio": line_on_path_ratio,
        "band_on_path_ratio": band_on_path_ratio,
        "bottom_center": bottom_center,
        "bottom_center_is_path": bottom_center_is_path,
        "bottom_band_path_ratio": bottom_band_path_ratio,
        "lateral_error_px": lateral_error_px,
        "heading_deg": heading_deg,
        "temporal_center_jump": temporal_center_jump,
        "heading_jump_deg": heading_jump_deg,
        "roi_start_y": y0,
    }

    return path_matrix, debug   

    

