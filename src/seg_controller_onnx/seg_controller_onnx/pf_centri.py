import numpy as np
import cv2


def _line(model, y):
    """Valuta il modello lineare x(y) = m*y + q."""
    return model["m"] * y + model["q"]


def _irls(ys, xs, weights, iters=4, sigma=18.0):
    """Fitta una retta con IRLS, riducendo il peso dei punti anomali."""
    ys = np.asarray(ys, np.float32)
    xs = np.asarray(xs, np.float32)
    weights = np.asarray(weights, np.float32)

    if len(ys) < 2:
        return None, None

    # Normalizza i pesi iniziali
    if np.sum(weights) > 1e-6:
        weights = weights / np.sum(weights) * len(weights)
    else:
        weights = np.ones_like(weights)

    coeff = np.polyfit(ys, xs, 1, w=weights)

    for _ in range(iters):
        residuals = xs - (coeff[0] * ys + coeff[1])
        robust_weights = 1.0 / (1.0 + (residuals / max(sigma, 1e-6)) ** 2)
        weights = weights * robust_weights

        if np.sum(weights) < 1e-6:
            break

        coeff = np.polyfit(ys, xs, 1, w=weights)

    model = {"m": float(coeff[0]), "q": float(coeff[1])}
    residuals = np.abs(xs - _line(model, ys))
    return model, residuals


def path_finder(
    pred_mask,
    path_class=1,
    min_row_ratio=0.30,
    n_strips=24,
    min_component_area=500,
    min_interval_width=8,
    max_interval_gap=30,
    border_margin=2,
    min_center_jump=28.0,
    robust_iters=4,
    robust_sigma=18.0,
    seed_bottom_strips=6,
    max_gap_strips=3,
    ok_conf_threshold=0.60,
    degraded_conf_threshold=0.38,
):
    """Trova e traccia il centro di un filare fittando i centri degli intervalli.

    Differenza da pf4: invece di fittare separatamente i bordi sinistro e destro,
    questo algoritmo calcola il centro di ogni intervallo e fitta una singola linea
    centrale usando IRLS per escludere outlier (cespugli, rumore).

    Args:
        pred_mask: Maschera 2D con le classi predette dal modello.
        path_class: Valore della classe che rappresenta il filare.
        min_row_ratio: Quota superiore esclusa dall'analisi.
        n_strips: Numero di strisce orizzontali usate per il tracking.
        min_component_area: Area minima per mantenere una componente connessa.
        min_interval_width: Larghezza minima di un intervallo candidato.
        max_interval_gap: Massimo vuoto orizzontale unito a un intervallo.
        border_margin: Margine per riconoscere intervalli tagliati ai bordi.
        min_center_jump: Salto minimo consentito tra centri di strisce consecutive.
        robust_iters: Iterazioni del fit robusto IRLS.
        robust_sigma: Scala dei residui per ridurre il peso degli outlier.
        seed_bottom_strips: Strisce inferiori candidate per il seed iniziale.
        max_gap_strips: Numero massimo di strisce consecutive senza tracking.
        ok_conf_threshold: Soglia minima di confidence per lo stato ``ok``.
        degraded_conf_threshold: Sotto questa soglia il risultato e' ``lost``.

    Returns:
        Una tupla con la maschera della centerline e un dizionario diagnostico.
        La maschera e' vuota quando il tracking o la validazione falliscono.
    """
    pred_mask = np.asarray(pred_mask)
    H, W = pred_mask.shape
    output = np.zeros((H, W), np.float32)

    # 1. ROI inferiore e binarizzazione
    y0 = int(H * min_row_ratio)
    path = (pred_mask[y0:] == path_class).astype(np.uint8)

    # 2. Morfologia e componenti
    path = cv2.morphologyEx(
        path, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    path = cv2.morphologyEx(
        path, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(path, 8)
    clean = np.zeros_like(path)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            clean[labels == label] = 1
    path = clean

    # 3. Suddivisione in strip
    roi_h = path.shape[0]
    strip_h = max(1, roi_h // n_strips)
    strips = []

    for idx in range(n_strips):
        ys = max(0, roi_h - (idx + 1) * strip_h)
        ye = min(roi_h, roi_h - idx * strip_h)

        if ye <= ys:
            continue

        strip = path[ys:ye]
        profile = strip.sum(axis=0)
        binary = (profile > 0).astype(np.int8)
        diff = np.diff(np.pad(binary, (1, 1)))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        # 4. Estrazione e fusione intervalli
        spans = []
        for start, end in zip(starts, ends):
            if spans and start - spans[-1][1] - 1 <= max_interval_gap:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))

        intervals = []
        for start, end in spans:
            if end - start + 1 < min_interval_width:
                continue
            if start <= border_margin and end >= W - 1 - border_margin:
                if np.mean(strip) < 0.7:
                    continue

            # Calcola il centro e il peso (score) dell'intervallo
            center = 0.5 * (start + end)
            width = end - start + 1
            # Il peso è basato sulla densità di pixel nella strip per questo intervallo
            score = float(profile[start:end+1].sum())

            intervals.append({
                "center": float(center),
                "width": float(width),
                "score": score,
                "start": float(start),
                "end": float(end),
            })

        strips.append({
            "idx": idx,
            "y": y0 + 0.5 * (ys + ye),
            "ys": ys,
            "ye": ye,
            "intervals": intervals,
        })

    if not strips:
        return output, {"status": "lost", "reason": "no_strips"}

    # 6. Scelta del seed
    seeds = []
    for strip in strips[:min(seed_bottom_strips, len(strips))]:
        for interval in strip["intervals"]:
            seeds.append((interval["score"], interval, strip))

    if not seeds:
        return output, {"status": "lost", "reason": "no_seed_candidate"}

    _, current, seed_strip = max(seeds, key=lambda item: (item[0], item[1]["width"]))

    # 7. Tracking bottom-up basato sui centri
    points = [{
        "y": seed_strip["y"],
        "strip_idx": seed_strip["idx"],
        "center": current["center"],
        "width": current["width"],
        "score": current["score"],
    }]

    prev_center = current["center"]
    prev_width = current["width"]
    missed = 0
    max_missed = 0

    for i in range(seed_strip["idx"] + 1, len(strips)):
        jump_limit = max(min_center_jump, 0.5 * prev_width)

        choices = []
        for interval in strips[i]["intervals"]:
            jump = abs(interval["center"] - prev_center)
            if jump <= jump_limit:
                cost = jump + 0.4 * abs(interval["width"] - prev_width)
                choices.append((cost, interval))

        if not choices:
            missed += 1
            max_missed = max(max_missed, missed)
            if missed > max_gap_strips:
                break
            continue

        _, current = min(choices, key=lambda item: item[0])
        strip = strips[i]
        points.append({
            "y": strip["y"],
            "strip_idx": strip["idx"],
            "center": current["center"],
            "width": current["width"],
            "score": current["score"],
        })
        prev_center = current["center"]
        prev_width = current["width"]
        missed = 0

    if len(points) < 3:
        return output, {"status": "lost", "reason": "tracking_failed"}

    # 8. Fit IRLS dei centri
    ys = np.array([p["y"] for p in points], np.float32)
    xs = np.array([p["center"] for p in points], np.float32)
    weights = np.array([p["score"] for p in points], np.float32)

    center_model, residuals = _irls(ys, xs, weights, robust_iters, robust_sigma)
    if center_model is None:
        return output, {"status": "lost", "reason": "fit_failed"}

    # 9. Metriche di qualità
    residual = float(np.mean(residuals))
    tracking = len(points) / max(len(strips), 1)
    fit_score = np.clip(1.0 - residual / max(robust_sigma, 1e-6), 0.0, 1.0)
    gap_score = np.clip(1.0 - max_missed / max(max_gap_strips, 1), 0.0, 1.0)

    # Calcola variabilità della larghezza come metrica di coerenza
    widths = np.array([p["width"] for p in points], dtype=np.float32)
    if len(widths) >= 3:
        diff1 = np.diff(widths)
        diff2 = np.diff(diff1)
        width_error_90 = float(np.percentile(np.abs(diff2), 90))
        width_error = width_error_90 / max(np.median(np.abs(widths)), 1.0)
    else:
        width_error = 0.0

    width_error_scale = 0.30
    width_score = np.clip(1.0 - width_error / max(width_error_scale, 1e-6), 0.0, 1.0)

    confidence = float(
        0.30 * tracking      # % di strip tracciate
        + 0.35 * fit_score   # coerenza geometrica (fit IRLS)
        + 0.25 * width_score # regolarità larghezza
        + 0.10 * gap_score   # continuità tracking
    )

    # Classifica il risultato
    if residual > 2.0 * robust_sigma:
        status = "lost"
    elif confidence < degraded_conf_threshold:
        status = "lost"
    elif confidence < ok_conf_threshold:
        status = "degraded"
    else:
        status = "ok"

    if status == "lost":
        roi_binary = np.zeros((H, W), np.uint8)
        roi_binary[y0:] = path * 255

        return output, {
            "status": status,
            "confidence": confidence,
            "mean_residual": residual,
            "tracking": float(tracking),
            "width_error": width_error,
            "max_gap": max_missed,
            "centerline_model": center_model,
            "line_start_y": None,
            "line_end_y": None,
            "roi_binary": roi_binary,
        }

    # 10. Rendering della centerline
    last_tracked_strip = strips[points[-1]["strip_idx"]] if "strip_idx" in points[-1] else None
    y_start = (
        y0 + last_tracked_strip["ys"]
        if last_tracked_strip is not None
        else int(np.floor(ys.min()))
    )
    y_end = H

    for y in range(y_start, y_end):
        x = int(np.clip(_line(center_model, y), 0, W - 1))
        output[y, x] = 1.0

    # Dilata la linea
    output = cv2.dilate(
        output,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    roi_binary = np.zeros((H, W), np.uint8)
    roi_binary[y0:] = path * 255

    return output, {
        "status": status,
        "confidence": confidence,
        "mean_residual": residual,
        "tracking": float(tracking),
        "width_error": width_error,
        "max_gap": max_missed,
        "centerline_model": center_model,
        "line_start_y": y_start,
        "line_end_y": y_end,
        "roi_binary": roi_binary,
    }


# Alias compatibile retroattivo
path_finder_filari_straight_prior = path_finder
