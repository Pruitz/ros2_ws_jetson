import numpy as np
import cv2

# pf5: come pf4, con tre differenze.
# 1. Niente supporto dei bordi: per un bordo pulito valeva al massimo 6/11,
#    mentre un bordo tagliato dall'immagine arrivava a 1.0, quindi il seed
#    preferiva proprio gli intervalli che non danno campioni ai bordi.
# 2. Seed scelto in base alla catena: si fa partire l'associazione da ogni
#    intervallo delle strip inferiori e si tiene la catena con piu' estremi
#    non tagliati (= piu' campioni per i due fit), a parita' quella con costo minore.
# 3. Il gap della confidence conta solo i buchi interni alla catena,
#    non le strip perse dopo la fine del corridoio.


def _line(model, y):
    """Valuta il modello lineare x(y) = m*y + q."""
    return model["m"] * y + model["q"]


def _irls(ys, xs, iters=4, sigma=18.0):
    """Fitta una retta con IRLS, riducendo il peso dei punti anomali."""
    ys = np.asarray(ys, np.float32)
    xs = np.asarray(xs, np.float32)
    if len(ys) < 2:
        return None, None

    weights = np.ones(len(ys), np.float32)
    coeff = np.polyfit(ys, xs, 1, w=weights)
    for _ in range(iters):
        residuals = xs - (coeff[0] * ys + coeff[1])
        weights = 1.0 / (1.0 + (residuals / max(sigma, 1e-6)) ** 2)
        coeff = np.polyfit(ys, xs, 1, w=weights)

    model = {"m": float(coeff[0]), "q": float(coeff[1])}
    residuals = np.abs(xs - _line(model, ys))
    return model, residuals


def _point(strip, interval):
    """Punto della catena: un intervallo accettato con la sua strip."""
    return {
        "y": strip["y"],
        "strip_idx": strip["idx"],
        "x0": interval["start"],
        "x1": interval["end"],
        "touches_left": interval["touches_left"],
        "touches_right": interval["touches_right"],
        "interval": interval,
    }


def _track(strips, seed_strip, seed, min_edge_jump, max_gap_strips):
    """Associazione bottom-up di pf4 a partire da un seed.

    Restituisce i punti della catena e il costo totale delle transizioni.
    """
    points = [_point(seed_strip, seed)]
    prev_left, prev_right, prev_width = seed["start"], seed["end"], seed["width"]
    missed = 0
    total_cost = 0.0

    for i in range(seed_strip["idx"] + 1, len(strips)):
        jump_limit = max(min_edge_jump, 0.5 * prev_width)
        choices = []
        for interval in strips[i]["intervals"]:
            jump = max(abs(interval["start"] - prev_left),
                       abs(interval["end"] - prev_right))
            if jump <= jump_limit:
                cost = jump + 0.4 * abs(interval["width"] - prev_width)
                choices.append((cost, interval))

        if not choices:
            missed += 1
            if missed > max_gap_strips:
                break
            continue

        cost, current = min(choices, key=lambda item: item[0])
        points.append(_point(strips[i], current))
        total_cost += cost
        prev_left, prev_right, prev_width = current["start"], current["end"], current["width"]
        missed = 0

    return points, total_cost


def path_finder(
    pred_mask,
    path_class=1,
    min_row_ratio=0.30,          # quota superiore esclusa (ROI dalla riga 67 su 224)
    n_strips=16,                 # 157 // 16 = 9 px: le strip coprono le 144 righe in basso della ROI
    min_component_area=500,
    min_interval_width=8,
    max_interval_gap=40,
    border_margin=2,
    min_edge_jump=28.0,
    robust_iters=4,
    robust_sigma=18.0,
    seed_bottom_strips=6,
    max_gap_strips=3,
    ok_conf_threshold=0.60,
    degraded_conf_threshold=0.38,
    min_valid_rows_ratio=0.10,
    cross_tolerance_ratio=0.05,
):
    """Trova la mezzeria del corridoio fittando separatamente i due bordi.

    Stessa interfaccia e stesse uscite di pf4.path_finder.
    """
    pred_mask = np.asarray(pred_mask)
    H, W = pred_mask.shape
    output = np.zeros((H, W), np.float32)

    # 1. ROI inferiore, maschera binaria del path e pulizia morfologica.
    y0 = int(H * min_row_ratio)
    path = (pred_mask[y0:] == path_class).astype(np.uint8)
    path = cv2.morphologyEx(path, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    path = cv2.morphologyEx(path, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(path, 8)
    clean = np.zeros_like(path)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            clean[labels == label] = 1
    path = clean
    roi_binary = np.zeros((H, W), np.uint8)
    roi_binary[y0:] = path * 255

    # 2. Strip dal basso verso l'alto (idx 0 = la piu' bassa) e intervalli.
    roi_h = path.shape[0]
    strip_h = max(1, roi_h // n_strips)
    strips = []
    for idx in range(n_strips):
        ys = max(0, roi_h - (idx + 1) * strip_h)
        ye = min(roi_h, roi_h - idx * strip_h)
        if ye <= ys:
            continue
        strip = path[ys:ye]
        binary = (strip.sum(axis=0) > 0).astype(np.int8)
        diff = np.diff(np.pad(binary, (1, 1)))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        # Fonde gli intervalli separati da buchi piccoli.
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
            touches_left = start <= border_margin
            touches_right = end >= W - 1 - border_margin
            if touches_left and touches_right and np.mean(strip) < 0.7:
                continue
            intervals.append({
                "start": float(start),
                "end": float(end),
                "width": float(end - start + 1),
                "touches_left": touches_left,
                "touches_right": touches_right,
            })
        strips.append({"idx": idx, "y": y0 + 0.5 * (ys + ye), "ys": ys, "intervals": intervals})

    if not strips:
        return output, {"status": "lost", "reason": "no_strips", "roi_binary": roi_binary}

    # 3. Seed scelto in base alla catena che genera: si prova ogni intervallo
    # delle strip inferiori e si tiene la catena con piu' estremi non tagliati
    # (campioni per i fit), a parita' quella con costo totale minore.
    # Un intervallo gia' contenuto in una catena precedente non viene provato:
    # la sua catena sarebbe solo la parte finale di quella gia' trovata.
    best_points, best_key = None, None
    used = set()
    for strip in strips[:min(seed_bottom_strips, len(strips))]:
        for seed in strip["intervals"]:
            if id(seed) in used:
                continue
            chain, total_cost = _track(strips, strip, seed, min_edge_jump, max_gap_strips)
            used.update(id(p["interval"]) for p in chain)
            samples = sum((not p["touches_left"]) + (not p["touches_right"]) for p in chain)
            key = (samples, -total_cost)
            if best_key is None or key > best_key:
                best_points, best_key = chain, key

    if best_points is None:
        return output, {"status": "lost", "reason": "no_seed_candidate", "roi_binary": roi_binary}
    points = best_points
    if len(points) < 3:
        return output, {"status": "lost", "reason": "tracking_failed", "roi_binary": roi_binary}

    # 4. Se le rette si incrociano dentro la zona osservata, scarta i punti
    # sopra l'incrocio e rifitta (come in pf4).
    for _ in range(3):
        pl = [p for p in points if not p["touches_left"]]
        pr = [p for p in points if not p["touches_right"]]
        if len(pl) < 2 or len(pr) < 2:
            break
        ml, _r = _irls([p["y"] for p in pl], [p["x0"] for p in pl], robust_iters, robust_sigma)
        mr, _r = _irls([p["y"] for p in pr], [p["x1"] for p in pr], robust_iters, robust_sigma)
        if ml is None or mr is None:
            break
        d_m = mr["m"] - ml["m"]
        d_q = mr["q"] - ml["q"]
        if d_m <= 1e-6:
            break
        y_cut = (min_interval_width - d_q) / d_m
        if y_cut <= min(p["y"] for p in points):
            break
        trimmed = [p for p in points if p["y"] >= y_cut]
        if len(trimmed) < 3:
            break
        points = trimmed

    # 5. Fit IRLS separato dei due bordi sugli estremi non tagliati.
    left = [p for p in points if not p["touches_left"]]
    right = [p for p in points if not p["touches_right"]]
    if len(left) < 2 or len(right) < 2:
        return output, {"status": "lost", "reason": "insufficient_edge_points", "roi_binary": roi_binary}
    yl = np.array([p["y"] for p in left], np.float32)
    xl = np.array([p["x0"] for p in left], np.float32)
    yr = np.array([p["y"] for p in right], np.float32)
    xr = np.array([p["x1"] for p in right], np.float32)
    left_model, left_residuals = _irls(yl, xl, robust_iters, robust_sigma)
    right_model, right_residuals = _irls(yr, xr, robust_iters, robust_sigma)
    if left_model is None or right_model is None:
        return output, {"status": "lost", "reason": "fit_failed", "roi_binary": roi_binary}

    # 6. Verifica geometrica (come in pf4): la larghezza w(y) = dm*y + dq deve
    # restare >= min_interval_width nel tratto valido.
    y_obs_top = float(min(yl.min(), yr.min()))
    y_obs_bottom = float(max(yl.max(), yr.max()))
    dm = right_model["m"] - left_model["m"]
    dq = right_model["q"] - left_model["q"]
    y_valid_top = y_obs_top
    y_valid_bottom = float(H - 1)
    if abs(dm) < 1e-6:
        width_ok = dq >= min_interval_width
    else:
        y_thr = (min_interval_width - dq) / dm
        if dm > 0:
            y_valid_top = max(y_valid_top, y_thr)
        else:
            y_valid_bottom = min(y_valid_bottom, y_thr)
        crossing_in_observed = y_obs_top + cross_tolerance_ratio * H < y_thr < y_obs_bottom
        width_ok = (
            not crossing_in_observed
            and y_valid_bottom - y_valid_top >= min_valid_rows_ratio * (H - y0)
        )
    # Strip attese nel tratto valido, confrontate con y_valid_top prima
    # dell'arrotondamento (con strip_h dispari i centri cadono su .5).
    expected_strips = [s for s in strips if s["y"] >= y_valid_top] if width_ok else strips
    y_valid_top = int(np.ceil(y_valid_top))
    if width_ok:
        y_eval = np.linspace(y_valid_top, y_valid_bottom, 16)
    else:
        y_eval = np.linspace(y_obs_top, y_obs_bottom, 16)
    width = float(np.median(_line(right_model, y_eval) - _line(left_model, y_eval)))

    # Regolarita' della larghezza: derivata seconda sui punti osservati non
    # tagliati (la larghezza di un intervallo tagliato non e' quella del corridoio).
    widths_obs = np.array([p["x1"] - p["x0"] for p in points
                           if not (p["touches_left"] or p["touches_right"])], dtype=np.float32)
    if len(widths_obs) >= 3:
        diff2 = np.diff(widths_obs, n=2)
        width_error = float(np.percentile(np.abs(diff2), 90)) / max(np.median(np.abs(widths_obs)), 1.0)
    else:
        width_error = 0.0

    # 7. Confidence (senza supporto: pesi di pf4 riscalati di 1/0.8).
    residual = float(np.mean(np.r_[left_residuals, right_residuals]))
    tracking = min(1.0, len(points) / max(len(expected_strips), 1))
    fit_score = np.clip(1.0 - residual / max(robust_sigma, 1e-6), 0.0, 1.0)
    width_score = np.clip(1.0 - width_error / 0.30, 0.0, 1.0)
    # Buchi interni alla catena: strip saltate tra due intervalli accettati.
    strip_ids = [p["strip_idx"] for p in points]
    max_gap = int(max(np.diff(strip_ids)) - 1) if len(strip_ids) > 1 else 0
    gap_score = np.clip(1.0 - max_gap / max(max_gap_strips, 1), 0.0, 1.0)
    confidence = float(
        0.3125 * tracking
        + 0.3125 * fit_score
        + 0.25 * width_score
        + 0.125 * gap_score
    )

    if not width_ok or residual > 2.0 * robust_sigma or confidence < degraded_conf_threshold:
        status = "lost"
    elif confidence < ok_conf_threshold:
        status = "degraded"
    else:
        status = "ok"

    diagnostics = {
        "status": status,
        "confidence": confidence,
        "mean_residual": residual,
        "tracking": float(tracking),
        "width": width,
        "width_error": width_error,
        "max_gap": max_gap,
        "left_edge_model": left_model,
        "right_edge_model": right_model,
        "centerline_model": None,
        "line_start_y": None,
        "line_end_y": None,
        "roi_binary": roi_binary,
    }
    if status == "lost":
        return output, diagnostics

    # 8. Mezzeria = media delle due rette, disegnata dal tratto seguito fino in fondo.
    center_model = {
        "m": 0.5 * (left_model["m"] + right_model["m"]),
        "q": 0.5 * (left_model["q"] + right_model["q"]),
    }
    y_start = max(int(y0 + strips[points[-1]["strip_idx"]]["ys"]), y_valid_top)
    y_end = int(y_valid_bottom) + 1
    for y in range(y_start, y_end):
        output[y, int(np.clip(_line(center_model, y), 0, W - 1))] = 1.0
    output = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    diagnostics.update({"centerline_model": center_model, "line_start_y": y_start, "line_end_y": y_end})
    return output, diagnostics
