import cv2
import numpy as np


def _widest_run_center(mask_1d):
    """Centro (indice intero) della sequenza contigua di True piu' lunga."""
    padded = np.concatenate(([0], mask_1d.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    if len(starts) == 0:
        return int(len(mask_1d) // 2)
    best = np.argmax(ends - starts)
    return int(round(0.5 * (starts[best] + ends[best])))


def path_finder(
    pred_mask,
    path_class=1,
    min_row_ratio=0.30,
    min_component_area=300,
    ok_conf_threshold=0.60,
    degraded_conf_threshold=0.38,
    **kwargs,
):
    """Path finder a istogramma: la via percorribile e' la colonna con piu' pixel di path.

    Restituisce esattamente la stessa interfaccia e le stesse strutture dati di pf4:
    (output, pf_debug).
    """
    pred_mask = np.asarray(pred_mask)
    H, W = pred_mask.shape
    output = np.zeros((H, W), dtype=np.float32)

    # 1. ROI inferiore (esclude la parte alta rumorosa come in pf4)
    y0 = int(H * min_row_ratio)
    path = (pred_mask[y0:] == path_class).astype(np.uint8)
    roi_h = path.shape[0]

    # 2. Pulizia morfologica (rimuove rumore e chiude buchi)
    path = cv2.morphologyEx(
        path, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    path = cv2.morphologyEx(
        path, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )

    # 3. Filtro componenti connesse piccole
    n, labels, stats, _ = cv2.connectedComponentsWithStats(path, 8)
    clean = np.zeros_like(path)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            clean[labels == label] = 1
    path = clean

    # ROI binaria completa per i debug visuali
    roi_binary = np.zeros((H, W), dtype=np.uint8)
    roi_binary[y0:] = path * 255

    # 4. Calcolo istogramma verticale (somma delle colonne)
    column_sums = path.sum(axis=0)
    peak_count = int(column_sums.max()) if column_sums.size else 0

    # 5. Calcolo confidence e validazione dello stato
    confidence = float(np.clip(peak_count / max(roi_h, 1), 0.0, 1.0))

    if peak_count == 0 or confidence < degraded_conf_threshold:
        status = "lost"
        return output, {
            "status": status,
            "reason": "no_path_pixels" if peak_count == 0 else "low_confidence",
            "confidence": confidence,
            "centerline_model": None,
            "line_start_y": None,
            "line_end_y": None,
            "roi_binary": roi_binary,
            "left_edge_model": None,
            "right_edge_model": None,
            "strip_points": [],
            "chosen_points": [],
            "mean_residual": 0.0,
            "support": 0.0,
            "tracking": 0.0,
            "width": 0.0,
            "width_error": 0.0,
            "max_gap": 0,
            "band_on_path_ratio": 0.0,
            "strip_tracking_ratio": 0.0,
            "histogram_x": None,
            "histogram_peak_count": peak_count,
        }

    status = "ok" if confidence >= ok_conf_threshold else "degraded"

    # 6. Trova la colonna con il massimo numero di pixel
    # In caso di plateau a pari altezza, prende il centro della sequenza
    x = _widest_run_center(column_sums == peak_count)

    # Stima dei bordi sinistro e destro a metà altezza del picco (per visualizzazione debug)
    half_peak = 0.5 * peak_count
    significant_cols = np.flatnonzero(column_sums >= half_peak)
    if significant_cols.size > 0:
        left_x = float(significant_cols[0])
        right_x = float(significant_cols[-1])
        width = float(right_x - left_x)
    else:
        left_x = float(max(0, x - 20))
        right_x = float(min(W - 1, x + 20))
        width = float(right_x - left_x)

    # 7. Genera la maschera output (retta verticale da y0 a H dilatata come in pf4)
    output[y0:, x] = 1.0
    output = cv2.dilate(
        output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    # 8. Modello lineare della linea centrale: x(y) = 0 * y + x
    center_model = {"m": 0.0, "q": float(x)}

    return output, {
        "status": status,
        "confidence": confidence,
        "mean_residual": 0.0,
        "support": 1.0,
        "tracking": 1.0,
        "width": width,
        "width_error": 0.0,
        "max_gap": 0,
        "left_edge_model": None,
        "right_edge_model": None,
        "centerline_model": center_model,
        "line_start_y": y0,
        "line_end_y": H,
        "roi_binary": roi_binary,
        "strip_points": [],
        "chosen_points": [],
        "band_on_path_ratio": confidence,
        "strip_tracking_ratio": 1.0,
        "histogram_x": x,
        "histogram_peak_count": peak_count,
    }


# Alias retrocompatibile
path_finder_filari_histogram = path_finder
