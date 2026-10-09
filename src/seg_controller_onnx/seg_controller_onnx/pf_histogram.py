import numpy as np
import cv2


def _widest_run_center(mask_1d):
    """Centro (indice intero) della sequenza contigua di True piu' lunga."""
    padded = np.concatenate(([0], mask_1d.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    best = np.argmax(ends - starts)
    return int(round(0.5 * (starts[best] + ends[best])))


def path_finder_filari_histogram(
    pred_mask,
    path_class=1,
    min_row_ratio=0.0,
    ok_conf_threshold=0.60,
    degraded_conf_threshold=0.38,
):
    """Path finder a istogramma: la via percorribile e' la colonna con piu' pixel di path.

    Equivale a far scendere i pixel di path verso il basso e prendere la colonna
    piu' alta. Restituisce la stessa interfaccia degli altri PF: (path_matrix, debug).
    """
    pred_mask = np.asarray(pred_mask)
    H, W = pred_mask.shape
    path_matrix = np.zeros((H, W), dtype=np.float32)

    y0 = int(H * min_row_ratio)
    roi_path = (pred_mask[y0:] == path_class).astype(np.uint8)
    roi_h = roi_path.shape[0]

    roi_binary = np.zeros((H, W), dtype=np.uint8)
    roi_binary[y0:] = roi_path * 255

    column_sums = roi_path.sum(axis=0)
    peak_count = int(column_sums.max()) if column_sums.size else 0
    if peak_count == 0:
        return path_matrix, {
            "status": "lost",
            "reason": "no_path_pixels",
            "confidence": 0.0,
            "roi_binary": roi_binary,
            "centerline_model": None,
            "line_start_y": None,
            "line_end_y": None,
        }

    # Su un plateau di colonne a pari altezza si usa il centro, non il bordo sinistro.
    x = _widest_run_center(column_sums == peak_count)

    path_matrix[y0:, x] = 1.0
    path_matrix = cv2.dilate(
        path_matrix, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    confidence = float(np.clip(peak_count / max(roi_h, 1), 0.0, 1.0))
    if confidence < degraded_conf_threshold:
        status = "lost"
    elif confidence < ok_conf_threshold:
        status = "degraded"
    else:
        status = "ok"

    return path_matrix, {
        "status": status,
        "confidence": confidence,
        "roi_binary": roi_binary,
        "centerline_model": {"m": 0.0, "q": float(x)},
        "line_start_y": y0,
        "line_end_y": H,
        "histogram_x": x,
        "histogram_peak_count": peak_count,
    }
