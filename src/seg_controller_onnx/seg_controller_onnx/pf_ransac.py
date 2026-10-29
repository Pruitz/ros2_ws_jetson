"""pf_ransac.py

Estrazione geometrica della linea di navigazione basata su RANSAC secondo la
letteratura (Li et al., Sensors 2025; Bai et al., 2023).

A differenza dei metodi basati su decomposizione a strisce (strip), questo
algoritmo estrae direttamente le nuvole di punti dei bordi laterali (sinistro
e destro) della maschera del corridoio navigabile su tutte le righe di scansione,
ed esegue il fitting robusto di due rette indipendenti tramite RANSAC (Random
Sample Consensus). La linea di mezzeria (centerline) viene quindi ottenuta come
media analitica delle rette stimate per i due bordi.
"""

import numpy as np
import cv2


def _line(model, y):
    """Valuta il modello lineare x(y) = m*y + q."""
    return model["m"] * y + model["q"]


def _ransac_line_fit(ys, xs, residual_threshold=18.0, min_samples=2, max_trials=100, random_state=42):
    """Fitta una retta x(y) = m*y + q tramite RANSAC puro NumPy su una nuvola di punti.

    Segue l'algoritmo canonico di Fischler & Bolles (1981) e Li et al. (2025):
    campiona coppie di punti casuali, valuta gli inlier e riesegue il refit ai minimi
    quadrati sull'insieme massimo di inlier identificato.

    Args:
        ys: Array 1D delle coordinate verticali (y).
        xs: Array 1D delle coordinate orizzontali (x).
        residual_threshold: Distanza massima (in pixel) per considerare un punto inlier.
        min_samples: Numero minimo di campioni per fittare il modello (2 per una retta).
        max_trials: Numero massimo di iterazioni di campionamento stocastico.
        random_state: Seed per rendere il campionamento stocastico controllato/riproducibile.

    Returns:
        model: Dizionario {"m": float, "q": float} o None se il fitting fallisce.
        residuals: Array dei residui assoluti per ciascun punto.
        inlier_mask: Maschera booleana dei punti classificati come inlier.
    """
    ys = np.asarray(ys, np.float32)
    xs = np.asarray(xs, np.float32)
    n_points = len(ys)

    if n_points < min_samples:
        return None, None, None

    rng = np.random.default_rng(random_state)
    best_inlier_count = -1
    best_inlier_mask = None
    best_model = None

    for _ in range(max_trials):
        idx = rng.choice(n_points, size=min_samples, replace=False)
        y1, y2 = ys[idx[0]], ys[idx[1]]
        x1, x2 = xs[idx[0]], xs[idx[1]]

        dy = y2 - y1
        if abs(dy) < 1e-4:
            continue

        m_cand = float((x2 - x1) / dy)
        q_cand = float(x1 - m_cand * y1)

        residuals_cand = np.abs(xs - (m_cand * ys + q_cand))
        inliers_cand = residuals_cand <= residual_threshold
        inlier_count = int(np.sum(inliers_cand))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inliers_cand
            best_model = {"m": m_cand, "q": q_cand}

    if best_model is None or best_inlier_count < min_samples:
        return None, None, None

    # Refit Ordinary Least Squares (OLS) su tutti gli inlier trovati
    ys_in = ys[best_inlier_mask]
    xs_in = xs[best_inlier_mask]
    if len(ys_in) >= 2 and (ys_in.max() - ys_in.min()) > 1e-3:
        coeff = np.polyfit(ys_in, xs_in, 1)
        final_model = {"m": float(coeff[0]), "q": float(coeff[1])}
    else:
        final_model = best_model

    final_residuals = np.abs(xs - (final_model["m"] * ys + final_model["q"]))
    return final_model, final_residuals, best_inlier_mask


def path_finder(
    pred_mask,
    path_class=1,
    min_row_ratio=0.30,
    min_component_area=500,
    min_corridor_width=10,
    ransac_residual_threshold=18.0,
    ransac_max_trials=100,
    ok_conf_threshold=0.60,
    degraded_conf_threshold=0.38,
    random_state=42,
):
    """Estrae la mezzeria del filare fittando i contorni laterali con RANSAC (Li et al., 2025; Bai et al., 2023).

    Pipeline:
      1. Selezione della ROI inferiore (y >= min_row_ratio * H).
      2. Pulizia morfologica ed estrazione della componente connessa principale del suolo.
      3. Estrazione diretta dei punti di bordo laterale (x_left, x_right) per ogni riga di scansione (senza strip).
      4. Fitting RANSAC indipendente sul bordo sinistro e sul bordo destro.
      5. Calcolo della mezzeria analitica come media delle due rette.
      6. Calcolo di confidenza e rendering della maschera.

    Args:
        pred_mask: Maschera 2D con le classi predette dal modello di segmentazione.
        path_class: Indice della classe che rappresenta il corridoio navigabile.
        min_row_ratio: Quota verticale superiore esclusa dall'analisi (es. 0.30 = 30% superiore).
        min_component_area: Area minima in pixel per mantenere una componente connessa.
        min_corridor_width: Larghezza minima del corridoio su una riga per accettare i punti di bordo.
        ransac_residual_threshold: Soglia residuo per classificazione inlier (in pixel).
        ransac_max_trials: Numero massimo di tentativi stocastici per RANSAC.
        ok_conf_threshold: Soglia minima di confidenza per stato 'ok'.
        degraded_conf_threshold: Sotto questa soglia lo stato diventa 'lost'.
        random_state: Seed casuale per RANSAC (utile per analisi di riproducibilità/jitter).

    Returns:
        output: Maschera 2D float32 con la centerline tracciata (0.0 ovunque, 1.0 sulla linea).
        diagnostics: Dizionario con diagnostica completa, modelli delle rette, residui e confidenza.
    """
    pred_mask = np.asarray(pred_mask)
    H, W = pred_mask.shape
    output = np.zeros((H, W), np.float32)

    # 1. ROI inferiore e isolamento classe corridoio
    y0 = int(H * min_row_ratio)
    path = (pred_mask[y0:] == path_class).astype(np.uint8)

    # 2. Filtraggio morfologico
    path = cv2.morphologyEx(
        path, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    path = cv2.morphologyEx(
        path, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )

    # Rimozione componenti piccole e selezione della componente di guida principale
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(path, 8)
    clean = np.zeros_like(path)
    valid_labels = []
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            valid_labels.append(label)

    if not valid_labels:
        roi_binary = np.zeros((H, W), np.uint8)
        roi_binary[y0:] = path * 255
        return output, {"status": "lost", "reason": "no_valid_components", "roi_binary": roi_binary}

    # Seleziona la componente con area maggiore che si estende nella parte inferiore
    best_label = max(
        valid_labels,
        key=lambda lbl: stats[lbl, cv2.CC_STAT_AREA] + stats[lbl, cv2.CC_STAT_TOP] * 10
    )
    clean[labels == best_label] = 1
    path = clean

    # 3. Estrazione diretta dei punti di contorno laterali riga per riga (senza decomposizione in strip)
    ys_left, xs_left = [], []
    ys_right, xs_right = [], []
    widths = []

    roi_h = path.shape[0]
    for r in range(roi_h):
        row_pixels = np.where(path[r] == 1)[0]
        if len(row_pixels) == 0:
            continue

        x_l = float(row_pixels[0])
        x_r = float(row_pixels[-1])
        w = x_r - x_l + 1.0

        if w < min_corridor_width:
            continue

        abs_y = float(y0 + r)
        ys_left.append(abs_y)
        xs_left.append(x_l)
        ys_right.append(abs_y)
        xs_right.append(x_r)
        widths.append(w)

    min_required_points = 10
    if len(ys_left) < min_required_points or len(ys_right) < min_required_points:
        roi_binary = np.zeros((H, W), np.uint8)
        roi_binary[y0:] = path * 255
        return output, {"status": "lost", "reason": "insufficient_boundary_points", "roi_binary": roi_binary}

    ys_left = np.array(ys_left, np.float32)
    xs_left = np.array(xs_left, np.float32)
    ys_right = np.array(ys_right, np.float32)
    xs_right = np.array(xs_right, np.float32)
    widths = np.array(widths, np.float32)

    # 4. Fitting RANSAC indipendente sul bordo sinistro e sul bordo destro
    left_edge_model, left_residuals, left_inliers = _ransac_line_fit(
        ys_left, xs_left,
        residual_threshold=ransac_residual_threshold,
        max_trials=ransac_max_trials,
        random_state=random_state,
    )
    right_edge_model, right_residuals, right_inliers = _ransac_line_fit(
        ys_right, xs_right,
        residual_threshold=ransac_residual_threshold,
        max_trials=ransac_max_trials,
        random_state=random_state,
    )

    if left_edge_model is None or right_edge_model is None:
        roi_binary = np.zeros((H, W), np.uint8)
        roi_binary[y0:] = path * 255
        return output, {"status": "lost", "reason": "ransac_fit_failed", "roi_binary": roi_binary}

    # 5. Mezzeria analitica come media delle due rette
    center_model = {
        "m": 0.5 * (left_edge_model["m"] + right_edge_model["m"]),
        "q": 0.5 * (left_edge_model["q"] + right_edge_model["q"]),
    }

    # 6. Diagnostica e valutazione di confidenza
    mean_left_res = float(np.mean(left_residuals[left_inliers])) if np.any(left_inliers) else float(np.mean(left_residuals))
    mean_right_res = float(np.mean(right_residuals[right_inliers])) if np.any(right_inliers) else float(np.mean(right_residuals))
    mean_residual = 0.5 * (mean_left_res + mean_right_res)

    inlier_ratio_left = float(np.sum(left_inliers)) / max(len(left_inliers), 1)
    inlier_ratio_right = float(np.sum(right_inliers)) / max(len(right_inliers), 1)
    inlier_ratio = 0.5 * (inlier_ratio_left + inlier_ratio_right)

    vertical_coverage = float(len(ys_left)) / max(float(roi_h), 1.0)
    fit_score = float(np.clip(1.0 - mean_residual / max(ransac_residual_threshold, 1e-6), 0.0, 1.0))

    if len(widths) >= 5:
        diff_w = np.abs(np.diff(widths))
        width_jitter = float(np.median(diff_w)) / max(float(np.median(widths)), 1.0)
        width_score = float(np.clip(1.0 - width_jitter / 0.30, 0.0, 1.0))
    else:
        width_score = 0.5

    confidence = float(
        0.35 * inlier_ratio
        + 0.30 * fit_score
        + 0.20 * vertical_coverage
        + 0.15 * width_score
    )

    # Classificazione stato
    if mean_residual > 2.0 * ransac_residual_threshold or inlier_ratio < 0.30:
        status = "lost"
    elif confidence < degraded_conf_threshold:
        status = "lost"
    elif confidence < ok_conf_threshold:
        status = "degraded"
    else:
        status = "ok"

    roi_binary = np.zeros((H, W), np.uint8)
    roi_binary[y0:] = path * 255

    diagnostics = {
        "status": status,
        "confidence": confidence,
        "mean_residual": mean_residual,
        "left_residual": mean_left_res,
        "right_residual": mean_right_res,
        "inlier_ratio": inlier_ratio,
        "inlier_ratio_left": inlier_ratio_left,
        "inlier_ratio_right": inlier_ratio_right,
        "vertical_coverage": vertical_coverage,
        "left_edge_model": left_edge_model,
        "right_edge_model": right_edge_model,
        "centerline_model": center_model,
        "roi_binary": roi_binary,
        "line_start_y": None,
        "line_end_y": None,
    }

    if status == "lost":
        return output, diagnostics

    # 7. Rendering della centerline
    y_start = int(np.floor(min(ys_left.min(), ys_right.min())))
    y_end = H
    diagnostics["line_start_y"] = y_start
    diagnostics["line_end_y"] = y_end

    for y in range(y_start, y_end):
        x = int(np.clip(_line(center_model, y), 0, W - 1))
        output[y, x] = 1.0

    output = cv2.dilate(
        output,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    return output, diagnostics


# Alias retrocompatibile
path_finder_filari_ransac = path_finder
