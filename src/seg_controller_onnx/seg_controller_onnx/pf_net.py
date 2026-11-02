"""Path finder basato sulla rete dei bordi (stessa interfaccia di pf5.path_finder).

La rete (rete_bordi.onnx) stima la x dei due bordi alle righe 156 e 223 e la
probabilita' che il corridoio ci sia. Da qui:
  - rette dei bordi x = m*y + q dai due punti di ciascun bordo;
  - mezzeria = media delle due rette (centerline_model, usata dal controllore);
  - confidence = p_valid * accordo, dove accordo e' l'IoU, nella ROI, tra il
    corridoio compreso fra le due rette e i pixel di path della maschera;
  - status: lost se p_valid < valid_threshold, se la larghezza alla riga
    width_row (156 o 223) e' minore di min_width o se confidence < degraded_conf_threshold; degraded se
    confidence < ok_conf_threshold; altrimenti ok.
Le soglie 0.60/0.38 sono quelle di pf5 e vanno ricalibrate per questa confidence.
Con il generatore v2 (sintetico2.py) le etichette sopra l'apice sono il prolungamento delle
rette, quindi con apice sotto la riga 156 i bordi a y=156 si incrociano: per questi modelli
usare width_row=223.
"""
from pathlib import Path

import numpy as np

Y_LAB = (156.0, 223.0)
Y0 = 64          # ROI usata per confidence e visualizzazione, non per l'ingresso della rete
NORM = 112.0
_SESSIONS = {}


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + np.exp(-value))
    exp_value = np.exp(value)
    return exp_value / (1.0 + exp_value)


def _session(path):
    if path not in _SESSIONS:
        import onnxruntime as ort
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        sess = ort.InferenceSession(str(path), providers=providers)
        _SESSIONS[path] = (sess, sess.get_inputs()[0].name)
    return _SESSIONS[path]


def _line(p_a, p_b):
    (ya, xa), (yb, xb) = p_a, p_b
    m = (xb - xa) / (yb - ya)
    return {"m": float(m), "q": float(xa - m * ya)}


def path_finder(
    pred_mask,
    path_class=1,
    model_path=str(Path(__file__).resolve().parent / "models" / "rete_bordi.onnx"),
    valid_threshold=0.5,
    min_width=8.0,
    width_row=223,
    ok_conf_threshold=0.60,
    degraded_conf_threshold=0.38,
):
    pred_mask = np.asarray(pred_mask)
    H, W = pred_mask.shape
    output = np.zeros((H, W), np.float32)
    path = (pred_mask == path_class).astype(np.float32)
    roi_binary = np.zeros((H, W), np.uint8)
    roi_binary[Y0:] = path[Y0:] * 255

    sess, name = _session(model_path)
    out = sess.run(None, {name: path[None, None]})[0][0]
    xs = out[:4] * NORM + NORM
    p_valid = float(_sigmoid(float(out[4])))

    left = _line((Y_LAB[0], xs[0]), (Y_LAB[1], xs[2]))
    right = _line((Y_LAB[0], xs[1]), (Y_LAB[1], xs[3]))
    center = {"m": 0.5 * (left["m"] + right["m"]), "q": 0.5 * (left["q"] + right["q"])}

    # Corridoio fra le due rette nella ROI e accordo con la maschera (IoU)
    ys = np.arange(Y0, H, dtype=np.float32)
    xl = left["m"] * ys + left["q"]
    xr = right["m"] * ys + right["q"]
    cols = np.arange(W, dtype=np.float32)
    region = (cols[None, :] >= xl[:, None]) & (cols[None, :] <= xr[:, None])
    observed = path[Y0:] > 0
    union = np.logical_or(region, observed).sum()
    iou = float(np.logical_and(region, observed).sum() / union) if union else 0.0
    confidence = p_valid * iou

    width_ref = float(xs[1] - xs[0]) if width_row == 156 else float(xs[3] - xs[2])
    if p_valid < valid_threshold or width_ref < min_width or confidence < degraded_conf_threshold:
        status = "lost"
    elif confidence < ok_conf_threshold:
        status = "degraded"
    else:
        status = "ok"

    # Tratto disegnato: righe della ROI dove la larghezza modellata e' >= min_width
    dm, dq = right["m"] - left["m"], right["q"] - left["q"]
    y_start = Y0
    if dm > 1e-6:
        y_start = int(np.clip(np.ceil((min_width - dq) / dm), Y0, H - 1))
    diag = {
        "status": status,
        "confidence": confidence,
        "p_valid": p_valid,
        "iou": iou,
        "width": width_ref,
        "left_edge_model": left,
        "right_edge_model": right,
        "centerline_model": center if status != "lost" else None,
        "line_start_y": y_start if status != "lost" else None,
        "line_end_y": H if status != "lost" else None,
        "roi_binary": roi_binary,
    }
    if status != "lost":
        for y in range(y_start, H):
            output[y, int(np.clip(center["m"] * y + center["q"], 0, W - 1))] = 1.0
    return output, diag
