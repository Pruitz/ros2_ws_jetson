"""debug_visualizer.py

Modulo separato per il rendering e la compressione delle immagini di debug
(maschera di segmentazione con mezzeria, ROI binarizzata con modelli delle rette,
e proiezione sul frame a colori della camera).
"""

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage


def mask_to_bgr(mask):
    """Converte la maschera di segmentazione delle classi in un'immagine BGR a colori."""
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[mask == 0] = (0, 0, 0)       # Background
    vis[mask == 1] = (0, 255, 0)     # Filare / Path (verde)
    vis[mask == 2] = (0, 0, 255)     # Terreno (rosso)
    vis[mask == 3] = (255, 0, 0)     # Cielo / Altro (blu)
    return vis


def draw_model_projection(image, model, mask_shape, y_start, y_end, color, thickness=2):
    """Traccia la proiezione della retta x = m*y + q sull'immagine."""
    if model is None:
        return

    mask_h, mask_w = mask_shape
    image_h, image_w = image.shape[:2]
    scale_x = image_w / mask_w
    scale_y = image_h / mask_h
    previous_point = None

    for y in range(max(0, int(y_start)), min(mask_h, int(y_end))):
        x = model["m"] * y + model["q"]
        if not 0 <= x < mask_w:
            previous_point = None
            continue

        point = (int(round(x * scale_x)), int(round(y * scale_y)))
        if previous_point is not None:
            cv2.line(image, previous_point, point, color, thickness, cv2.LINE_AA)
        previous_point = point


def draw_path_models(image, pf_debug, mask_shape, path_matrix=None):
    """Traccia i modelli dei bordi sinistro (blu), destro (verde) e mezzeria (rosso)."""
    mask_h, _ = mask_shape
    y_start = pf_debug.get("line_start_y", int(mask_h * 0.30))
    y_end = pf_debug.get("line_end_y", mask_h)
    if y_start is None:
        y_start = int(mask_h * 0.30)
    if y_end is None:
        y_end = mask_h

    # Bordi laterali opzionali: supporta left_edge_model e left_model (RANSAC).
    # Se None (es. in pf_centri o pf_histogram), non disegna nulla e non crasha.
    left_edge = pf_debug.get("left_edge_model") or pf_debug.get("left_model")
    right_edge = pf_debug.get("right_edge_model") or pf_debug.get("right_model")
    center_model = pf_debug.get("centerline_model")

    if left_edge is not None:
        draw_model_projection(image, left_edge, mask_shape, y_start, y_end, (255, 0, 0))
    if right_edge is not None:
        draw_model_projection(image, right_edge, mask_shape, y_start, y_end, (0, 255, 0))

    if center_model is not None:
        draw_model_projection(image, center_model, mask_shape, y_start, y_end, (0, 0, 255))
    elif path_matrix is not None:
        # Fallback se manca centerline_model analitico: disegna i punti di path_matrix
        image_h, image_w = image.shape[:2]
        scale_x = image_w / mask_shape[1]
        scale_y = image_h / mask_shape[0]
        for y, x in np.argwhere(path_matrix > 0):
            pt = (int(round(x * scale_x)), int(round(y * scale_y)))
            cv2.circle(image, pt, 2, (0, 0, 255), -1)


def draw_lateral_error(image, mask_shape, raw_error_px, side, error_row_ratio=0.50):
    """Disegna la linea orizzontale magenta e il puntino bianco indicanti l'errore laterale."""
    mask_h, mask_w = mask_shape
    image_h, image_w = image.shape[:2]
    scale_x = image_w / mask_w
    scale_y = image_h / mask_h
    y_ref = int(mask_h * error_row_ratio)
    center_x = mask_w / 2.0

    if raw_error_px is not None:
        error_text = f"{raw_error_px:+.1f}px"
    else:
        error_text = "n/d"

    # Banner di stato in alto a sinistra
    label = f"Errore laterale: {error_text} ({side})"
    cv2.rectangle(image, (0, 0), (min(image_w - 1, 330), 27), (0, 0, 0), -1)
    cv2.putText(
        image, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (255, 255, 255), 1, cv2.LINE_AA,
    )

    if raw_error_px is not None:
        target_x = center_x + raw_error_px
        start = (int(round(center_x * scale_x)), int(round(y_ref * scale_y)))
        end = (
            int(round(np.clip(target_x, 0, mask_w - 1) * scale_x)),
            start[1],
        )
        # Linea orizzontale magenta
        cv2.line(image, start, end, (255, 0, 255), 2, cv2.LINE_AA)
        # Puntino bianco al centro del corridoio
        cv2.circle(image, end, 4, (255, 255, 255), -1, cv2.LINE_AA)


def build_debug_image(
    pred_mask,
    path_matrix,
    raw_error_px,
    filtered_error,
    side,
    confidence,
    pf_status,
    error_row_ratio=0.50,
):
    """Costruisce l'immagine di debug basata sulla maschera di segmentazione."""
    debug = mask_to_bgr(pred_mask)
    h, w = debug.shape[:2]
    center_x = int((w - 1) // 2)

    # Linea verticale centrale bianca
    cv2.line(debug, (center_x, 0), (center_x, h - 1), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Polilinea gialla della mezzeria
    points = []
    for y in range(h):
        row = path_matrix[y]
        if np.any(row > 0):
            xs = np.where(row > 0)[0]
            x = int(np.round(xs.mean()))
            points.append((x, y))

    if len(points) > 1:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(debug, [pts], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    elif len(points) == 1:
        x, y = points[0]
        cv2.circle(debug, (x, y), 3, (0, 255, 255), -1)

    # Errore laterale magenta + puntino bianco
    y_ref = int(h * error_row_ratio)
    if raw_error_px is not None:
        raw_x = int(round(center_x + raw_error_px))
        raw_x = max(0, min(w - 1, raw_x))
        cv2.line(debug, (center_x, y_ref), (raw_x, y_ref), (255, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(debug, (raw_x, y_ref), 4, (255, 255, 255), -1)

    cv2.putText(debug, f"side={side}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(debug, f"conf={confidence:.2f}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(debug, f"pf_status={pf_status}", (10, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if filtered_error is not None:
        cv2.putText(debug, f"filtered_error={filtered_error:.2f}px", (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if raw_error_px is not None:
        cv2.putText(debug, f"raw_error={raw_error_px:.2f}px", (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return debug


def build_debug_roi_image(roi_binary, pf_debug, path_matrix, raw_error_px, side, error_row_ratio=0.50):
    """Costruisce l'immagine di debug ROI binaria con i modelli delle rette sovrapposti."""
    debug = cv2.cvtColor(roi_binary, cv2.COLOR_GRAY2BGR)
    draw_path_models(debug, pf_debug, path_matrix.shape, path_matrix=path_matrix)
    draw_lateral_error(debug, path_matrix.shape, raw_error_px, side, error_row_ratio=error_row_ratio)
    return debug


def build_debug_color_path_image(frame, pf_debug, path_matrix, raw_error_px, side, error_row_ratio=0.50):
    """Sovrappone i modelli delle rette e l'errore laterale direttamente sul frame RGB della camera."""
    debug = frame.copy()
    draw_path_models(debug, pf_debug, path_matrix.shape, path_matrix=path_matrix)
    draw_lateral_error(debug, path_matrix.shape, raw_error_px, side, error_row_ratio=error_row_ratio)
    return debug


def cv2_to_compressed_msg(img, stamp):
    """Codifica un'immagine BGR in JPEG e la converte in messaggio ROS2 CompressedImage."""
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.format = "jpeg"
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    msg.data = encoded.tobytes()
    return msg
