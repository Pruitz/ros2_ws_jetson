# 29/06/2026
# Nodo ROS2 per inferenza ONNX e controllo di follower di filari agricoli.
# Pipeline: camera compressed -> segmentation ONNX -> path finder robusto ->
# stima errore laterale -> filtro temporale -> cmd_vel.
#
# Scopo:
# seguire filari rettilinei in modo stabile, privilegiando robustezza a buchi
# di segmentazione, rumore e reacquisition errate.
#
# Strategia:
# usare il lateral error geometrico del path finder come sorgente primaria,
# con fallback sulla path_matrix solo se necessario.
#
# Tuning principale:
# kp_lat, soglie di confidence, deadband, alpha del filtro errore e limiti
# di velocità/sterzata.
#
# Parametri di tuning iniziali:
# - kp_lat: guadagno sull'errore laterale, regola quanto aggressivamente
#   il robot rientra verso il centro;
# - alpha filtro errore: coefficiente di smoothing dell'errore filtrato,
#   più alto = risposta più rapida ma più rumore.
# Se il robot sterza in modo nervoso, riduci kp_lat o aumenta poco la deadband.
# Se rientra troppo lentamente, aumenta leggermente kp_lat.
# Se l'errore filtrato oscilla troppo, riduci alpha del filtro.
from pathlib import Path
from time import time, perf_counter




import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist




import numpy as np
import cv2




from seg_controller_onnx.onnx_inference import (
    build_onnx_session,
    run_inference_on_frame,
)
from .pf4 import path_finder





TARGET_W = 224
TARGET_H = 224
ERROR_ROW_RATIO = 0.70




def resize_frame(frame):
    return cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)





def mask_to_bgr(mask):
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[mask == 0] = (0, 0, 0)
    vis[mask == 1] = (0, 255, 0)
    vis[mask == 2] = (0, 0, 255)
    vis[mask == 3] = (255, 0, 0)
    return vis


def draw_model_projection(image, model, mask_shape, y_start, y_end, color, thickness=2):
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


def draw_path_models(image, pf_debug, mask_shape):
    mask_h, _ = mask_shape
    y_start = pf_debug.get("line_start_y", int(mask_h * 0.30))
    y_end = pf_debug.get("line_end_y", mask_h)
    if y_start is None or y_end is None:
        return

    for key, color in (
        ("left_edge_model", (255, 0, 0)),
        ("right_edge_model", (0, 255, 0)),
        ("centerline_model", (0, 0, 255)),
    ):
        draw_model_projection(
            image, pf_debug.get(key), mask_shape, y_start, y_end, color
        )


def draw_lateral_error(image, mask_shape, raw_error_px, side):
    mask_h, mask_w = mask_shape
    image_h, image_w = image.shape[:2]
    scale_x = image_w / mask_w
    scale_y = image_h / mask_h
    y_ref = int(mask_h * ERROR_ROW_RATIO)
    center_x = mask_w / 2.0

    if raw_error_px is not None:
        target_x = center_x + raw_error_px
        start = (int(round(center_x * scale_x)), int(round(y_ref * scale_y)))
        end = (
            int(round(np.clip(target_x, 0, mask_w - 1) * scale_x)),
            start[1],
        )
        cv2.line(image, start, end, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(image, end, 4, (255, 255, 255), -1, cv2.LINE_AA)
        error_text = f"{raw_error_px:+.1f}px"
    else:
        error_text = "n/d"

    label = f"Errore laterale: {error_text} ({side})"
    cv2.rectangle(image, (0, 0), (min(image_w - 1, 330), 27), (0, 0, 0), -1)
    cv2.putText(
        image, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (255, 255, 255), 1, cv2.LINE_AA,
    )





# def central_pixel_side(path_matrix):
#     """
#     Calcola l'errore laterale cercando la centerline su un intervallo di righe
#     invece che su una riga fissa. Più robusto quando la centerline disegnata
#     da PF parte più in basso.
#     """
#     h, w = path_matrix.shape

#     # Riga di riferimento centrale (70% dall'alto)
#     row_center = int(h * ERROR_ROW_RATIO)

#     # Intervallo di righe da scandire (±15% intorno al centro)
#     row_span = int(h * 0.15)
#     row_start = max(0, row_center - row_span)
#     row_end = min(h - 1, row_center + row_span)

#     # Scorro le righe dall'alto verso il basso e prendo la prima con pixel
#     for row in range(row_start, row_end + 1):
#         cols = np.where(path_matrix[row] > 0)[0]
#         if len(cols) == 0:
#             continue

#         pixel_col = float(np.mean(cols))
#         error = pixel_col - (w / 2.0)

#         if error > 0:
#             side = "destra"
#         elif error < 0:
#             side = "sinistra"
#         else:
#             side = "centro"

#         return error, side

#     # Nessuna riga con pixel trovata
#     return None, "no_pixel"


def central_pixel_side_from_model(pf_debug, path_matrix):
    """
    Calcola l'errore laterale usando direttamente il modello della centerline.
    """
    center_model = pf_debug.get("centerline_model")
    if center_model is None:
        return None, "no_model"

    h, w = path_matrix.shape

    # y di riferimento: 70% dall'alto (stessa logica di prima)
    y_ref = int(h * ERROR_ROW_RATIO)
    y_ref = max(0, min(h - 1, y_ref))

    # x della centerline a quella y: x = m * y + q
    x_center = center_model["m"] * y_ref + center_model["q"]

    error = x_center - (w / 2.0)

    if error > 0:
        side = "destra"
    elif error < 0:
        side = "sinistra"
    else:
        side = "centro"

    return error, side



def build_debug_image(
    pred_mask,
    path_matrix,
    raw_error_px,
    filtered_error,
    side,
    confidence,
    pf_status,
):
    # Base: segmentation mask colored
    debug = mask_to_bgr(pred_mask)
    h, w = debug.shape[:2]
    center_x = int((w - 1) // 2)

    # draw vertical center line
    cv2.line(debug, (center_x, 0), (center_x, h - 1), (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Draw path_matrix as a smooth yellow polyline (one x per y) — OPAQUE, disegnata direttamente su debug
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

    # Draw lateral error as a single horizontal line at ref_y
    y_ref = int(h * ERROR_ROW_RATIO)

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

def build_debug_roi_image(roi_binary, pf_debug, path_matrix, raw_error_px, side):
    # Base: roi binaria pulita (bianco = path pulito, nero = resto)
    debug = cv2.cvtColor(roi_binary, cv2.COLOR_GRAY2BGR)
    draw_path_models(debug, pf_debug, path_matrix.shape)
    draw_lateral_error(debug, path_matrix.shape, raw_error_px, side)

    return debug


def build_debug_color_path_image(frame, pf_debug, path_matrix, raw_error_px, side):
    debug = frame.copy()
    draw_path_models(debug, pf_debug, path_matrix.shape)
    draw_lateral_error(debug, path_matrix.shape, raw_error_px, side)

    return debug

def cv2_to_compressed_msg(img, stamp):
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.format = "jpeg"
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    msg.data = encoded.tobytes()
    return msg





class SegInferenceNode(Node):
    def __init__(self):
        super().__init__('seg_inference_cmdvel_node_onnx')




        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )




        debug_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )




        model_path = (
            Path(__file__).resolve().parent
            / 'models'
            / 'lraspp_224x224_allPlants.onnx'
        )




        (
            self.session,
            self.input_name,
            self.output_name,
            available_providers,
            requested_providers,
            active_providers,
            ort_device,
        ) = build_onnx_session(model_path=model_path)




        self.get_logger().info(f'Provider disponibili ONNX Runtime: {available_providers}')
        self.get_logger().info(f'Provider richiesti: {requested_providers}')
        self.get_logger().info(f'Provider attivi sessione: {active_providers}')
        self.get_logger().info(f'Dispositivo ONNX Runtime: {ort_device}')
        self.get_logger().info('Inferenza forzata su CPU')
        self.get_logger().info(f'Modello ONNX: {model_path}')




        self.frame_count = 0
        self.process_every_n = 1




        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.callback,
            sensor_qos
        )




        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)




        self.debug_img_pub = self.create_publisher(
            CompressedImage,
            '/debug/segmentation/compressed',
            debug_qos
        )
        self.debug_roi_pub = self.create_publisher(
            CompressedImage,
            '/debug/roi/compressed',
            debug_qos
        )
        self.debug_path_pub = self.create_publisher(
            CompressedImage,
            '/debug/path/compressed',
            debug_qos
        )




        self.filtered_error = None
        self.last_valid_error = None
        self.last_valid_time = 0.0
        self.lost_count = 0




        self.last_status = "lost"




        self.deadband_px = 6.0
        self.hold_timeout = 0.30
        self.max_lost_frames = 3




        self.alpha_min = 0.08
        self.alpha_max = 0.45
        self.jump_thr_min = 20.0
        self.jump_thr_max = 60.0




        self.kp_lat = 0.85
        self.v_max = 10.0
        self.v_min = 0.07
        self.w_max = 0.25




        self.log_every_n = 5




        self.get_logger().info(
            f'Nodo ONNX avviato, analizzo 1 frame ogni {self.process_every_n}'
        )




    def filter_error(self, error_px, confidence, pf_status):
        """
        Filtro dell'errore laterale:
        - esponenziale con alpha fisso (due regimi: ok / degraded);
        - threshold fisso sul salto per rejectare outlier grossolani;
        - hold dell'ultimo errore valido se error_px è None;
        - deadband sull'errore filtrato.

        Non si fida ciecamente di confidence, usa pf_status solo per due regimi.
        """
        now = time()

        # Se PF è perso, non produco errore filtrato
        if pf_status == "lost":
            return None, "lost_by_pf_status"

        # Se non c'è errore grezzo (es. centerline assente)
        if error_px is None:
            if (
                self.last_valid_error is not None
                and (now - self.last_valid_time) <= self.hold_timeout
            ):
                # Mantengo l'ultimo errore filtrato per un breve periodo
                return self.filtered_error, "hold"
            return None, "lost"

        # Parametri in base allo stato PF (due regimi semplici)
        if pf_status == "degraded":
            alpha = 0.12        # più prudente
            jump_thr = 30.0     # soglia di salto più severa
        else:
            alpha = 0.20        # reattivo
            jump_thr = 40.0     # soglia di salto normale

        # Inizializzo filtered_error se è la prima volta
        if self.filtered_error is None:
            self.filtered_error = float(error_px)
        else:
            jump = abs(error_px - self.filtered_error)

            if jump <= jump_thr:
                # Salto normale: filtro esponenziale classico
                self.filtered_error = (
                    (1.0 - alpha) * self.filtered_error + alpha * float(error_px)
                )
            else:
                # Salto grande: probabile outlier, accetto solo il 5% del nuovo errore
                self.filtered_error = (
                    0.95 * self.filtered_error + 0.05 * float(error_px)
                )

        # Deadband: se l'errore filtrato è piccolo, lo porto a zero
        if abs(self.filtered_error) < self.deadband_px:
            self.filtered_error = 0.0

        # Aggiorno ultimo errore valido e tempo
        self.last_valid_error = float(self.filtered_error)
        self.last_valid_time = now

        # Determino lo stato del filtro
        if pf_status == "degraded":
            filter_status = "valid_degraded"
        else:
            filter_status = "valid"

        return self.filtered_error, filter_status



    def compute_cmd(self, error_px, img_width, confidence, pf_status):
        if error_px is None or pf_status == "lost":
            return 0.0, 0.0




        error_norm = error_px / (img_width / 2.0)
        w_gain = 0.6 + 0.4 * float(np.clip(confidence, 0.0, 1.0))
        w = -(self.kp_lat * w_gain * error_norm)




        if pf_status == "degraded":
            w *= 0.7




        w = float(np.clip(w, -self.w_max, self.w_max))
        v = 0.2 * self.v_max
        return v, w




    def publish_stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)




    def callback(self, msg):
        t_cb_0 = perf_counter()
        self.frame_count += 1




        if self.frame_count % self.process_every_n != 0:
            return




        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)




        if frame is None:
            t_cb_1 = perf_counter()
            self.get_logger().warn(
                f'Immagine non decodificata | total={(t_cb_1 - t_cb_0)*1000:.1f} ms'
            )
            self.publish_stop()
            return




        camera_frame = frame.copy()
        frame = resize_frame(frame)
        t1 = perf_counter()




        pred_mask = run_inference_on_frame(
            self.session,
            self.input_name,
            self.output_name,
            frame
        )
        t2 = perf_counter()




        path_matrix, pf_debug = path_finder(pred_mask)



        confidence = float(pf_debug.get("confidence", 0.0))
        pf_status = pf_debug.get("status", "lost")
        band_on_path_ratio = float(pf_debug.get("band_on_path_ratio", 0.0))
        strip_tracking_ratio = float(pf_debug.get("strip_tracking_ratio", 0.0))
        # per immagine debug roi
        roi_binary = pf_debug.get("roi_binary", None)
        strip_points = pf_debug.get("strip_points", [])
        chosen_points = pf_debug.get("chosen_points", [])

        self.last_status = pf_status


        raw_error, side = central_pixel_side_from_model(pf_debug, path_matrix)


        filtered_error, filter_status = self.filter_error(raw_error, confidence, pf_status)


        if pf_status == "lost":
            self.lost_count += 1
        else:
            self.lost_count = 0


        debug_img = build_debug_image(
            pred_mask,
            path_matrix,
            raw_error,
            filtered_error,
            side,
            confidence,
            pf_status,
        )



        # immagine debug seg
        debug_msg = cv2_to_compressed_msg(debug_img, msg.header.stamp)
        if debug_msg is not None:
            self.debug_img_pub.publish(debug_msg)
        # immagine debug roi
        if roi_binary is not None:
            debug_roi_img = build_debug_roi_image(
                roi_binary, pf_debug, path_matrix, raw_error, side
            )
            debug_roi_msg = cv2_to_compressed_msg(debug_roi_img, msg.header.stamp)
            if debug_roi_msg is not None:
                self.debug_roi_pub.publish(debug_roi_msg)

        debug_path_img = build_debug_color_path_image(
            camera_frame, pf_debug, path_matrix, raw_error, side
        )
        debug_path_msg = cv2_to_compressed_msg(debug_path_img, msg.header.stamp)
        if debug_path_msg is not None:
            self.debug_path_pub.publish(debug_path_msg)



        if self.lost_count >= self.max_lost_frames:
            self.publish_stop()
            t_cb_1 = perf_counter()
            if self.frame_count % self.log_every_n == 0:
                self.get_logger().warn(
                    f'Frame {self.frame_count} | '
                    f'total={(t_cb_1 - t_cb_0)*1000:.1f} ms | '
                    f'inference={(t2 - t1)*1000:.1f} ms | '
                    f'STOP | pf_status={pf_status} | conf={confidence:.2f} | '
                    f'band_ratio={band_on_path_ratio:.2f} | strip_ratio={strip_tracking_ratio:.2f} | '
                    f'filter={filter_status} | lost_count={self.lost_count}'
                )
            return


        img_width = path_matrix.shape[1]
        v, w = self.compute_cmd(filtered_error, img_width, confidence, pf_status)




        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w




        self.cmd_pub.publish(cmd)




        t_cb_1 = perf_counter()




        if self.frame_count % self.log_every_n == 0:
            raw_error_txt = f"{raw_error:.2f}" if raw_error is not None else "None"
            filtered_error_txt = f"{filtered_error:.2f}" if filtered_error is not None else "None"
            self.get_logger().info(
                f'Frame {self.frame_count} | '
                f'total={(t_cb_1 - t_cb_0)*1000:.1f} ms | '
                f'inference={(t2 - t1)*1000:.1f} ms | '
                f'pf_status={pf_status} | conf={confidence:.2f} | '
                f'band_ratio={band_on_path_ratio:.2f} | strip_ratio={strip_tracking_ratio:.2f} | '
                f'side={side} | raw_error={raw_error_txt} | '
                f'filtered_error={filtered_error_txt}px | '
                f'filter={filter_status} | v={v:.3f} | w={w:.3f}'
            )



def main(args=None):
    rclpy.init(args=args)
    node = SegInferenceNode()




    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()




if __name__ == '__main__':
    main()