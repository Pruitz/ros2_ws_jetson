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
import csv




import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import String




import numpy as np
import cv2




from seg_controller_onnx.onnx_inference import (
    build_onnx_session,
    run_inference_on_frame,
)
# Scegli l'algoritmo di path-finding desiderato (tutti hanno la stessa interfaccia):
from .pf4 import path_finder
# from .pf_histogram import path_finder
# from .pf_centri import path_finder
# from .pf_ransac import path_finder





from .debug_visualizer import (
    build_debug_image,
    build_debug_roi_image,
    build_debug_color_path_image,
    cv2_to_compressed_msg,
)

TARGET_W = 224
TARGET_H = 224
ERROR_ROW_RATIO = 0.8

# VARIABILE DI DEBUG VIDEO:
# True  = genera e pubblica i 3 topic video (/debug/segmentation, /debug/roi, /debug/path) - identico ad adesso
# False = disattiva il rendering grafico e la compressione JPEG, massimizzando gli FPS (risparmia ~15ms di CPU)
ENABLE_DEBUG_IMAGES = True

# LOG BENCHMARK PER TESI:
# True  = registra i tempi frame per frame (totale, inferenza, path extractor) in un file CSV per la tesi
# False = disattiva la scrittura su file
SAVE_BENCHMARK_CSV = True
BENCHMARK_CSV_PATH = "/mnt/nvme/emanuele_project/benchmark_tempi_tesi.csv"




def resize_frame(frame):
    return cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)





def central_pixel_side_from_model(pf_debug, path_matrix):
    """
    Calcola l'errore laterale usando primariamente il modello analitico della centerline.
    Se non presente, usa come fallback la scansione diretta della path_matrix.
    """
    center_model = pf_debug.get("centerline_model")
    h, w = path_matrix.shape
    y_ref = int(h * ERROR_ROW_RATIO)
    y_ref = max(0, min(h - 1, y_ref))

    if center_model is not None:
        x_center = center_model["m"] * y_ref + center_model["q"]
    else:
        # Fallback robusto su path_matrix intorno a y_ref
        row_span = int(h * 0.15)
        row_start = max(0, y_ref - row_span)
        row_end = min(h - 1, y_ref + row_span)
        found_x = []
        for row in range(row_start, row_end + 1):
            cols = np.where(path_matrix[row] > 0)[0]
            if len(cols) > 0:
                found_x.append(float(np.mean(cols)))
        if not found_x:
            return None, "no_model"
        x_center = float(np.mean(found_x))

    error = x_center - (w / 2.0)

    if error > 0:
        side = "destra"
    elif error < 0:
        side = "sinistra"
    else:
        side = "centro"

    return error, side






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
            / 'lraspp_224x224.onnx'
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
        self.process_every_n = 1   # 30fps / 3 = 10 Hz di analisi (era 1 = 30 Hz)




        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.callback,
            sensor_qos
        )




        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 20)

        self.declare_parameter('enable_debug_images', ENABLE_DEBUG_IMAGES)
        self.enable_debug_images = bool(self.get_parameter('enable_debug_images').value)
        self.get_logger().info(f'Debug video/immagini abilitato: {self.enable_debug_images}')

        self.declare_parameter('save_benchmark_csv', SAVE_BENCHMARK_CSV)
        self.save_benchmark_csv = bool(self.get_parameter('save_benchmark_csv').value)
        self.declare_parameter('benchmark_csv_path', BENCHMARK_CSV_PATH)
        self.benchmark_csv_path = str(self.get_parameter('benchmark_csv_path').value)

        self.csv_file = None
        self.csv_writer = None
        if self.save_benchmark_csv:
            csv_path = Path(self.benchmark_csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = csv_path.exists() and csv_path.stat().st_size > 0
            self.csv_file = open(self.benchmark_csv_path, 'a', newline='', buffering=1)
            self.csv_writer = csv.writer(self.csv_file)
            if not file_exists:
                self.csv_writer.writerow([
                    'frame', 'debug_video', 'tempo_totale_ms', 'tempo_inferenza_ms', 'tempo_path_extractor_ms'
                ])
            self.get_logger().info(f'Registrazione benchmark tesi attiva su: {self.benchmark_csv_path}')




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
        self.debug_text_pub = self.create_publisher(
            String,
            '/debug/text',
            debug_qos
        )




        self.filtered_error = None
        self.last_valid_error = None
        self.last_valid_time = 0.0
        self.lost_count = 0
        self.consecutive_jumps = 0




        self.last_status = "lost"




        self.deadband_px = 4.0
        self.hold_timeout = 0.50   # mantenimento ultimo errore valido su buchi temporanei (~500ms)
        self.max_lost_frames = 15  # a 30 FPS = ~500ms prima di fermarsi del tutto




        self.alpha_min = 0.08
        self.alpha_max = 0.45
        self.jump_thr_min = 20.0
        self.jump_thr_max = 60.0




        self.kp_lat = 1.2   # 30° a ~54px di errore (più fluido, evita oscillazioni)

        # === PROFILI VELOCITÀ / STERZO (30° max) ===
        # --- Profilo REALE 1 (Standard): v=0.30 m/s, w=0.87 rad/s ---
        # self.v_max = 0.30
        # self.v_min = 0.07
        # self.w_max = 0.87

        # --- Profilo REALE 2 (Moderato): v=0.20 m/s, w=0.58 rad/s ---
        self.v_max = 0.20
        self.v_min = 0.07
        self.w_max = 0.58

        # --- Profilo TEST ATTIVO (Molto lento): v=0.07 m/s, w=0.21 rad/s ---
        #self.v_max = 0.07
        #self.v_min = 0.07
        #self.w_max = 0.21   # w per ottenere 30° a v=0.07: atan2(0.4*0.21, 2*0.07) >= 30°
        # COME SCEGLIERE v_max E w_max
        # Il robot non usa w direttamente: il driver lo converte in angolo delle ruote (sterzo):
        #   phi = atan2(wheel_base * w, 2 * v)   con wheel_base = 0.4
        # A parità di w, più v è alta e meno il robot sterza. Se lo sterzo è
        # troppo piccolo (pochi gradi) il robot sembra non girare.
        # Lo sterzo massimo del robot è 30°.
        #
        # La tabella dice: "con velocità v e w, di quanti gradi sterzano le ruote?"
        #   - RIGA   = la tua v_max (velocità in avanti, m/s)
        #   - COLONNA = un valore di w (rad/s)
        #   - CELLA  = gradi di sterzo che ne risultano
        #
        # Come usarla:
        #   1. Scegli v_max (es. 0.3 m/s) e trova la sua riga.
        #   2. Guarda l'ultima colonna "w per 30°": è il w_max che serve per
        #      far sterzare il robot al massimo (30°). Usa quel valore (o poco meno).
        #   3. Per curve più dolci scegli un w_max più basso: leggi nella riga
        #      la colonna che ti dà i gradi che vuoi.
        #
        # Esempio: v_max = 0.3 e w_max = 0.50 -> sterzo max 18.4°.
        #          Per arrivare a 30° servirebbe w_max = 0.87 (ultima colonna).
        # Esempio: w = 0.16 con v = 2.0 -> sterzo 0.9° (quasi nullo, il robot non gira).
        #
        # (wheel_base = 0.4 è il valore di driver_params.yaml; i gradi sono calcolati
        #  con la formula del driver, non misurati sul robot)
        #
        #  v \ w |  0.05 |  0.10 |  0.16 |  0.25 |  0.50 |  1.00 | w per 30°
        # -------+-------+-------+-------+-------+-------+-------+----------
        #   0.2  |  2.9  |  5.7  |  9.1  | 14.0  | 26.6  | 45.0  |   0.58
        #   0.3  |  1.9  |  3.8  |  6.1  |  9.5  | 18.4  | 33.7  |   0.87
        #   0.5  |  1.1  |  2.3  |  3.7  |  5.7  | 11.3  | 21.8  |   1.44
        #   0.7  |  0.8  |  1.6  |  2.6  |  4.1  |  8.1  | 15.9  |   2.02
        #   1.0  |  0.6  |  1.1  |  1.8  |  2.9  |  5.7  | 11.3  |   2.89
        #   1.5  |  0.4  |  0.8  |  1.2  |  1.9  |  3.8  |  7.6  |   4.33
        #   2.0  |  0.3  |  0.6  |  0.9  |  1.4  |  2.9  |  5.7  |   5.77
        #
        # Più v_max è alta, più w_max deve crescere per ottenere lo stesso sterzo.




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

        # Se PF è perso, mantengo temporaneamente l'ultimo errore valido se entro hold_timeout
        if pf_status == "lost":
            if (
                self.last_valid_error is not None
                and (now - self.last_valid_time) <= self.hold_timeout
            ):
                cmd_error = self.filtered_error
                if abs(cmd_error) < self.deadband_px:
                    cmd_error = 0.0
                return cmd_error, "hold_lost"
            self.filtered_error = None
            self.last_valid_error = None
            self.consecutive_jumps = 0
            return None, "lost_by_pf_status"

        # Se non c'è errore grezzo (es. centerline assente)
        if error_px is None:
            if (
                self.last_valid_error is not None
                and (now - self.last_valid_time) <= self.hold_timeout
            ):
                # Mantengo l'ultimo errore filtrato per un breve periodo
                cmd_error = self.filtered_error
                if abs(cmd_error) < self.deadband_px:
                    cmd_error = 0.0
                return cmd_error, "hold"
            self.filtered_error = None
            self.last_valid_error = None
            self.consecutive_jumps = 0
            return None, "lost"

        # Parametri in base allo stato PF (due regimi semplici)
        if pf_status == "degraded":
            alpha = 0.18        # quasi uguale a ok
            jump_thr = 38.0     # soglia rilassata
        else:
            alpha = 0.20        # reattivo
            jump_thr = 40.0     # soglia di salto normale

        # Inizializzo filtered_error se è la prima volta o dopo uno stop/lost
        if self.filtered_error is None:
            self.filtered_error = float(error_px)
            self.consecutive_jumps = 0
        else:
            jump = abs(error_px - self.filtered_error)

            if jump <= jump_thr:
                # Salto normale: filtro esponenziale classico
                self.filtered_error = (
                    (1.0 - alpha) * self.filtered_error + alpha * float(error_px)
                )
                self.consecutive_jumps = 0
            else:
                self.consecutive_jumps += 1
                if self.consecutive_jumps >= 2:
                    # Se il salto persiste per >= 2 frame consecutivi, è la linea reale
                    self.filtered_error = float(error_px)
                    self.consecutive_jumps = 0
                else:
                    # Singolo frame isolato: integrazione prudente senza azzeramento
                    self.filtered_error = (
                        0.80 * self.filtered_error + 0.20 * float(error_px)
                    )

        # Aggiorno ultimo errore valido e tempo
        self.last_valid_error = float(self.filtered_error)
        self.last_valid_time = now

        # Deadband applicata all'OUTPUT di comando per non oscillare a centro,
        # senza azzerare lo stato interno del filtro (evita il deadlock a 0)
        cmd_error = self.filtered_error
        if abs(cmd_error) < self.deadband_px:
            cmd_error = 0.0

        # Determino lo stato del filtro
        if pf_status == "degraded":
            filter_status = "valid_degraded"
        else:
            filter_status = "valid"

        return cmd_error, filter_status



    def compute_cmd(self, error_px, img_width, confidence, pf_status):
        if error_px is None:
            return 0.0, 0.0




        error_norm = error_px / (img_width / 2.0)
        w_gain = 0.6 + 0.4 * float(np.clip(confidence, 0.0, 1.0))
        w = -(self.kp_lat * w_gain * error_norm)




        # Penalità degraded rimossa: se degraded è la condizione normale,
        # ridurre w al 70% porta il robot a sterzare sempre meno del necessario.
        # if pf_status == "degraded":
        #     w *= 0.7




        w = float(np.clip(w, -self.w_max, self.w_max))
        v = self.v_max
        return v, w




    def publish_stop(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)




    def callback(self, msg):
        t_cb_0 = perf_counter()
        self.frame_count += 1




        if self.frame_count % self.process_every_n != 0:
            return




        t_decode_0 = perf_counter()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        t_decode_1 = perf_counter()




        if frame is None:
            t_cb_1 = perf_counter()
            self.get_logger().warn(
                f'Immagine non decodificata | callback_total={(t_cb_1 - t_cb_0)*1000:.1f} ms'
            )
            self.publish_stop()
            return




        camera_frame = frame.copy() if self.enable_debug_images else None
        t0 = perf_counter()
        frame = resize_frame(frame)
        t1 = perf_counter()




        pred_mask = run_inference_on_frame(
            self.session,
            self.input_name,
            self.output_name,
            frame
        )
        t2 = perf_counter()




        t3 = perf_counter()
        path_matrix, pf_debug = path_finder(pred_mask)
        t4 = perf_counter()



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


        # === 1. CALCOLA E PUBBLICA CMD_VEL SUBITO (MASSIMA PRIORITÀ CONTROLLO) ===
        is_stopped = False
        if self.lost_count >= self.max_lost_frames or filtered_error is None:
            self.filtered_error = None
            self.last_valid_error = None
            self.consecutive_jumps = 0
            v, w = 0.0, 0.0
            t_cmd_pub_0 = perf_counter()
            self.publish_stop()
            t_cmd_pub_1 = perf_counter()
            is_stopped = True
        else:
            img_width = path_matrix.shape[1]
            v, w = self.compute_cmd(filtered_error, img_width, confidence, pf_status)
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            t_cmd_pub_0 = perf_counter()
            self.cmd_pub.publish(cmd)
            t_cmd_pub_1 = perf_counter()

        # === 2. GENERAZIONE E PUBBLICAZIONE DEBUG (SE ABILITATO) ===
        if self.enable_debug_images:
            debug_img = build_debug_image(
                pred_mask,
                path_matrix,
                raw_error,
                filtered_error,
                side,
                confidence,
                pf_status,
                error_row_ratio=ERROR_ROW_RATIO,
            )

            # immagine debug seg
            debug_msg = cv2_to_compressed_msg(debug_img, msg.header.stamp)
            if debug_msg is not None:
                self.debug_img_pub.publish(debug_msg)

            # immagine debug roi
            if roi_binary is None:
                roi_y0 = int(pred_mask.shape[0] * 0.30)
                roi_binary = np.zeros_like(pred_mask, dtype=np.uint8)
                roi_binary[roi_y0:, :] = (pred_mask[roi_y0:, :] == 1).astype(np.uint8) * 255

            debug_roi_img = build_debug_roi_image(
                roi_binary, pf_debug, path_matrix, raw_error, side,
                error_row_ratio=ERROR_ROW_RATIO,
            )
            debug_roi_msg = cv2_to_compressed_msg(debug_roi_img, msg.header.stamp)
            if debug_roi_msg is not None:
                self.debug_roi_pub.publish(debug_roi_msg)

            if camera_frame is not None:
                debug_path_img = build_debug_color_path_image(
                    camera_frame, pf_debug, path_matrix, raw_error, side,
                    error_row_ratio=ERROR_ROW_RATIO,
                )
                debug_path_msg = cv2_to_compressed_msg(debug_path_img, msg.header.stamp)
                if debug_path_msg is not None:
                    self.debug_path_pub.publish(debug_path_msg)

        t_cb_1 = perf_counter()

        # Registrazione su file CSV per la Tesi di Laurea (se attiva)
        if self.save_benchmark_csv and self.csv_writer:
            debug_flag = "ON" if self.enable_debug_images else "OFF"
            self.csv_writer.writerow([
                self.frame_count,
                debug_flag,
                f"{(t_cb_1 - t_cb_0)*1000:.2f}",
                f"{(t2 - t1)*1000:.2f}",
                f"{(t4 - t3)*1000:.2f}",
            ])

        if is_stopped:
            raw_error_txt = f"{raw_error:.2f}" if raw_error is not None else "None"
            log_str = (
                f'Frame {self.frame_count} | '
                f'callback_total={(t_cb_1 - t_cb_0)*1000:.1f} ms | '
                f'decode={(t_decode_1 - t_decode_0)*1000:.1f} ms | '
                f'preprocess={(t1 - t0)*1000:.1f} ms | '
                f'inference={(t2 - t1)*1000:.1f} ms | '
                f'path_finding={(t4 - t3)*1000:.1f} ms | '
                f'STOP | pf_status={pf_status} | conf={confidence:.2f} | '
                f'band_ratio={band_on_path_ratio:.2f} | strip_ratio={strip_tracking_ratio:.2f} | '
                f'raw_error={raw_error_txt} | side={side} | '
                f'filter={filter_status} | lost_count={self.lost_count}'
            )
            # Pubblica la stringa di debug sul topic
            txt_msg = String()
            txt_msg.data = log_str
            self.debug_text_pub.publish(txt_msg)

            if self.frame_count % self.log_every_n == 0:
                self.get_logger().warn(log_str)
            return

        raw_error_txt = f"{raw_error:.2f}" if raw_error is not None else "None"
        filtered_error_txt = f"{filtered_error:.2f}" if filtered_error is not None else "None"
        log_str = (
            f'Frame {self.frame_count} | '
            f'callback_total={(t_cb_1 - t_cb_0)*1000:.1f} ms | '
            f'decode={(t_decode_1 - t_decode_0)*1000:.1f} ms | '
            f'preprocess={(t1 - t0)*1000:.1f} ms | '
            f'inference={(t2 - t1)*1000:.1f} ms | '
            f'path_finding={(t4 - t3)*1000:.1f} ms | '
            f'cmd_publish={(t_cmd_pub_1 - t_cmd_pub_0)*1000:.1f} ms | '
            f'pf_status={pf_status} | conf={confidence:.2f} | '
            f'band_ratio={band_on_path_ratio:.2f} | strip_ratio={strip_tracking_ratio:.2f} | '
            f'side={side} | raw_error={raw_error_txt} | '
            f'filtered_error={filtered_error_txt}px | '
            f'filter={filter_status} | v={v:.3f} | w={w:.3f}'
        )
        # Pubblica la stringa di debug sul topic
        txt_msg = String()
        txt_msg.data = log_str
        self.debug_text_pub.publish(txt_msg)

        if self.frame_count % self.log_every_n == 0:
            self.get_logger().info(log_str)



    def destroy_node(self):
        if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()
        super().destroy_node()


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