#!/usr/bin/env python3
"""
Simulatore Camera Virtuale Fedele alla Cinematica Reale del Robot Demetra
========================================================================
Modella fedelmente:
1. La geometria e cinematica del robot reale da demetra_serial_driver/kinematics.cpp:
   - Passo del robot (wheelbase): L = 0.40 m
   - Sterzatura simmetrica a 4 ruote (Dual Ackermann / 4WS):
       delta = atan2(L * omega, 2 * v)
       omega_eff = (2 * v * tan(delta)) / L
   - Limite fisico reale di sterzo: +-30.0 gradi
2. L'integrazione della posa del robot nel piano:
   - theta(t+dt) = theta(t) + omega_eff * dt
   - y(t+dt) = y(t) - v * sin(theta) * dt
3. La proiezione ottica prospettica della RealSense D435:
   - Focale fx = ~465 px a 640x480 (HFOV ~69 gradi)
   - Distanza lookahead D = ~1.25 m (riga 70% dell'immagine dove calcola l'errore)
   - dx_pixel = - (y * K_y) + (theta * K_theta)

Comandi interattivi da tastiera (nella finestra di questo terminale):
  [ a ] oppure [ <- ] : Perturba la posa del robot di 4 cm verso SINISTRA
  [ d ] oppure [ -> ] : Perturba la posa del robot di 4 cm verso DESTRA
  [ A ] / [ D ]       : Perturbazione rapida di 10 cm (+/- 0.10 m)
  [ r ] o [ Spazio ]  : Ripristina il robot al CENTRO (y = 0, theta = 0)
  [ q ]               : Esci
"""

from pathlib import Path
import math
import sys
import select
import tty
import termios
import threading
from time import perf_counter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import numpy as np
import cv2


# === SPECIFICHE GEOMETRICHE E CINEMATICHE DEL VERO ROBOT DEMETRA ===
# Fonte: demetra_serial_driver/config/driver_params.yaml e kinematics.cpp
WHEEL_BASE_M = 0.40                   # Passo assi ruote: 40 cm
MAX_STEERING_RAD = math.radians(30.0) # Limite massimo di sterzatura meccanica: +-30° (0.5236 rad)
WHEEL_DIAMETER_M = 0.192              # Diametro ruote: 19.2 cm

# === SPECIFICHE OTTICHE REALSENSE D435 ===
# Fonte: RealSense D435 nominal a risoluzione 640x480 (HFOV 69°)
FOCAL_LENGTH_PX = 465.0               # fx in pixel
LOOKAHEAD_DIST_M = 1.25               # Punto di mira sul terreno a y_ref (70% dell'immagine)
PX_PER_METER = FOCAL_LENGTH_PX / LOOKAHEAD_DIST_M   # ~372 px per metro di scostamento
PX_PER_RAD = FOCAL_LENGTH_PX                        # ~465 px per radiante di imbardata (yaw)


class VirtualCameraSimulator(Node):
    def __init__(self):
        super().__init__('virtual_camera_simulator')

        # === PARAMETRI CONFIGURABILI ===
        default_img = str(Path(__file__).resolve().parent / "frame000021.png")
        self.declare_parameter('image_path', default_img)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('max_lateral_m', 0.45) # Limite larghezza corsia serra (+- 45 cm)

        image_path = Path(self.get_parameter('image_path').value)
        self.fps = float(self.get_parameter('fps').value)
        self.dt = 1.0 / self.fps
        self.max_lat_m = float(self.get_parameter('max_lateral_m').value)

        # Caricamento immagine base del filare
        if not image_path.is_file():
            self.get_logger().error(f"Immagine non trovata: {image_path}")
            raise FileNotFoundError(f"File non trovato: {image_path}")

        raw_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_img is None:
            raise RuntimeError(f"Impossibile decodificare l'immagine: {image_path}")

        self.img_h, self.img_w = raw_img.shape[:2]

        # Padding orizzontale riflettente per consentire scostamenti ampi senza bordi neri
        self.pad_x = 400
        self.padded_img = cv2.copyMakeBorder(
            raw_img, 0, 0, self.pad_x, self.pad_x, cv2.BORDER_REFLECT_101
        )

        # === STATO CINEMATICO DEL ROBOT NEL FILARE ===
        # y: scostamento laterale rispetto al centro del filare (+ = robot a destra del centro, in metri)
        # theta: orientamento yaw rispetto all'asse del filare (+ = robot ruotato a sinistra, in radianti)
        self.y = 0.0
        self.theta = 0.0

        # Variabili dinamiche
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.last_cmd_time = 0.0
        self.current_steering_deg = 0.0
        self.current_dx = 0
        self.running = True

        # === COMUNICAZIONE ROS 2 ===
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Pubblica sia JPEG che RAW (per web_video_server / browser)
        self.pub_image = self.create_publisher(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            sensor_qos
        )
        self.pub_image_raw = self.create_publisher(
            Image,
            '/camera/camera/color/image_raw',
            sensor_qos
        )

        # Ascolta /cmd_vel inviato dal controller neurale ONNX
        self.sub_cmd = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.on_cmd_vel,
            10
        )

        # Servizio opzionale di reset posa
        self.srv_reset = self.create_service(
            Empty,
            '/simulation/reset',
            self.on_reset_service
        )

        # Timer loop simulazione a 30 FPS
        self.timer = self.create_timer(self.dt, self.on_step)
        self.log_counter = 0

        # Thread di lettura tastiera interattivo
        if sys.stdin.isatty():
            self.kb_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
            self.kb_thread.start()

        self.print_banner()

    def print_banner(self):
        banner = f"""
====================================================================
 SIMULATORE DEMETRA SITL - CINEMATICA REALE 4WS (Dual-Ackermann)
====================================================================
 Parametri Piattaforma Demetra (fedeli al firmware seriale):
   * Passo robot (wheelbase):      {WHEEL_BASE_M*100:.1f} cm
   * Limite sterzo meccanico:      +-30.0 gradi ({math.degrees(MAX_STEERING_RAD):.1f} deg)
   * Cinematica di sterzo:         4WS simmetrica [omega = 2*v*tan(delta)/L]
   * Camera:                       RealSense D435 (HFOV 69 deg, fx={FOCAL_LENGTH_PX:.0f} px)

 Comandi da tastiera (nella finestra di questo terminale):
   [ a ] oppure [ <- Freccia Sinistra ] : Perturba posa di 4 cm a SINISTRA
   [ d ] oppure [ -> Freccia Destra   ] : Perturba posa di 4 cm a DESTRA
   [ A ] / [ D ]                        : Perturba posa di 10 cm (+/- 10 cm)
   [ r ] oppure [ Barra Spaziatrice   ] : Centra robot (y = 0 cm, yaw = 0 deg)
   [ q ]                                : Esci
====================================================================
"""
        print(banner, flush=True)

    def on_cmd_vel(self, msg: Twist):
        self.cmd_v = float(msg.linear.x)
        self.cmd_w = float(msg.angular.z)
        self.last_cmd_time = perf_counter()

    def keyboard_listener(self):
        """Legge i tasti in tempo reale senza dover premere Invio."""
        fd = sys.stdin.fileno()
        try:
            old_settings = termios.tcgetattr(fd)
        except Exception:
            return

        try:
            tty.setcbreak(fd)
            while self.running and rclpy.ok():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not rlist:
                    continue
                c = sys.stdin.read(1)
                if not c:
                    break

                if c == '\x1b':  # Sequenza frecce
                    seq = sys.stdin.read(2)
                    if seq == '[D':  # Freccia Sinistra
                        self.y -= 0.04
                        self.print_status()
                    elif seq == '[C':  # Freccia Destra
                        self.y += 0.04
                        self.print_status()
                    elif seq in ('[A', '[B'):
                        self.y = 0.0
                        self.theta = 0.0
                        self.print_status()
                elif c in ('a', 'j'):
                    self.y -= 0.04
                    self.print_status()
                elif c in ('d', 'l'):
                    self.y += 0.04
                    self.print_status()
                elif c in ('A', 'J'):
                    self.y -= 0.10
                    self.print_status()
                elif c in ('D', 'L'):
                    self.y += 0.10
                    self.print_status()
                elif c in ('r', 'R', ' ', 'c', 'C', '0'):
                    self.y = 0.0
                    self.theta = 0.0
                    self.print_status()
                elif c in ('q', 'Q', '\x03'):
                    print("\nUscita dal simulatore...", flush=True)
                    self.running = False
                    break
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def print_status(self):
        self.y = max(-self.max_lat_m, min(self.max_lat_m, self.y))
        bar_len = 16
        norm = int(((self.y + self.max_lat_m) / (2.0 * self.max_lat_m)) * bar_len)
        bar = list("-" * bar_len)
        if 0 <= norm < bar_len:
            bar[norm] = "|"
        bar_str = "".join(bar)

        now = perf_counter()
        is_active = (now - self.last_cmd_time) < 0.6 and (abs(self.cmd_v) > 0.001 or abs(self.cmd_w) > 0.001)

        sat_warn = " [SAT 30°]" if abs(abs(self.current_steering_deg) - 30.0) < 0.5 else ""

        if is_active:
            ctrl_str = f"CMD: v={self.cmd_v:.2f} w={self.cmd_w:+.2f} | sterzo: {self.current_steering_deg:+5.1f}°{sat_warn}"
        else:
            ctrl_str = "CMD: in attesa controller..."

        print(
            f"\r[DEMETRA SITL] [{bar_str}] y: {self.y*100:+5.1f}cm | yaw: {math.degrees(self.theta):+4.1f}° | dx: {self.current_dx:+4d}px | {ctrl_str:<42}",
            end='',
            flush=True
        )

    def on_reset_service(self, request, response):
        self.y = 0.0
        self.theta = 0.0
        self.print_status()
        return response

    def on_step(self):
        now = perf_counter()
        time_since_cmd = now - self.last_cmd_time

        # Watchdog: se il controller non trasmette comandi da > 0.5s, robot fermo
        if time_since_cmd > 0.5:
            v_eff = 0.0
            w_cmd = 0.0
        else:
            v_eff = max(0.0, self.cmd_v)
            w_cmd = self.cmd_w

        # --- 1. CONVERSIONE TWIST -> ANGOLO DI STERZO DEMETRA (da kinematics.cpp) ---
        # Nel driver C++ di Demetra:
        # phi = atan2(wheel_base * omega, 2 * v)
        # omega_eff = 2 * v * tan(phi) / wheel_base
        if abs(v_eff) < 1e-4:
            phi = 0.0
            omega_eff = 0.0
        else:
            # Calcolo angolo di sterzo desiderato
            phi_raw = math.atan2(WHEEL_BASE_M * w_cmd, 2.0 * v_eff)
            # Saturazione meccanica a +-30°
            phi = max(-MAX_STEERING_RAD, min(MAX_STEERING_RAD, phi_raw))
            # Velocità angolare effettiva permessa dalla cinematica a 4 ruote sterzanti
            omega_eff = (2.0 * v_eff * math.tan(phi)) / WHEEL_BASE_M

        self.current_steering_deg = math.degrees(phi)

        # --- 2. INTEGRAZIONE DELLA CINEMATICA NEL PIANO (Odom filare) ---
        # theta: orientamento yaw rispetto al filare (+ = robot rivolto a sinistra)
        self.theta += omega_eff * self.dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # y: scostamento laterale (+ = robot a destra del centro filare)
        # Se theta > 0 (robot rivolto a sinistra), avanzare porta il robot verso sinistra -> dy negativo
        dy = - v_eff * math.sin(self.theta) * self.dt
        self.y += dy
        self.y = max(-self.max_lat_m, min(self.max_lat_m, self.y))

        # --- 3. PROIEZIONE OTTICA REALSENSE D435 ---
        # Se robot a destra (+y) -> filare a sinistra (-dx)
        # Se robot rivolto a sinistra (+theta) -> scena nella camera ruota a destra (+dx)
        dx_meters = - (self.y * PX_PER_METER)
        dx_theta = + (self.theta * PX_PER_RAD)
        self.current_dx = int(round(dx_meters + dx_theta))

        # Aggiornamento riga di telemetria nel terminale ogni ~10 cicli (3 Hz)
        self.log_counter += 1
        if self.log_counter % 10 == 0:
            self.print_status()

        # --- 4. RITAGLIO FRAME TRASLATO (PULITO, SENZA SOVRIMPRESSIONI) ---
        # start_x rispetto all'immagine con padding
        start_x = self.pad_x - self.current_dx
        start_x = max(0, min(self.padded_img.shape[1] - self.img_w, start_x))
        frame = self.padded_img[:, start_x:start_x + self.img_w].copy()

        # --- 5. PUBBLICAZIONE SU ROS 2 ---
        try:
            now_msg = self.get_clock().now().to_msg()

            # Compressa per controller_node
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                comp_msg = CompressedImage()
                comp_msg.header.stamp = now_msg
                comp_msg.header.frame_id = "camera_color_optical_frame"
                comp_msg.format = "jpeg"
                comp_msg.data = encoded.tobytes()
                self.pub_image.publish(comp_msg)

            # RAW per visualizzazione web / browser (QoS RELIABLE)
            raw_msg = Image()
            raw_msg.header.stamp = now_msg
            raw_msg.header.frame_id = "camera_color_optical_frame"
            raw_msg.height = self.img_h
            raw_msg.width = self.img_w
            raw_msg.encoding = "bgr8"
            raw_msg.is_bigendian = False
            raw_msg.step = self.img_w * 3
            raw_msg.data = frame.tobytes()
            self.pub_image_raw.publish(raw_msg)
        except Exception:
            return


def main(args=None):
    rclpy.init(args=args)
    sim = VirtualCameraSimulator()
    try:
        rclpy.spin(sim)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        sim.running = False
        sim.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
