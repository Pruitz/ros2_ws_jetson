cd /mnt/nvme/emanuele_project/ros2_ws_emanuele
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 run seg_controller_onnx controller_node

PER RUNNARE LA BAG
ros2 bag play "database_bags/insalata_grande1" --clock

//per usare le bag che pubblicano in raw e non in raw/compressed
Prima avvi ros bag play come ros2 bag play "/nas/7 - Experimental data/OrtaggiAgrion/2026_06_11/data_converted/2026-06-11_10-41-54" --clock che pubblica in raw, e poi questo in un nuovo terminale:
ros2 run image_transport republish raw compressed \
  --ros-args \
  -r in:=/camera/camera/color/image_raw \
  -r out/compressed:=/camera/camera/color/image_raw/compressed

PER REGISTRARE NUOVI TOPICS
ros2 bag record -o debug_filter4 \
  /camera/camera/color/image_raw/compressed \
  /cmd_vel \
  /debug/segmentation/compressed \
  /debug/roi/compressed \
  /debug/path/compressed \
  /debug/text

  STREAMING
ros2 run web_video_server web_video_server --ros-args -p port:=8080 -p address:=0.0.0.0

COMUNICAZIONE SERIALE (da avviare per mandare i comandi al robot)
source /opt/ros/humble/setup.bash
source /mnt/nvme/emanuele_project/ros2_ws_emanuele/install/setup.bash
ros2 run demetra_serial_driver driver_node --ros-args --params-file /mnt/nvme/emanuele_project/ros2_ws_emanuele/src/demetra_robot/demetra_serial_driver/config/driver_params.yaml

PER LANCIARE LA CAMERA
ros2 launch realsense2_camera rs_launch.py camera_name:=camera camera_namespace:=camera

(per vedere quali pacchetti hanno il nome realsense)
ros2 pkg list | grep realsense
(per vedere i dispositivi collegati usb)
001 002


TEST
timeout 10 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.8}}" ; ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"

TEST 360 (sterzo 30°)
timeout 10.9 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.58}}" ; ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"

TEST 360 (sterzo 26.5°)
timeout 12.6 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.5}}" ; ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"


GUIDA TELEOP DA TASTIERA
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.2 -p turn:=0.58


RESET ODOMETRIA
ros2 service call /reset_odometry std_srvs/srv/Empty
REGISTRA ODOMETRIA
ros2 bag record /odom


SIMULAZIONE CAMERA DA TASTIERA (SPOSTA FOTO SENZA ROBOT)
1) In un terminale container avvia la camera virtuale:
python3 /mnt/nvme/emanuele_project/ros2_ws_emanuele/simulation_singleframe/sim_camera.py

2) Premi direttamente i tasti nella finestra dello script:
   [ a ] oppure [ <- Freccia Sinistra ] : sposta l'immagine a SINISTRA
   [ d ] oppure [ -> Freccia Destra   ] : sposta l'immagine a DESTRA
   [ A ] / [ D ]                        : spostamento rapido (+/- 40 px)
   [ r ] oppure [ Barra Spaziatrice   ] : ricentra l'immagine (offset 0 px)

3) In un secondo terminale puoi avviare il controller per vedere come calcola l'errore:
ros2 run seg_controller_onnx controller_node
