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
  /debug/path/compressed

  STREAMING
ros2 run web_video_server web_video_server --ros-args -p port:=8080 -p address:=0.0.0.0

COMUNICAZIONE SERIALE (da avviare per mandare i comandi al robot)
source /opt/ros/humble/setup.bash
source /mnt/nvme/emanuele_project/ros2_ws_emanuele/install/setup.bash
ros2 run demetra_serial_driver driver_node

PER LANCIARE LA CAMERA
ros2 launch realsense2_camera rs_launch.py camera_name:=camera camera_namespace:=camera

(per vedere quali pacchetti hanno il nome realsense)
ros2 pkg list | grep realsense
(per vedere i dispositivi collegati usb)
001 002