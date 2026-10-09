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