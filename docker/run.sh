#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
WS_DIR="$(cd "$DIR/.." && pwd)"

# Se ci troviamo nel path standard della Jetson (/mnt/nvme/emanuele_project), usa il path padre
if [ -d "/mnt/nvme/emanuele_project" ]; then
  PROJ=/mnt/nvme/emanuele_project
else
  PROJ="$WS_DIR"
fi

CAM_ARGS=()
for d in /dev/video*; do
  [ -e "$d" ] && CAM_ARGS+=(--device "$d")
done
[ -S /tmp/argus_socket ] && CAM_ARGS+=(-v /tmp/argus_socket:/tmp/argus_socket)

SERIAL=/dev/serial/by-id/usb-FTDI_TTL232R-3V3_FTCZQMIG-if00-port0
if [ -e "$SERIAL" ]; then
  SERIAL_DEVICE=$(readlink -f "$SERIAL")
elif [ -c /dev/ttyUSB0 ]; then
  SERIAL_DEVICE=/dev/ttyUSB0
else
  SERIAL_DEVICE=
fi

if [ -n "$SERIAL_DEVICE" ]; then
  CAM_ARGS+=(--device "$SERIAL_DEVICE:/dev/ttyUSB0")
else
  echo "ATTENZIONE: seriale robot non trovata ($SERIAL o /dev/ttyUSB0), avvio senza" >&2
fi

# Rimuove container fermi precedenti per evitare conflitti di nome
if [ "$(docker ps -aq -f status=exited -f name=emanuele-ros)" ]; then
  echo "Rimuovo container fermo emanuele-ros..."
  docker rm emanuele-ros >/dev/null 2>&1 || true
fi

echo "Avvio container Docker emanuele-ros (volume: $PROJ)..."
docker run -it -d \
  --name emanuele-ros \
  --runtime nvidia \
  --network host --ipc host \
  --group-add video --group-add dialout \
  "${CAM_ARGS[@]}" \
  -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42} \
  -v ${PROJ}:${PROJ} \
  -w ${PROJ} \
  --privileged -v /dev:/dev \
  emanuele-ros:humble-gpu "$@"

echo "Container emanuele-ros avviato con successo in background!"
echo "Per entrare: docker exec -it emanuele-ros bash"
