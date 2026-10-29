#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "Costruzione immagine emanuele-ros:humble-gpu dal Dockerfile in $DIR..."
docker build --progress=plain \
  --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) \
  -t emanuele-ros:humble-gpu .
