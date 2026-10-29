# Ambiente Docker per Jetson (Nvidia JetPack 6 / ROS 2 Humble)

Questa cartella contiene la configurazione Docker per replicare l'ambiente di runtime e sviluppo del robot su Nvidia Jetson.

## Contenuto
- **`Dockerfile`**: Immagine basata su `l4t-jetpack:r36.4.0` con ROS 2 Humble, OpenCV e `onnxruntime-gpu`.
- **`build.sh`**: Script per compilare l'immagine Docker (`emanuele-ros:humble-gpu`).
- **`run.sh`**: Script per avviare il container con accelerazione GPU, videocamere USB/RealSense e porta seriale del robot mappate.

## Utilizzo
1. **Compilazione dell'immagine** (solo la prima volta o su nuova Jetson):
   ```bash
   ./docker/build.sh
   ```

2. **Avvio del container**:
   ```bash
   ./docker/run.sh
   ```

3. **Accesso alla shell del container**:
   ```bash
   docker exec -it emanuele-ros bash
   ```
