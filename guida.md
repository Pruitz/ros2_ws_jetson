# Guida: ripartenza completa sul Jetson

Questa guida avvia il robot da una nuova connessione SSH: container Docker, camera
RealSense, driver seriale e controller. I comandi indicano esplicitamente dove
eseguirli:

- **Terminale Jetson (host):** shell SSH sul Jetson, fuori dal container.
- **Terminale container:** shell Linux dentro `emanuele-ros`.

Sostituisci `<IP_JETSON>` con l'indirizzo IP o il nome host effettivo del Jetson.

## 1. Collegarsi al Jetson

Dal terminale del proprio computer:

```bash
ssh dev@<IP_JETSON>
```

Controlla che camera e seriale del robot siano collegate. Questi comandi si
eseguono sul **Terminale Jetson (host)**:

```bash
ls -l /dev/video*
ls -l /dev/serial/by-id/
```

La camera deve comparire come uno o più dispositivi `/dev/video*`. Per la
seriale è atteso il dispositivo FTDI usato dal robot; lo script usa
automaticamente `/dev/ttyUSB0` se il link `/dev/serial/by-id/...` non è
disponibile.

## 2. Riprendere o avviare il container

È consigliato usare `tmux` sull'host: mantiene in esecuzione il container anche
se la connessione SSH cade.

Sul **Terminale Jetson (host)**:

```bash
tmux new -s robot
```

Se la sessione esiste già, invece di crearla ricollegati con:

```bash
tmux attach -t robot
```

Una volta dentro `tmux`, controlla se il container è già attivo:

```bash
docker ps --filter name=emanuele-ros
```

- Se compare `emanuele-ros`, **non** avviare un secondo container. Apri una
  shell con `docker exec` come descritto sotto e controlla `ros2 node list`:
  dopo una semplice disconnessione SSH i processi dentro `tmux` possono essere
  ancora attivi, quindi riavvia solo i nodi che non sono in esecuzione.
- Se non compare, avvialo dal **Terminale Jetson (host)**:

```bash
cd /mnt/nvme/emanuele_project
./scripts/run.sh
```

Questo comando apre una shell dentro il container `emanuele-ros` e gli rende
accessibili camera e seriale.

Se Docker segnala che l'immagine `emanuele-ros:humble-gpu` non esiste, esegui
sull'host il build una volta sola, poi avvia il container:

```bash
cd /mnt/nvme/emanuele_project
./scripts/build.sh
./scripts/run.sh
```

Per aprire altri terminali nel container, crea una nuova finestra `tmux` con
`Ctrl+b`, poi `c`; nella nuova finestra, sul **Terminale Jetson (host)**, esegui:

```bash
docker exec -it emanuele-ros bash
```

Ripeti questo comando per ciascun terminale aggiuntivo. Per passare tra le
finestre `tmux`, usa `Ctrl+b` seguito da `n` (successiva) o `p` (precedente).

### Usare l'interfaccia grafica di VS Code

1. Apri VS Code sul tuo computer e connettiti al Jetson con **Remote-SSH**:
   apri la palette comandi (`Ctrl+Shift+P`), scegli **Remote-SSH: Connect to
   Host...** e seleziona `dev@<IP_JETSON>`. Se richiesto, inserisci la password
   o usa la chiave SSH configurata.
2. Nella finestra VS Code connessa al Jetson, scegli **File > Open Folder...**
   e apri `/mnt/nvme/emanuele_project`.
3. Apri il terminale integrato con **Terminal > New Terminal**. Il terminale
   iniziale è sull'host Jetson. Per mantenere i processi attivi se VS Code o SSH
   si disconnettono, avvia `tmux` da questo terminale:

   ```bash
   tmux new -s robot
   ```

   Se la sessione esiste già, usa invece `tmux attach -t robot`.
4. Nel terminale integrato collegato a `tmux`, avvia il container come descritto
   sopra. Se il container è già attivo, non rilanciarlo: usa una shell aggiuntiva
   come al punto successivo.
5. Per ogni processo ROS aggiuntivo, crea una finestra `tmux` con `Ctrl+b`,
   poi `c`. La nuova finestra parte sull'host Jetson: entra nel container con:

   ```bash
   docker exec -it emanuele-ros bash
   ```

   Per passare tra le finestre usa `Ctrl+b`, poi `n` o `p`. Puoi anche aprire
   altri terminali con **Terminal > New Terminal**, ma per mantenere i processi
   attivi durante una disconnessione usa le finestre `tmux` create dal terminale
   integrato collegato alla sessione.
6. Nei terminali container distinti, incolla i comandi dei passaggi 3, 4 e 5
   per avviare rispettivamente camera, driver seriale e controller. Lascia
   aperto ciascun terminale mentre il relativo comando è in esecuzione. Se
   riapri VS Code dopo una disconnessione, riconnettiti via Remote-SSH, riapri
   la cartella e usa `tmux attach -t robot` per ritrovare le finestre e i
   processi già avviati.

## 3. Terminale A: avviare la camera RealSense

Se stai ripartendo da un Jetson riavviato o da un container nuovo, nel
**Terminale container** principale avvia la camera e lascia il comando in
esecuzione. Se hai solo perso la connessione SSH e il nodo camera è ancora
attivo, non avviarlo una seconda volta.

```bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py
```

Apri un nuovo terminale container (passaggio 2) e verifica che arrivino le
immagini:

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /camera/camera/color/image_raw/compressed
```

Attendi qualche secondo: il topic deve mostrare una frequenza diversa da zero.
Il controller usa proprio `/camera/camera/color/image_raw/compressed`.

## 4. Terminale B: avviare il driver seriale

Se il driver non è già attivo, avvialo in un nuovo **Terminale container**:

```bash
source /opt/ros/humble/setup.bash
source /mnt/nvme/emanuele_project/ros2_ws_emanuele/install/setup.bash
ros2 run demetra_serial_driver driver_node
```

Lascia il driver in esecuzione. Nei log cerca:
`Serial port opened successfully: /dev/ttyUSB0`.
Il messaggio iniziale `[ERROR] Serial port is not open, trying to reopen...`
può comparire mentre il driver prova ad aprire la porta.

Dopo modifiche al driver (incluse quelle che applicano i limiti di sterzo e
velocità), fermalo e ricompila il pacchetto prima di riavviarlo. Esegui in un
**Terminale container**:

```bash
source /opt/ros/humble/setup.bash
cd /mnt/nvme/emanuele_project/ros2_ws_emanuele
colcon build --symlink-install --packages-select demetra_serial_driver
source install/setup.bash
```

## 5. Terminale C: avviare il controller

Se il controller non è già attivo, dopo aver verificato camera e driver apri un
altro **Terminale container**:

```bash
source /opt/ros/humble/setup.bash
source /mnt/nvme/emanuele_project/ros2_ws_emanuele/install/setup.bash
ros2 run seg_controller_onnx controller_node
```

Il controller può inviare comandi di movimento al robot. Prima di avviarlo,
metti il robot in un'area libera e tieni a portata di mano lo stop di emergenza.

### Regolare velocità e sterzata

I valori sono definiti nel codice del controller:
`ros2_ws_emanuele/src/seg_controller_onnx/seg_controller_onnx/controller_node.py`,
nel blocco `self.kp_lat`, `self.v_max` e `self.w_max`.

- `v_max = 0.3` imposta la velocità in avanti a 0,3 m/s.
- `w_max = 0.87` (già impostato nel controller) limita il comando angolare in rad/s.
  Con la geometria attuale (`wheel_base = 0.4 m`), a `v = 0.3 m/s` equivale
  teoricamente a circa 30° di sterzo:
  `phi = atan2(0.4 * w, 2 * 0.3)`.
- `w_max` è un **limite massimo**, non aumenta da solo la sterzata durante le
  correzioni normali. Aumentarlo oltre `0.87` non dà più sterzo fisico: il driver
  limita l'angolo delle ruote a 30°. La risposta alle deviazioni dipende
  soprattutto da `kp_lat`, che è stato portato a `1.0` per rendere le correzioni
  più energiche. Se il robot oscilla o supera il centro del filare, abbassalo
  gradualmente; se corregge ancora troppo poco, aumentalo con cautela e prova
  ogni volta in un'area libera.

Per cambiare i valori con VS Code, apri il file indicato, modifica i numeri,
salva e nel terminale container ricompila il workspace:

```bash
source /opt/ros/humble/setup.bash
cd /mnt/nvme/emanuele_project/ros2_ws_emanuele
colcon build --symlink-install --packages-select seg_controller_onnx
source install/setup.bash
```

Ferma il controller con `Ctrl+C` e rilancialo perché carichi i nuovi valori.
`v_max`, `w_max` e `kp_lat` sono impostati nel codice, non sono parametri ROS:
non basta passarli al comando `ros2 run`. Dopo una modifica, ferma il controller,
ricompila e rilancialo.

### Se lo sterzo vibra

Non aumentare ancora `kp_lat` o `w_max`. Durante il funzionamento del controller,
controlla prima se il comando oscilla:

```bash
ros2 topic echo /cmd_vel
```

Se `angular.z` cambia rapidamente segno o valore, la vibrazione può venire dalle
correzioni che cambiano col rilevamento visivo. Se invece resta quasi costante
ma lo sterzo vibra, prova a isolare il controller: fermalo con `Ctrl+C` e, con
il robot immobilizzato e le ruote motrici sollevate in sicurezza, invia per
pochi secondi un comando angolare fisso:

```bash
timeout 5 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}, angular: {z: 0.5}}"
```

Con questi valori il comando corrisponde teoricamente a circa 18° di sterzo.
Se vibra anche con il comando fisso, il problema probabilmente non è il filtro
visivo: controlla servo, alimentazione, cablaggi e leveraggi e non lasciare il
test in esecuzione. Se il test è regolare ma vibra solo col controller, annota
`raw_error`, `filtered_error` e `w` dai log del controller: servono a capire se
oscilla la stima o il comando generato.

## 6. Verifiche e streaming video (facoltativo)

Da un ulteriore **Terminale container**, controlla i topic:

```bash
source /opt/ros/humble/setup.bash
source /mnt/nvme/emanuele_project/ros2_ws_emanuele/install/setup.bash
ros2 topic hz /camera/camera/color/image_raw/compressed
ros2 topic info /cmd_vel
```

Per vedere il video da un browser, avvia il server in un altro **Terminale
container**:

```bash
source /opt/ros/humble/setup.bash
ros2 run web_video_server web_video_server --ros-args -p port:=8080 -p address:=0.0.0.0
```

Dal computer collegato alla stessa rete, apri:

```text
http://<IP_JETSON>:8080/stream?topic=/camera/camera/color/image_raw
```

## 7. Fermare i processi

Premi `Ctrl+C` nei terminali del controller, del driver, della camera e,
se avviato, del server video. Per terminare anche il container, torna alla
shell principale avviata da `./scripts/run.sh` e digita:

```bash
exit
```

Lo script avvia Docker con `--rm`, quindi il container viene rimosso quando
termina; il codice e il workspace restano sul Jetson. Per uscire dalla sessione
`tmux` senza fermare i processi, premi `Ctrl+b`, poi `d`. Alla prossima
connessione usa `tmux attach -t robot`.

## Comandi aggiuntivi

### Riprodurre una bag

Dal **Terminale container**, dalla cartella del progetto:

```bash
cd /mnt/nvme/emanuele_project
ros2 bag play database_bags/insalata_grande1 --clock
```

Se la bag pubblica immagini raw e il controller richiede quelle compresse,
avvia prima la bag e poi, in un altro terminale, esegui:

```bash
ros2 run image_transport republish raw compressed \
  --ros-args \
  -r in:=/camera/camera/color/image_raw \
  -r out/compressed:=/camera/camera/color/image_raw/compressed
```

### Registrare i topic principali

```bash
ros2 bag record -o debug_robot \
  /camera/camera/color/image_raw/compressed \
  /cmd_vel \
  /debug/segmentation/compressed \
  /debug/roi/compressed \
  /debug/path/compressed
```

### Test manuale del movimento

Questo test invia velocità al robot per 5 secondi: eseguilo solo con area libera,
robot sotto controllo e stop di emergenza a portata di mano.

```bash
timeout 5 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.0}}"
```
