# Controller callback benchmark

The measurements use commit `717f299` (including the final extractor code from
`8067e72`) and the bag `database_bags/insalata_grande1`, played once per case
at normal speed. Each result uses frames 6--105: five warm-up callbacks are
discarded and the next 100 are analyzed. The raw `/debug/text` logs are in
`raw/`; the standard aggregation is in `tempi_callback.txt`, and the extended
statistics are in `tempi_callback_stats.csv`.

The ten cases were run one at a time with only the controller, bag player, and
`ros2 topic echo` active. Every controller log reports
`CUDAExecutionProvider` as an active provider. The EdgeNet model is
`src/seg_controller_onnx/seg_controller_onnx/models/rete_bordi.onnx`,
MD5 `3c3d6423534aca8a92c8d208e9a6b32f`, with `width_row=223`.

Jetson conditions:

- `nvpmodel -q`: `NV Power Mode: MAXN`, mode `0`, at start and end.
- Temperatures and system state are recorded in `start_tegrastats.txt` and
  `end_tegrastats.txt`.
- `sudo jetson_clocks` could not be applied from this session because sudo
  required an interactive password. `jetson_clocks --show` consequently
  reported that it must be run as root.

The initial Center points OFF, Column histogram OFF, and EdgeNet OFF runs did
not meet the requested `9.9 +/- 0.1 ms` inference check. Each was repeated
once; the initial logs are retained with the `_initial` suffix and the final
CSV uses the repeated logs. The repeated values remained systematic rather
than outliers (approximately 10.34, 11.26, and 10.65 ms), so they are marked
invalid in the CSV rather than silently discarded.
