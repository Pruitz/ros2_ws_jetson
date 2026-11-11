# Locked-clock callback benchmark

This is a new benchmark result, separate from the unlocked-clock data in
`benchmark_tempi/raw/`. It uses the same workspace and bag
`database_bags/insalata_grande1`, played at normal speed, with one case at a
time. Each case contains at least 110 raw callbacks; frames 6--105 (five
warm-up frames plus 100 measured frames) are analyzed.

The Jetson was set by the user to:

```text
sudo nvpmodel -m 0
sudo jetson_clocks
```

The user then verified `NV Power Mode: MAXN` and fixed CPU, GPU, and EMC
frequencies with `sudo jetson_clocks --show`. Every controller log reports
`CUDAExecutionProvider` as active.

Raw logs are in `raw_locked/`. The aggregate statistics are in
`tempi_callback_stats_locked.csv`. This run is intentionally not compared
against the approximately 9.9 ms inference check from the unlocked-clock
benchmark: with clocks locked, inference is approximately 5.3--6.0 ms across
the cases.
