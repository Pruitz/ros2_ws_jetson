#!/usr/bin/env python3
"""
analizza_benchmark_tesi.py

Script di elaborazione statistica dei tempi di calcolo registrati dal nodo di controllo.
Genera tabelle comparative (Con Debug Video vs Senza Debug Video) formattate
sia in Markdown che in LaTeX per la stesura della Tesi di Laurea.

Uso:
    python3 scripts/analizza_benchmark_tesi.py
    python3 scripts/analizza_benchmark_tesi.py /percorso/al/file.csv
"""

import sys
import csv
from pathlib import Path
import numpy as np

DEFAULT_CSV = "/mnt/nvme/emanuele_project/benchmark_tempi_tesi.csv"

def analyze_group(records):
    if not records:
        return None

    totals = np.array([r["total_ms"] for r in records])
    inferences = np.array([r["inference_ms"] for r in records])
    pfs = np.array([r["path_finding_ms"] for r in records])
    decodes = np.array([r["decode_ms"] for r in records])
    preprocesses = np.array([r["preprocess_ms"] for r in records])
    cmd_pubs = np.array([r["cmd_pub_ms"] for r in records])

    return {
        "count": len(records),
        "total": {
            "mean": np.mean(totals),
            "std": np.std(totals),
            "min": np.min(totals),
            "max": np.max(totals),
            "p95": np.percentile(totals, 95),
            "fps": 1000.0 / np.mean(totals) if np.mean(totals) > 0 else 0.0,
        },
        "inference": {
            "mean": np.mean(inferences),
            "std": np.std(inferences),
            "min": np.min(inferences),
            "max": np.max(inferences),
        },
        "path_finding": {
            "mean": np.mean(pfs),
            "std": np.std(pfs),
            "min": np.min(pfs),
            "max": np.max(pfs),
        },
        "decode": {
            "mean": np.mean(decodes),
            "std": np.std(decodes),
        },
        "preprocess": {
            "mean": np.mean(preprocesses),
            "std": np.std(preprocesses),
        },
        "cmd_pub": {
            "mean": np.mean(cmd_pubs),
            "std": np.std(cmd_pubs),
        },
    }

def print_summary(stats_on, stats_off):
    print("=" * 80)
    print("               REPORT STATISTICO TEMPI DI ESECUZIONE (PER TESI)              ")
    print("=" * 80)

    for name, stats in [("CON DEBUG VIDEO (ON)", stats_on), ("SENZA DEBUG VIDEO (OFF)", stats_off)]:
        if stats is None:
            print(f"\nModalità {name}: Nessun dato registrato.")
            continue
        print(f"\n--- {name} (Campioni: {stats['count']} frame) ---")
        t = stats['total']
        inf = stats['inference']
        pf = stats['path_finding']
        print(f"  • TEMPO TOTALE:       {t['mean']:.2f} ± {t['std']:.2f} ms  (Min: {t['min']:.2f}, Max: {t['max']:.2f}, 95° perc: {t['p95']:.2f})")
        print(f"    -> FPS teorici:     {t['fps']:.1f} FPS")
        print(f"  • INFERENZA ONNX:     {inf['mean']:.2f} ± {inf['std']:.2f} ms  (Min: {inf['min']:.2f}, Max: {inf['max']:.2f})")
        print(f"  • PATH EXTRACTOR:     {pf['mean']:.2f} ± {pf['std']:.2f} ms  (Min: {pf['min']:.2f}, Max: {pf['max']:.2f})")
        print(f"  • Decodifica Camera:  {stats['decode']['mean']:.2f} ± {stats['decode']['std']:.2f} ms")
        print(f"  • Preprocessing:      {stats['preprocess']['mean']:.2f} ± {stats['preprocess']['std']:.2f} ms")
        print(f"  • Invio cmd_vel:      {stats['cmd_pub']['mean']:.2f} ± {stats['cmd_pub']['std']:.2f} ms")

    # Confronto diretto se presenti entrambi
    if stats_on and stats_off:
        diff_ms = stats_on['total']['mean'] - stats_off['total']['mean']
        speedup = (stats_on['total']['mean'] / stats_off['total']['mean'] - 1.0) * 100.0
        print("\n" + "=" * 80)
        print(f"RISPARMIO TEMPO TOTALE: -{diff_ms:.2f} ms per frame ({speedup:.1f}% più veloce)")
        print(f"CAPACITÀ FPS: da {stats_on['total']['fps']:.1f} FPS a {stats_off['total']['fps']:.1f} FPS")
        print("=" * 80)

    # Stampa tabella Markdown per Tesi
    print("\n### Tabella Markdown per Documentazione / Tesi:")
    print("| Fase della Pipeline | Con Debug Video (ms) | Senza Debug Video (ms) | Delta (ms) |")
    print("| :--- | :---: | :---: | :---: |")

    def fmt(st, key):
        if st is None:
            return "N/D"
        return f"{st[key]['mean']:.2f} ± {st[key]['std']:.2f}"

    def delta(k):
        if stats_on and stats_off:
            d = stats_off[k]['mean'] - stats_on[k]['mean']
            return f"{d:+.2f}"
        return "-"

    print(f"| **Tempo Totale Callback** | {fmt(stats_on, 'total')} | {fmt(stats_off, 'total')} | {delta('total')} |")
    print(f"| **Inferenza Rete Neurale** | {fmt(stats_on, 'inference')} | {fmt(stats_off, 'inference')} | {delta('inference')} |")
    print(f"| **Path Extractor (pf4)** | {fmt(stats_on, 'path_finding')} | {fmt(stats_off, 'path_finding')} | {delta('path_finding')} |")
    print(f"| Decodifica Frame Camera | {fmt(stats_on, 'decode')} | {fmt(stats_off, 'decode')} | {delta('decode')} |")
    print(f"| Preprocessing & Resize | {fmt(stats_on, 'preprocess')} | {fmt(stats_off, 'preprocess')} | {delta('preprocess')} |")
    print(f"| Invio Comando cmd_vel | {fmt(stats_on, 'cmd_pub')} | {fmt(stats_off, 'cmd_pub')} | {delta('cmd_pub')} |")

    fps_on = f"{stats_on['total']['fps']:.1f}" if stats_on else "N/D"
    fps_off = f"{stats_off['total']['fps']:.1f}" if stats_off else "N/D"
    print(f"| **Frequenza Massima (FPS)** | **{fps_on} FPS** | **{fps_off} FPS** | - |")

    # Stampa tabella LaTeX per Tesi (Overleaf)
    print("\n### Tabella LaTeX per Overleaf / Tesi:")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Confronto prestazioni ed efficienza computazionale con e senza pipeline di debug video.}")
    print(r"\label{tab:benchmark_pipeline}")
    print(r"\begin{tabular}{lccc}")
    print(r"\hline")
    print(r"\textbf{Fase della Pipeline} & \textbf{Con Debug Video (ms)} & \textbf{Senza Debug Video (ms)} & \textbf{$\Delta$ (ms)} \\")
    print(r"\hline")
    print(f"Tempo Totale Callback       & {fmt(stats_on, 'total')} & {fmt(stats_off, 'total')} & {delta('total')} \\\\")
    print(f"Inferenza Rete Neurale     & {fmt(stats_on, 'inference')} & {fmt(stats_off, 'inference')} & {delta('inference')} \\\\")
    print(f"Path Extractor (pf4)        & {fmt(stats_on, 'path_finding')} & {fmt(stats_off, 'path_finding')} & {delta('path_finding')} \\\\")
    print(f"Decodifica Frame Camera     & {fmt(stats_on, 'decode')} & {fmt(stats_off, 'decode')} & {delta('decode')} \\\\")
    print(f"Preprocessing \\& Resize    & {fmt(stats_on, 'preprocess')} & {fmt(stats_off, 'preprocess')} & {delta('preprocess')} \\\\")
    print(f"Invio Comando cmd\\_vel     & {fmt(stats_on, 'cmd_pub')} & {fmt(stats_off, 'cmd_pub')} & {delta('cmd_pub')} \\\\")
    print(r"\hline")
    print(f"Frequenza Massima (FPS)     & {fps_on} FPS & {fps_off} FPS & - \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analisi benchmark tempi pipeline di navigazione")
    parser.add_argument("csv_file", nargs="?", default=DEFAULT_CSV, help="Percorso al file CSV")
    parser.add_argument("--skip-warmup", type=int, default=1, help="Numero di frame iniziali da scartare per escludere il warmup CUDA (default: 1)")
    args = parser.parse_args()

    path = Path(args.csv_file)
    if not path.exists():
        print(f"File non trovato: {path}")
        print("Avvia prima il nodo per registrare i dati di benchmark!")
        sys.exit(1)

    records_on = []
    records_off = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < args.skip_warmup:
                continue
            try:
                rec = {
                    "frame": int(row["frame"]),
                    "total_ms": float(row["total_ms"]),
                    "inference_ms": float(row["inference_ms"]),
                    "path_finding_ms": float(row["path_finding_ms"]),
                    "decode_ms": float(row["decode_ms"]),
                    "preprocess_ms": float(row["preprocess_ms"]),
                    "cmd_pub_ms": float(row["cmd_pub_ms"]),
                    "debug_video_enabled": str(row["debug_video_enabled"]).lower() in ["true", "1"],
                }
                if rec["debug_video_enabled"]:
                    records_on.append(rec)
                else:
                    records_off.append(rec)
            except Exception:
                continue

    if args.skip_warmup > 0:
        print(f"[Info] Scartati i primi {args.skip_warmup} frame per escludere il warmup iniziale della GPU.")

    stats_on = analyze_group(records_on)
    stats_off = analyze_group(records_off)

    print_summary(stats_on, stats_off)

if __name__ == "__main__":
    main()
