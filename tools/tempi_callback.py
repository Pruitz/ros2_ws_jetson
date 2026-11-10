"""Media e deviazione standard dei tempi della callback del controller_node.

Legge il testo pubblicato su /debug/text (una riga per frame) e calcola, per
ogni file: callback_total, inference, path_finding e il resto
(callback_total - inference - path_finding).

Registrazione, una volta per ogni path finder e per debug video on/off,
con gli stessi frame (per esempio la stessa bag di insalata_grande1):

    ros2 run seg_controller_onnx controller_node --ros-args \
        -p path_finder:=pf5 -p enable_debug_images:=false
    ros2 topic echo --field data /debug/text > tempi_pf5_off.txt
    ros2 bag play <bag_insalata_grande1>

Analisi (salta i primi 5 frame di riscaldamento, usa i 100 successivi):

    python3 tools/tempi_callback.py tempi_*.txt --skip 5 --n 100
"""
import argparse
import re
import statistics as st

KEYS = ("callback_total", "inference", "path_finding")
PATTERN = {k: re.compile(rf"{k}=([0-9.]+) ms") for k in KEYS}


def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            vals = {k: PATTERN[k].search(line) for k in KEYS}
            if all(vals.values()):
                rows.append({k: float(v.group(1)) for k, v in vals.items()})
    return rows


def fmt(values):
    if len(values) < 2:
        return f"{values[0]:.2f}" if values else "-"
    return f"{st.mean(values):.2f} +- {st.stdev(values):.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--skip", type=int, default=5, help="frame iniziali scartati")
    ap.add_argument("--n", type=int, default=100, help="frame usati dopo --skip")
    a = ap.parse_args()
    print(f"{'file':32s} {'frame':>5s}  {'callback_total':>16s}  {'inference':>14s}  {'path_finding':>14s}  {'resto':>14s}")
    for path in a.files:
        rows = read(path)[a.skip:a.skip + a.n]
        col = {k: [r[k] for r in rows] for k in KEYS}
        rest = [r["callback_total"] - r["inference"] - r["path_finding"] for r in rows]
        print(f"{path:32s} {len(rows):5d}  {fmt(col['callback_total']):>16s}  {fmt(col['inference']):>14s}  "
              f"{fmt(col['path_finding']):>14s}  {fmt(rest):>14s}")


if __name__ == "__main__":
    main()
