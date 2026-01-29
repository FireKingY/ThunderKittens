#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict


def read_rows(path):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def to_int(value, default=0):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def build_series(rows):
    series = defaultdict(list)
    meta = {}
    for row in rows:
        op = row.get("op", "unknown")
        blocks = to_int(row.get("blocks", 0))
        gbps = to_float(row.get("gbps", 0.0))
        series[op].append((blocks, gbps))
        if op not in meta:
            meta[op] = {
                "peak_gbps": to_float(row.get("peak_gbps", 0.0)),
                "peak_blocks": to_int(row.get("peak_blocks", 0)),
                "saturation_ratio": to_float(row.get("saturation_ratio", 0.0)),
                "saturation_blocks": to_int(row.get("saturation_blocks", 0)),
            }
    for op in series:
        series[op].sort(key=lambda item: item[0])
    return series, meta


def main():
    parser = argparse.ArgumentParser(description="Plot TMA bandwidth sweep.")
    parser.add_argument("--csv", default="tma_bandwidth.csv", help="CSV file from tma_bandwidth test")
    parser.add_argument("--out", default="tma_bandwidth.png", help="Output image path")
    parser.add_argument("--title", default=None, help="Optional plot title override")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    series, meta = build_series(rows)

    device = rows[0].get("device", "").strip()
    title = args.title or f"TMA bandwidth sweep ({device})"

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for op, points in series.items():
        blocks = [b for b, _ in points]
        gbps = [g for _, g in points]
        ax.plot(blocks, gbps, marker="o", label=f"{op}")
        info = meta.get(op, {})
        sat_blocks = info.get("saturation_blocks", 0)
        if sat_blocks:
            ax.axvline(sat_blocks, linestyle="--", linewidth=1.0, alpha=0.4)

    ax.set_title(title)
    ax.set_xlabel("Active SMs (1 CTA per SM)")
    ax.set_ylabel("GB/s")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote plot: {args.out}")


if __name__ == "__main__":
    main()
