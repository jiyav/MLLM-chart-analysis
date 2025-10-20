#!/usr/bin/env python3
"""
Generate N random (x, y) datasets and three visualizations per dataset:
  - Line chart of (x, y) with a natural, typical title (e.g., "Average monthly rainfall").
  - Pie chart over fixed-width integer y-bins (either counts or sums).
  - Bar chart over the same fixed-width integer y-bins (either counts or sums).

Key features:
  * Bins use FIXED integer widths (e.g., 1, 2, 5, 10, 20), not percentiles.
  * Annotations are saved in JSON (one annotations.json per DOMAIN subfolder).
  * Outputs are grouped into per-domain subfolders:
        <out>/<domain>/{images,data,annotations.json}

Usage:
  python gen_fixed_bins_by_domain.py --out ./multi_graphs --n 60 --seed 7
"""

import argparse
import json
import math
import os
import random
import string
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


DOMAINS = [
    "healthcare", "politics", "environment", "technology", "entertainment",
    "animal", "linguistics", "internet", "miscellaneous"
]

REP_MODES = ["counts", "sums"]  # how pie/bar summarize bins

# Domain-specific, natural-sounding base titles for the line chart.
BASE_TITLES = {
    "healthcare": [
        "Average daily heart rate",
        "Hospital admissions per day",
        "Blood glucose readings over time",
        "Clinic wait times per hour",
        "Vaccinations administered per week",
    ],
    "politics": [
        "Voter registrations per week",
        "Campaign donations per day",
        "Debate mentions over time",
        "Approval rating over time",
        "Ballots processed per hour",
    ],
    "environment": [
        "Average monthly rainfall",
        "Air quality index over time",
        "River water level per day",
        "Daily CO₂ concentration",
        "Wildfire incidents per week",
    ],
    "technology": [
        "Website visits per hour",
        "API requests per minute",
        "CPU utilization over time",
        "Network latency over time",
        "Bugs reported per day",
    ],
    "entertainment": [
        "Tickets sold per day",
        "Streams per hour",
        "Box office revenue per day",
        "Podcast downloads per day",
        "Playlist additions per hour",
    ],
    "animal": [
        "Bird sightings per day",
        "Whale calls detected per hour",
        "Hotdogs sold per day",
        "Zoo attendance per day",
        "Butterfly counts per week",
    ],
    "linguistics": [
        "New words added per day",
        "Corpus tokens per hour",
        "Transcriptions completed per day",
        "Dictionary lookups per hour",
        "Grammar fixes per day",
    ],
    "internet": [
        "Tweets posted per minute",
        "Search queries per second",
        "New users per day",
        "Page views per hour",
        "DNS queries per second",
    ],
    "miscellaneous": [
        "Steps walked per day",
        "Coffee cups consumed per day",
        "Packages delivered per day",
        "Energy usage per hour",
        "Tasks completed per day",
    ],
}


@dataclass
class DatasetRecord:
    dataset_id: str
    domain: str
    title: str                    # base dataset title (line chart)
    line_title: str               # explicit line chart title
    pie_title: str                # explicit pie chart title
    bar_title: str                # explicit bar chart title
    n_points: int
    x_path: str
    y_path: str
    line_image: str
    pie_image: str
    bar_image: str
    bin_width: int
    bin_start: int
    bin_end: int
    bin_edges: List[int]             # integer edges
    bin_labels: List[str]            # string labels per bin
    mode: str                        # "counts" or "sums"
    per_bin_counts: List[int]
    per_bin_sums: List[float]
    mean: float
    min: float
    max: float
    y_range: float


# -------------------------- Helpers -----------------------------------------

def _rand_id(k: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


def _choose_domain_and_title() -> Tuple[str, str]:
    domain = random.choice(DOMAINS)
    title = random.choice(BASE_TITLES[domain])
    return domain, title


def _gen_xy(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a diverse (x, y) signal with trend + seasonality + noise and (sometimes) steps."""
    x = np.arange(n_points).astype(float)

    # Trend
    trend_slope = np.random.uniform(-0.05, 0.2)
    trend = trend_slope * x

    # Seasonality
    period = np.random.uniform(6, max(7, n_points / np.random.uniform(1.5, 5.0)))
    seasonal_amp = np.random.uniform(0.0, 3.0)
    seasonal = seasonal_amp * np.sin(2 * np.pi * x / period + np.random.uniform(0, 2*np.pi))

    # Noise
    noise = np.random.normal(0.0, np.random.uniform(0.2, 1.0), size=n_points)

    y = trend + seasonal + noise

    # Optional step
    if random.random() < 0.4:
        cut = random.randint(n_points//4, (3*n_points)//4)
        step = np.random.uniform(-3.0, 3.0)
        y[cut:] += step

    # Scale/shift
    y = y * np.random.uniform(0.8, 2.5) + np.random.uniform(-2.0, 2.0)

    # Optional smoothing
    if random.random() < 0.3:
        win = random.randint(3, 7)
        y = np.convolve(y, np.ones(win)/win, mode="same")

    return x, y


def _fixed_int_bins(y: np.ndarray, candidate_widths=(1, 2, 5, 10, 20)) -> Tuple[np.ndarray, int, int, int]:
    """Create fixed-width INTEGER bins spanning min..max of y.
    Returns (edges, bin_width, start, end) where edges are integers.
    """
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    width = int(random.choice(candidate_widths))

    start = math.floor(y_min / width) * width
    end = math.ceil(y_max / width) * width
    # ensure at least 3 bins
    while (end - start) / width < 3:
        end += width

    # integer edges
    edges = np.arange(start, end + width, width, dtype=int)
    return edges, width, start, end


def _bin_stats_fixed(y: np.ndarray, edges: np.ndarray) -> Tuple[List[int], List[float]]:
    """Counts and sums per fixed integer bin. Left-closed, right-open except last inclusive."""
    nb = len(edges) - 1
    counts = [0] * nb
    sums = [0.0] * nb
    # Map y -> bin index using np.digitize vs integer edges
    idx = np.digitize(y, bins=edges[1:-1], right=False)  # 0..nb-1
    for v, b in zip(y, idx):
        counts[b] += 1
        sums[b] += float(v)
    return counts, sums


def _labels_from_int_edges(edges: np.ndarray) -> List[str]:
    labels = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i+1]
        if i < len(edges) - 2:
            labels.append(f"[{a}, {b})")
        else:
            labels.append(f"[{a}, {b}]")
    return labels


def _stats(y: np.ndarray) -> Tuple[float, float, float, float]:
    vmin = float(np.min(y))
    vmax = float(np.max(y))
    mean = float(np.mean(y))
    return mean, vmin, vmax, float(vmax - vmin)


def _save_fig(fig: plt.Figure, out_path: str, dpi: int = 150) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_line(x: np.ndarray, y: np.ndarray, title: str, out_path: str, figsize: Tuple[float, float]) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_fig(fig, out_path)


def _plot_pie(labels: List[str], values: List[float], title: str, out_path: str, figsize: Tuple[float, float]) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    eps = 1e-12
    safe_vals = [v if v > 0 else eps for v in values]
    ax.pie(safe_vals, labels=labels, autopct=lambda p: f"{p:.0f}%")
    ax.set_title(title)
    _save_fig(fig, out_path)


def _plot_bar(labels: List[str], values: List[float], ylabel: str, title: str, out_path: str, figsize: Tuple[float, float]) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _save_fig(fig, out_path)


def _make_rep_titles(base_title: str, mode: str) -> Tuple[str, str, str]:
    line_title = base_title
    if mode == "counts":
        pie_title = f"Share of observations by fixed y-range — {base_title}"
        bar_title = f"Counts by fixed y-range — {base_title}"
    else:
        pie_title = f"Share of total y by fixed y-range — {base_title}"
        bar_title = f"Total y by fixed y-range — {base_title}"
    return line_title, pie_title, bar_title


# -------------------------- Main generation ----------------------------------

def generate_dataset(out_dir: str, n: int, width: int, height: int, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    # Prepare domain folders
    domain_dirs = {}
    for d in DOMAINS:
        d_root = os.path.join(out_dir, d)
        domain_dirs[d] = {
            "root": d_root,
            "images": os.path.join(d_root, "images"),
            "data": os.path.join(d_root, "data"),
            "ann_path": os.path.join(d_root, "annotations.json"),
            "records": [],  # will collect per-domain records
        }
        os.makedirs(domain_dirs[d]["images"], exist_ok=True)
        os.makedirs(domain_dirs[d]["data"], exist_ok=True)

    for _ in range(n):
        dataset_id = _rand_id()
        domain, base_title = _choose_domain_and_title()
        ddirs = domain_dirs[domain]

        # Generate (x, y)
        n_points = random.randint(20, 120)
        x, y = _gen_xy(n_points)

        # Persist raw arrays (paths are relative to <out>/<domain>)
        x_rel = os.path.join("data", f"{dataset_id}_x.npy")
        y_rel = os.path.join("data", f"{dataset_id}_y.npy")
        np.save(os.path.join(ddirs["root"], x_rel), x)
        np.save(os.path.join(ddirs["root"], y_rel), y)

        # Fixed-width integer bins & stats
        edges, width_int, start_int, end_int = _fixed_int_bins(y)
        labels = _labels_from_int_edges(edges)
        counts, sums = _bin_stats_fixed(y, edges)
        mode = random.choice(REP_MODES)
        bar_vals = counts if mode == "counts" else sums
        pie_vals = counts if mode == "counts" else sums

        # Titles
        line_title, pie_title, bar_title = _make_rep_titles(base_title, mode)

        # Figure size (inches) derived from pixel args
        figsize = (width / 100.0, height / 100.0)

        # Filepaths (relative to domain root)
        line_img_rel = os.path.join("images", f"{dataset_id}_line.png")
        pie_img_rel  = os.path.join("images", f"{dataset_id}_pie.png")
        bar_img_rel  = os.path.join("images", f"{dataset_id}_bar.png")

        # Plot & save
        _plot_line(x, y, line_title, os.path.join(ddirs["root"], line_img_rel), figsize)
        _plot_pie(labels, pie_vals, pie_title, os.path.join(ddirs["root"], pie_img_rel), figsize)
        _plot_bar(labels, bar_vals, ylabel=mode.capitalize(), title=bar_title, out_path=os.path.join(ddirs["root"], bar_img_rel), figsize=figsize)

        # Stats
        mean, y_min, y_max, y_rng = _stats(y)

        rec = DatasetRecord(
            dataset_id=dataset_id,
            domain=domain,
            title=base_title,
            line_title=line_title,
            pie_title=pie_title,
            bar_title=bar_title,
            n_points=n_points,
            x_path=x_rel,
            y_path=y_rel,
            line_image=line_img_rel,
            pie_image=pie_img_rel,
            bar_image=bar_img_rel,
            bin_width=int(width_int),
            bin_start=int(start_int),
            bin_end=int(end_int),
            bin_edges=[int(e) for e in edges.tolist()],
            bin_labels=labels,
            mode=mode,
            per_bin_counts=counts,
            per_bin_sums=[float(s) for s in sums],
            mean=float(mean),
            min=float(y_min),
            max=float(y_max),
            y_range=float(y_rng),
        )

        ddirs["records"].append(asdict(rec))

    # Write per-domain annotations.json
    for d, info in domain_dirs.items():
        with open(info["ann_path"], "w", encoding="utf-8") as f:
            json.dump(info["records"], f, ensure_ascii=False, indent=2)

    # Root README
    readme_path = os.path.join(out_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "Fixed-width integer bin chart dataset (per-domain)\n"
            "---------------------------------------------------\n"
            "Each generated dataset yields three charts (line, pie, bar) from the same (x, y).\n"
            "Pie/Bar use fixed-width INTEGER y-bins (e.g., width 1/2/5/10/20). Mode is either COUNTS or SUMS.\n"
            "Annotations are saved as JSON in <out>/<domain>/annotations.json.\n"
            "Raw arrays are stored under <out>/<domain>/data, and images under <out>/<domain>/images.\n"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate N random (x,y) datasets with fixed-width integer bins and per-domain JSON annotations.")
    p.add_argument("--out", type=str, default="./multi_graphs", help="Output directory (domains are subfolders)")
    p.add_argument("--n", type=int, default=60, help="Number of datasets to generate (spread across domains)")
    p.add_argument("--width", type=int, default=640, help="Image width in pixels (for figsize scaling)")
    p.add_argument("--height", type=int, default=480, help="Image height in pixels (for figsize scaling)")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(out_dir=args.out, n=args.n, width=args.width, height=args.height, seed=args.seed)


if __name__ == "__main__":
    main()
