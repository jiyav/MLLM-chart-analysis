#!/usr/bin/env python3
"""

python gen_graph_dataset_with_annotations.py \
  --out ./dataset \
  --n 30 \
  --img_size 256 \
  --seed 42 \
  --layout flat

(if u wanna generate by layout)
python gen_graph_dataset_with_annotations.py \
  --out ./dataset_by_group \
  --n 30 \
  --img_size 256 \
  --seed 42 \
  --layout by_group

  
Generate N random (x, y) datasets and three visualizations per dataset:

  - Line chart of (x, y) with a natural, typical title (e.g., "Average monthly rainfall").
  - Pie chart over fixed-width integer y-bins (counts shown as percentages).
  - Bar chart over the same fixed-width integer y-bins (counts per bin).

Each visualization gets:
  - A creative title derived from one of the domains:
      [economics, healthcare, politics, environment, technology,
       entertainment, animal, linguistics, internet, agriculture,
       society, misc.]
  - A domain label (capitalized in annotations).

We also create a single annotations.jsonl in the ROOT directory with entries:
  {"image":"<relative_path>","prefix":"What is the maximum?","suffix":"<max>"}
  {"image":"<relative_path>","prefix":"What is the minimum?","suffix":"<min>"}
  {"image":"<relative_path>","prefix":"What is the range of the y-axis? Format as min-max (No spaces)","suffix":"<ymin>-<ymax>"}
  {"image":"<relative_path>","prefix":"What is the title?","suffix":"<title>"}"
  {"image":"<relative_path>","prefix":"What is the domain?","suffix":"<Domain>"}

Directory layout is configurable via --layout:
  * flat (default):
        /dataset_root
            image_<uuid>.png
            ...
            annotations.jsonl

  * by_group:
        /dataset_root
            /line
            /pie
            /bar
            annotations.jsonl
"""

import argparse
import json
import math
import os
import random
import string
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


DOMAINS = [
    "economics", "healthcare", "politics", "environment", "technology",
    "entertainment", "animal", "linguistics", "internet", "agriculture",
    "society", "misc."
]

BASE_TITLES: Dict[str, List[str]] = {
    "economics": [
        "Monthly unemployment rate",
        "Quarterly GDP growth",
        "Average hourly wage",
        "Consumer spending per month",
        "Inflation rate over time",
    ],
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
    "agriculture": [
        "Crop yield per hectare",
        "Rainfall on farmland per week",
        "Fertilizer usage per month",
        "Cattle weight gain per day",
        "Irrigation volume per day",
    ],
    "society": [
        "Steps walked per day",
        "Public transit rides per day",
        "School attendance per day",
        "Community events per month",
        "Volunteer hours per week",
    ],
    "misc.": [
        "Coffee cups consumed per day",
        "Packages delivered per day",
        "Energy usage per hour",
        "Tasks completed per day",
        "Random sensor readings over time",
    ],
}


@dataclass
class AnnotationItem:
    image: str
    prefix: str
    suffix: str



def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _gen_xy(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a diverse (x, y) signal with trend + seasonality + noise and (sometimes) steps."""
    x = np.arange(n_points).astype(float)

    trend_slope = np.random.uniform(-0.05, 0.2)
    trend = trend_slope * x

    period = np.random.uniform(6, max(7, n_points / np.random.uniform(1.5, 5.0)))
    seasonal_amp = np.random.uniform(0.0, 3.0)
    seasonal = seasonal_amp * np.sin(2 * np.pi * x / period + np.random.uniform(0, 2*np.pi))

    noise = np.random.normal(0.0, np.random.uniform(0.2, 1.0), size=n_points)

    y = trend + seasonal + noise

    if random.random() < 0.4:
        cut = random.randint(n_points//4, (3*n_points)//4)
        step = np.random.uniform(-3.0, 3.0)
        y[cut:] += step

    y = y * np.random.uniform(0.8, 2.5) + np.random.uniform(-2.0, 2.0)

    if random.random() < 0.3:
        win = random.randint(3, 7)
        y = np.convolve(y, np.ones(win)/win, mode="same")

    return x, y


def _choose_domain_and_title() -> Tuple[str, str]:
    domain = random.choice(DOMAINS)
    base_titles = BASE_TITLES[domain]
    title = random.choice(base_titles)
    return domain, title


def _fixed_int_bins(y: np.ndarray, candidate_widths=(1, 2, 5, 10, 20)) -> Tuple[np.ndarray, int, int, int]:
    """Create fixed-width INTEGER bins spanning min..max of y."""
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    width = int(random.choice(candidate_widths))

    start = math.floor(y_min / width) * width
    end = math.ceil(y_max / width) * width
    while (end - start) / width < 3:
        end += width

    edges = np.arange(start, end + width, width, dtype=int)
    return edges, width, start, end


def _bin_counts_and_sums(y: np.ndarray, edges: np.ndarray) -> Tuple[List[int], List[float]]:
    """Counts and sums per fixed integer bin. Left-closed, right-open except last inclusive."""
    nb = len(edges) - 1
    counts = [0] * nb
    sums = [0.0] * nb
    idx = np.digitize(y, bins=edges[1:-1], right=False)  # 0..nb-1
    for v, b in zip(y, idx):
        counts[b] += 1
        sums[b] += float(v)
    return counts, sums


def _labels_from_edges(edges: np.ndarray) -> List[str]:
    labels = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i+1]
        if i < len(edges) - 2:
            labels.append(f"[{a}, {b})")
        else:
            labels.append(f"[{a}, {b}]")
    return labels


def _stats(y: np.ndarray) -> Tuple[float, float, float]:
    vmin = float(np.min(y))
    vmax = float(np.max(y))
    return vmin, vmax, float(vmax - vmin)


def _make_figsize(px_width: int, px_height: int, dpi: int = 100) -> Tuple[float, float, int]:
    return px_width / dpi, px_height / dpi, dpi


def _save_fig(fig: plt.Figure, out_path: str, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_line(x: np.ndarray, y: np.ndarray, title: str, ylim: Tuple[float, float], out_path: str, figsize: Tuple[float, float], dpi: int) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o", markersize=2, linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save_fig(fig, out_path, dpi)


def _plot_pie(labels: List[str], counts: List[int], title: str, out_path: str, figsize: Tuple[float, float], dpi: int) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    eps = 1e-12
    safe_vals = [c if c > 0 else eps for c in counts]
    ax.pie(safe_vals, labels=labels, autopct=lambda p: f"{p:.0f}%")
    ax.set_title(title)
    _save_fig(fig, out_path, dpi)


def _plot_bar(labels: List[str], counts: List[int], title: str, out_path: str, figsize: Tuple[float, float], dpi: int) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    _save_fig(fig, out_path, dpi)


def _domain_label_for_annotations(domain: str) -> str:
    base = domain.rstrip(".")
    return base[:1].upper() + base[1:]


# -------------------------- Main generation ----------------------------------

def generate_dataset(
    out_dir: str,
    n_datasets: int,
    img_size: int,
    seed: int,
    layout: str,
) -> None:
    """
    layout: 'flat' or 'by_group'
    """
    assert layout in ("flat", "by_group"), "layout must be 'flat' or 'by_group'"

    _seed_everything(seed)
    os.makedirs(out_dir, exist_ok=True)

    width_px = height_px = img_size
    fig_w_in, fig_h_in, dpi = _make_figsize(width_px, height_px, dpi=100)

    if layout == "flat":
        line_dir = pie_dir = bar_dir = out_dir
    else:
        line_dir = os.path.join(out_dir, "line")
        pie_dir = os.path.join(out_dir, "pie")
        bar_dir = os.path.join(out_dir, "bar")
        os.makedirs(line_dir, exist_ok=True)
        os.makedirs(pie_dir, exist_ok=True)
        os.makedirs(bar_dir, exist_ok=True)

    annotations: List[AnnotationItem] = []

    for _ in range(n_datasets):
        domain, base_title = _choose_domain_and_title()
        domain_label = _domain_label_for_annotations(domain)

        n_points = random.randint(20, 120)
        x, y = _gen_xy(n_points)

        edges, bin_width, bin_start, bin_end = _fixed_int_bins(y)
        labels = _labels_from_edges(edges)
        counts, sums = _bin_counts_and_sums(y, edges)

        y_axis_min = float(bin_start)
        y_axis_max = float(bin_end)

        line_title = base_title
        pie_title = f"Share of observations by fixed y-range — {base_title}"
        bar_title = f"Counts by fixed y-range — {base_title}"

        def new_image_name() -> str:
            return f"image_{uuid.uuid4().hex}.png"

        line_name = new_image_name()
        line_path = os.path.join(line_dir, line_name)
        line_rel = os.path.relpath(line_path, out_dir)
        _plot_line(
            x, y, line_title, (y_axis_min, y_axis_max),
            out_path=line_path,
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
        )

        vmin, vmax, vrange = _stats(y)
        y_axis_range_str = f"{int(y_axis_min)}-{int(y_axis_max)}"

        annotations.extend([
            AnnotationItem(line_rel, "What is the maximum?", f"{vmax:.6f}".rstrip("0").rstrip(".")),
            AnnotationItem(line_rel, "What is the minimum?", f"{vmin:.6f}".rstrip("0").rstrip(".")),
            AnnotationItem(line_rel, "What is the range of the y-axis? Format as min-max (No spaces)", y_axis_range_str),
            AnnotationItem(line_rel, "What is the title?", line_title),
            AnnotationItem(line_rel, "What is the domain?", domain_label),
        ])

        # ----- PIE CHART -----
        pie_name = new_image_name()
        pie_path = os.path.join(pie_dir, pie_name)
        pie_rel = os.path.relpath(pie_path, out_dir)
        _plot_pie(
            labels, counts, pie_title,
            out_path=pie_path,
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
        )

        # For bar/pie, we can reuse the same y stats and axis range
        annotations.extend([
            AnnotationItem(pie_rel, "What is the maximum?", f"{vmax:.6f}".rstrip("0").rstrip(".")),
            AnnotationItem(pie_rel, "What is the minimum?", f"{vmin:.6f}".rstrip("0").rstrip(".")),
            AnnotationItem(pie_rel, "What is the range of the y-axis? Format as min-max (No spaces)", y_axis_range_str),
            AnnotationItem(pie_rel, "What is the title?", pie_title),
            AnnotationItem(pie_rel, "What is the domain?", domain_label),
        ])

        # ----- BAR CHART -----
        bar_name = new_image_name()
        bar_path = os.path.join(bar_dir, bar_name)
        bar_rel = os.path.relpath(bar_path, out_dir)
        _plot_bar(
            labels, counts, bar_title,
            out_path=bar_path,
            figsize=(fig_w_in, fig_h_in),
            dpi=dpi,
        )

        annotations.extend([
            AnnotationItem(bar_rel, "What is the maximum?", f"{vmax:.6f}".rstrip("0").rstrip(".")),
            AnnotationItem(bar_rel, "What is the minimum?", f"{vmin:.6f}".rstrip("0").rstrip(".")),
            AnnotationItem(bar_rel, "What is the range of the y-axis? Format as min-max (No spaces)", y_axis_range_str),
            AnnotationItem(bar_rel, "What is the title?", bar_title),
            AnnotationItem(bar_rel, "What is the domain?", domain_label),
        ])

    # Write annotations.jsonl in root
    ann_path = os.path.join(out_dir, "annotations.jsonl")
    with open(ann_path, "w", encoding="utf-8") as f:
        for item in annotations:
            obj = {
                "image": item.image.replace("\\", "/"),
                "prefix": item.prefix,
                "suffix": item.suffix,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate random (x,y) datasets with line/pie/bar charts and JSONL annotations.")
    p.add_argument("--out", type=str, default="./dataset", help="Output root directory")
    p.add_argument("--n", type=int, default=30, help="Number of datasets to generate")
    p.add_argument("--img_size", type=int, default=256, help="Image size in pixels (width=height)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--layout", type=str, default="flat", choices=["flat", "by_group"],
                   help="Directory layout: 'flat' (all images in root) or 'by_group' (line/pie/bar subfolders)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        out_dir=args.out,
        n_datasets=args.n,
        img_size=args.img_size,
        seed=args.seed,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
