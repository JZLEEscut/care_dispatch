"""Matplotlib plotting helpers used by all new experiments."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.common.utils import ensure_dir


def _clean_labels(vals):
    return [str(v) for v in vals]


def plot_bar_with_error(summary_df: pd.DataFrame, x: str, y_mean: str, y_std: str, title: str, ylabel: str, output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)
    df = summary_df.copy()
    labels = _clean_labels(df[x].tolist())
    means = df[y_mean].astype(float).tolist()
    stds = df[y_std].astype(float).tolist() if y_std in df.columns else [0.0] * len(df)
    plt.figure(figsize=(max(7, len(labels) * 1.2), 4))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_grouped_bar(summary_df: pd.DataFrame, x: str, hue: str, y_mean: str, y_std: str, title: str, ylabel: str, output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)
    xs = list(summary_df[x].drop_duplicates())
    hues = list(summary_df[hue].drop_duplicates())
    width = 0.8 / max(1, len(hues))
    positions = list(range(len(xs)))
    plt.figure(figsize=(max(8, len(xs) * 1.5), 4))
    for hi, hv in enumerate(hues):
        sub = summary_df[summary_df[hue] == hv].set_index(x)
        means = [float(sub.loc[xv, y_mean]) if xv in sub.index else 0.0 for xv in xs]
        stds = [float(sub.loc[xv, y_std]) if xv in sub.index and y_std in sub.columns else 0.0 for xv in xs]
        offs = [p + (hi - (len(hues)-1)/2) * width for p in positions]
        plt.bar(offs, means, width=width, yerr=stds, capsize=3, label=str(hv))
    plt.xticks(positions, _clean_labels(xs), rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out, s = [], 0.0
    for i, v in enumerate(values):
        s += v
        if i >= window:
            s -= values[i-window]
            out.append(s / window)
        else:
            out.append(s / (i+1))
    return out


def plot_training_curve(log_paths: dict[str, str | Path], metric: str, output_path: str | Path, rolling_window: int = 100) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)
    plt.figure(figsize=(10, 4))
    for label, p in log_paths.items():
        p = Path(p)
        if not p.exists():
            continue
        xs, ys = [], []
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    xs.append(int(float(row.get("episode_index", i))))
                    ys.append(float(row.get(metric, 0.0)))
                except Exception:
                    pass
        if xs:
            plt.plot(xs, _rolling_mean(ys, rolling_window), label=label)
    plt.xlabel("episode")
    plt.ylabel(metric)
    plt.title(f"{metric} curve | rolling={rolling_window}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
