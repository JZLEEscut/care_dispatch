"""Experiment summary and optional statistical tests."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

SUMMARY_METRICS = [
    "total_cost", "episode_reward", "distance_cost", "tardiness_cost",
    "reject_rate", "emergency_reject_rate", "mean_emergency_response_time",
    "preemption_count", "illegal_action_count", "illegal_action_rate",
    "mean_decision_time", "completion_rate",
]


def summarize_results(raw_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, grp in raw_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        for metric in SUMMARY_METRICS:
            if metric not in grp.columns:
                continue
            row[f"{metric}_mean"] = float(grp[metric].mean())
            row[f"{metric}_std"] = float(grp[metric].std(ddof=1)) if len(grp) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_wilcoxon(
    raw_df: pd.DataFrame,
    target_method: str,
    baseline_methods: list[str],
    metric: str,
    paired_cols: list[str] = ["seed", "episode"],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        from scipy.stats import wilcoxon
    except Exception:
        for b in baseline_methods:
            rows.append({"target_method": target_method, "baseline_method": b, "metric": metric, "p_value": math.nan, "statistic": math.nan, "note": "scipy_unavailable"})
        return pd.DataFrame(rows)

    target = raw_df[raw_df["method"] == target_method][paired_cols + [metric]].rename(columns={metric: "target_value"})
    for b in baseline_methods:
        base = raw_df[raw_df["method"] == b][paired_cols + [metric]].rename(columns={metric: "baseline_value"})
        merged = target.merge(base, on=paired_cols, how="inner")
        if len(merged) < 2:
            rows.append({"target_method": target_method, "baseline_method": b, "metric": metric, "p_value": math.nan, "statistic": math.nan, "note": "insufficient_pairs"})
            continue
        diff = merged["target_value"] - merged["baseline_value"]
        try:
            stat, p = wilcoxon(diff)
            rows.append({"target_method": target_method, "baseline_method": b, "metric": metric, "p_value": float(p), "statistic": float(stat), "note": "ok"})
        except Exception as exc:
            rows.append({"target_method": target_method, "baseline_method": b, "metric": metric, "p_value": math.nan, "statistic": math.nan, "note": str(exc)})
    return pd.DataFrame(rows)
