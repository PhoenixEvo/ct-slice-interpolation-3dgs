"""
Paired statistical tests and ablation aggregation helpers.

Used by `06_visualization.ipynb` / `05_benchmark_ablation.ipynb` and by any
offline analysis script that wants a publication-ready comparison table
(means ± std, paired t-test, Wilcoxon, Cohen's d) between a reference
method and a set of competitor methods on per-case results.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def paired_comparison(
    ref_values: Sequence[float],
    other_values: Sequence[float],
    ref_name: str = "ref",
    other_name: str = "other",
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """Paired t-test + Wilcoxon + Cohen's d between two matched score series.

    Both series must have the same length and ordering (e.g. per case_idx).
    NaN rows are dropped pairwise. Returns a dict ready to append as a row.

    Args:
        ref_values: Scores for the reference method.
        other_values: Scores for the competitor method.
        ref_name: Label for the reference.
        other_name: Label for the competitor.
        higher_is_better: If True, a positive (ref - other) means ref wins.

    Returns:
        Dict with means, deltas, p-values, and effect size.
    """
    a = _as_array(ref_values)
    b = _as_array(other_values)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    out: Dict[str, float] = {
        "ref": ref_name,
        "other": other_name,
        "n": int(a.size),
        f"mean_{ref_name}": float(np.mean(a)) if a.size else float("nan"),
        f"mean_{other_name}": float(np.mean(b)) if b.size else float("nan"),
        f"std_{ref_name}": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        f"std_{other_name}": float(np.std(b, ddof=1)) if b.size > 1 else 0.0,
    }

    if a.size < 2:
        out.update({
            "delta_mean": float("nan"),
            "t_stat": float("nan"),
            "p_ttest": float("nan"),
            "w_stat": float("nan"),
            "p_wilcoxon": float("nan"),
            "cohen_d": float("nan"),
            "ref_wins_frac": float("nan"),
        })
        return out

    diff = a - b
    out["delta_mean"] = float(np.mean(diff))

    # Paired t-test (scipy required).
    try:
        from scipy import stats
        t_res = stats.ttest_rel(a, b)
        out["t_stat"] = float(t_res.statistic)
        out["p_ttest"] = float(t_res.pvalue)
        if np.any(diff != 0):
            w_res = stats.wilcoxon(a, b, zero_method="wilcox", correction=False)
            out["w_stat"] = float(w_res.statistic)
            out["p_wilcoxon"] = float(w_res.pvalue)
        else:
            out["w_stat"] = float("nan")
            out["p_wilcoxon"] = float("nan")
    except Exception:
        out["t_stat"] = float("nan")
        out["p_ttest"] = float("nan")
        out["w_stat"] = float("nan")
        out["p_wilcoxon"] = float("nan")

    # Cohen's d for paired samples: mean(diff) / std(diff).
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    out["cohen_d"] = float(out["delta_mean"] / sd) if sd > 0 else 0.0

    if higher_is_better:
        wins = (diff > 0).sum()
    else:
        wins = (diff < 0).sum()
    out["ref_wins_frac"] = float(wins) / max(1, diff.size)
    return out


def build_comparison_table(
    per_case_results: Dict[str, pd.DataFrame],
    ref_method: str,
    metric: str = "psnr",
    case_key: str = "case_idx",
    ratio_key: Optional[str] = "sparse_ratio",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Build a comparison table across methods for a given metric.

    Args:
        per_case_results: Mapping method_name -> DataFrame with at least
            [case_key, metric] (and optionally `ratio_key` to keep per-R rows).
        ref_method: Key in per_case_results used as the paired reference.
        metric: Column name to compare (e.g. 'psnr', 'ssim', 'lpips').
        case_key: Column that identifies a case (used as the pairing key).
        ratio_key: Optional column to group results per sparse ratio.
        higher_is_better: Direction of the metric.

    Returns:
        DataFrame with one row per (ratio, other_method).
    """
    if ref_method not in per_case_results:
        raise KeyError(f"ref_method '{ref_method}' not in per_case_results")

    ref_df = per_case_results[ref_method].copy()
    if ratio_key and ratio_key in ref_df.columns:
        ratio_values: List = sorted(ref_df[ratio_key].dropna().unique().tolist())
    else:
        ratio_values = [None]

    rows: List[Dict[str, float]] = []
    for ratio in ratio_values:
        if ratio is None:
            ref_sub = ref_df
        else:
            ref_sub = ref_df[ref_df[ratio_key] == ratio]
        ref_sub = ref_sub[[case_key, metric]].dropna()

        for other_name, other_df in per_case_results.items():
            if other_name == ref_method:
                continue
            other_sub = other_df
            if ratio is not None and ratio_key in other_sub.columns:
                other_sub = other_sub[other_sub[ratio_key] == ratio]
            other_sub = other_sub[[case_key, metric]].dropna()
            merged = ref_sub.merge(
                other_sub, on=case_key, how="inner",
                suffixes=("_ref", "_other"),
            )
            if merged.empty:
                continue
            row = paired_comparison(
                merged[f"{metric}_ref"].values,
                merged[f"{metric}_other"].values,
                ref_name=ref_method,
                other_name=other_name,
                higher_is_better=higher_is_better,
            )
            row["sparse_ratio"] = ratio
            row["metric"] = metric
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_ablation(
    ablation_df: pd.DataFrame,
    full_variant: str = "full",
    variant_col: str = "variant",
    case_col: str = "case_idx",
    ratio_col: str = "sparse_ratio",
    metrics: Sequence[str] = ("psnr", "ssim"),
) -> pd.DataFrame:
    """Produce paired comparisons for every ablation variant vs `full_variant`.

    Expects `ablation_df` to be a long table with columns
    `[variant, case_idx, sparse_ratio, psnr, ssim, ...]`.

    Returns a wide DataFrame (one row per (variant, ratio)) with paired
    stats against `full_variant` for each metric.
    """
    required = {variant_col, case_col}
    missing = required - set(ablation_df.columns)
    if missing:
        raise KeyError(f"Ablation df missing required columns: {missing}")

    if full_variant not in ablation_df[variant_col].unique():
        raise ValueError(
            f"full_variant='{full_variant}' not found in column '{variant_col}'"
        )

    variants = [v for v in ablation_df[variant_col].unique() if v != full_variant]
    if ratio_col in ablation_df.columns:
        ratios = sorted(ablation_df[ratio_col].dropna().unique().tolist())
    else:
        ratios = [None]

    rows: List[Dict[str, float]] = []
    for variant in variants:
        for ratio in ratios:
            row: Dict[str, float] = {"variant": variant, "ratio": ratio}
            sub_full = ablation_df[ablation_df[variant_col] == full_variant]
            sub_var = ablation_df[ablation_df[variant_col] == variant]
            if ratio is not None:
                sub_full = sub_full[sub_full[ratio_col] == ratio]
                sub_var = sub_var[sub_var[ratio_col] == ratio]
            for metric in metrics:
                if metric not in ablation_df.columns:
                    continue
                merged = sub_full[[case_col, metric]].merge(
                    sub_var[[case_col, metric]],
                    on=case_col, how="inner",
                    suffixes=("_full", "_ablat"),
                )
                merged = merged.dropna()
                if merged.empty:
                    row[f"delta_{metric}"] = float("nan")
                    row[f"p_{metric}"] = float("nan")
                    continue
                stats = paired_comparison(
                    merged[f"{metric}_full"].values,
                    merged[f"{metric}_ablat"].values,
                    ref_name="full",
                    other_name=variant,
                    higher_is_better=(metric not in ("mae", "lpips", "hfen", "gmsd")),
                )
                row[f"mean_full_{metric}"] = stats["mean_full"]
                row[f"mean_ablat_{metric}"] = stats.get(f"mean_{variant}", float("nan"))
                row[f"delta_{metric}"] = stats["delta_mean"]
                row[f"p_{metric}"] = stats.get("p_ttest", float("nan"))
                row[f"d_{metric}"] = stats["cohen_d"]
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "paired_comparison",
    "build_comparison_table",
    "summarize_ablation",
]
