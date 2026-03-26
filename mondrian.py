"""
Mondrian Multidimensional K-Anonymity
=====================================
Based on: K. LeFevre, D. J. DeWitt, R. Ramakrishnan,
"Mondrian Multidimensional K-Anonymity," ICDE 2006.

Implements greedy recursive partitioning on numeric quasi-identifiers.
Generalization replaces QI values with the equivalence class mean.
"""

import pandas as pd
import numpy as np
from utils import NUMERICAL_QI


def _get_span(partition, dim):
    """Compute the normalized span (range / global max) for a dimension."""
    col = partition[dim]
    col_min, col_max = col.min(), col.max()
    if col_max == col_min:
        return 0.0
    return (col_max - col_min) / max(abs(col_max), 1)


def _choose_dimension(partition, qi_columns):
    """Choose the QI dimension with the largest normalized span."""
    best_dim = None
    best_span = -1.0
    for col in qi_columns:
        span = _get_span(partition, col)
        if span > best_span:
            best_span = span
            best_dim = col
    return best_dim


def _split_numerical(df, col):
    """Split a numerical column at its median."""
    median = df[col].median()
    lhs = df[df[col] <= median]
    rhs = df[df[col] > median]
    return lhs, rhs


def _mondrian_partition(df, qi_columns, k):
    """Recursively partition the data using Mondrian algorithm."""
    if len(df) < 2 * k:
        return [df]

    # Try dimensions in order of span, largest first
    tried = set()
    while len(tried) < len(qi_columns):
        remaining = [d for d in qi_columns if d not in tried]
        dim = _choose_dimension(df, remaining)
        if dim is None:
            break
        tried.add(dim)

        lhs, rhs = _split_numerical(df, dim)

        # Check if split produces valid partitions
        if len(lhs) >= k and len(rhs) >= k:
            return _mondrian_partition(lhs, qi_columns, k) + _mondrian_partition(rhs, qi_columns, k)

    # No valid split found — this is a leaf partition
    return [df]


def _generalize_partition(partition, qi_columns):
    """Generalize QI values within a partition by replacing with the mean."""
    result = partition.copy()
    for col in qi_columns:
        mean_val = partition[col].mean()
        result[col] = mean_val
    return result


def compute_information_loss(partitions, total_records, k):
    """
    Compute information loss metrics from the Mondrian paper.

    Returns:
        dict with:
        - C_DM: Discernability metric (sum of |E|^2 for each equivalence class E)
        - C_AVG: Normalized average equivalence class size
        - num_classes: Number of equivalence classes
        - min_size: Smallest equivalence class
        - max_size: Largest equivalence class
        - avg_size: Average equivalence class size
    """
    sizes = [len(p) for p in partitions]
    c_dm = sum(s ** 2 for s in sizes)
    num_classes = len(sizes)
    avg_size = total_records / num_classes if num_classes > 0 else 0
    c_avg = avg_size / k if k > 0 else 0

    return {
        "C_DM": c_dm,
        "C_AVG": c_avg,
        "num_classes": num_classes,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": avg_size,
    }


def mondrian_k_anonymity(df, k, qi_columns):
    """
    Apply Mondrian multidimensional K-anonymity.

    Args:
        df: Input DataFrame.
        k: Privacy parameter (minimum group size).
        qi_columns: List of quasi-identifier column names.

    Returns:
        tuple: (anonymized_df, partitions, info_loss_metrics)
    """
    # Preserve original index for correct assignment
    df_work = df.copy()
    df_work["_orig_idx"] = df_work.index

    partitions = _mondrian_partition(df_work, qi_columns, k)

    # Validate: every partition has >= k records
    for i, part in enumerate(partitions):
        assert len(part) >= k, f"Partition {i} has {len(part)} records, expected >= {k}"

    # Compute information loss before generalization
    info_loss = compute_information_loss(partitions, len(df), k)

    # Generalize: replace QI values with partition mean
    anon_df = df.copy()
    for qi in qi_columns:
        anon_df[qi] = anon_df[qi].astype(float)

    for part in partitions:
        indices = part["_orig_idx"].values
        for qi in qi_columns:
            mean_val = part[qi].mean()
            anon_df.loc[indices, qi] = mean_val

    return anon_df, partitions, info_loss
