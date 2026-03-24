import pandas as pd
import numpy as np
from utils import NUMERICAL_QI, CATEGORICAL_QI


def _normalize_range(series):
    """Compute normalized range for a column (0-1 scale)."""
    if not pd.api.types.is_numeric_dtype(series):
        return len(series.unique()) / max(len(series), 1)
    else:
        col_min, col_max = series.min(), series.max()
        if col_max == col_min:
            return 0.0
        return (col_max - col_min) / col_max


def _choose_dimension(df, qi_columns):
    """Choose the QI dimension with the largest normalized range."""
    best_dim = None
    best_range = -1.0
    for col in qi_columns:
        r = _normalize_range(df[col])
        if r > best_range:
            best_range = r
            best_dim = col
    return best_dim


def _split_numerical(df, col):
    """Split a numerical column at its median."""
    median = df[col].median()
    lhs = df[df[col] <= median]
    rhs = df[df[col] > median]
    return lhs, rhs


def _split_categorical(df, col):
    """Split a categorical column into two roughly equal halves by frequency."""
    freq = df[col].value_counts()
    values = freq.index.tolist()

    # Sort by frequency and split into two groups
    half = len(df) // 2
    cumsum = 0
    split_idx = 0
    for i, v in enumerate(values):
        cumsum += freq[v]
        if cumsum >= half:
            split_idx = i + 1
            break

    if split_idx == 0:
        split_idx = 1
    if split_idx >= len(values):
        split_idx = len(values) - 1

    left_vals = set(values[:split_idx])
    lhs = df[df[col].isin(left_vals)]
    rhs = df[~df[col].isin(left_vals)]
    return lhs, rhs


def _mondrian_partition(df, qi_columns, k):
    """Recursively partition the data using Mondrian algorithm."""
    if len(df) < 2 * k:
        return [df]

    dim = _choose_dimension(df, qi_columns)
    if dim is None:
        return [df]

    if dim in NUMERICAL_QI:
        lhs, rhs = _split_numerical(df, dim)
    else:
        lhs, rhs = _split_categorical(df, dim)

    # If split fails to produce valid partitions, return current partition
    if len(lhs) < k or len(rhs) < k:
        return [df]

    return _mondrian_partition(lhs, qi_columns, k) + _mondrian_partition(rhs, qi_columns, k)


def _generalize_partition(partition, qi_columns):
    """Generalize QI values within a partition."""
    result = partition.copy()
    for col in qi_columns:
        if col in NUMERICAL_QI:
            col_min = partition[col].min()
            col_max = partition[col].max()
            if col_min == col_max:
                result[col] = str(col_min)
            else:
                result[col] = f"{col_min}-{col_max}"
        else:
            unique_vals = sorted(partition[col].unique())
            if len(unique_vals) == 1:
                result[col] = unique_vals[0]
            else:
                result[col] = ",".join(unique_vals)
    return result


def mondrian_k_anonymity(df, k, qi_columns):
    """Apply Mondrian multidimensional K-anonymity.

    Args:
        df: Input DataFrame.
        k: Privacy parameter (minimum group size).
        qi_columns: List of quasi-identifier column names.

    Returns:
        Anonymized DataFrame with generalized QI values.
    """
    partitions = _mondrian_partition(df, qi_columns, k)

    anonymized_parts = []
    for part in partitions:
        anonymized_parts.append(_generalize_partition(part, qi_columns))

    result = pd.concat(anonymized_parts, ignore_index=True)
    return result
