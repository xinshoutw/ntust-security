import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

# Use numeric-only quasi-identifiers for clean Mondrian partitioning
QUASI_IDENTIFIERS = [
    "age",
    "education-num",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
]

# All QIs are numeric in this configuration
NUMERICAL_QI = QUASI_IDENTIFIERS[:]
CATEGORICAL_QI = []

# Features used for ML (includes non-QI columns for richer signal)
FEATURE_COLUMNS = QUASI_IDENTIFIERS + [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_adult_dataset(data_dir="data"):
    """Load the Adult dataset from UCI repository or local cache."""
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, "adult.data")

    if os.path.exists(local_path):
        df = pd.read_csv(local_path, header=None, names=DATA_COLUMNS, skipinitialspace=True)
    else:
        df = pd.read_csv(DATA_URL, header=None, names=DATA_COLUMNS, skipinitialspace=True)
        df.to_csv(local_path, index=False, header=False)

    # Remove rows with missing values
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)

    # Drop fnlwgt (sampling weight, not useful for classification)
    df = df.drop(columns=["fnlwgt"])

    # Binary encode income
    df["income"] = (df["income"].str.strip().str.rstrip(".") == ">50K").astype(int)

    return df


def prepare_ml_data(df, test_size=0.2, random_state=42, is_anonymized=False):
    """Prepare features and labels for ML, returning encoded and scaled data."""
    features = df[FEATURE_COLUMNS].copy()
    labels = df["income"].values

    if is_anonymized:
        # Anonymized QI columns are already floats (partition means)
        for col in NUMERICAL_QI:
            if col in features.columns:
                features[col] = features[col].astype(float)

    # Label-encode categorical columns
    cat_cols = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    for col in cat_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))

    features = features.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        features.values, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred, y_prob=None):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    metrics = {
        "Accuracy": acc,
        "Misclassification Rate": 1.0 - acc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["AUC"] = float("nan")
    return metrics
