import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    load_adult_dataset,
    prepare_ml_data,
    evaluate_model,
    QUASI_IDENTIFIERS,
)
from mondrian import mondrian_k_anonymity
from ml_models import train_svm, train_neural_network

K_VALUES = [2, 4, 10, 20, 50, 100]
RESULTS_DIR = "results"


def run_experiment_on_data(df, label, is_anonymized=False):
    """Train both models on the given dataframe and return metrics."""
    print(f"  Training on: {label}")
    X_train, X_test, y_train, y_test = prepare_ml_data(df, is_anonymized=is_anonymized)

    # SVM
    print(f"    SVM...", end=" ", flush=True)
    svm_pred, svm_prob = train_svm(X_train, y_train, X_test)
    svm_metrics = evaluate_model(y_test, svm_pred, svm_prob)
    print(f"Accuracy={svm_metrics['Accuracy']:.4f}")

    # Neural Network
    print(f"    Neural Network...", end=" ", flush=True)
    nn_pred, nn_prob = train_neural_network(X_train, y_train, X_test)
    nn_metrics = evaluate_model(y_test, nn_pred, nn_prob)
    print(f"Accuracy={nn_metrics['Accuracy']:.4f}")

    return {"SVM": svm_metrics, "NN": nn_metrics}


def plot_results(all_results, original_results):
    """Generate comparison charts."""
    metrics_names = ["Accuracy", "Misclassification Rate", "Precision", "Recall", "AUC"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]

        for model_name in ["SVM", "NN"]:
            values = [all_results[k][model_name][metric] for k in K_VALUES]
            ax.plot(K_VALUES, values, marker="o", label=f"{model_name} (K-anon)")

            # Baseline
            baseline = original_results[model_name][metric]
            ax.axhline(y=baseline, linestyle="--", alpha=0.5,
                        label=f"{model_name} (Original)")

        ax.set_xlabel("K value")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([str(k) for k in K_VALUES])

    # Hide unused subplot
    axes[-1].set_visible(False)

    plt.suptitle("Impact of K-anonymity on ML Model Performance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_chart.png"), dpi=150)
    plt.close()
    print(f"\nChart saved to {RESULTS_DIR}/comparison_chart.png")


def save_metrics_table(all_results, original_results):
    """Save all metrics to a CSV file."""
    rows = []
    for model_name in ["SVM", "NN"]:
        row = {"Model": model_name, "K": "Original"}
        row.update(original_results[model_name])
        rows.append(row)

    for k in K_VALUES:
        for model_name in ["SVM", "NN"]:
            row = {"Model": model_name, "K": k}
            row.update(all_results[k][model_name])
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "metrics_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    print("\n" + df.to_string(index=False))


def generate_report(all_results, original_results):
    """Generate a simple experiment report."""
    lines = [
        "# K-anonymity Experiment Report\n",
        "## Method",
        "- **Algorithm**: Mondrian Multidimensional K-anonymity",
        "- **Dataset**: UCI Adult Dataset",
        f"- **Quasi-identifiers**: {', '.join(QUASI_IDENTIFIERS)}",
        "- **Sensitive attribute**: income (>50K / <=50K)\n",
        "## ML Models",
        "### SVM",
        "- Kernel: RBF",
        "- Library: scikit-learn SVC with default hyperparameters",
        "- probability=True for AUC computation\n",
        "### Neural Network",
        "- Architecture: 3-layer MLP (Input -> 64 -> 32 -> 1)",
        "- Activation: ReLU (hidden), Sigmoid (output)",
        "- Loss: BCELoss",
        "- Optimizer: Adam (lr=0.001)",
        "- Epochs: 50, Batch size: 64",
        "- Framework: PyTorch\n",
        f"## K Values Tested: {K_VALUES}\n",
        "## Results\n",
        "| K | Model | Accuracy | Misclass. Rate | Precision | Recall | AUC |",
        "|---|-------|----------|---------------|-----------|--------|-----|",
    ]

    for model_name in ["SVM", "NN"]:
        m = original_results[model_name]
        lines.append(
            f"| Original | {model_name} | {m['Accuracy']:.4f} | "
            f"{m['Misclassification Rate']:.4f} | {m['Precision']:.4f} | "
            f"{m['Recall']:.4f} | {m['AUC']:.4f} |"
        )

    for k in K_VALUES:
        for model_name in ["SVM", "NN"]:
            m = all_results[k][model_name]
            lines.append(
                f"| {k} | {model_name} | {m['Accuracy']:.4f} | "
                f"{m['Misclassification Rate']:.4f} | {m['Precision']:.4f} | "
                f"{m['Recall']:.4f} | {m['AUC']:.4f} |"
            )

    lines.extend([
        "\n## Analysis",
        "As K increases, the level of privacy protection grows but information loss also increases.",
        "This is reflected in the general trend of declining ML model performance (lower accuracy,",
        "precision, recall, and AUC) with higher K values. The trade-off between privacy and utility",
        "is a fundamental challenge in data anonymization.",
    ])

    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("K-anonymity Experiment")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading Adult dataset...")
    df = load_adult_dataset()
    print(f"  Dataset size: {len(df)} records, {len(df.columns)} columns")

    # Run on original data
    print("\n[2/4] Training on original data...")
    original_results = run_experiment_on_data(df, "Original Data")

    # Run on anonymized data for each K
    print("\n[3/4] Running K-anonymity experiments...")
    all_results = {}
    for k in K_VALUES:
        print(f"\n--- K = {k} ---")
        anon_df = mondrian_k_anonymity(df, k, QUASI_IDENTIFIERS)
        all_results[k] = run_experiment_on_data(anon_df, f"K={k} anonymized", is_anonymized=True)

    # Generate outputs
    print("\n[4/4] Generating results...")
    save_metrics_table(all_results, original_results)
    plot_results(all_results, original_results)
    generate_report(all_results, original_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
