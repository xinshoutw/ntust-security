# K-Anonymity Experiment

Implementation of the **Mondrian Multidimensional K-Anonymity** algorithm with ML impact analysis.

## Reference

K. LeFevre, D. J. DeWitt, R. Ramakrishnan, "Mondrian Multidimensional K-Anonymity," ICDE, Vol. 6, 2006.

## Setup

```bash
# Using uv (recommended)
uv sync
uv run python main.py

# Or pip
pip install pandas numpy scikit-learn torch matplotlib
python main.py
```

## What It Does

1. Loads the **UCI Adult** dataset (Census Income, ~45K records)
2. Applies Mondrian K-anonymity for K = {2, 5, 10, 25, 50, 100}
3. Trains **SVM**, **Neural Network** (PyTorch), and **Random Forest** on both original and anonymized data
4. Measures: Accuracy, Misclassification Rate, Precision, Recall, AUC
5. Computes information loss metrics (**C_DM**, **C_AVG**) from the Mondrian paper
6. Runs 3 seeds per configuration for robustness (mean ± std)
7. Generates comparison charts and a full report

## Output

Results are saved to `results/`:
- `comparison_chart.png` — ML metrics across K values
- `information_loss.png` — C_DM, C_AVG, partition count
- `metrics_table.csv` — all ML metrics
- `information_loss.csv` — anonymization quality metrics
- `report.md` — full experiment report
- `anonymized_sample_k2.csv` — sample of anonymized data

## Project Structure

```
main.py          — experiment orchestration, plotting, reporting
mondrian.py      — Mondrian algorithm + information loss metrics
ml_models.py     — SVM, Neural Network, Random Forest
utils.py         — data loading, preprocessing, evaluation
```

## Quasi-Identifiers

| Attribute      | Type    | Description          |
|---------------|---------|----------------------|
| age           | Numeric | Age in years         |
| education-num | Numeric | Education level      |
| hours-per-week| Numeric | Weekly work hours    |
| capital-gain  | Numeric | Capital gains        |
| capital-loss  | Numeric | Capital losses       |

**Sensitive attribute**: income (>50K / ≤50K)

## Perturbation Method

Mondrian recursively splits data along the QI dimension with the largest span at the median. Within each resulting equivalence class (≥ K records), QI values are replaced with the partition mean (generalization).
