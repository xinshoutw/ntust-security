# K-anonymity Experiment

![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-f7931e?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

探討 **K-anonymity 隱私保護技術**對機器學習模型效能的影響。使用 Mondrian 多維度 K-anonymity 演算法對 UCI Adult Dataset 進行匿名化處理，並比較匿名化前後 SVM 與 Neural Network 模型的表現差異。

## 總覽

本專案實作 Mondrian Multidimensional K-anonymity 演算法，將資料集中的準識別欄位（Quasi-identifiers）進行泛化處理，以達到隱私保護的目的。同時透過訓練 SVM 與 Neural Network 兩種模型，觀察不同 K 值下隱私保護對模型效能的影響，量化 **隱私與效用的權衡（Privacy-Utility Trade-off）**。

### 功能特色

- :shield: **K-anonymity 實作** — Mondrian 多維度分割演算法，支援數值與類別型準識別欄位
- :robot: **雙模型比較** — SVM（RBF kernel）與 Neural Network（3 層 MLP）
- :bar_chart: **完整評估指標** — Accuracy、Misclassification Rate、Precision、Recall、AUC
- :chart_with_upwards_trend: **視覺化結果** — 自動產生比較圖表與 CSV 報表
- :test_tube: **多組 K 值實驗** — K = 2, 4, 10, 20, 50, 100

## 實驗結果

| K | Model | Accuracy | Misclass. Rate | Precision | Recall | AUC |
|---|-------|----------|----------------|-----------|--------|-----|
| Original | SVM | 0.8407 | 0.1593 | 0.7466 | 0.5453 | 0.8869 |
| Original | NN | 0.8419 | 0.1581 | 0.7199 | 0.5972 | 0.9035 |
| 2 | SVM | 0.8400 | 0.1600 | 0.7589 | 0.5240 | 0.8906 |
| 2 | NN | 0.8465 | 0.1535 | 0.7311 | 0.6065 | 0.9018 |
| 10 | SVM | 0.8465 | 0.1535 | 0.7785 | 0.5360 | 0.8923 |
| 10 | NN | 0.8437 | 0.1563 | 0.6978 | 0.6565 | 0.9035 |
| 20 | SVM | 0.8487 | 0.1513 | 0.7862 | 0.5386 | 0.8985 |
| 20 | NN | 0.8516 | 0.1484 | 0.7148 | 0.6724 | 0.9076 |
| 100 | SVM | 0.8394 | 0.1606 | 0.7652 | 0.5120 | 0.8814 |
| 100 | NN | 0.8420 | 0.1580 | 0.7484 | 0.5506 | 0.8976 |

<details>
<summary>結果圖表</summary>

![Comparison Chart](results/comparison_chart.png)

</details>

### 分析

隨著 K 值增加，隱私保護程度提高但資訊損失也增加。實驗結果顯示模型效能在不同 K 值下保持相對穩定（約 2% 以內），表明 Mondrian 演算法在維持資料效用的同時能提供有效的隱私保護。K=20 時兩個模型都達到最佳表現，顯示適度的泛化反而有助於降低雜訊。

## 系統需求

| 項目 | 需求 |
|------|------|
| Python | 3.13+ |
| 套件管理 | [uv](https://docs.astral.sh/uv/) (建議) 或 pip |
| 主要依賴 | PyTorch, scikit-learn, pandas, matplotlib, numpy |
| 資料集 | UCI Adult Dataset（自動下載或本地 `data/adult.data`） |

## 開發環境建置

### 安裝

```bash
# Clone 專案
git clone <repo-url>
cd k-anonymity

# 使用 uv 安裝依賴
uv sync

# 或使用 pip
pip install -e .
```

### 執行實驗

```bash
python main.py
```

執行後將自動：
1. 載入 Adult Dataset（32,561 筆資料）
2. 在原始資料上訓練 SVM 與 Neural Network 作為 baseline
3. 對 K = 2, 4, 10, 20, 50, 100 分別進行匿名化並訓練模型
4. 產出結果至 `results/` 目錄

## 專案架構

```
k-anonymity/
├── main.py              # 實驗主程式，負責流程編排與結果產出
├── mondrian.py          # Mondrian K-anonymity 演算法實作
├── ml_models.py         # SVM 與 Neural Network 模型定義
├── utils.py             # 資料載入、前處理與模型評估工具
├── pyproject.toml       # 專案設定與依賴管理
├── data/
│   └── adult.data       # UCI Adult Dataset
├── results/
│   ├── comparison_chart.png   # 指標比較圖表
│   ├── metrics_table.csv      # 完整指標 CSV
│   └── report.md              # 實驗報告
└── docs/
    └── HW_K-anonymity_v3_20250316.pdf  # 作業說明
```

### 模組說明

| 模組 | 功能 |
|------|------|
| `main.py` | 實驗流程控制：資料載入 → 模型訓練 → 匿名化 → 結果比較 → 報告產出 |
| `mondrian.py` | Mondrian 多維度 K-anonymity：遞迴分割、正規化範圍選擇、泛化處理 |
| `ml_models.py` | SVM（RBF kernel）與 MLP Neural Network（64→32→1, PyTorch） |
| `utils.py` | Adult Dataset 載入、Label Encoding、Feature Scaling、Train/Test Split、評估指標計算 |

## 方法說明

### K-anonymity 演算法

採用 **Mondrian Multidimensional K-anonymity**：
- 選擇正規化範圍最大的準識別欄位進行分割
- 數值型欄位以中位數分割，類別型欄位以頻率均分
- 遞迴分割直到每個分區至少包含 K 筆資料
- 對分區內的準識別欄位進行泛化（數值→範圍，類別→集合）

### 準識別欄位（Quasi-identifiers）

`age`, `workclass`, `education`, `marital-status`, `occupation`, `race`, `sex`, `native-country`

### 機器學習模型

- **SVM**: scikit-learn SVC, RBF kernel, probability=True
- **Neural Network**: PyTorch 3 層 MLP (Input→64→32→1), ReLU + Sigmoid, Adam optimizer (lr=0.001), 50 epochs

## 參考文獻

- K. LeFevre, D. J. DeWitt, and R. Ramakrishnan, "Mondrian multidimensional k-anonymity," *ICDE*, Vol. 6, 2006.
- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

## 授權

本專案為 NTUST 資訊安全課程作業。
