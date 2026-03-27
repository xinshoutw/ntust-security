# K-Anonymity 隱私保護實驗

實作 **Mondrian 多維度 K-匿名演算法**，並分析其對機器學習模型效能的影響。

## 參考文獻

K. LeFevre, D. J. DeWitt, R. Ramakrishnan, "Mondrian Multidimensional K-Anonymity," ICDE, Vol. 6, 2006.

## 環境建置

```bash
# 使用 uv（推薦）
uv sync
uv run python main.py

# 或使用 pip
pip install pandas numpy scikit-learn torch matplotlib
python main.py
```

## 實驗流程

1. 載入 **UCI Adult** 資料集（人口普查收入資料，約 45,000 筆）
2. 對不同 K 值（K = 2, 5, 10, 25, 50, 100）套用 Mondrian K-匿名演算法
3. 分別以 **SVM**、**類神經網路**（PyTorch）、**隨機森林** 訓練原始與匿名化資料
4. 評估指標：Accuracy、Misclassification Rate、Precision、Recall、AUC
5. 計算 Mondrian 論文中的資訊損失指標（**C_DM**、**C_AVG**）
6. 每組設定執行 3 個隨機種子，報告 mean ± std 以確保穩健性
7. 產生比較圖表與完整實驗報告

## 輸出結果

結果儲存於 `results/` 目錄：

| 檔案 | 說明 |
|------|------|
| `comparison_chart.png` | 各 K 值下 ML 指標比較圖 |
| `information_loss.png` | C_DM、C_AVG、分割數量變化圖 |
| `metrics_table.csv` | 所有 ML 指標數據 |
| `information_loss.csv` | 匿名化品質指標 |
| `report.md` | 完整實驗報告 |
| `anonymized_sample_k2.csv` | 匿名化資料樣本（K=2） |

## 專案結構

```
main.py          — 實驗流程控制、繪圖、報告產生
mondrian.py      — Mondrian 演算法 + 資訊損失指標
ml_models.py     — SVM、類神經網路、隨機森林
utils.py         — 資料載入、前處理、評估函數
```

## 準識別符（Quasi-Identifiers）

| 屬性 | 類型 | 說明 |
|------|------|------|
| age | 數值 | 年齡 |
| education-num | 數值 | 教育程度（數值化） |
| hours-per-week | 數值 | 每週工作時數 |
| capital-gain | 數值 | 資本利得 |
| capital-loss | 數值 | 資本損失 |

**敏感屬性**：income（年收入 >50K / ≤50K）

## 資料擾動方式

Mondrian 演算法以遞迴方式，沿著範圍最大的準識別符維度在中位數處進行切割。當切割會導致任一分組少於 K 筆紀錄時停止。最終在每個等價類（≥ K 筆紀錄）中，將準識別符數值替換為該組平均值（泛化），確保同一等價類中所有紀錄的準識別符完全相同，達成 K-匿名要求。

## 機器學習模型

### SVM（支持向量機）
- 核函數：RBF（高斯徑向基函數）
- 正則化參數 C = 1.0，Gamma = scale
- 函式庫：scikit-learn

### Neural Network（類神經網路）
- 架構：3 層全連接 MLP（Input → 64 → 32 → 1）
- 激活函數：ReLU（隱藏層）、Sigmoid（輸出層）
- Dropout：0.3（第一隱藏層）、0.2（第二隱藏層）
- 優化器：Adam（lr=0.001）
- 訓練：30 epochs，batch size 256
- 框架：PyTorch

### Random Forest（隨機森林）
- 決策樹數量：100
- 最大深度：15
- 函式庫：scikit-learn

## 評估指標

| 指標 | 說明 |
|------|------|
| Accuracy（準確率） | 正確預測的比例 |
| Misclassification Rate（誤分類率） | 1 − Accuracy |
| Precision（精確率） | 預測為正類中，實際為正類的比例 |
| Recall（召回率） | 實際正類中，被正確預測的比例 |
| AUC | ROC 曲線下面積，衡量排序能力（1.0 = 完美，0.5 = 隨機） |

## 資訊損失指標

源自 Mondrian 論文：

- **C_DM（Discernability Metric）**：各等價類大小平方和（Σ|E|²），越低越好
- **C_AVG（Normalized Average）**：（總筆數 / 等價類數）/ K，最佳值為 1.0
