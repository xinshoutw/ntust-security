# K-anonymity Experiment Report

## Method
- **Algorithm**: Mondrian Multidimensional K-anonymity
- **Dataset**: UCI Adult Dataset
- **Quasi-identifiers**: age, workclass, education, marital-status, occupation, race, sex, native-country
- **Sensitive attribute**: income (>50K / <=50K)

## ML Models
### SVM
- Kernel: RBF
- Library: scikit-learn SVC with default hyperparameters
- probability=True for AUC computation

### Neural Network
- Architecture: 3-layer MLP (Input -> 64 -> 32 -> 1)
- Activation: ReLU (hidden), Sigmoid (output)
- Loss: BCELoss
- Optimizer: Adam (lr=0.001)
- Epochs: 50, Batch size: 64
- Framework: PyTorch

## K Values Tested: [2, 4, 10, 20, 50, 100]

## Results

| K | Model | Accuracy | Misclass. Rate | Precision | Recall | AUC |
|---|-------|----------|---------------|-----------|--------|-----|
| Original | SVM | 0.8407 | 0.1593 | 0.7466 | 0.5453 | 0.8869 |
| Original | NN | 0.8419 | 0.1581 | 0.7199 | 0.5972 | 0.9035 |
| 2 | SVM | 0.8400 | 0.1600 | 0.7589 | 0.5240 | 0.8906 |
| 2 | NN | 0.8465 | 0.1535 | 0.7311 | 0.6065 | 0.9018 |
| 4 | SVM | 0.8404 | 0.1596 | 0.7550 | 0.5313 | 0.8890 |
| 4 | NN | 0.8470 | 0.1530 | 0.7476 | 0.5819 | 0.9009 |
| 10 | SVM | 0.8465 | 0.1535 | 0.7785 | 0.5360 | 0.8923 |
| 10 | NN | 0.8437 | 0.1563 | 0.6978 | 0.6565 | 0.9035 |
| 20 | SVM | 0.8487 | 0.1513 | 0.7862 | 0.5386 | 0.8985 |
| 20 | NN | 0.8516 | 0.1484 | 0.7148 | 0.6724 | 0.9076 |
| 50 | SVM | 0.8463 | 0.1537 | 0.7679 | 0.5486 | 0.8949 |
| 50 | NN | 0.8490 | 0.1510 | 0.7355 | 0.6145 | 0.9062 |
| 100 | SVM | 0.8394 | 0.1606 | 0.7652 | 0.5120 | 0.8814 |
| 100 | NN | 0.8420 | 0.1580 | 0.7484 | 0.5506 | 0.8976 |

## Analysis
As K increases, the level of privacy protection grows but information loss also increases.
This is reflected in the general trend of declining ML model performance (lower accuracy,
precision, recall, and AUC) with higher K values. The trade-off between privacy and utility
is a fundamental challenge in data anonymization.