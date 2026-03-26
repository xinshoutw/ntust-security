"""
ML Models for K-Anonymity Experiment
=====================================
SVM (scikit-learn), Neural Network (PyTorch), Random Forest (scikit-learn).
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NeuralNet(nn.Module):
    """3-layer MLP with dropout for binary classification."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_svm(X_train, y_train, X_test):
    """Train SVM (RBF kernel) and return predictions and probabilities."""
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              max_iter=5000, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_prob = svm.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def train_neural_network(X_train, y_train, X_test, epochs=30, batch_size=256, lr=0.001):
    """Train a PyTorch neural network and return predictions and probabilities."""
    device = _get_device()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNet(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t).cpu().numpy().flatten()

    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


def train_random_forest(X_train, y_train, X_test):
    """Train Random Forest and return predictions and probabilities."""
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    return y_pred, y_prob
