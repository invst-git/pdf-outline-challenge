# pdf_outline/classify.py
import numpy as np
from pathlib import Path
import pickle

LABELS = ["body", "heading"]           # binary first
HIDDEN_DIM = 1024                      # Donut-base output size

def train_head(embeds: np.ndarray, weak_labels: np.ndarray) -> np.ndarray:
    """
    Simple logistic regression: W (HIDDEN_DIM,) + b  -> prob heading
    Closed‑form via least‑squares on logit space.
    """
    # logit ~ linear; add bias term
    X = np.hstack([embeds, np.ones((embeds.shape[0], 1))])
    y = weak_labels.reshape(-1, 1)
    # ridge λ=1e-3 to avoid singular
    W = np.linalg.inv(X.T @ X + 1e-3 * np.eye(X.shape[1])) @ X.T @ y
    return W.flatten()

def save_head(weights: np.ndarray, path: Path):
    with open(path, "wb") as f: pickle.dump(weights, f)

def load_head(path: Path) -> np.ndarray:
    with open(path, "rb") as f: return pickle.load(f)

def predict(embeds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    X = np.hstack([embeds, np.ones((embeds.shape[0], 1))])
    logits = X @ weights
    return 1 / (1 + np.exp(-logits))   # sigmoid
