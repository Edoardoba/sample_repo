import numpy as np

def load_dataset():
    """
    Synthetic Integer Parity Dataset
    X: numbers 0-99
    y: 0 = even, 1 = odd
    """
    X = np.array([[i] for i in range(100)])
    y = np.array([i % 2 for i in range(100)])
    return X, y

# Governance info (Article 10)
DATA_DESCRIPTION = {
    "name": "Synthetic Integer Parity Dataset",
    "size": 100,
    "features": 1,
    "labels": [0,1],
    "source": "programmatic generation",
    "lineage": "raw → numpy.arange → reshape → label i%2",
    "limitations": "synthetic, deterministic, not representative of real-world demographics"
}
