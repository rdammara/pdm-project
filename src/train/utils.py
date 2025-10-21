
import numpy as np
import random
import os

def set_global_seed(seed: int = 42):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0

    def step(self, value):
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            return False
        else:
            self.count += 1
            return self.count >= self.patience
