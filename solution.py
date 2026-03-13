import pandas as pd
import numpy as np
import random
from pathlib import Path

# Read data
base_dir = Path('./task2_data')
ratings = pd.read_csv(base_dir / 'ratings.csv')
ratings.columns = ['u', 'i', 'r']

np.random.seed(42)
random.seed(42)

U = ratings['u'].max() + 1
I = ratings['i'].max() + 1
K = 10

P = np.random.normal(0, 0.1, (U, K))
Q = np.random.normal(0, 0.1, (I, K))

lr = 5e-3
reg = .1

data = ratings.to_numpy()

for epoch in range(1, 21):
    np.random.shuffle(data)
    for u, i, r in data:
        pred = P[u] @ Q[i]
        err = r - pred
        pu_old = P[u].copy()
        P[u] += lr*(err*Q[i] - reg*P[u])
        Q[i] += lr*(err*pu_old - reg*Q[i])

    if epoch==5:
        np.save("checkpoint_epoch_5.npy",P)

    if epoch==10:
        np.save("checkpoint_epoch_10.npy",P)
