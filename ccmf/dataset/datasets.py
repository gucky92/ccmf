import numpy as np
import pandas as pd

from .generate import gaussian_tuning


def load_fake_dm9():
    """Generate a data set for a Dm9-like linear recurrent circuit.

    Returns
    -------

    """
    W = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])

    M = np.array([[0, 0, -.5, 0, .5],
                  [0, 0, 0, -.5, .5],
                  [-.5, 0, 0, 0, .5],
                  [0, -.5, 0, 0, .5],
                  [-.2, -.1, -.3, -.3, 0]])

    s = np.linspace(300, 550, 10)
    U = np.stack([gaussian_tuning(s, s_a, 50) for s_a in [350, 400, 450, 500]])
    V = np.linalg.inv(np.eye(len(M)) - M) @ W @ U
    X = np.concatenate([U, V])

    inputs = ['Rh3', 'Rh4', 'Rh5', 'Rh6']
    outputs =['pR7', 'yR7', 'pR8', 'yR8', 'Dm9']
    columns = pd.Index(s, name='stimulus')
    input_index = pd.Index(inputs, name='cell')
    all_index = pd.Index(inputs + outputs, name='cell')
    return \
        pd.DataFrame(W, index=outputs, columns=inputs),\
        pd.DataFrame(M, index=outputs, columns=outputs),\
        pd.DataFrame(U, index=input_index, columns=columns),\
        pd.DataFrame(X, index=all_index, columns=columns)