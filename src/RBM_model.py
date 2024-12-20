import numpy as np

class RBM:
    def __init__(self, p: int, q: int):
        self.p = p
        self.q = q
        self.A = np.zeros(p)
        self.B = np.zeros(q)
        self.W = np.random.normal(loc=0, scale=0.1, size=(p, q))

    def entree_sortie_RBM(self, X: np.array) -> np.array:
        H = 1/(1 + np.exp(np.dot(X, self.W) + self.B))
        return H

