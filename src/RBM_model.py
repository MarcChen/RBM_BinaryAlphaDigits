import numpy as np

class RBM:
    def __init__(self, p: int, q: int) -> None:
        self.p = p
        self.q = q
        self.A = np.zeros(p)
        self.B = np.zeros(q)
        self.W = np.random.normal(loc=0, scale=0.1, size=(p, q))

    def entree_sortie_RBM(self, X: np.array) -> np.array:
        H = 1/(1 + np.exp(np.dot(X, self.W) + self.B))
        return H

    def sortie_entree_RBM(self, H: np.array) -> np.array:
        X = 1/(1 + np.exp(np.dot(H, self.W.T) + self.A))
        return X

    def train_RBM(self, X: np.array, epochs: int, batch_size: int, learning_rate: float) -> 'RBM':

        for epoch in range(epochs):
            np.random.shuffle(X)
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i: min(i+batch_size, X.shape[0])]
                t_b = X_batch.shape[0]
                v_0 = X_batch
                p_h_v_0 = self.entree_sortie_RBM(v_0)
                h_0 = np.random.binomial(1, p_h_v_0)
                p_v_h_0 = self.sortie_entree_RBM(h_0)
                v_1 = np.random.binomial(1, p_v_h_0)
                ph_v_1 = self.entree_sortie_RBM(v_1)

                grad_a = np.sum(v_0 - v_1, axis=0)
                grad_b = np.sum(p_h_v_0 - ph_v_1, axis=0)
                grad_w = np.dot(v_0.T, p_h_v_0) - np.dot(v_1.T, ph_v_1)

                self.A += learning_rate * grad_a / t_b
                self.B += learning_rate * grad_b / t_b
                self.W += learning_rate * grad_w / t_b

                H = self.entree_sortie_RBM(X)
                X_reconstruit = self.sortie_entree_RBM(H)
                erreur = np.mean((X - X_reconstruit)**2)

                print(f"Epoch: {epoch+1}/{epochs}, Batch: {i//batch_size+1}/{X.shape[0]//batch_size}, Erreur: {erreur}")

        return self

