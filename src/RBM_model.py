import numpy as np
from tqdm import tqdm

class RBM:
    def __init__(self, p: int, q: int) -> None:
        self.p = p
        self.q = q
        self.A = np.zeros(p)
        self.B = np.zeros(q)
        self.W = np.random.normal(loc=0, scale=0.1, size=(p, q))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def entree_sortie_RBM(self, X: np.array) -> np.array:
        return self.sigmoid(X @ self.W + self.B)

    def sortie_entree_RBM(self, H: np.array) -> np.array:
        return self.sigmoid(H @ self.W.T + self.A)

    def free_energy(self, v: np.array) -> float:
        return -v @ self.A - np.sum(np.log(1 + np.exp(v @ self.W + self.B)), axis=1)

    def train_RBM(self, X: np.array, epochs: int, batch_size: int, learning_rate: float) -> tuple:
        weights, losses, free_energies, weight_snapshots, gradients, avg_activations, hidden_probs_snapshots = [], [], [], [], [], [], []
        n = X.shape[0]

        for epoch in tqdm(range(epochs)):
            np.random.shuffle(X)
            for i in range(0, n, batch_size):
                X_batch = X[i: min(i+batch_size, n), :]
                t_b = X_batch.shape[0]
                v_0 = X_batch
                p_h_v_0 = self.entree_sortie_RBM(v_0)
                h_0 = np.random.binomial(1, p_h_v_0)
                p_v_h_0 = self.sortie_entree_RBM(h_0)
                v_1 = np.random.binomial(1, p_v_h_0)
                ph_v_1 = self.entree_sortie_RBM(v_1)

                grad_a = np.sum(v_0 - v_1, axis=0)
                grad_b = np.sum(p_h_v_0 - ph_v_1, axis=0)
                grad_w = v_0.T @ p_h_v_0 - v_1.T @ ph_v_1

                self.A += learning_rate * grad_a / t_b
                self.B += learning_rate * grad_b / t_b
                self.W += learning_rate * grad_w / t_b

                weights.append(np.mean(abs(self.W)))
                gradients.append(np.mean(abs(grad_w)))

            H = self.entree_sortie_RBM(X)
            losses.append(np.mean((X - self.sortie_entree_RBM(H)) ** 2))
            free_energies.append(np.mean(self.free_energy(X)))
            avg_activations.append(np.mean(H))

            if epoch % 400 == 0 or epoch == epochs - 1:
                weight_snapshots.append(self.W.copy())
                hidden_probs_snapshots.append(H.copy())

        return losses, weights, free_energies, weight_snapshots, gradients, avg_activations, hidden_probs_snapshots