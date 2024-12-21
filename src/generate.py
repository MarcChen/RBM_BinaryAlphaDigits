import numpy as np
import matplotlib.pyplot as plt

from src.RBM_model import RBM

def display_images(images: list, size: tuple) -> None:
    for image in images:
        image = image.reshape(size)
        plt.imshow(image, cmap='gray')
        plt.show()

def generer_image_RBM(model: RBM, nb_images: int, nb_iter: int, size_img: tuple) -> list:
    p, q = model.W.shape
    images = []
    for _ in range(nb_images):
        v = (np.random.rand(p) < 0.5) * 1
        for _ in range(nb_iter):
            h = (np.random.rand(q) < model.entree_sortie_RBM(v)) * 1
            v = (np.random.rand(p) < model.sortie_entree_RBM(h)) * 1
        images.append(v.reshape(size_img))
    return images