import scipy.io
import numpy as np
from torchvision import datasets, transforms
import os

from src.utilis.utilis import colored_print, bcolors

CLASS_INDEX_TABLE = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
}

def get_image_size(chemin: str ="data/binaryalphadigs.mat") -> tuple:
    try:
        data = scipy.io.loadmat(chemin)
    except:
        raise ValueError("Fichier non existant !")

    return data['dat'][0][0].shape

def lire_alpha_digit(caracteres: list, chemin: str ="data/binaryalphadigs.mat") -> np.array:
    if not caracteres:
        raise ValueError("Aucun caractère fourni en paramètre !")
    if not all(elem in CLASS_INDEX_TABLE for elem in caracteres):
        raise ValueError("Utiliser des caractères valides !")

    try:
        data = scipy.io.loadmat(chemin)
    except:
        raise ValueError("Fichier non existant !")

    X = np.concatenate([data['dat'][CLASS_INDEX_TABLE[char]].flatten() for char in caracteres], axis=0)
    X = np.array([element.flatten() for element in X])

    return X

def lire_mnist_digits(characters: list , data_dir: str ="../data", examples_per_char=39) -> np.ndarray:
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, "MNIST")

    if not os.path.exists(dataset_path):
        colored_print(bcolors.WARNING, "Downloading MNIST dataset...")
        transform = transforms.Compose([transforms.ToTensor()])
        datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    else:
        colored_print(bcolors.OKGREEN, "MNIST dataset already exists.")

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=False)
    binary_arrays = []
    char_count = {char: 0 for char in characters}

    for image, label in mnist_dataset:
        if label in characters and char_count[label] < examples_per_char:
            image = transforms.ToPILImage()(image)
            image_np = np.array(image)
            binary_image = (image_np > 128).astype(np.uint8)
            flattened_image = binary_image.flatten()
            binary_arrays.append(flattened_image)
            char_count[label] += 1
            if all(count >= examples_per_char for count in char_count.values()):
                break

    binary_matrix = np.array(binary_arrays)

    return binary_matrix

