import scipy.io
from typing import List
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import os

from utilis.utilis import colored_print, bcolors


CLASS_INDEX_TABLE = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
}

def lire_alpha_digit( caracteres: List, chemin: str ="../data/binaryalphadigs.mat") -> np.array:
    """
    :param caracteres: Caractères à traiter Ex: ['0', 'A', 'D']
    :param chemin: Chemin du dataset
    :return: Array des 36 elements pour chaque caractères demandé
    """

    if not caracteres:
        raise ValueError("Aucun caractère fourni en paramètre")
    if not all(elem in CLASS_INDEX_TABLE for elem in caracteres)

    mat_file = scipy.io.loadmat(chemin)
    print(mat_file.keys())

    data = mat_file['classlabels']
    print(data)

def lire_mnist_digits(characters : list , data_dir : str ="../data", examples_per_char=39) -> np.ndarray:
    """
    Downloads the MNIST dataset if not already present, converts images to binary (0 and 1),
    resizes to 32x10, flattens to a 320-dimensional numpy array, and returns a matrix of specified characters.

    Args:
        data_dir (str): Path to the directory where the dataset will be stored.
        characters (list): List of characters (digits or letters) to extract.
        examples_per_char (int): Number of examples per character.

    Returns:
        np.ndarray: A matrix of dimensions [n * examples_per_char, 320], where n is the number of selected characters.
    """
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Check if the dataset is already downloaded
    dataset_path = os.path.join(data_dir, "MNIST")
    if not os.path.exists(dataset_path):
        colored_print(bcolors.WARNING, "Downloading MNIST dataset...")
        transform = transforms.Compose([transforms.ToTensor()])
        datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    else:
        colored_print(bcolors.OKGREEN, "MNIST dataset already exists.")

    # Load the dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=False)

    binary_arrays = []

    # Initialize a counter to track examples per character
    char_count = {char: 0 for char in characters}

    # Process each image
    for image, label in mnist_dataset:
        if label in characters and char_count[label] < examples_per_char:
            # Convert image to PIL format
            image = transforms.ToPILImage()(image)

            # Resize to 32x10
            image = image.resize((32, 10), resample=Image.Resampling.LANCZOS)

            # Convert to binary (threshold at 0.5)
            image_np = np.array(image)
            binary_image = (image_np > 128).astype(np.uint8)  # Convert to binary using threshold

            # Flatten to 320-dimensional array
            flattened_image = binary_image.flatten()

            # Append to the list
            binary_arrays.append(flattened_image)

            # Update character count
            char_count[label] += 1

            # Break if we have enough examples for all characters
            if all(count >= examples_per_char for count in char_count.values()):
                break

    # Convert list to numpy array
    binary_matrix = np.array(binary_arrays)

    return binary_matrix

if __name__ == "__main__":
    lire_alpha_digit()

    ### Testing MNIST Data Import ###
    characters = [0, 1, 2, 3, 4]
    MNIST_DIGITS = lire_mnist_digits(characters)
    print(MNIST_DIGITS[0])