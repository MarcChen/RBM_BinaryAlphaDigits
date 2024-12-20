import scipy.io
from typing import List
import numpy as np

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
        raise ValueError("Aucun caractère fourni en paramètre !")
    if not all(elem in CLASS_INDEX_TABLE for elem in caracteres):
        raise ValueError("Utiliser des caractères valides !")

    try:
        dataset_mat = scipy.io.loadmat(chemin)
    except:
        raise ValueError("Fichier non existant !")

    dataset = np.array([])
    for caractere in caracteres:
        class_data = dataset_mat['dat'][CLASS_INDEX_TABLE[caractere]]
        dataset = np.concatenate((dataset, class_data))

    return dataset

