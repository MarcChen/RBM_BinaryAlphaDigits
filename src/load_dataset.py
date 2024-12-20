import scipy.io
from typing import List

CLASS_INDEX_TABLE = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
}

def lire_alpha_digit( caracteres: List, chemin: str ="../data/binaryalphadigs.mat"):
    if not caracteres:
        raise ValueError("Aucun caractère fourni en paramètre")
    if not all(elem in CLASS_INDEX_TABLE for elem in caracteres)

    mat_file = scipy.io.loadmat(chemin)
    print(mat_file.keys())

    data = mat_file['classlabels']
    print(data)


if __name__ == "__main__":
    lire_alpha_digit()
