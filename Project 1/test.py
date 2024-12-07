# Written by Jack Rowe
# October 2024

import numpy as np
import logging
from main import SVD,SMS

def main():
    logging.warning(f'====================')
    logging.warning(f'Singular Value Decomposition')

    matrices = [
        np.array([[3, 2, 2], [2, 3, -2]]),
        np.array([[-3, 1], [6, -2], [6, -2]]),
        np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]]),
        np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[4, 0, 0], [0, 3, 0], [0, 0, 2]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[4, 1, 2], [1, 3, 0], [2, 0, 5]]),
        np.array([[2, -1, 3], [0, 5, -2], [4, 2, 1]])
    ]

    for i, A in enumerate(matrices):
        try:
            logging.warning(f'====================')
            USVT, k, A_inv = SVD(A)
            U=USVT[0]; S=USVT[1]; VT=USVT[2]
            NSVD = np.linalg.svd(A)
            logging.warning(f'{i}:\nU:\n{U}\nS:\n{S}\nVT:\n{VT}\nnpU:\n{NSVD[0]}\nnpS:\n{NSVD[1]}\nnpVT:\n{NSVD[2]}')
        except:
            continue

    logging.warning(f'====================')
    logging.warning(f'Spring Mass System')

    ms = [
        ([1, 2, 3], [1, 2, 3, 4]),
        ([5, 6, 7], [8, 9, 10]),
        ([11, 12, 13], [14, 15])
        ]

    for i, (m,s) in enumerate(ms):
        try:
            logging.warning(f'====================')
            u,e,w = SMS(m,s)
            logging.warning(f'{i}:\nu: {u}\ne: {e}\nw: {w}')
        except:
            continue

    logging.warning(f'====================')
    return

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    logging.basicConfig(level=logging.WARNING)
    main()
