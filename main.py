# Written by Jack Rowe
# October 2024

import numpy as np
import logging

def SVD(A: np.ndarray, eps: float=1e-10) -> [[np.ndarray, np.ndarray, np.ndarray], float, np.ndarray]:

    logging.info(f'Running SVD...')

    # (1) Find Singular Values:
    # - get eigenvalues of A^TA-LI
    # - sort in descending order
    # - construct S using sqrt(l)

    eigenvalues, eigenvectors = np.linalg.eig(A.T @ A)
    i = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[i].real
    eigenvectors = eigenvectors[:, i].real

    singular_values = np.sqrt(eigenvalues[abs(eigenvalues) > eps]).real

    S = np.zeros((A.shape[0], A.shape[1]), dtype=float)
    for i in range(len(singular_values)):
        S[i,i] = singular_values[i]
    logging.info(f'S:\n{S}')

    # (2) Find V:
    # - get eigenvectors (available from step 1)
    # - store in columns of V

    V = eigenvectors
    logging.info(f'V:\n{V}')

    # (3) Find U:
    # - back-solve using U=AVS^-1
    # (easy math for S^-1 since its diagonal)

    S_inv = np.zeros((S.shape[1], S.shape[0]), dtype=float)
    for i in range(len(singular_values)):
        S_inv[i, i] = 1 / singular_values[i]

    U = A @ V @ S_inv
    logging.info(f'U:\n{U}')

    # (4) Calculate condition number:

    k = max(singular_values) / min(singular_values)
    logging.info(f'k:\n{k}')

    # (5) Calculate inverse of A:

    A_inv = np.zeros_like(A, dtype=float)
    if (A_inv.shape[0] != A_inv.shape[1]):
        logging.error(f'Matrix A is non-square (not invertible)')
    elif np.any(eigenvalues < eps):
        logging.error(f'Matrix A has singular value close to 0 (not invertible)')
    else:
        A_inv = V @ S_inv @ U.T
        logging.info(f'A_inv:\n{A_inv}')

    logging.info(f'A=USV^T:\n{U}\n{S}\n{V.T}')
    logging.info(f'A:\n{U@S@V.T}')

    return [U,S,V.T], k, A_inv

def SMS(masses: list, spring_constants: list, eps: float=1e-10) -> [list, list, list]:

    # FREE/FREE if n_springs = n_masses - 1
    # FREE/FIXED if n_springs = n_masses
    # FIXED/FIXED if n_springs = n_masses + 1

    n_masses = len(masses)
    n_springs = len(spring_constants)

    # (1) Construct A
    # e(n_springs*1) = A(n_springs*n_masses)u(n_masses*1)
    A = np.zeros((n_springs, n_masses))
    for i in range(min(n_masses,n_springs)):
        A[i,i] = 1
    for i in range(1,n_springs):
        A[i,i-1] = -1
    logging.info(f'\n{A}')

    # (2) Construct C
    # C = diagonal matrix of 1/k_i
    C = np.zeros((n_springs, n_springs))
    for i in range(n_springs):
        C[i,i] = 1 / spring_constants[i]

    logging.info(f'\n{C}')

    # (3) Construct K
    # K = A^TCA
    K = A.T @ C @ A
    logging.info(f'\n{K}')

    # (4) Run SVD on K
    USVT, k, K_inv = SVD(K)

    # (5) Solve Ku=f
    # u = K^-1f
    # assuming gravity as external force
    f = [m * 9.8 for m in masses]
    u = K_inv @ f
    logging.info(f'\n{u}')

    # (6) Solve e=Au & w=Ce
    e = A @ u
    w = C @ e
    logging.info(f'\n{e}\n{u}')

    return u,e,w

def main():

    matrices = [
        #np.array([[3, 2, 2], [2, 3, -2]]),
        np.array([[-3, 1], [6, -2], [6, -2]])
        #np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]]),
        #np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
        #np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        #np.array([[4, 0, 0], [0, 3, 0], [0, 0, 2]]),
        #np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        #np.array([[4, 1, 2], [1, 3, 0], [2, 0, 5]]),
        #np.array([[2, -1, 3], [0, 5, -2], [4, 2, 1]])
    ]

    for i, A in enumerate(matrices):
        USVT, k, A_inv = SVD(A)
        try:
            logging.warning(f'\n{i}:\n{A}\n{USVT[0] @ USVT[1] @ USVT[2]}')
        except:
            continue

    #m = [1,2,3]
    #s = [4,5,6,7]
    #SMS(m,s)

    return

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    logging.basicConfig(level=logging.INFO)
    main()
