# Saikat Chakraborty
# Department of EEE, KUET

import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import statsmodels.api as sm

def X_to_TS(X_i:np.ndarray) -> np.ndarray:
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    # Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])

def main(y_tran:np.ndarray, t_tran:np.ndarray) -> None:
    """Main function of the program"""
    N=y_tran.shape[0]
    L = 2000 # The window length.
    d = 10 # No of SVD components
    K = N - L + 1 # The number of columns in the trajectory matrix.
    audio_data = np.column_stack([y_tran[i:i+L] for i in range(0,K)]) # Trajectory Matrix
    U, Sigma, V = np.linalg.svd(audio_data, full_matrices=False)
    V = V.T
    X_elem = np.array( [Sigma[i] * np.outer(U[:,i], V[:,i]) for i in range(0,d)] )
    for i in range(d):
        F_i = X_to_TS(X_elem[i])
        #cross_corr=sm.tsa.stattools.ccf(y_tran, F_i, adjusted=False)
        print(f"Cy{i + 1}: {np.sum(F_i * y_tran) / N}") # Cross correlation


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python svd.py <filename>.wav <sample size:optional>")
        sys.exit()
    if not sys.argv[1].endswith(".wav"):
        print("Usage: python svd.py <filename>.wav <sample size:optional>")
        sys.exit()
    try:
        y, sr = librosa.load(sys.argv[1], sr=44100)
        ss = 5000 if len(sys.argv) == 2 else int(sys.argv[2])
        t = np.linspace(0, (y.shape[0] - 1), y.shape[0]) / sr
        y_tran = y[0 : ss]
        t_tran = t[0 : ss]
    except Exception as E:
        print(E)
        sys.exit()
    main(y_tran, t_tran)