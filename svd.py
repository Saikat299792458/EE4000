# Saikat Chakraborty
# Department of EEE, KUET

import sys
import getopt
import librosa
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

class ThreadWithReturn(Thread):
    """Modified Threading with return functionality"""
    def run(self) -> None:
        self.result = self._target(*self._args, **self._kwargs)

def X_to_TS(X_i:np.ndarray) -> np.ndarray:
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    # Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])

def correlate(org:np.ndarray, comp:np.ndarray=None) -> None:
    """Main function of the program"""
    d = 10
    N = 5000
    print("Correlation with self:")
    for i in range(d):
        num = np.sum(org['org'] * org['comp'][i])/N
        den = np.std(org['org']) * np.std(org['comp'][i])
        print(f"{num / den}")
    
    if comp:
        print("Correlation with comparand:")
        for i in range(d):
            num = np.sum(org['org'] * comp['comp'][i])/N
            den = np.std(org['org']) * np.std(comp['comp'][i])
            print(f"Cy{i + 1}: {num / den}")
    

def svd(filename:str, ss:int=5000) -> any:
    """Loads audio file and decomposes it"""
    try:
        y, sr = librosa.load(filename, sr=44100)
        y_tran = y[0 : ss]
        t_tran = np.linspace(0, ss - 1, ss) / sr
        L = 2000 # The window length.
        d = 10 # No of SVD components
        K = ss - L + 1 # The number of columns in the trajectory matrix.
        audio_data = np.column_stack([y_tran[i:i+L] for i in range(0,K)]) # Trajectory Matrix
        U, Sigma, V = np.linalg.svd(audio_data, full_matrices=False)
        V = V.T
        X_elem = np.array( [Sigma[i] * np.outer(U[:,i], V[:,i]) for i in range(0,d)] )
        return {'org':y_tran, 'comp':[X_to_TS(X_elem[i]) for i in range(d)]}
    except Exception as E:
        return str(E)


def main() -> None:
    """Main function of the program"""
    try:
        arg_dict = {}
        args, opts = getopt.getopt(sys.argv[1:],'ho:c:s:')
        for arg, opt in args:
            arg_dict[arg] = opt
    except:
        print('Invalid command. Use -h for help.')
        sys.exit(2)
    
    if '-h' in arg_dict:
        print("Usage: svd.py <original>.wav <compare:optional>.wav \
              <sample size:optional>")
        sys.exit()

    sample_size = arg_dict['-s'] if '-s' in arg_dict else 5000

    thread1 = ThreadWithReturn(target=svd, 
        args=[arg_dict['-c'], sample_size]) if '-c' in arg_dict else None
    if thread1:thread1.start()

    original = svd(arg_dict['-o'], sample_size)
    comparand = None
    if thread1:
        thread1.join()
        comparand = thread1.result

    if type(original) == str or type(comparand) == str:
        print(str)
        sys.exit()

    
    correlate(original, comparand)


if __name__ == "__main__":
    main()