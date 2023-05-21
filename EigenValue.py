import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rcParams['figure.figsize'] = (10, 5)

def main() -> None:
    "Main function of the program"
    M = 30;    # window length = embedding dimension
    N = 11000;   # length of generated time series

    if len(sys.argv) != 2 and not sys.argv[1].endswith(".wav"):
        print("Invalid parameter")
        sys.exit(2)

    X, sr = librosa.load(sys.argv[1], sr=N)
    t=np.linspace(0, (X.shape[0]-1), X.shape[0]) / sr
    print(sr)

    Y = np.zeros((N-M+1,M));
    for m in range(M):
        Y[:, m] = X[m:N-M+m+1];
    C = Y.T @ Y / (N-M+1);


    LAMBDA,RHO = np.linalg.eig(C);
    SI = np.flip(np.argsort(LAMBDA))
    LAMBDA = LAMBDA[SI]
    RHO = RHO[:, SI];


    plt.title('Eigenvalues LAMBDA')
    plt.plot(LAMBDA,'o-');
    plt.show()


if __name__ == "__main__":
    main()