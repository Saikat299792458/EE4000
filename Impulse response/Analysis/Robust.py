import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import linalg

def impulse_response(f, BW, Fs, len):
    r = 1 - (BW / Fs) * np.pi
    theta = (f * 2 * np.pi) / Fs
    A = 1 / (2j * r * np.sin(theta))

    # Calculate inverse z transform
    n = np.arange(0, len)  # Adjust the range as needed
    h_n = A * ((r * np.exp(1j * theta))**n - (r * np.exp(-1j * theta))**n)

    return n, np.real(h_n)


def main() -> None:
    parser = argparse.ArgumentParser(description='Perform Convolution, and SVD.')
    parser.add_argument('-Fs', type=float, help='Sampling frequency in Hz', required=True)
    parser.add_argument('-n', type=int, help='Number of Formant Frequencies', required=True)
    parser.add_argument('-f', nargs='+', type=float, help='Formant frequency in Hz', required=True)
    parser.add_argument('-BW', nargs='+', type=float, help='Formant bandwidth in dB', required=True)
    parser.add_argument('-len', type=float, help='Total number of samples', default=2000)
    parser.add_argument('-wl', type=float, help='Window Length', default=200)
    args = parser.parse_args()

    # User mistake handling
    if len(args.BW) != args.n:
        parser.error(f"Expected {args.n} Bandwidth values, but found {len(args.BW)}")
    
    if len(args.f) != args.n:
        parser.error(f"Expected {args.n} Formant frequency values, but found {len(args.f)}")
    
    # Composite signal generation
    composite = np.zeros(args.len)
    plt.figure(figsize=(10,7))
    plt.subplot(3, 1, 1)
    for i in range(args.n):
        n, h_n = impulse_response(args.f[i], args.BW[i], args.Fs, args.len)
        plt.plot(n, h_n, label=f"{args.f[i]}Hz")
        # Convolve h(n) with Gaussian noise g(n)
        g_n = np.random.normal(0, 1, len(n))
        s_n = np.convolve(h_n, g_n, mode='same')
        # Add with composite signal
        composite += s_n
    plt.legend()
    plt.title("Formant frequencies")
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    composite /= args.n # Normalization
    plt.subplot(3, 1, 2)
    plt.plot(composite)
    plt.title("Composite signal")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Singular Spectrum Analysis
    L = args.wl
    N = args.len
    K = N - L + 1 # The number of columns in the trajectory matrix.
    # Create the trajectory matrix by pulling the relevant subseries of F, and stacking them as columns.
    X = np.column_stack([s_n[i:i+L] for i in range(0,K)])
    d = np.linalg.matrix_rank(X) # The intrinsic dimensionality of the trajectory space.
    U, S, V = np.linalg.svd(X)
    plt.subplot(3, 1, 3)
    plt.plot(S[:20], 'o-')
    plt.title('Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')

    plt.tight_layout()
    #plt.savefig(f'FF_{int(args.formant_frequency)}BW_{int(args.bandwidth)}.png')
    plt.show()

if __name__ == "__main__":
    main()