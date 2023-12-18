""" Changelog:
 SSA.py is my original code
 Dr. Md. Mahbub Hasan correction:
 SSA_update.py.
 Fix the singular value pair shift problem.
 Include trajectory matrix instead of toeplitz matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import linalg

def impulse_response(f, Fs, BW, len):
    r = 1 - (BW / Fs) * np.pi
    theta = (f * 2 * np.pi) / Fs
    A = 1 / (2j * r * np.sin(theta))

    # Calculate inverse z transform
    n = np.arange(0, len)  # Adjust the range as needed
    h_n = A * ((r * np.exp(1j * theta))**n - (r * np.exp(-1j * theta))**n)

    return n, np.real(h_n)

def main(args) -> None:
    n, h_n = impulse_response(args.formant_frequency, args.sampling_frequency, args.bandwidth, args.ts_length)

    # Convolve h(n) with Gaussian noise g(n)
    g_n = np.random.normal(0, 1, len(n))
    s_n = np.convolve(h_n, g_n, mode='same')

    # Singular Spectrum Analysis
    L = args.window_length
    N = args.ts_length
    K = N - L + 1 # The number of columns in the trajectory matrix.
    # Create the trajectory matrix by pulling the relevant subseries of F, and stacking them as columns.
    X = np.column_stack([s_n[i:i+L] for i in range(0,K)])
    d = np.linalg.matrix_rank(X) # The intrinsic dimensionality of the trajectory space.
    U, S, V = np.linalg.svd(X)

    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(n, h_n)
    plt.title('Impulse Response (h(n))')
    plt.xlabel('n')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 2)
    plt.plot(g_n)
    plt.title('Gaussian Noise (g(n))')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 3)
    plt.plot(s_n)
    plt.title('Convolution Result (s(n))')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 4)
    plt.plot(S[:20], 'o-')
    plt.title('Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')

    plt.tight_layout()
    plt.savefig(f'FF_{int(args.formant_frequency)}BW_{int(args.bandwidth)}.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Convolution, and SVD.')
    parser.add_argument('-f', dest='formant_frequency', type=float, help='Formant frequency in Hz', required=True)
    parser.add_argument('-Fs', dest='sampling_frequency', type=float, help='Sampling frequency in Hz', required=True)
    parser.add_argument('-BW', dest='bandwidth', type=float, help='Formant bandwidth in dB', required=True)
    parser.add_argument('-len', dest='ts_length', type=float, help='Total number of samples', default=2000)
    parser.add_argument('-wl', dest='window_length', type=float, help='Window Length', default=200)

    args = parser.parse_args()
    main(args)
