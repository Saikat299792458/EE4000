import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import linalg

def impulse_response(f, Fs, BW):
    r = 1 - (BW / Fs) * np.pi
    theta = (f * 2 * np.pi) / Fs
    A = 1 / (2j * r * np.sin(theta))

    # Calculate inverse z transform
    n = np.arange(0, 200)  # Adjust the range as needed
    h_n = A * ((r * np.exp(1j * theta))**n - (r * np.exp(-1j * theta))**n)

    return n, np.real(h_n)

def main(args) -> None:
    n, h_n = impulse_response(args.formant_frequency, args.sampling_frequency, args.bandwidth)

    # Convolve h(n) with Gaussian noise g(n)
    g_n = np.random.normal(0, 1, len(n))
    s_n = np.convolve(h_n, g_n, mode='full')

    # Create a Toeplitz matrix
    s_n_2d = linalg.toeplitz(s_n)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(s_n_2d, full_matrices=False)

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
    plt.savefig(f'{int(args.formant_frequency)}.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Convolution, and SVD.')
    parser.add_argument('-f', dest='formant_frequency', type=float, help='Formant frequency in Hz', required=True)
    parser.add_argument('-Fs', dest='sampling_frequency', type=float, help='Sampling frequency in Hz', required=True)
    parser.add_argument('-BW', dest='bandwidth', type=float, help='Formant bandwidth in dB', required=True)

    args = parser.parse_args()
    main(args)
