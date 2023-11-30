import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse


def plot_response(f, Fs, BW, p):
    r = 1 - (BW / Fs) * np.pi
    theta = (f * 2 * np.pi) / Fs
    A = 1 / (2j * r * np.sin(theta))

    # Time domain
    n = np.arange(0, 100)  # Adjust the range as needed
    h_n = A * ((r * np.exp(1j * theta))**n - (r * np.exp(-1j * theta))**n)

    #plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(n, np.real(h_n))
    plt.axhline(0)
    plt.title('Impulse Response (h(n))')
    plt.xlabel('n')
    plt.ylabel('Amplitude')

    # Frequency domain
    w, H_f = signal.freqz([0, 0, 1],  [1, -2 * r * np.cos(theta), r**2])

    plt.subplot(2, 1, 2)
    plt.plot(w, np.abs(H_f))
    plt.title('Frequency Response |H(f)|')
    plt.xlabel('Frequency (radians/sample)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.savefig(p)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot impulse and frequency response.')
    parser.add_argument('-f', type=float, help='Formant frequency in Hz', required=True)
    parser.add_argument('-Fs', type=float, help='Sampling frequency in Hz', required=True)
    parser.add_argument('-BW', type=float, help='Formant bandwidth in dB', required=True)

    args = parser.parse_args()
    f = args.f
    Fs = args.Fs
    BW = args.BW

    plot_response(f, Fs, BW, f'{int(f)}.png')