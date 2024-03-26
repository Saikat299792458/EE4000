import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import linalg

def impulse_response(f, BW, Fs, len):
    r = 1 - (BW / Fs) * np.pi
    theta = (f * 2 * np.pi) / Fs
    A = 1 / (2j * r) # We need to multiply the denominator with np.sin(theta)
    # But that causes a variable magnitude which is dependednt on formant frequency.

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
    parser.add_argument('-wl', type=float, help='Window Length', default=240)
    parser.add_argument('-out', type=str, help='Output figure name', default="output.png")
    args = parser.parse_args()

    # User mistake handling
    if len(args.BW) != args.n:
        parser.error(f"Expected {args.n} Bandwidth values, but found {len(args.BW)}")
    
    if len(args.f) != args.n:
        parser.error(f"Expected {args.n} Formant frequency values, but found {len(args.f)}")
    
    # Composite signal generation
    composite = np.zeros(args.len)
    plt.figure(figsize=(10,7))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.subplot(3, 1, 1)
    np.random.seed(0)
    for i in range(args.n):
        n, h_n = impulse_response(args.f[i], args.BW[i], args.Fs, args.len)
        plt.plot(n, h_n, label=f"{args.f[i]}Hz")
        # Convolve h(n) with Gaussian noise g(n)
        g_n = np.random.normal(0, 1, len(n)) # Calculate variance
        s_n = np.convolve(h_n, g_n, mode='same')
        # Add with composite signal
        composite += s_n
    plt.legend()
    plt.title("Formant Impulse Response")
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    composite /= args.n # Normalization
    plt.subplot(3, 1, 2)
    plt.plot(composite)
    plt.title("Convolved with noise")
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    # Singular Spectrum Analysis
    L = args.wl
    N = args.len
    K = N - L + 1 # The number of columns in the trajectory matrix.
    # Create the trajectory matrix by pulling the relevant subseries of F, and stacking them as columns.
    X = np.column_stack([composite[i:i+L] for i in range(0,K)])
    U, S, V = np.linalg.svd(X)
    S = [round(i, 2) for i in S]
    MPP = None
    print("Pair No.\tSV1\t\tSV2\t\tIntraPair Deviation\tInterPair Deviation")
    for i in range(args.n):
        IPD = S[i*2] - S[i*2+1]
        MP1 = (S[i*2] + S[i*2+1]) / 2
        MPD = round(MPP - MP1, 2) if MPP else None
        print(f"{i+1}\t\t{S[i*2]}\t\t{S[i*2+1]}\t\t{round(IPD,2)}\t\t\t{MPD}")
        MPP = MP1
    print(f"Threshold SVs:\t{S[args.n*2]}\t\t{S[args.n*2+1]}") # Or do I calculate midpoint of the first insignificant pair?
    plt.subplot(3, 1, 3)
    plt.plot(S[:20], 'o-')
    plt.title('Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')

    plt.tight_layout()
    plt.savefig(args.out)
    plt.show()

if __name__ == "__main__":
    main()