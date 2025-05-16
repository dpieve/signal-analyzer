import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, freqz, lfilter
from enum import Enum

class Mode(Enum):
    LOAD_JSON = 'load'
    GENERATE  = 'generate'

FILTER     = True
MODE       = Mode.GENERATE
JSON_PATH  = 'signal.json'
OUTPUT_PNG = 'signal_analysis.png'

def load_signal(path):
    """Load time & amplitude arrays from JSON file, or exit if not found."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{path}'.")
        sys.exit(1)
    
    t = np.array(data['time'])
    x = np.array(data['amplitude'])
    
    if t.size != x.size:
        n = min(t.size, x.size)
        print(f"Warning: time/amplitude length mismatch ({t.size} vs {x.size}), trimming to {n} samples.")
        t = t[:n]
        x = x[:n]

    return t, x, (1.0 / (t[1] - t[0]))

def generate_signal(fs=1000, duration=1.0):
    """
    Generate a wave.
    """
    t = np.arange(0, duration, 1/fs)
    clean = np.sin(2 * np.pi * 10 * t)         # 10 Hz useful signal
    noise = 0.5 * np.sin(2 * np.pi * 60 * t)   # 60 Hz noise
    return t, clean + noise, (1.0 / (t[1] - t[0]))

def compute_fft(x, fs):
    N = len(x)
    X = np.fft.fft(x)
    X_mag = np.abs(X) / N
    freqs = np.fft.fftfreq(N, d=1/fs)
    idx = freqs >= 0
    return freqs[idx], X_mag[idx]

def filter_signal(x, fs, lowcut=58, highcut=62, order=4):
    """
    Apply a filter.
    Returns filtered_signal and b, a filter coefficients.
    """
    nyq = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq

    if not (0 < low < high < 1):
        print(f"Error: cannot design a {lowcut}–{highcut} Hz stopband at fs={fs} Hz "
              f"(normalized low={low:.2f}, high={high:.2f}).")
        sys.exit(1)

    b, a = butter(order, [low, high], btype='bandstop', analog=False)
    x_filt = lfilter(b, a, x)
    return x_filt, b, a

def plot_and_save(t, x, fs, freqs, X_mag, x_filt, b, a, out_png):
    w, h = freqz(b, a, worN=8000)
    phase = np.unwrap(np.angle(h))

    plt.figure(figsize=(10, 14))

    # 1) Raw signal (time-domain), now showing mode in title
    plt.subplot(5, 1, 1)
    plt.plot(t, x, marker='o', markersize=3)
    mode_label = MODE.value.upper()  # → 'GENERATE' or 'LOAD'
    plt.title(f'Original Time-Domain Signal ({mode_label})')
    plt.xlabel('Time [s]'); plt.ylabel('Amplitude'); plt.grid(True)

    # 2) FFT magnitude
    plt.subplot(5, 1, 2)
    plt.stem(freqs, X_mag, basefmt=" ")
    plt.title('FFT Magnitude Spectrum')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude'); plt.grid(True)

    # 3) Filter magnitude response
    plt.subplot(5, 1, 3)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h))
    plt.title(f'Filter Magnitude Response ({FILTER})')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Gain'); plt.grid(True)

    # 4) Filter phase response
    plt.subplot(5, 1, 4)
    plt.plot(0.5 * fs * w / np.pi, phase)
    plt.title(f'Filter Phase Response ({FILTER})')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Phase [rad]'); plt.grid(True)

    # 5) Filtered signal (time-domain)
    plt.subplot(5, 1, 5)
    plt.plot(t, x_filt, marker='x', markersize=2)
    plt.title('Filtered Time-Domain Signal')
    plt.xlabel('Time [s]'); plt.ylabel('Amplitude'); plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")

def main():
    if MODE is Mode.GENERATE:
        t, x, fs = generate_signal()
    else:
        t, x, fs = load_signal(JSON_PATH)

    freqs, X_mag = compute_fft(x, fs)

    if FILTER:
        x_filt, b, a = filter_signal(x, fs)
    else:
        # No filtering: use identity filter (y = x)
        x_filt = x
        b = np.array([1.0])
        a = np.array([1.0])

    plot_and_save(t, x, fs, freqs, X_mag, x_filt, b, a, OUTPUT_PNG)

if __name__ == '__main__':
    main()
