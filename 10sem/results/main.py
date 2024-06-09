import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy.io import wavfile


def get_peaks(freq, t, spec):
    delta_t = int(0.1 * len(t))
    delta_freq = int(50 / (freq[1] - freq[0]))
    filtered = maximum_filter(spec, size=(delta_freq, delta_t))
    peaks_mask = (spec == filtered)
    peak_values = spec[peaks_mask]
    peak_frequencies = freq[peaks_mask.any(axis=1)]
    top_indices = np.argsort(peak_values)[-3:]
    top_frequencies = peak_frequencies[top_indices]
    return list(top_frequencies)


def spectrogram(samples, sample_rate, filepath):
    freq, t, spec = signal.spectrogram(samples, sample_rate, window=('hann'))
    spec = np.log10(spec + 1)
    plt.pcolormesh(t, freq, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())
    plt.ylabel('Frequncy (Hz)')
    plt.xlabel('Time (Seconds)')

    plt.savefig(filepath)

    return freq, t, spec


def get_max_tembr(filepath):
    data, sample_rate = librosa.load(filepath)
    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    f0 = librosa.piptrack(y=data, sr=sample_rate, S=chroma)[0]
    max_f0 = np.argmax(f0)
    return max_f0

def get_max_min(voice_path):
    y, sr = librosa.load(voice_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    frequencies = librosa.fft_frequencies(sr=sr)
    mean_spec = np.mean(D, axis=1)
    idx_min = np.argmax(mean_spec > -80)
    idx_max = len(mean_spec) - np.argmax(mean_spec[::-1] > -80) - 1
    min_freq = frequencies[idx_min]
    max_freq = frequencies[idx_max]

    return max_freq, min_freq

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    os.makedirs(output_path, exist_ok=True)
    input_path = os.path.join(current_dir, 'input')
    humiliations = ['voice_a', 'voice_gav', 'voice_i']
    humiliation_voice_paths = [
        (humiliation, os.path.join(input_path, f'{humiliation}.wav'))
        for humiliation in humiliations
    ]
    with open(os.path.join(output_path, 'res.txt'), 'w') as res_file:
        for humiliation, voice_path in humiliation_voice_paths:
            rate, samples = wavfile.read(voice_path)
            freq, t, spec = spectrogram(samples, rate, os.path.join(output_path, f'{humiliation}.png'))
            max_freq, min_freq = get_max_min(voice_path)

            res_file.write(f'{humiliation}:\n')
            res_file.write(f'\tMaximal Frequncy: {max_freq}\n')
            res_file.write(f'\tMinimnal Frequency: {min_freq}\n')
            res_file.write(f"\tThe most timbral-colored basic tone: {get_max_tembr(voice_path)}\n")
            if 'letter' in humiliation:
                res_file.write(f"\tThe three most powerful formants: {get_peaks(freq, t, spec)}\n")

if __name__ == "__main__":
    main()