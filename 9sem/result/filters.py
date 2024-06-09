import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm
from main import spectrogram

def butter_filter(sample_rate, data, output_dir):
    b, a = signal.butter(10, 0.1, btype='lowpass')
    filtered_signal = signal.filtfilt(b, a, data)
    wavfile.write(os.path.join(output_dir, 'butter.wav'), sample_rate, filtered_signal.astype(np.int16))
    spectrogram(filtered_signal, sample_rate, os.path.join(output_dir, 'butter.png'))


def savgol_filter(sample_rate, data, output_dir):
    denoised_savgol = signal.savgol_filter(data, 75, 5)
    wavfile.write(os.path.join(output_dir, 'savgol.wav'), sample_rate, denoised_savgol.astype(np.int16))
    spectrogram(denoised_savgol, sample_rate, os.path.join(output_dir, 'savgol.png'))
