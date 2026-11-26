import numpy as np                        # массивы
from scipy import signal                  # спектрограмма
from scipy.fft import dct                 # дискретное косинусное пр.
from scipy.io import wavfile              # чтение WAV

fs, x = wavfile.read("sounds/voice.wav")  # сигнал из примера 3.10
nfft, hop, n_mels = 512, 160, 40          # БПФ, шаг 10 мс, полос


def hz2mel(f):                            # герцы -> мелы
    return 2595 * np.log10(1 + f / 700)   # формула O'Shaughnessy


def mel2hz(m):                            # мелы -> герцы
    return 700 * (10 ** (m / 2595) - 1)   # обратная формула

mels = np.linspace(0, hz2mel(fs / 2), n_mels + 2)  # равномерно в мел
bins = np.floor((nfft + 1) * mel2hz(mels) / fs).astype(int)  # бины
fb = np.zeros((n_mels, nfft // 2 + 1))    # матрица фильтров
for m in range(1, n_mels + 1):            # треугольные фильтры
    l, c, r = bins[m - 1], bins[m], bins[m + 1]  # левая/центр/правая
    fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1)  # подъём
    fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)  # спад

f, t, S = signal.spectrogram(x, fs, window="hamming",  # мощность
                             nperseg=400, noverlap=400 - hop,  # 25 мс
                             nfft=nfft, mode="psd")  # спектр мощности
mel_spec = fb @ S                         # мел-спектрограмма
log_mel = np.log(mel_spec + 1e-10)        # логарифм энергии
mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:13]  # 13 коэф.

print("спектрограмма:", S.shape)          # (257, кадры)
print("мел-спектрограмма:", mel_spec.shape)  # (40, кадры)
print("MFCC:", mfcc.shape)                # (13, кадры)
print("среднее MFCC:", np.round(mfcc.mean(axis=1)[:5], 2))  # 5 первых
