import numpy as np                        # массивы
from scipy import signal                  # STFT и ISTFT
from scipy.io import wavfile              # чтение и запись WAV

fs, data = wavfile.read("sounds/chord.wav")  # полезный сигнал
clean = np.concatenate([np.zeros(fs // 2),  # 0,5 с тишины,
                        0.5 * data / 32767.0])  # затем аккорд
rng = np.random.default_rng(8)            # генератор с seed
x = clean + 0.05 * rng.standard_normal(clean.size)  # + шум

nper, hop = 512, 128                      # кадр 32 мс, шаг 8 мс
f, t, Z = signal.stft(x, fs, nperseg=nper,  # прямое STFT
                      noverlap=nper - hop)  # перекрытие 75 %
mag, phase = np.abs(Z), np.angle(Z)       # модуль и фаза
quiet = t < 0.4                           # кадры без сигнала
noise = mag[:, quiet].mean(axis=1, keepdims=True)  # оценка шума
alpha, beta = 2.0, 0.02                   # перевычитание и порог
clean_mag = np.maximum(mag - alpha * noise,  # вычитание шума
                       beta * mag)        # спектральный «пол»
_, y = signal.istft(clean_mag * np.exp(1j * phase), fs,  # обратное
                    nperseg=nper, noverlap=nper - hop)  # STFT
y = y[:x.size]                            # выравниваем длину


def snr_db(ref, test):                    # SNR относительно эталона
    err = test - ref                      # остаточная ошибка
    return 10 * np.log10(np.sum(ref**2) / np.sum(err**2))  # дБ


print(f"до обработки:    SNR = {snr_db(clean, x):5.1f} дБ")  # исходный
print(f"после обработки: SNR = {snr_db(clean, y):5.1f} дБ")  # итог
wavfile.write("sounds/specsub.wav", fs,   # запись результата
              y.astype(np.float32))       # в формате float32
