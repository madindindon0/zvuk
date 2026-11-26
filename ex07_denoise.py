import numpy as np                        # массивы
from scipy import signal                  # фильтры
from scipy.io import wavfile              # чтение и запись WAV

fs, data = wavfile.read("sounds/noisy.wav")  # тон + шум + 50 Гц
x = data / 32767.0                        # нормированный сигнал
t = np.arange(x.size) / fs                # ось времени
clean = 0.5 * np.sin(2 * np.pi * 440 * t) # эталон без помех


def snr_db(ref, test):                    # SNR относительно эталона
    err = test - ref                      # остаточная помеха
    return 10 * np.log10(np.sum(ref**2) / np.sum(err**2))  # дБ


def report(label, y, part=slice(None)):   # печать SNR этапа
    s = snr_db(clean[part], y[part])      # SNR на участке
    print(f"{label:15s} SNR = {s:5.1f} дБ")  # строка отчёта


report("исходный:", x)                    # до обработки
b, a = signal.iirnotch(50, Q=30, fs=fs)   # режекторный фильтр 50 Гц
y1 = signal.filtfilt(b, a, x)             # прямой и обратный проход
report("после режекции:", y1)             # подавлены 50 Гц
sos = signal.butter(4, [400, 480],        # полосовой фильтр
                    btype="bandpass", fs=fs, output="sos")  # 400...480
y2 = signal.sosfiltfilt(sos, y1)          # без фазового сдвига
report("после полосы:", y2)               # подавлен шум

edge = int(0.1 * fs)                      # переходные процессы
mid = slice(edge, -edge)                  # центральная часть
report("центр. часть:", y2, mid)          # без краёв

out = np.clip(y2, -1, 1)                  # защита от перегрузки
wavfile.write("sounds/denoised.wav", fs,  # запись результата
              (out * 32767).astype(np.int16))  # в 16-бит PCM
