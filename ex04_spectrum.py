import numpy as np                        # массивы и БПФ
import matplotlib                         # настройка headless-окружения
matplotlib.use("Agg")                     # графики без GUI
matplotlib.rcParams["font.family"] = "Noto Sans"  # кириллические подписи
import matplotlib.pyplot as plt           # графики
from pathlib import Path                  # работа с путями
from scipy import signal                  # окна и поиск пиков
from scipy.io import wavfile              # чтение WAV

Path("results/examples").mkdir(parents=True, exist_ok=True)  # каталог для графиков

fs, data = wavfile.read("sounds/chord.wav")  # аккорд 262/330/392 Гц
x = data / 32768.0                        # нормированный сигнал
N = 2048                                  # длина фрагмента
frag = x[:N]                              # первые 128 мс
freqs = np.fft.rfftfreq(N, d=1 / fs)      # частоты бинов, Гц
print(f"шаг по частоте: {fs / N:.2f} Гц") # разрешение ДПФ

fig, ax = plt.subplots(figsize=(9, 4))    # одна панель
styles = {"boxcar": "k:", "hann": "k-",   # стиль линии
          "blackman": "k--"}              # для каждого окна
for name, style in styles.items():        # обходим окна
    w = signal.get_window(name, N)        # оконная функция
    X = np.fft.rfft(frag * w)             # спектр окна сигнала
    mag = 2 * np.abs(X) / np.sum(w)       # амплитудный спектр
    db = 20 * np.log10(mag + 1e-12)       # в децибелах
    pk, _ = signal.find_peaks(db, height=-20, distance=4)  # пики
    print(f"{name:8s}:", np.round(freqs[pk], 1),  # частоты
          np.round(mag[pk], 3))           # и амплитуды пиков
    ax.plot(freqs, db, style, lw=1, label=name)  # кривая
ax.set_xlim(150, 500)                     # область аккорда
ax.set_ylim(-100, 0)                      # диапазон уровней
ax.set_xlabel("Частота, Гц")              # подпись оси
ax.set_ylabel("Уровень, дБ")              # подпись оси
ax.legend()                               # легенда
fig.savefig("results/examples/ex04_spectrum.png")  # сохранение вместо вывода на экран
