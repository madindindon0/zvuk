"""Повтор примера 3.4 (амплитудный спектр) на реальной записи."""
import matplotlib                         # настройка headless-окружения
matplotlib.use("Agg")                     # графики без GUI
matplotlib.rcParams["font.family"] = "Noto Sans"  # кириллические подписи
import matplotlib.pyplot as plt           # графики
import numpy as np                        # массивы и БПФ
from pathlib import Path                  # работа с путями
from scipy import signal                  # окна и поиск пиков

from real_signal_common import load_real_wav  # общая загрузка 24-битного WAV

Path("results/real_signal").mkdir(parents=True, exist_ok=True)  # каталог результатов

fs, x, path = load_real_wav()             # чтение и нормировка на 2^23
N = 2048                                  # длина фрагмента
frag = x[int(0.5 * fs):int(0.5 * fs) + N]  # 128 мс с ударом метронома
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
    pk, _ = signal.find_peaks(db, height=-80, distance=4)  # пики (щелчки широкополосные)
    order = np.argsort(mag[pk])[::-1][:5]  # пять самых сильных
    print(f"{name:8s}:", np.round(freqs[pk][order], 1),  # частоты
          np.round(mag[pk][order], 5))    # и амплитуды пиков
    ax.plot(freqs, db, style, lw=1, label=name)  # кривая
ax.set_xlim(0, 8000)                      # область анализа
ax.set_ylim(-100, 0)                      # диапазон уровней
ax.set_xlabel("Частота, Гц")              # подпись оси
ax.set_ylabel("Уровень, дБ")              # подпись оси
ax.legend()                               # легенда
fig.tight_layout()                        # компоновка
fig.savefig("results/real_signal/real_spectrum.png")  # сохранение графика
