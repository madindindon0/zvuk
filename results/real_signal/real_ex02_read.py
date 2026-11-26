"""Повтор примера 3.2 (чтение и уровни) на реальной записи."""
import matplotlib                         # настройка headless-окружения
matplotlib.use("Agg")                     # графики без GUI
matplotlib.rcParams["font.family"] = "Noto Sans"  # кириллические подписи
import matplotlib.pyplot as plt           # графики
import numpy as np                        # массивы и вычисления
from pathlib import Path                  # работа с путями

from real_signal_common import load_real_wav  # общая загрузка 24-битного WAV

Path("results/real_signal").mkdir(parents=True, exist_ok=True)  # каталог результатов

fs, x, path = load_real_wav()             # чтение и нормировка на 2^23
print("длительность:", x.size / fs, "с")  # 8.0 с
print("число каналов: 1")                  # моно

peak = np.max(np.abs(x))                  # пиковое значение
rms = np.sqrt(np.mean(x ** 2))            # среднеквадратичное
print(f"пик: {peak:.3f} ({20*np.log10(peak):.1f} дБFS)")  # уровень
print(f"RMS: {rms:.3f} ({20*np.log10(rms):.1f} дБFS)")  # уровень
print(f"пик-фактор: {peak / rms:.2f}")    # отношение пик/RMS

t = np.arange(x.size) / fs                # ось времени, с
fig, ax = plt.subplots(2, 1, figsize=(9, 5))  # две панели
ax[0].plot(t, x, color="k", lw=0.5)       # вся осциллограмма
ax[0].set_xlabel("Время, с")              # подпись оси
ax[0].set_ylabel("Амплитуда")             # подпись оси
n = int(0.02 * fs)                        # отсчётов в 20 мс
ax[1].plot(t[:n] * 1e3, x[:n], "k.-", lw=0.8)  # фрагмент 20 мс
ax[1].set_xlabel("Время, мс")             # подпись оси
ax[1].set_ylabel("Амплитуда")             # подпись оси
fig.tight_layout()                        # компоновка
fig.savefig("results/real_signal/real_levels.png")  # сохранение графика
