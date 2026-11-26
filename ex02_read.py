import numpy as np                        # массивы и вычисления
import matplotlib                         # настройка headless-окружения
matplotlib.use("Agg")                     # графики без GUI
matplotlib.rcParams["font.family"] = "Noto Sans"  # кириллические подписи
import matplotlib.pyplot as plt           # графики
from pathlib import Path                  # работа с путями
from scipy.io import wavfile              # чтение WAV

Path("results/examples").mkdir(parents=True, exist_ok=True)  # каталог для графиков

path = "sounds/chord.wav"                 # путь к файлу
fs, data = wavfile.read(path)             # частота и отсчёты
print("частота дискретизации:", fs, "Гц") # 16000 Гц
print("тип отсчётов:", data.dtype)        # int16
print("форма массива:", data.shape)       # (32000,) - моно
ch = 1 if data.ndim == 1 else data.shape[1]  # число каналов
print("число каналов:", ch)               # 1
print("длительность:", data.shape[0] / fs, "с")  # 2.0 с

x = data.astype(np.float64)               # в вещественный тип
if np.issubdtype(data.dtype, np.integer): # целочисленный PCM
    x /= np.iinfo(data.dtype).max         # к диапазону -1...1
if x.ndim == 2:                           # стереозапись
    x = x.mean(axis=1)                    # сведение в моно

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
fig.savefig("results/examples/ex02_read.png")  # сохранение вместо вывода на экран
