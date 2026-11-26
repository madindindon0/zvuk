import numpy as np                        # массивы
import matplotlib                         # настройка headless-окружения
matplotlib.use("Agg")                     # графики без GUI
matplotlib.rcParams["font.family"] = "Noto Sans"  # кириллические подписи
import matplotlib.pyplot as plt           # графики
from pathlib import Path                  # работа с путями
from scipy import signal                  # спектрограмма
from scipy.io import wavfile              # чтение WAV

Path("results/examples").mkdir(parents=True, exist_ok=True)  # каталог для графиков

fs, data = wavfile.read("sounds/chirp.wav")  # ЛЧМ-сигнал
x = data / 32768.0                        # нормированный сигнал
click = np.zeros_like(x)                  # «щелчки» для сравнения
click[[8000, 8400, 24000]] = 1.0          # три коротких импульса
x = x + click                             # сумма сигналов

fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # две панели
for a, nper in zip(ax, (128, 1024)):      # короткое и длинное окно
    f, t, S = signal.spectrogram(         # кратковременный спектр
        x, fs, window="hann",             # окно Ханна
        nperseg=nper, noverlap=nper * 3 // 4)  # перекрытие 75 %
    S_db = 10 * np.log10(S + 1e-12)       # мощность в дБ
    a.pcolormesh(t, f, S_db, cmap="gray_r",  # полутоновая карта
                 vmin=S_db.max() - 80, shading="auto")  # 80 дБ
    a.set_title(f"окно {nper} отсчётов")  # заголовок панели
    a.set_xlabel("Время, с")              # подпись оси
    a.set_ylabel("Частота, Гц")           # подпись оси
    hop = nper - nper * 3 // 4            # шаг между кадрами
    df, dt = fs / nper, 1e3 * hop / fs    # разрешения по f и t
    print(f"окно {nper}: df={df:.1f} Гц, dt={dt:.1f} мс")  # вывод
fig.tight_layout()                        # компоновка
fig.savefig("results/examples/ex05_spectrogram.png")  # сохранение вместо вывода на экран
