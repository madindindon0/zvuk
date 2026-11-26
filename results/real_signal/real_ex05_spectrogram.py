"""Повтор примера 3.5 (спектрограмма) на реальной записи."""
import matplotlib                         # настройка headless-окружения
matplotlib.use("Agg")                     # графики без GUI
matplotlib.rcParams["font.family"] = "Noto Sans"  # кириллические подписи
import matplotlib.pyplot as plt           # графики
import numpy as np                        # массивы и логарифм
from pathlib import Path                  # работа с путями
from scipy import signal                  # спектрограмма

from real_signal_common import load_real_wav  # общая загрузка 24-битного WAV

Path("results/real_signal").mkdir(parents=True, exist_ok=True)  # каталог результатов

fs, x, path = load_real_wav()             # чтение и нормировка на 2^23

fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # две панели
for a, nper in zip(ax, (256, 1024)):      # короткое и длинное окно
    f, t, S = signal.spectrogram(         # кратковременный спектр
        x, fs, window="hann",             # окно Ханна
        nperseg=nper, noverlap=nper * 3 // 4)  # перекрытие 75 %
    S_db = 10 * np.log10(S + 1e-12)       # мощность в дБ
    a.pcolormesh(t, f, S_db, cmap="gray_r",  # полутоновая карта
                 vmin=S_db.max() - 80, shading="auto")  # 80 дБ
    a.set_title(f"окно {nper} отсчётов")  # заголовок панели
    a.set_xlabel("Время, с")              # подпись оси
    a.set_ylabel("Частота, Гц")           # подпись оси
    a.set_ylim(0, 8000)                   # область анализа
    hop = nper - nper * 3 // 4            # шаг между кадрами
    df, dt = fs / nper, 1e3 * hop / fs    # разрешения по f и t
    print(f"окно {nper}: df={df:.1f} Гц, dt={dt:.1f} мс")  # вывод
fig.tight_layout()                        # компоновка
fig.savefig("results/real_signal/real_spectrogram.png")  # сохранение графика
