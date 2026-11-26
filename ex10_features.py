import numpy as np                        # массивы
from scipy.io import wavfile              # запись WAV

fs = 16000                                # частота дискретизации
plan = [(0, .30), (120, .40), (0, .25),   # (F0, длительность):
        (180, .50), (0, .30), (150, .35), (0, .30)]  # 0 - пауза
rng = np.random.default_rng(10)           # генератор с seed
parts = []                                # фрагменты сигнала
for f0, d in plan:                        # синтез «слогов»
    t = np.arange(int(d * fs)) / fs       # ось времени фрагмента
    s = 0.005 * rng.standard_normal(t.size)  # фоновый шум
    if f0:                                # звонкий фрагмент
        for k in range(1, 20):            # 19 гармоник
            s += np.sin(2 * np.pi * k * f0 * t) / k  # спад 1/k
        s *= np.hanning(t.size) ** 0.3    # мягкая огибающая
    parts.append(s)                       # добавляем фрагмент
x = np.concatenate(parts)                 # весь сигнал
x /= np.max(np.abs(x)) * 1.25             # пик -2 дБFS
wavfile.write("sounds/voice.wav", fs, x.astype(np.float32))  # запись

flen, hop = int(0.025 * fs), int(0.010 * fs)  # кадр 25, шаг 10 мс
n = 1 + (x.size - flen) // hop            # число кадров
idx = np.arange(flen)[None, :] + hop * np.arange(n)[:, None]  # индексы
frames = x[idx] * np.hamming(flen)        # матрица кадров с окном
energy = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)  # дБ
sgn = np.sign(x[idx])                     # знаки отсчётов
zcr = np.mean(np.abs(np.diff(sgn, axis=1)) > 0, axis=1)  # ZCR

thr = energy.max() - 30                   # порог: -30 дБ от макс.
voiced = energy > thr                     # решение VAD по кадрам
edges = np.flatnonzero(np.diff(voiced.astype(int)))  # смены
print("порог энергии:", round(thr, 1), "дБ")  # значение порога
for a, b in zip(edges[::2] + 1, edges[1::2] + 1):  # сегменты
    f0s = []                              # оценки F0 в сегменте
    for fr in frames[a:b]:                # обходим кадры
        r = np.correlate(fr, fr, "full")[flen - 1:]  # АКФ
        lo, hi = fs // 400, fs // 60      # поиск F0 в 60...400 Гц
        lag = lo + np.argmax(r[lo:hi])    # положение максимума
        f0s.append(fs / lag)              # F0 = fs / задержка
    print(f"речь {a * hop / fs:5.2f}-{b * hop / fs:5.2f} с, "  # время
          f"F0 = {np.median(f0s):6.1f} Гц, "  # медиана F0
          f"ZCR = {zcr[a:b].mean():.3f}") # итог по сегменту
