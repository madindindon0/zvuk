import numpy as np                        # массивы и вычисления
from pathlib import Path                  # работа с путями
from scipy.io import wavfile              # чтение и запись WAV

fs = 16000                                # частота дискретизации, Гц
dur = 2.0                                 # длительность, с
t = np.arange(int(fs * dur)) / fs         # моменты отсчётов, с
Path("sounds").mkdir(exist_ok=True)       # каталог для файлов

tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # тон «ля» первой октавы
chord = sum(np.sin(2 * np.pi * f * t)     # сумма трёх тонов
            for f in (262, 330, 392)) / 3  # до-мажор, нормировка
f0, f1 = 100, 4000                        # границы ЛЧМ-сигнала, Гц
k = (f1 - f0) / dur                       # скорость роста частоты
chirp = 0.5 * np.sin(2 * np.pi * (f0 * t + k * t**2 / 2))  # ЛЧМ
rng = np.random.default_rng(19)           # генератор с seed
noise = 0.1 * rng.standard_normal(t.size) # белый гауссов шум
hum = 0.2 * np.sin(2 * np.pi * 50 * t)    # сетевая помеха 50 Гц
noisy = tone + noise + hum                # тон с помехами


def to_int16(x):                          # перевод в 16-битный PCM
    x = np.clip(x, -1.0, 1.0)             # защита от переполнения
    return np.round(x * 32767).astype(np.int16)  # масштаб и тип


signals = {"tone": tone, "chord": chord,  # словарь «имя - сигнал»
           "chirp": chirp, "noisy": noisy}  # четыре тестовых файла
for name, x in signals.items():           # обходим сигналы
    wavfile.write(f"sounds/{name}.wav", fs, to_int16(x))  # запись
    peak = np.max(np.abs(x))              # пиковое значение
    print(f"{name:6s}: {x.size} отсч., пик {peak:.3f}")  # отчёт
