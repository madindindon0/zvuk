"""Общие функции для повтора примеров на реальной 24-битной записи."""
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load_real_wav():
    """Читает внешний WAV и нормирует 24-битные отсчёты на 2**23."""
    candidates = (Path("../audio/metronome_120bpm_16beats.wav"),  # путь из задания
                  Path("audio/metronome_120bpm_16beats.wav"))    # фактический путь
    path = next((p for p in candidates if p.exists()), None)     # первый существующий
    if path is None:                                            # защита
        raise FileNotFoundError("metronome_120bpm_16beats.wav не найден")

    fs, data = wavfile.read(path)                               # частоты и отсчёты
    print("файл:", path)                                        # использованный путь
    print("частота дискретизации:", fs, "Гц")                   # 48000 Гц
    print("тип отсчётов:", data.dtype)                          # int32
    print("форма массива:", data.shape)                         # (384000,)

    if data.dtype == np.int32:                                  # 24 бита в контейнере int32
        # scipy.io.wavfile хранит 24-битные отсчёты в int32,
        # сдвинутыми влево на 8 бит: сначала сдвиг, затем делитель 2**23
        data = data >> 8                                        # вернуть 24-битные значения
        div = float(2 ** 23)                                    # делитель 2^23, не int32 max
    elif np.issubdtype(data.dtype, np.integer):                # другой целочисленный PCM
        div = float(np.iinfo(data.dtype).max)                   # штатный делитель
    else:                                                       # уже вещественный формат
        div = 1.0                                               # нормировка не нужна

    x = data.astype(np.float64) / div                           # приведение к -1...1
    if x.ndim == 2:                                             # стереозапись
        x = x.mean(axis=1)                                      # сведение в моно
    print("делитель нормировки:", div)                          # 8388608.0 для 24 бит
    return fs, x, path
