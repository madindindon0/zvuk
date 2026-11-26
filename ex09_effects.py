import numpy as np                        # массивы
from scipy import signal                  # быстрая свёртка
from scipy.io import wavfile              # чтение и запись WAV

fs, data = wavfile.read("sounds/chord.wav")  # исходный аккорд
x = data / 32767.0                        # нормированный сигнал


def level_db(x):                          # уровень RMS в дБFS
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)  # дБFS


def normalize_rms(x, target_db=-20.0):    # нормализация по RMS
    gain = 10 ** ((target_db - level_db(x)) / 20)  # коэффициент
    return x * gain                       # усиление сигнала

y = normalize_rms(x, -20.0)               # целевой уровень -20 дБFS
print(f"уровень: {level_db(x):.1f} -> {level_db(y):.1f} дБFS")  # итог

n = int(0.3 * fs)                         # длительность фейда 0,3 с
env = np.ones_like(y)                     # огибающая усиления
env[:n] = np.linspace(0, 1, n)            # плавное нарастание
env[-n:] = np.linspace(1, 0, n)           # плавное затухание
y = y * env                               # применяем огибающую

h = np.zeros(int(0.6 * fs))               # импульсная характеристика
h[0] = 1.0                                # прямой сигнал
h[int(0.25 * fs)] = 0.5                   # первое эхо через 250 мс
h[int(0.50 * fs) - 1] = 0.25              # второе эхо через 500 мс
echo = signal.fftconvolve(y, h)           # свёртка = эхо

rng = np.random.default_rng(9)            # генератор с seed
tt = np.arange(int(1.2 * fs)) / fs        # длительность ИХ 1,2 с
ir = rng.standard_normal(tt.size) * np.exp(-tt / 0.2)  # хвост
ir[0] = 4.0                               # прямой звук сильнее
reverb = signal.fftconvolve(y, ir)        # «зал» по свёртке
reverb *= np.max(np.abs(y)) / np.max(np.abs(reverb))  # пик

print("длины:", y.size, echo.size, reverb.size)  # рост длины
for name, s in (("echo", echo), ("reverb", reverb)):  # запись
    wavfile.write(f"sounds/{name}.wav", fs, s.astype(np.float32))  # WAV
