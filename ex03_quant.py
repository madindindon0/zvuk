import numpy as np                        # массивы и вычисления
from scipy import signal                  # обработка сигналов
from scipy.io import wavfile              # чтение и запись WAV

fs, data = wavfile.read("sounds/tone.wav")  # тон 440 Гц, A = 0,5
x = data / 32768.0                        # нормированный сигнал


def quantize(x, bits):                    # равномерное квантование
    q = 2.0 ** (1 - bits)                 # шаг квантования
    return np.round(x / q) * q            # округление к уровню


def snr_db(ref, test):                    # отношение сигнал/шум
    err = test - ref                      # ошибка = шум
    return 10 * np.log10(np.sum(ref**2) / np.sum(err**2))  # в дБ


for bits in (4, 6, 8, 10, 12):            # разные разрядности
    y = quantize(x, bits)                 # квантованный сигнал
    theory = 6.02 * bits + 1.76           # для полной шкалы
    msg = f"{bits:2d} бит: SNR = {snr_db(x, y):5.1f} дБ"  # измерено
    print(msg, f"(теория {theory:5.1f} дБ)")  # сравнение с формулой

fs, data = wavfile.read("sounds/chirp.wav")  # ЛЧМ 100...4000 Гц
c = data / 32768.0                        # нормированный сигнал
naive = c[::4]                            # прореживание без ФНЧ
poly = signal.resample_poly(c, up=1, down=4)  # с ФНЧ внутри
fs2 = fs // 4                             # новая частота 4000 Гц
print("новая частота:", fs2, "Гц")        # 4000 Гц
print("частота Найквиста:", fs2 / 2, "Гц")  # 2000 Гц
wavfile.write("sounds/chirp_naive.wav", fs2,  # с наложением
              naive.astype(np.float32))   # float32 допустим в WAV
wavfile.write("sounds/chirp_poly.wav", fs2,  # без наложения
              poly.astype(np.float32))    # запись результата
