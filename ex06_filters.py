import numpy as np                        # массивы
import matplotlib                         # настройка headless-окружения
matplotlib.use("Agg")                     # графики без GUI
matplotlib.rcParams["font.family"] = "Noto Sans"  # кириллические подписи
import matplotlib.pyplot as plt           # графики
from pathlib import Path                  # работа с путями
from scipy import signal                  # проектирование фильтров
from scipy.io import wavfile              # чтение WAV

Path("results/examples").mkdir(parents=True, exist_ok=True)  # каталог для графиков

fs = 16000                                # частота дискретизации
fc = 1000                                 # частота среза ФНЧ, Гц
fir = signal.firwin(101, fc, fs=fs,       # КИХ-фильтр 101 отвод
                    window="hamming")     # метод окна Хэмминга
sos = signal.butter(4, fc, btype="low",   # БИХ Баттерворта 4-го
                    fs=fs, output="sos")  # порядка, секции 2-го пор.

w, h_fir = signal.freqz(fir, worN=4096, fs=fs)  # АЧХ КИХ
w, h_iir = signal.sosfreqz(sos, worN=4096, fs=fs)  # АЧХ БИХ

for name, h in (("КИХ", h_fir), ("БИХ", h_iir)):  # оба фильтра
    db = 20 * np.log10(np.abs(h) + 1e-12)  # АЧХ в дБ
    i3 = np.argmin(np.abs(db + 3))        # точка уровня -3 дБ
    a2 = db[np.argmin(np.abs(w - 2000))]  # затухание на 2 кГц
    print(f"{name}: -3 дБ на {w[i3]:.1f} Гц,",  # частота среза
          f"на 2 кГц {a2:.1f} дБ")        # затухание
print("задержка КИХ:", 1e3 * 50 / fs, "мс")  # (N-1)/2 отсчётов

fs, data = wavfile.read("sounds/chirp.wav")  # ЛЧМ 100...4000 Гц
x = data / 32768.0                        # нормированный сигнал
y_fir = signal.lfilter(fir, 1.0, x)       # фильтрация КИХ
y_iir = signal.sosfiltfilt(sos, x)        # БИХ без фазового сдвига

fig, ax = plt.subplots(2, 1, figsize=(9, 6))  # две панели
curves = ((h_fir, "k-", "КИХ"), (h_iir, "k--", "БИХ"))  # стили
for h, st, lb in curves:                  # обе АЧХ на одном графике
    ax[0].plot(w, 20 * np.log10(np.abs(h)), st, label=lb)  # в дБ
ax[0].set_ylim(-90, 5)                    # диапазон уровней
ax[0].set_xlabel("Частота, Гц")           # подпись оси
ax[0].set_ylabel("АЧХ, дБ")               # подпись оси
ax[0].legend()                            # легенда
t = np.arange(x.size) / fs                # ось времени
ax[1].plot(t, y_fir, "k", lw=0.3)         # отклик на ЛЧМ
ax[1].set_xlabel("Время, с")              # подпись оси
fig.tight_layout()                        # компоновка
fig.savefig("results/examples/ex06_filters.png")  # сохранение вместо вывода на экран
