"""Вспомогательные функции лабораторной работы 19 (адаптация приложения А).

Модуль сохраняет интерфейс из методических указаний, но дополнен
аннотациями типов, docstring и более аккуратной обработкой крайних случаев.
Чтение/запись файлов вынесены в :mod:`audioio`, здесь остаются только
математические операции над уже загруженными массивами.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal


def db(x: float | np.ndarray, ref: float = 1.0) -> float | np.ndarray:
    """Перевести амплитуду (или массив амплитуд) в децибелы относительно *ref*.

    Args:
        x: Амплитуда или массив амплитуд.
        ref: Опорное значение амплитуды.

    Returns:
        Уровень в дБ. Малое слагаемое внутри логарифма защищает от log(0).
    """
    return 20.0 * np.log10(np.abs(x) / ref + 1e-12)


def peak_db(x: np.ndarray) -> float:
    """Пиковый уровень сигнала в дБFS."""
    return float(db(float(np.max(np.abs(x)))))


def rms_db(x: np.ndarray) -> float:
    """Среднеквадратический уровень сигнала в дБFS."""
    return float(db(float(np.sqrt(np.mean(np.square(x))))))


def crest_factor_db(x: np.ndarray) -> float:
    """Пик-фактор в децибелах: разница пикового и RMS-уровней."""
    return peak_db(x) - rms_db(x)


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    """Отношение сигнал/шум тестового сигнала относительно эталона.

    Сравнивается только общая длина массивов. Значение ``inf`` означает,
    что остаточная ошибка равна нулю.
    """
    n = min(ref.size, test.size)
    if n == 0:
        raise ValueError("snr_db: эталон и тест не должны быть пустыми")
    err = test[:n] - ref[:n]
    p_err = float(np.sum(err**2))
    p_ref = float(np.sum(ref[:n] ** 2))
    if p_err <= 0.0:
        return float("inf")
    if p_ref <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(p_ref / p_err))


def frames(
    x: np.ndarray,
    fs: int,
    flen: float = 0.025,
    hop: float = 0.010,
) -> np.ndarray:
    """Разбить сигнал на кадры с окном Ханна (совместимо с приложением А).

    Args:
        x: Одноканальный сигнал в float64.
        fs: Частота дискретизации, Гц.
        flen: Длина кадра, с.
        hop: Шаг между кадрами, с.

    Returns:
        Матрица формы ``(n_frames, n_samples_per_frame)``.
    """
    if fs <= 0 or flen <= 0 or hop <= 0:
        raise ValueError("frames: fs, flen и hop должны быть положительными")
    n_len = max(2, int(round(flen * fs)))
    n_hop = max(1, int(round(hop * fs)))
    if x.size < n_len:
        x = np.pad(x, (0, n_len - x.size))
    n = 1 + (x.size - n_len) // n_hop
    starts = n_hop * np.arange(n)[:, None]
    idx = np.arange(n_len)[None, :] + starts
    return x[idx] * np.hamming(n_len)


def spectrum(
    x: np.ndarray,
    fs: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Односторонний амплитудный спектр с нормировкой на сумму окна.

    Returns:
        ``(frequencies_hz, amplitude)``; амплитуда сингулярной составляющей
        удвоена, как и в примере из приложения А.
    """
    w = signal.get_window(window, x.size)
    mag = 2.0 * np.abs(np.fft.rfft(x * w)) / np.sum(w)
    return np.fft.rfftfreq(x.size, 1.0 / fs), mag


def plot_overview(
    x: np.ndarray,
    fs: int,
    title: str = "",
    path: str | Path | None = None,
) -> Path | None:
    """Построить обзорную иллюстрацию «осциллограмма — спектр — спектрограмма».

    Если *path* не задан, рисунок показывается на экране. Функция
    оставлена для совместимости с приложением А; основной конвейер
    использует модуль :mod:`plots`.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    t = np.arange(x.size) / fs
    ax[0].plot(t, x, "k", lw=0.4)
    ax[0].set(xlabel="Время, с", ylabel="Амплитуда", title=title)
    f, mag = spectrum(x, fs)
    ax[1].plot(f, db(mag), "k", lw=0.6)
    ax[1].set(xlabel="Частота, Гц", ylabel="Уровень, дБFS")
    ax[2].specgram(x, NFFT=512, Fs=fs, noverlap=384, cmap="gray_r")
    ax[2].set(xlabel="Время, с", ylabel="Частота, Гц")
    fig.tight_layout()
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
        plt.close(fig)
        return out
    plt.show()
    return None
