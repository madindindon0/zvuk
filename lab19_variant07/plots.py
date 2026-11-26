"""Построение графиков для отчёта по варианту 7.

Все рисунки сохраняются в PNG, все оси подписаны с указанием единиц
измерения. Используется неинтерактивный backend Agg.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal as sp_signal  # noqa: E402

from audioio import SignalInfo  # noqa: E402
from audioutils import db  # noqa: E402
from processing import DetectionResult  # noqa: E402


def _save(fig: plt.Figure, path: str | Path) -> Path:
    """Сохранить figures и освободить память."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_overview(
    x: np.ndarray,
    fs: int,
    info: SignalInfo,
    name: str,
    out_dir: str | Path,
) -> Path:
    """Осциллограмма всего сигнала и фрагмента, спектр и спектрограмма."""
    fig, ax = plt.subplots(4, 1, figsize=(10, 11))
    t = np.arange(x.size) / fs
    ax[0].plot(t, x, color="black", lw=0.35)
    ax[0].set(
        title=f"{name}: осциллограмма всего сигнала",
        xlabel="Время, с",
        ylabel="Амплитуда (д.ед.)",
    )
    ax[0].grid(alpha=0.25)

    frag = min(1.0, info.duration_s)
    n_frag = min(x.size, int(round(frag * fs)))
    t_frag = np.arange(n_frag) / fs
    ax[1].plot(t_frag, x[:n_frag], color="navy", lw=0.6)
    ax[1].set(
        title=f"Фрагмент 0–{frag:.2f} с",
        xlabel="Время, с",
        ylabel="Амплитуда (д.ед.)",
    )
    ax[1].grid(alpha=0.25)

    w = np.hanning(x.size)
    mag = 2.0 * np.abs(np.fft.rfft(x * w)) / np.sum(w)
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    ax[2].plot(freqs, db(mag), color="darkgreen", lw=0.6)
    ax[2].set(
        title="Амплитудный спектр (окно Ханна)",
        xlabel="Частота, Гц",
        ylabel="Уровень, дБFS",
    )
    ax[2].grid(alpha=0.25)

    n_fft = min(1024, max(64, 2 ** int(np.floor(np.log2(max(x.size, 64))))))
    f, t_s, z = sp_signal.spectrogram(
        x, fs=fs, window="hann", nperseg=n_fft, noverlap=3 * n_fft // 4
    )
    im = ax[3].pcolormesh(t_s, f, db(np.abs(z)), shading="auto", cmap="magma")
    ax[3].set(
        title="Спектрограмма",
        xlabel="Время, с",
        ylabel="Частота, Гц",
    )
    fig.colorbar(im, ax=ax[3], label="Уровень, дБFS")
    return _save(fig, Path(out_dir) / f"{name}_overview.png")


def plot_flux(
    result: DetectionResult,
    name: str,
    out_dir: str | Path,
) -> Path:
    """График спектрального потока с отмеченными атаками и порогом."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        result.flux_times_s,
        result.smoothed_flux,
        color="steelblue",
        lw=0.9,
        label="Spectral flux (сглаженный)",
    )
    ax.axhline(
        result.threshold_level,
        color="darkorange",
        ls="--",
        lw=1.0,
        label=f"Порог = {result.threshold_level:.3g}",
    )
    if result.n_onsets:
        ax.plot(
            result.onsets_s,
            np.interp(result.onsets_s, result.flux_times_s, result.smoothed_flux),
            "rv",
            ms=7,
            label=f"Атаки (n={result.n_onsets})",
        )
    ax.set(
        title=f"{name}: спектральный поток и обнаруженные атаки",
        xlabel="Время, с",
        ylabel="Spectral flux (норм.)",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, Path(out_dir) / f"{name}_flux.png")


def plot_acf(
    result: DetectionResult,
    name: str,
    out_dir: str | Path,
) -> Path:
    """График АКФ огибающей атак с найденным периодом."""
    max_lag = min(2.0, float(result.acf_lags_s[-1]))
    mask = result.acf_lags_s <= max_lag
    lags_ms = result.acf_lags_s[mask] * 1000.0
    acf = result.acf[mask]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags_ms, acf, color="purple", lw=1.0, label="АКФ огибающей")
    period_ms = result.period_s * 1000.0
    ax.axvline(period_ms, color="crimson", ls="--", lw=1.2,
               label=f"Период {period_ms:.1f} мс ({result.bpm:.1f} BPM)")
    ax.axvspan(300.0, min(1000.0, max_lag * 1000.0), color="gold", alpha=0.10,
               label="Диапазон 60–200 BPM")
    ax.set(
        title=f"{name}: автокорреляция огибающей атак",
        xlabel="Задержка, мс",
        ylabel="АКФ (норм.)",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, Path(out_dir) / f"{name}_acf.png")


def plot_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    fs: int,
    name: str,
    out_dir: str | Path,
) -> Path:
    """Сравнение исходного и обработанного WAV при одинаковых масштабах."""
    n = min(original.size, processed.size, int(round(1.5 * fs)))
    t = np.arange(n) / fs
    limit = float(max(np.max(np.abs(original[:n])), np.max(np.abs(processed[:n]))))
    limit = max(limit, 1e-6) * 1.05
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].plot(t, original[:n], color="black", lw=0.5)
    ax[1].plot(t, processed[:n], color="darkred", lw=0.5)
    ax[0].set_ylim(-limit, limit)
    ax[1].set_ylim(-limit, limit)
    ax[0].set(title="Исходный сигнал", ylabel="Амплитуда (д.ед.)")
    ax[1].set(title="Сигнал с отметками обнаруженных атак",
              xlabel="Время, с", ylabel="Амплитуда (д.ед.)")
    fig.suptitle(f"{name}: сравнение при одинаковом масштабе осей")
    ax[0].grid(alpha=0.25)
    ax[1].grid(alpha=0.25)
    return _save(fig, Path(out_dir) / f"{name}_comparison.png")


def _mean_by_param(
    rows: Sequence[dict[str, Any]], param: str, metric: str
) -> list[tuple[float, float]]:
    """Сгруппировать эксперименты по параметру и усреднить метрику."""
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        try:
            p = float(row[param])
            v = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(p) and np.isfinite(v):
            groups[p].append(v)
    return [(p, float(np.mean(groups[p]))) for p in sorted(groups)]


def plot_experiments(
    rows: Sequence[dict[str, Any]],
    plots_dir: str | Path,
) -> list[Path]:
    """Построить графики зависимостей метрик от параметров экспериментов."""
    out_dir = Path(plots_dir)
    specs: tuple[tuple[str, str, str, str, str], ...] = (
        ("hop_ms", "f1", "Зависимость F1 от шага STA", "Шаг STA, мс", "F1 (д.ед.)"),
        ("threshold", "f1", "Зависимость F1 от порога", "Порог (отн. ед.)", "F1 (д.ед.)"),
        ("window_ms", "onset_error_ms", "Погрешность атак и длина окна",
         "Длина окна, мс", "Средняя погрешность, мс"),
        ("hop_ms", "bpm_error_pct", "Ошибка BPM и шаг STA",
         "Шаг STA, мс", "Ошибка BPM, %"),
        ("window_ms", "processing_time_s", "Время обработки и длина окна",
         "Длина окна, мс", "Время обработки, с"),
    )
    paths: list[Path] = []
    for param, metric, title, xlabel, ylabel in specs:
        data = _mean_by_param(rows, param, metric)
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        ax.plot(x, y, marker="o", color="teal", lw=1.4)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.3)
        paths.append(_save(fig, out_dir / f"exp_{metric}_vs_{param}.png"))
    return paths
