"""Алгоритмы варианта 7: спектральный поток, атаки и оценка темпа.

Ключевые операции реализованы на NumPy/SciPy:

1. STA и положительная разница модулей спектра между соседними кадрами
   (spectral flux) с последующим сглаживанием;
2. адаптивный порог + поиск локальных максимумов + минимальная дистанция;
3. автокорреляция огибающей атак в диапазоне 60–200 BPM с устранением
   октавной ошибки (выбор более короткого периода, если он не хуже).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal as sp_signal


@dataclass(frozen=True)
class OnsetParams:
    """Параметры детектора атак и оценщика темпа."""

    hop_ms: float = 10.0
    window_ms: float = 40.0
    threshold: float = 1.5
    min_distance_ms: float = 200.0
    smooth_ms: float = 5.0
    bpm_min: float = 60.0
    bpm_max: float = 200.0


@dataclass
class DetectionResult:
    """Результат детекции атак и оценки темпа."""

    flux: np.ndarray
    flux_times_s: np.ndarray
    smoothed_flux: np.ndarray
    threshold_level: float
    onset_frames: np.ndarray
    onsets_s: np.ndarray
    envelope: np.ndarray
    acf: np.ndarray
    acf_lags_s: np.ndarray
    bpm: float
    period_s: float

    @property
    def n_onsets(self) -> int:
        """Число обнаруженных атак."""
        return int(self.onsets_s.size)


def compute_spectral_flux(
    x: np.ndarray,
    fs: int,
    params: OnsetParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вычислить спектральный поток по STA.

    Args:
        x: Монофонический сигнал ``float64``.
        fs: Частота дискретизации, Гц.
        params: Параметры обработки.

    Returns:
        ``(raw_flux, smoothed_flux, times_s)``. Время относится к центру
        каждого кадра; при поиске атак центры сдвигаются на половину окна.
    """
    if x.size < 2:
        raise ValueError("Сигнал слишком короткий для STA")
    hop = max(1, int(round(params.hop_ms * fs / 1000.0)))
    n_win = int(round(params.window_ms * fs / 1000.0))
    n_win = max(8, min(n_win, x.size))
    n_hop = max(1, min(hop, n_win - 1))
    # Нулевая подложка по краям нужна, чтобы атака в самый первый момент
    # попала в разность соседних кадров, а не оказалась «за кадром».
    pad = 2 * n_win
    padded = np.pad(x, (pad, pad))
    _, times, z = sp_signal.stft(
        padded,
        fs=fs,
        window="hann",
        nperseg=n_win,
        noverlap=n_win - n_hop,
        boundary=None,
        padded=False,
    )
    mag = np.abs(z)
    raw = np.sum(np.maximum(np.diff(mag, axis=1), 0.0), axis=0) / n_win
    times = times[1:] - pad / fs
    smooth_frames = max(1, int(round(params.smooth_ms / params.hop_ms)))
    if smooth_frames > 1 and raw.size >= smooth_frames:
        smoothed = ndimage.uniform_filter1d(raw, size=smooth_frames, mode="nearest")
    else:
        smoothed = raw.copy()
    return raw, smoothed, times


def detect_onsets(
    flux: np.ndarray,
    times_s: np.ndarray,
    params: OnsetParams,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Обнаружить атаки по адаптивному порогу и локальным пикам.

    Порог равен ``threshold`` × робастный уровень шума (75-й перцентиль
    флюкса) с нижней ограничением в 2 % от максимума. Такая нормализация
    делает один и тот же порог применимым к разным длинам окна и уровням
    сигнала.

    Returns:
        ``(onset_frames, onset_times_s, threshold_level)``.
    """
    if flux.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), 0.0
    peak_value = float(np.max(flux))
    baseline = float(np.percentile(flux, 75))
    level = max(baseline * params.threshold, 0.02 * peak_value)
    distance = max(1, int(round(params.min_distance_ms / params.hop_ms)))
    found, _ = sp_signal.find_peaks(flux, height=level, distance=distance)
    if found.size == 0 and peak_value >= level * 0.5:
        # Резервный случай: очень короткая запись, где пик оказался на краю.
        found = np.array([int(np.argmax(flux))])
    # Для короткого импульса положительная разница максимизируется в кадре,
    # центр которого совпадает с атакой (окно Ханна симметрично).
    onset_times = np.maximum(times_s[found], 0.0)
    return found.astype(int), onset_times.astype(float), level


def estimate_tempo(
    onset_frames: np.ndarray,
    flux: np.ndarray,
    times_s: np.ndarray,
    params: OnsetParams,
    duration_s: float,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Оценить темп по АКФ огибающей атак.

    Returns:
        ``(bpm, period_s, envelope, acf, lags_s)``. Диапазон поиска
        соответствует ``bpm_min``…``bpm_max`` (по умолчанию 60–200).
    """
    hop_s = params.hop_ms / 1000.0
    env = np.zeros(flux.size, dtype=np.float64)
    if onset_frames.size:
        valid = onset_frames[(onset_frames >= 0) & (onset_frames < flux.size)]
        env[valid] = flux[valid]
    else:
        env[:] = flux
    sigma = max(0.6, 5.0 / params.hop_ms)
    env = ndimage.gaussian_filter1d(env, sigma=sigma, mode="nearest")
    centered = env - env.mean()
    acf = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    if acf[0] > 0:
        acf = acf / acf[0]
    lags_s = np.arange(acf.size, dtype=np.float64) * hop_s

    lower = int(np.floor((60.0 / params.bpm_max) / hop_s))
    upper = int(np.ceil(min(60.0 / params.bpm_min, duration_s * 0.9) / hop_s))
    lower = max(1, lower)
    upper = min(upper, acf.size - 2)
    if upper <= lower:
        # Слишком короткая запись: оценить темп по всему доступному диапазону.
        lower, upper = 1, max(2, acf.size - 2)
    segment = acf[lower : upper + 1]
    maxima, _ = sp_signal.find_peaks(segment)
    candidates = [lower + int(i) for i in maxima]
    if not candidates:
        candidates = [lower + int(np.argmax(segment))]
    # Октавная коррекция: АКФ метронома имеет пики и на T, и на 2T.
    # среди достаточно заметных пиков (≥60 % от максимума) берём самый
    # короткий период — это основной темп, а не его половинный вариант.
    best_acf = max(float(acf[i]) for i in candidates)
    valid = [i for i in candidates if acf[i] >= 0.60 * best_acf]
    best = int(min(valid))

    left, right = max(0, best - 1), min(acf.size - 1, best + 1)
    y0, y1, y2 = acf[left], acf[best], acf[right]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    period_s = (best + delta) * hop_s
    bpm = float(np.clip(60.0 / period_s, params.bpm_min, params.bpm_max))
    period_s = 60.0 / bpm
    return bpm, period_s, env, acf, lags_s


def analyze(x: np.ndarray, fs: int, params: OnsetParams) -> DetectionResult:
    """Полный конвейер: спектральный поток → атаки → темп по АКФ."""
    duration_s = x.size / fs
    min_duration = max(0.3, 4.0 * params.window_ms / 1000.0)
    if duration_s < min_duration:
        raise ValueError(
            f"Сигнал длится {duration_s:.3f} с; для анализа нужно не менее "
            f"{min_duration:.3f} с"
        )
    raw, smoothed, times = compute_spectral_flux(x, fs, params)
    frames, onsets, level = detect_onsets(smoothed, times, params)
    bpm, period, env, acf, lags = estimate_tempo(
        frames, smoothed, times, params, duration_s
    )
    return DetectionResult(
        flux=raw,
        flux_times_s=times,
        smoothed_flux=smoothed,
        threshold_level=level,
        onset_frames=frames,
        onsets_s=onsets,
        envelope=env,
        acf=acf,
        acf_lags_s=lags,
        bpm=bpm,
        period_s=period,
    )
