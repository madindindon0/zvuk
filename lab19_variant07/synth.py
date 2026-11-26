"""Детерминированный синтез метрономов и эталонных разметок атак.

Все случайные числа создаются с фиксированным ``seed`` (по умолчанию 42), поэтому
повторный запуск даёт битово идентичные WAV-файлы. Для каждого файла
сохраняются известные времена атак и эталонный темп.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from audioio import write_audio

DEFAULT_FS: int = 48_000
DEFAULT_SEED: int = 42
DEFAULT_BPMS: tuple[int, ...] = (100, 120, 140)
DEFAULT_BEATS: int = 16
EXTERNAL_NAME: str = "metronome_120bpm_16beats.wav"
REFERENCE_FILE: str = "references.json"


def make_click(fs: int = DEFAULT_FS, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Создать короткий широкополосный щелчок с экспоненциальным затуханием.

    Щелчок начинается с отсчёта 0 (атака в момент t=0), что позволяет
    использовать Round(t*fs) как эталонное время атаки.
    """
    rng = np.random.default_rng(seed)
    length = int(round(0.012 * fs))
    t = np.arange(length) / fs
    env = np.exp(-t / 0.0015)
    wave = (
        0.75 * np.sin(2 * np.pi * 1100.0 * t)
        + 0.35 * np.sin(2 * np.pi * 2650.0 * t)
        + 0.12 * rng.standard_normal(length)
    )
    click = env * wave
    peak = float(np.max(np.abs(click)))
    return 0.9 * click / peak if peak > 0 else click


def add_noise(clean: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Добавить белый гауссов шум с заданным отношением сигнал/шум."""
    rng = np.random.default_rng(seed)
    p_signal = float(np.mean(clean**2))
    if p_signal <= 0.0:
        raise ValueError("add_noise: эталонный сигнал имеет нулевую мощность")
    amplitude = np.sqrt(p_signal * 10.0 ** (-snr_db / 10.0))
    return clean + amplitude * rng.standard_normal(clean.size)


def synthesize_metronome(
    bpm: float,
    fs: int = DEFAULT_FS,
    beats: int = DEFAULT_BEATS,
    seed: int = DEFAULT_SEED,
    noise_snr_db: float | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Синтезировать метроном с известными временами атак.

    Args:
        bpm: Темп в ударах в минуту.
        fs: Частота дискретизации, Гц.
        beats: Число щелчков.
        seed: Seed генератора случайных чисел.
        noise_snr_db: Если задано, добавить шум с этим SNR (дБ).

    Returns:
        ``(signal, onsets_s)``. Времена атак заданы в секундах.
    """
    if bpm <= 0 or fs <= 0 or beats < 2:
        raise ValueError("bpm, fs должны быть положительными, beats >= 2")
    period = 60.0 / bpm
    onsets = [k * period for k in range(beats)]
    duration_s = beats * period
    n = int(round(duration_s * fs))
    x = np.zeros(n, dtype=np.float64)
    click = make_click(fs=fs, seed=seed)
    for t in onsets:
        start = int(round(t * fs))
        end = min(n, start + click.size)
        if start < n:
            x[start:end] += click[: end - start]
    # Тихий «комнатный» тон делает детектор чувствительным к порогу,
    # но не создаёт ложных атак.
    x = add_noise(x, 55.0, seed + 1)
    if noise_snr_db is not None:
        x = add_noise(x, noise_snr_db, seed + 2)
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x = x / peak * 0.95
    return x, onsets


def _reference_name(bpm: float) -> str:
    """Имя файла синтетического метронома для заданного темпа."""
    return f"metronome_{int(round(bpm))}bpm.wav"


def load_references(sounds_dir: str | Path) -> dict[str, dict]:
    """Загрузить ``sounds/references.json`` (пустой словарь, если файла нет)."""
    path = Path(sounds_dir) / REFERENCE_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_references(sounds_dir: str | Path, refs: dict[str, dict]) -> Path:
    """Сохранить эталонные времена атак и темп в ``sounds/references.json``."""
    out = Path(sounds_dir) / REFERENCE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(refs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def ensure_external_signal(
    sounds_dir: str | Path,
    source: str | Path | None = None,
) -> dict | None:
    """Скопировать открытый метроном 120 BPM в ``sounds/`` и описать его.

    Источник: github.com/mshark1/audio-mcp-assets, лицензия CC0 1.0
    (см. ``../audio/SOURCES.md``). Если файл уже скопирован, копирование
    пропускается.
    """
    sounds = Path(sounds_dir)
    sounds.mkdir(parents=True, exist_ok=True)
    src = (
        Path(source)
        if source is not None
        else Path(__file__).resolve().parent.parent / "audio" / EXTERNAL_NAME
    )
    dst = sounds / EXTERNAL_NAME
    if not dst.is_file():
        if not src.is_file():
            return None
        shutil.copy2(src, dst)
    refs = load_references(sounds)
    refs[EXTERNAL_NAME] = {
        "bpm": 120,
        "onsets_s": [round(0.5 * k, 6) for k in range(16)],
        "source": "https://github.com/mshark1/audio-mcp-assets",
        "license": "CC0 1.0 Universal (public domain)",
        "note": "16 щелчков каждые 0.5 с, первый на t=0",
    }
    save_references(sounds, refs)
    return refs[EXTERNAL_NAME]


def create_synthetic_signals(
    sounds_dir: str | Path,
    bpms: tuple[int, ...] = DEFAULT_BPMS,
    fs: int = DEFAULT_FS,
    beats: int = DEFAULT_BEATS,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict]:
    """Синтезировать метрономы 100/120/140 BPM и сохранить эталоны.

    Returns:
        Словарь ``refs`` из ``references.json`` с добавленными записями.
    """
    sounds = Path(sounds_dir)
    sounds.mkdir(parents=True, exist_ok=True)
    refs = load_references(sounds)
    for bpm in bpms:
        x, onsets = synthesize_metronome(bpm=bpm, fs=fs, beats=beats, seed=seed)
        name = _reference_name(bpm)
        write_audio(sounds / name, fs, x, bits=16)
        refs[name] = {
            "bpm": int(round(bpm)),
            "onsets_s": [round(t, 6) for t in onsets],
            "source": "synthetic",
            "seed": seed,
            "beats": beats,
        }
    save_references(sounds, refs)
    return refs
