"""Чтение, диагностика и запись аудиофайлов для варианта 7.

Поддерживаются любые разрядности и число каналов, которые умеет открыть
libsndfile (через :mod:`soundfile`); при отсутствии soundfile используется
``scipy.io.wavfile``. Отсчёты всегда приводятся к ``float64`` в [-1, 1]
и сводятся в моно.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy.io import wavfile

from audioutils import crest_factor_db, peak_db, rms_db

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".aiff", ".aif", ".ogg")
DEFAULT_MAX_SECONDS: float = 600.0

_SUBTYPE_BITS: dict[str, int] = {
    "PCM_S8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}


class AudioIOError(ValueError):
    """Понятная пользователю ошибка ввода/вывода аудиоданных."""


@dataclass(frozen=True)
class SignalInfo:
    """Параметры прочитанного сигнала (часть А задания)."""

    path: Path
    fs: int
    bits: int
    channels: int
    samples: int
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    overload: bool
    subtype: str

    def lines(self) -> list[str]:
        """Вернуть строки текстового отчёта о параметрах сигнала."""
        return [
            f"Файл: {self.path.name}",
            f"Частота дискретизации: {self.fs} Гц",
            f"Разрядность: {self.bits} бит ({self.subtype})",
            f"Число каналов: {self.channels}",
            f"Число отсчётов на канал: {self.samples}",
            f"Длительность: {self.duration_s:.6f} с",
            f"Пиковый уровень: {self.peak_dbfs:.2f} дБFS",
            f"Уровень RMS: {self.rms_dbfs:.2f} дБFS",
            f"Пик-фактор: {self.crest_db:.2f} дБ",
            f"Перегрузка (отсчёты на полной шкале): "
            f"{'да' if self.overload else 'нет'}",
        ]


def _scale_integer(data: np.ndarray, bits: int) -> np.ndarray:
    """Привести целочисленные PCM-отсчёты к диапазону [-1, 1].

    Для 24-битных данных, хранящихся в младших 24 битах ``int32``,
    делитель равен ``2**23``. Библиотеки libsndfile/scipy, однако,
    возвращают 24-битные значения, сдвинутые влево до 31 бита: в этом
    случае корректный делитель ``2**31``. Тип сдвига определяется
    по максимальной амплитуде, поэтому оба представления поддерживаются.
    """
    x = data.astype(np.float64)
    if bits == 24 and data.dtype == np.int32:
        divisor = float(2**31) if np.max(np.abs(x)) > float(2**23) else float(2**23)
        return x / divisor
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        divisor = float(max(abs(info.min), info.max))
        return x / divisor
    return x


def _bits_from_array(data: np.ndarray) -> tuple[int, str]:
    """Определить разрядность и подтип по типу массива (fallback для scipy)."""
    if data.dtype == np.int8 or data.dtype == np.uint8:
        return 8, "PCM_S8"
    if data.dtype == np.int16:
        return 16, "PCM_16"
    if data.dtype == np.int32:
        return 24, "PCM_24 (left-aligned)"
    if data.dtype == np.float32:
        return 32, "FLOAT"
    if data.dtype == np.float64:
        return 64, "DOUBLE"
    return 16, "unknown"


def read_audio(
    path: str | Path,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> tuple[int, np.ndarray, SignalInfo]:
    """Прочитать аудиофайл, свести в моно и вернуть метаданные.

    Args:
        path: Путь к аудиофайлу.
        max_seconds: Максимально допустимая длительность; защита от
            слишком больших файлов (по умолчанию 10 минут).

    Returns:
        ``(fs, x_mono, info)`` где ``x_mono`` имеет тип ``float64``.

    Raises:
        AudioIOError: файл отсутствует, формат не поддерживается, файл
            слишком длинный или пустой.
    """
    p = Path(path)
    if not p.is_file():
        raise AudioIOError(f"Файл не найден: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioIOError(
            f"Неподдерживаемый формат '{p.suffix}'. "
            f"Ожидаются: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if max_seconds <= 0:
        raise AudioIOError("--max-seconds должен быть положительным")

    try:
        import soundfile as sf

        sf_info = sf.info(str(p))
        frames = int(sf_info.frames)
        fs = int(sf_info.samplerate)
        channels = int(sf_info.channels)
        subtype = str(sf_info.subtype)
        bits = _SUBTYPE_BITS.get(subtype, 16)
        duration = frames / fs if fs > 0 else 0.0
        if duration > max_seconds:
            raise AudioIOError(
                f"Файл {p.name} длится {duration / 60:.1f} мин и превышает "
                f"--max-seconds={max_seconds:g} с. Увеличьте лимит явно, "
                f"если это действительно нужно."
            )
        data, _ = sf.read(str(p), dtype="float64", always_2d=False)
        x = np.asarray(data, dtype=np.float64)
    except AudioIOError:
        raise
    except Exception:  # soundfile отсутствует или не смог открыть файл
        try:
            fs, data = wavfile.read(str(p))
        except Exception as exc:  # pragma: no cover - зависит от окружения
            raise AudioIOError(f"Не удалось прочитать {p.name}: {exc}") from exc
        fs = int(fs)
        bits, subtype = _bits_from_array(data)
        x = _scale_integer(data, bits)
        channels = 1 if x.ndim == 1 else int(x.shape[1])
        duration = x.shape[0] / fs if fs > 0 else 0.0
        if duration > max_seconds:
            raise AudioIOError(
                f"Файл {p.name} длится {duration / 60:.1f} мин и превышает "
                f"--max-seconds={max_seconds:g} с"
            )

    if fs <= 0 or x.size == 0:
        raise AudioIOError(f"Файл {p.name} не содержит отсчётов")
    if x.ndim == 2:
        channels = int(x.shape[1])
        x = x.mean(axis=1)
    else:
        channels = 1
        x = np.asarray(x, dtype=np.float64).ravel()

    x = np.asarray(x, dtype=np.float64)
    samples = int(x.size)
    duration = samples / fs
    peak = float(np.max(np.abs(x))) if samples else 0.0
    info = SignalInfo(
        path=p,
        fs=fs,
        bits=bits,
        channels=channels,
        samples=samples,
        duration_s=duration,
        peak_dbfs=peak_db(x),
        rms_dbfs=rms_db(x),
        crest_db=crest_factor_db(x),
        overload=bool(peak >= 1.0),
        subtype=subtype,
    )
    return fs, x, info


def write_audio(
    path: str | Path,
    fs: int,
    x: np.ndarray,
    bits: int = 16,
) -> tuple[Path, bool]:
    """Записать WAV, ограничив отсчёты диапазоном [-1, 1].

    При наличии отсчётов вне диапазона выводится предупреждение и
    возвращается флаг ``clipped=True``.
    """
    if fs <= 0:
        raise AudioIOError("Частота дискретизации должна быть положительной")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(x, dtype=np.float64)
    clipped = bool(np.any(np.abs(y) > 1.0))
    if clipped:
        warnings.warn(
            f"Перегрузка перед записью {out.name}: отсчёты ограничены [-1, 1]",
            RuntimeWarning,
            stacklevel=2,
        )
    y = np.clip(y, -1.0, 1.0)
    if bits == 16:
        data = np.round(y * 32767.0).astype(np.int16)
    elif bits == 32:
        data = y.astype(np.float32)
    else:
        raise AudioIOError(f"Неподдерживаемая разрядность записи: {bits}")
    wavfile.write(str(out), fs, data)
    return out, clipped


def resolve_inputs(input_path: str | Path) -> list[Path]:
    """Преобразовать аргумент --input в отсортированный список файлов."""
    p = Path(input_path)
    if p.is_dir():
        files = sorted(
            f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise AudioIOError(
                f"Каталог {p} не содержит аудиофайлов "
                f"({', '.join(SUPPORTED_EXTENSIONS)})"
            )
        return files
    if not p.exists():
        raise AudioIOError(f"Путь не найден: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioIOError(
            f"Неподдерживаемый формат '{p.suffix}'. "
            f"Ожидаются: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return [p]


def write_info(path: str | Path, info: SignalInfo) -> Path:
    """Сохранить текстовый отчёт о параметрах сигнала (часть А)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(info.lines()) + "\n", encoding="utf-8")
    return out
