"""Точка входа конвейера варианта 7 (метроном, атаки, темп).

Пример полного запуска::

    .venv/bin/python main.py --method all --input sounds --output results \
        --create-synth --hop-ms 10 --window-ms 40 --threshold 1.5

Методы: ``analyze`` (часть А), ``onset_tempo`` (часть Б),
``experiments`` (часть В) и ``all`` (все три части).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

import audioutils as au
import audioio
import metrics
import plots
import processing
import synth


DEFAULT_SOUNDS = Path("sounds")
DEFAULT_OUTPUT = Path("results")
HOP_VALUES: tuple[float, ...] = (5.0, 10.0, 20.0)
WINDOW_VALUES: tuple[float, ...] = (20.0, 40.0, 60.0)
THRESHOLD_VALUES: tuple[float, ...] = (1.2, 1.5, 2.0)
NOISY_SNRS: tuple[float, ...] = (0.0, 5.0, 10.0)
TOLERANCE_MS: float = 50.0


@dataclass(frozen=True)
class ExperimentSignal:
    """Сигнал (или зашумлённое состояние) для экспериментов."""

    name: str
    x: np.ndarray
    fs: int
    reference_bpm: float | None
    reference_onsets_s: list[float]
    clean_x: np.ndarray | None = None
    input_snr_db: float | None = None
    bits: int = 16
    channels: int = 1


def build_parser() -> argparse.ArgumentParser:
    """Создать и полностью описать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Вариант 7: обнаружение атак по спектральному потоку и оценка "
            "темпа по АКФ огибающей. Читает WAV/FLAC/AIFF, сохраняет "
            "отчёты, графики, WAV и таблицу экспериментов."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_SOUNDS),
        help="Входной аудиофайл или каталог с аудиофайлами.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Каталог для всех результатов (создаётся при необходимости).",
    )
    parser.add_argument(
        "--method",
        choices=("analyze", "onset_tempo", "experiments", "all"),
        default="all",
        help=(
            "analyze — параметры сигнала и обзорные графики; "
            "onset_tempo — детекция атак, темп и метрики; "
            "experiments — перебор параметров; all — весь конвейер."
        ),
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=120.0,
        help="Эталонный темп (БПМ), если для файла нет разметки в references.json.",
    )
    parser.add_argument(
        "--hop-ms", type=float, default=10.0,
        help="Шаг STA в миллисекундах (положительный, не больше длины окна).",
    )
    parser.add_argument(
        "--window-ms", type=float, default=40.0,
        help="Длина окна Ханна STA в миллисекундах (положительная).",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.5,
        help="Множитель порога детекции над робастным уровнем шума (0 < порог ≤ 20).",
    )
    parser.add_argument(
        "--min-distance-ms", type=float, default=200.0,
        help="Минимальная дистанция между обнаруженными атаками, мс.",
    )
    parser.add_argument(
        "--seed", type=int, default=synth.DEFAULT_SEED,
        help="Seed генератора случайных чисел для воспроизводимости.",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=audioio.DEFAULT_MAX_SECONDS,
        help="Максимальная длительность входного файла, с (защита >10 мин).",
    )
    parser.add_argument(
        "--create-synth", action="store_true",
        help=(
            "Синтезировать в sounds/ метрономы 100/120/140 BPM и скопировать "
            "открытый метроном 120 BPM (источник и лицензия CC0 — в README.md)."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Проверить диапазоны всех числовых параметров."""
    for name in ("hop_ms", "window_ms", "min_distance_ms", "max_seconds"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} должен быть положительным числом")
    if not np.isfinite(args.threshold) or not (0 < args.threshold <= 20):
        raise ValueError("--threshold должен находиться в диапазоне (0, 20]")
    if not np.isfinite(args.bpm) or not (30 <= args.bpm <= 300):
        raise ValueError("--bpm должен находиться в диапазоне [30, 300]")
    if args.hop_ms > args.window_ms:
        raise ValueError("--hop-ms не должен превышать --window-ms")
    if args.seed < 0:
        raise ValueError("--seed должен быть неотрицательным целым")


def params_from_args(args: argparse.Namespace) -> processing.OnsetParams:
    """Построить параметры детектора из аргументов CLI."""
    return processing.OnsetParams(
        hop_ms=args.hop_ms,
        window_ms=args.window_ms,
        threshold=args.threshold,
        min_distance_ms=args.min_distance_ms,
    )


def reference_for(path: Path, sounds_dir: Path, fallback_bpm: float) -> tuple[float | None, list[float]]:
    """Найти эталонный темп и времена атак по ``sounds/references.json``."""
    refs = synth.load_references(sounds_dir)
    meta = refs.get(path.name)
    if meta is None:
        return float(fallback_bpm), []
    return (
        float(meta.get("bpm", fallback_bpm)),
        [float(t) for t in meta.get("onsets_s", [])],
    )


def click_track(
    onsets_s: Sequence[float],
    fs: int,
    duration_s: float,
    seed: int,
) -> np.ndarray:
    """Построить модельный метроном в момент обнаруженных атак."""
    n = int(round(duration_s * fs))
    out = np.zeros(n, dtype=np.float64)
    click = synth.make_click(fs=fs, seed=seed)
    for t in onsets_s:
        start = int(round(float(t) * fs))
        if 0 <= start < n:
            end = min(n, start + click.size)
            out[start:end] += click[: end - start]
    return out


def reconstruction_snr(
    x: np.ndarray,
    onsets_s: Sequence[float],
    fs: int,
    seed: int,
) -> float:
    """SNR реконструкции метронома по обнаруженным атакам относительно x."""
    model = click_track(onsets_s, fs, x.size / fs, seed)
    energy = float(np.dot(model, model))
    if energy <= 0:
        return float("-inf")
    scale = float(np.dot(x[: model.size], model) / energy)
    return au.snr_db(x, scale * model)


def signal_summary(
    path: Path,
    info: audioio.SignalInfo,
    reference_bpm: float | None,
    reference_onsets_s: list[float],
    output: Path,
) -> dict[str, Any]:
    """Сформировать запись ``signals`` для ``summary.json``."""
    info_file = output / f"{path.stem}_info.txt"
    return {
        "name": path.name,
        "fs": int(info.fs),
        "bits": int(info.bits),
        "channels": int(info.channels),
        "duration_s": round(float(info.duration_s), 6),
        "reference_bpm": reference_bpm,
        "reference_onsets_s": reference_onsets_s,
        "info_file": info_file.name if info_file.is_file() else None,
    }


def empty_best() -> dict[str, Any]:
    """Вернуть структуру ``best`` с пустыми значениями (для стабильности JSON)."""
    return {
        "signal": None,
        "hop_ms": None,
        "window_ms": None,
        "threshold": None,
        "detected_bpm": None,
        "bpm_error_pct": None,
        "onset_error_ms": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }


def write_summary(
    output: Path,
    signals: list[dict[str, Any]],
    best: dict[str, Any] | None,
    experiments_csv: Path | None,
    plots_paths: Sequence[Path],
    recommendations: str,
) -> Path:
    """Записать ``results/summary.json`` с фиксированной структурой."""

    def rel(p: Path | None) -> str | None:
        """Привести путь к относительному относительно каталога результатов."""
        if p is None:
            return None
        try:
            return str(p.relative_to(output))
        except ValueError:
            return str(p)

    payload = {
        "signals": signals,
        "best": best if best is not None else empty_best(),
        "experiments_csv": rel(experiments_csv),
        "plots": list(dict.fromkeys(rel(p) or "" for p in plots_paths)),
        "recommendations": recommendations or "",
    }
    out = output / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def run_prepare(args: argparse.Namespace) -> None:
    """Синтезировать тестовые сигналы и подготовить копию открытого файла."""
    if not args.create_synth:
        return
    sounds = DEFAULT_SOUNDS
    synth.create_synthetic_signals(sounds, seed=args.seed)
    meta = synth.ensure_external_signal(sounds)
    if meta is None:
        print("Предупреждение: внешний метроном не найден, используется только синтез",
              file=sys.stderr)


def run_analyze(
    paths: Sequence[Path],
    output: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Часть А: параметры сигнала и обзорные графики."""
    signals: list[dict[str, Any]] = []
    plot_paths: list[Path] = []
    for path in paths:
        fs, x, info = audioio.read_audio(path, max_seconds=args.max_seconds)
        audioio.write_info(output / f"{path.stem}_info.txt", info)
        plot_paths.append(plots.plot_overview(x, fs, info, path.stem, output))
        ref_bpm, ref_onsets = reference_for(path, DEFAULT_SOUNDS, args.bpm)
        signals.append(signal_summary(path, info, ref_bpm, ref_onsets, output))
        print(f"[A] {path.name}: fs={fs} Гц, {info.duration_s:.3f} с, "
              f"пик={info.peak_dbfs:.2f} дБFS, RMS={info.rms_dbfs:.2f} дБFS, "
              f"перегрузка={'да' if info.overload else 'нет'}")
    return signals, plot_paths


def run_onset_tempo(
    paths: Sequence[Path],
    output: Path,
    args: argparse.Namespace,
    params: processing.OnsetParams,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Часть Б: детекция атак, оценка темпа, метрики и графики."""
    signals: list[dict[str, Any]] = []
    plot_paths: list[Path] = []
    for path in paths:
        fs, x, info = audioio.read_audio(path, max_seconds=args.max_seconds)
        audioio.write_info(output / f"{path.stem}_info.txt", info)
        plot_paths.append(plots.plot_overview(x, fs, info, path.stem, output))
        result = processing.analyze(x, fs, params)
        ref_bpm, ref_onsets = reference_for(path, DEFAULT_SOUNDS, args.bpm)
        met = metrics.compute_metrics(ref_onsets, result.onsets_s, ref_bpm, result.bpm,
                                      tolerance_ms=TOLERANCE_MS)
        met["snr_reconstruction_db"] = reconstruction_snr(x, result.onsets_s, fs, args.seed)
        met["snr_input_db"] = None
        metrics.save_metrics_json(output / f"{path.stem}_metrics.json", met)
        metrics.save_onsets_csv(output / f"{path.stem}_onsets.csv",
                                ref_onsets, result.onsets_s)

        marked = x + 0.30 * click_track(result.onsets_s, fs, x.size / fs, args.seed)
        peak_marked = float(np.max(np.abs(marked))) if marked.size else 0.0
        if peak_marked > 1.0:
            # Метки не должны создавать перегрузку в выходном WAV.
            marked = marked * (0.98 / peak_marked)
        audioio.write_audio(output / f"{path.stem}_detected.wav", fs, marked, bits=16)
        plot_paths.append(plots.plot_flux(result, path.stem, output))
        plot_paths.append(plots.plot_acf(result, path.stem, output))
        plot_paths.append(plots.plot_comparison(x, marked, fs, path.stem, output))
        signals.append(signal_summary(path, info, ref_bpm, ref_onsets, output))
        err = met["onset_error_mean_ms"]
        bpm_err = met["bpm_error_pct"]
        print(f"[Б] {path.name}: атак={result.n_onsets}/{len(ref_onsets)}, "
              f"BPM={result.bpm:.2f}"
              + (f" (ошибка {bpm_err:.2f} %)" if bpm_err is not None else "")
              + (f", ошибка атак={err:.2f} мс" if err is not None else "")
              + f", F1={met['f1'] if met['f1'] is not None else 'н/д'}")
    return signals, plot_paths


def _base_signals(paths: Sequence[Path], args: argparse.Namespace) -> list[ExperimentSignal]:
    """Прочитать базовые файлы для экспериментов."""
    out: list[ExperimentSignal] = []
    for path in paths:
        fs, x, info = audioio.read_audio(path, max_seconds=args.max_seconds)
        ref_bpm, ref_onsets = reference_for(path, DEFAULT_SOUNDS, args.bpm)
        out.append(
            ExperimentSignal(
                path.name,
                x,
                fs,
                ref_bpm,
                ref_onsets,
                x,
                None,
                bits=int(info.bits),
                channels=int(info.channels),
            )
        )
    return out


def _noisy_signals(
    base: Sequence[ExperimentSignal], args: argparse.Namespace, output: Path
) -> list[ExperimentSignal]:
    """Добавить три состояния с шумом SNR 0/5/10 дБ к 120-BPM метроному."""
    candidates = [s for s in base if s.reference_bpm == 120 and s.reference_onsets_s]
    external = [s for s in candidates if s.name == synth.EXTERNAL_NAME]
    if external:
        candidates = external
    if not candidates:
        candidates = [s for s in base if s.reference_onsets_s] or list(base)
    if not candidates:
        return []
    source = candidates[0]
    result: list[ExperimentSignal] = []
    for snr in NOISY_SNRS:
        noisy = synth.add_noise(source.x, snr, args.seed + int(snr))
        name = f"{Path(source.name).stem}_snr{int(snr)}dB.wav"
        audioio.write_audio(output / name, source.fs, noisy, bits=16)
        result.append(
            ExperimentSignal(
                name=name,
                x=noisy,
                fs=source.fs,
                reference_bpm=source.reference_bpm,
                reference_onsets_s=list(source.reference_onsets_s),
                clean_x=np.asarray(source.x),
                input_snr_db=snr,
                bits=16,  # write_audio(..., bits=16)
                channels=source.channels,
            )
        )
    return result


def _score_row(row: dict[str, Any]) -> tuple[float, float, float]:
    """Ключ ранжирования экспериментов: F1, ошибка BPM, ошибка атак."""
    f1 = float(row["f1"]) if row.get("f1") not in (None, "") else -np.inf
    bpm_err = (
        -round(float(row["bpm_error_pct"]), 1)
        if row.get("bpm_error_pct") not in (None, "")
        else -np.inf
    )
    onset_err = (
        -float(row["onset_error_ms"])
        if row.get("onset_error_ms") not in (None, "")
        else -np.inf
    )
    return f1, bpm_err, onset_err


def choose_best(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Выбрать лучшую комбинацию параметров, усреднив метрики по сигналам."""
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["hop_ms"]), float(row["window_ms"]), float(row["threshold"]))
        groups.setdefault(key, []).append(row)
    if not groups:
        return empty_best(), "Эксперименты не дали корректных измерений."

    def agg_score(key: tuple[float, float, float]) -> tuple[float, float, float]:
        """Оценка комбинации параметров по усреднённым метрикам."""
        group = groups[key]
        f1s = [float(r["f1"]) for r in group if r.get("f1") not in (None, "")]
        berr = [float(r["bpm_error_pct"]) for r in group
                if r.get("bpm_error_pct") not in (None, "")]
        oerr = [float(r["onset_error_ms"]) for r in group
                if r.get("onset_error_ms") not in (None, "")]
        if not f1s:
            return (-np.inf, -np.inf, -np.inf)
        return (
            float(np.mean(f1s)),
            -round(float(np.mean(berr)), 1) if berr else -np.inf,
            -float(np.mean(oerr)) if oerr else -np.inf,
        )

    winner = max(groups, key=agg_score)
    best_row = max(groups[winner], key=_score_row)
    best = {
        "signal": best_row.get("filename"),
        "hop_ms": winner[0],
        "window_ms": winner[1],
        "threshold": winner[2],
        "detected_bpm": float(best_row["bpm"]) if best_row.get("bpm") not in (None, "") else None,
        "bpm_error_pct": (
            float(best_row["bpm_error_pct"])
            if best_row.get("bpm_error_pct") not in (None, "") else None
        ),
        "onset_error_ms": (
            float(best_row["onset_error_ms"])
            if best_row.get("onset_error_ms") not in (None, "") else None
        ),
        "precision": float(best_row["precision"]) if best_row.get("precision") not in (None, "") else None,
        "recall": float(best_row["recall"]) if best_row.get("recall") not in (None, "") else None,
        "f1": float(best_row["f1"]) if best_row.get("f1") not in (None, "") else None,
    }
    score = agg_score(winner)
    by_threshold: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("f1") not in (None, ""):
            by_threshold[float(row["threshold"])].append(float(row["f1"]))
    thr_summary = ", ".join(
        f"{t:g} → F1={np.mean(v):.3f}" for t, v in sorted(by_threshold.items())
    )
    text = (
        f"Рекомендуемые параметры: hop_ms={winner[0]:g}, window_ms={winner[1]:g}, "
        f"threshold={winner[2]:g}. Средний F1={score[0]:.3f}, средняя ошибка BPM="
        f"{-score[1]:.2f} %, средняя погрешность атак={-score[2]:.2f} мс. "
        "Малый шаг улучшает время атак, но увеличивает число кадров; слишком "
        "большое окно сглаживает короткие щелчки. Влияние порога: "
        f"{thr_summary}; при сильном шуме меньший порог повышает риск ложных "
        "атак, а слишком высокий — риск пропуска тихих атак."
    )
    return best, text


def run_experiments(
    paths: Sequence[Path],
    output: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path], str, Path]:
    """Часть В: перебор hop/window/threshold по трём и более сигналам."""
    base = _base_signals(paths, args)
    signals_out = _noisy_signals(base, args, output)
    all_signals = base + signals_out
    rows: list[dict[str, Any]] = []
    for sig in all_signals:
        for hop in HOP_VALUES:
            for win in WINDOW_VALUES:
                for thr in THRESHOLD_VALUES:
                    params = processing.OnsetParams(
                        hop_ms=hop,
                        window_ms=win,
                        threshold=thr,
                        min_distance_ms=args.min_distance_ms,
                    )
                    t0 = time.perf_counter()
                    try:
                        result = processing.analyze(sig.x, sig.fs, params)
                    except ValueError as exc:
                        print(f"Пропущен эксперимент {sig.name}: {exc}", file=sys.stderr)
                        continue
                    elapsed = time.perf_counter() - t0
                    met = metrics.compute_metrics(
                        sig.reference_onsets_s, result.onsets_s,
                        sig.reference_bpm, result.bpm, TOLERANCE_MS,
                    )
                    row: dict[str, Any] = {
                        "filename": sig.name,
                        "params": f"hop_ms={hop:g};window_ms={win:g};threshold={thr:g}",
                        "onset_error_ms": met["onset_error_mean_ms"],
                        "bpm": result.bpm,
                        "bpm_error_pct": met["bpm_error_pct"],
                        "precision": met["precision"],
                        "recall": met["recall"],
                        "f1": met["f1"],
                        "processing_time_s": elapsed,
                        "hop_ms": hop,
                        "window_ms": win,
                        "threshold": thr,
                        "reference_bpm": sig.reference_bpm,
                        "n_reference": len(sig.reference_onsets_s),
                        "n_detected": result.n_onsets,
                        "snr_db": sig.input_snr_db,
                    }
                    if sig.clean_x is not None and met["n_matched"] > 0:
                        row["snr_db"] = au.snr_db(sig.clean_x, sig.x)
                    rows.append(row)

    csv_path = metrics.save_experiments_csv(output / "experiments.csv", rows)
    plot_paths = plots.plot_experiments(rows, output / "plots")
    best, recommendations = choose_best(rows)
    print(f"[В] экспериментов: {len(rows)}; CSV: {csv_path.name}; "
          f"графиков: {len(plot_paths)}")
    return signals_for_experiments(all_signals, output), best, plot_paths, recommendations, csv_path


def signals_for_experiments(
    signals: Sequence[ExperimentSignal], output: Path
) -> list[dict[str, Any]]:
    """Подготовить ``signals`` для сводного JSON по всем состояниям."""
    result: list[dict[str, Any]] = []
    for sig in signals:
        info_file = output / f"{Path(sig.name).stem}_info.txt"
        result.append(
            {
                "name": sig.name,
                "fs": int(sig.fs),
                "bits": int(sig.bits),
                "channels": int(sig.channels),
                "duration_s": round(sig.x.size / sig.fs, 6),
                "reference_bpm": sig.reference_bpm,
                "reference_onsets_s": sig.reference_onsets_s,
                "info_file": info_file.name if info_file.is_file() else None,
            }
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить конвейер; вернуть 0 при успехе и 1 при ошибке."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if args.create_synth or args.method == "all":
            run_prepare(args)
        paths = audioio.resolve_inputs(args.input)
        params = params_from_args(args)

        signals: list[dict[str, Any]] = []
        all_plots: list[Path] = []
        best: dict[str, Any] | None = None
        csv_path: Path | None = None
        recommendation = ""

        if args.method in ("analyze", "onset_tempo", "all"):
            signals, p = run_analyze(paths, output, args)
            all_plots.extend(p)
        if args.method in ("onset_tempo", "all"):
            signals, p = run_onset_tempo(paths, output, args, params)
            all_plots.extend(p)
        if args.method in ("experiments", "all"):
            signals, best, p, recommendation, csv_path = run_experiments(
                paths, output, args
            )
            all_plots.extend(p)

        write_summary(output, signals, best, csv_path, all_plots, recommendation)
        print(f"Готово. Результаты: {output}/")
        return 0
    except (AudioError, ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


class AudioError(audioio.AudioIOError):
    """Ошибка входных аудиоданных (псевдоним для совместимости)."""


if __name__ == "__main__":
    raise SystemExit(main())
