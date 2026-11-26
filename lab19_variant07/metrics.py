"""Метрики варианта 7: погрешность атак, темп, precision/recall/F1, SNR.

Совпадение атак считается по жадному лучшему соответствию с допуском
50 мс; каждая эталонная атака может быть сопоставлена не более чем с одним
обнаруженной.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

EXPERIMENT_COLUMNS: tuple[str, ...] = (
    "filename",
    "params",
    "onset_error_ms",
    "bpm",
    "bpm_error_pct",
    "precision",
    "recall",
    "f1",
    "processing_time_s",
)
EXTRA_COLUMNS: tuple[str, ...] = (
    "hop_ms",
    "window_ms",
    "threshold",
    "reference_bpm",
    "n_reference",
    "n_detected",
    "snr_db",
)


def match_onsets(
    reference_s: Sequence[float],
    detected_s: Sequence[float],
    tolerance_ms: float = 50.0,
) -> tuple[list[tuple[float, float, float]], int, int]:
    """Сопоставить атаки по ближайшему соседу с допуском *tolerance_ms*.

    Returns:
        ``(pairs, n_false_positive, n_false_negative)`` где каждый элемент
        ``pairs`` — ``(эталон, обнаружение, ошибка в мс)``.
    """
    tolerance = tolerance_ms / 1000.0
    used: set[int] = set()
    pairs: list[tuple[float, float, float]] = []
    for ref in reference_s:
        best_idx = -1
        best_dist = float("inf")
        for i, det in enumerate(detected_s):
            if i in used:
                continue
            dist = abs(float(det) - float(ref))
            if dist < best_dist:
                best_dist, best_idx = dist, i
        if best_idx >= 0 and best_dist <= tolerance:
            used.add(best_idx)
            pairs.append((float(ref), float(detected_s[best_idx]), best_dist * 1000.0))
    fn = len(reference_s) - len(pairs)
    fp = len(detected_s) - len(pairs)
    return pairs, fp, fn


def compute_metrics(
    reference_onsets_s: Sequence[float] | None,
    detected_onsets_s: Sequence[float],
    reference_bpm: float | None,
    detected_bpm: float,
    tolerance_ms: float = 50.0,
) -> dict[str, Any]:
    """Рассчитать все показатели качества детекции.

    Недоступные значения (нет эталона, нет совпадений) записываются как
    ``None`` и сохраняются в JSON/CSV без исключений.
    """
    ref = list(reference_onsets_s or [])
    det = list(detected_onsets_s)
    pairs, fp, fn = match_onsets(ref, det, tolerance_ms)
    errors = [p[2] for p in pairs]
    tp = len(pairs)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    bpm_error = (
        abs(float(detected_bpm) - float(reference_bpm)) / float(reference_bpm) * 100.0
        if reference_bpm
        else None
    )
    return {
        "n_reference": len(ref),
        "n_detected": len(det),
        "n_matched": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "onset_error_mean_ms": float(np.mean(errors)) if errors else None,
        "onset_error_median_ms": float(np.median(errors)) if errors else None,
        "onset_error_max_ms": float(np.max(errors)) if errors else None,
        "onset_errors_ms": errors,
        "reference_bpm": float(reference_bpm) if reference_bpm else None,
        "detected_bpm": float(detected_bpm),
        "bpm_error_pct": bpm_error,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tolerance_ms": float(tolerance_ms),
    }


def save_metrics_json(path: str | Path, metrics: dict[str, Any]) -> Path:
    """Сохранить метрики в JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def save_onsets_csv(
    path: str | Path,
    reference_s: Sequence[float],
    detected_s: Sequence[float],
) -> Path:
    """Сохранить эталонные и обнаруженные времена атак в CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(reference_s), len(detected_s))
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "reference_s", "detected_s"])
        for i in range(n):
            writer.writerow(
                [
                    i,
                    f"{reference_s[i]:.6f}" if i < len(reference_s) else "",
                    f"{detected_s[i]:.6f}" if i < len(detected_s) else "",
                ]
            )
    return out


def _csv_value(value: Any) -> str:
    """Безопасно преобразовать значение метрики в строку CSV."""
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return str(value)


def save_experiments_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Сохранить таблицу экспериментов в CSV с фиксированным порядком колонок."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = EXPERIMENT_COLUMNS + EXTRA_COLUMNS
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    return out


def read_experiments_csv(path: str | Path) -> list[dict[str, Any]]:
    """Прочитать таблицу экспериментов, созданную :func:`save_experiments_csv`."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
