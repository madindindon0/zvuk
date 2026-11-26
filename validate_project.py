#!/usr/bin/env python3
"""Финальная проверка состава проекта лабораторной работы №19."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"

EXAMPLES = [
    "ex01_generate.py",
    "ex02_read.py",
    "ex03_quant.py",
    "ex04_spectrum.py",
    "ex05_spectrogram.py",
    "ex06_filters.py",
    "ex07_denoise.py",
    "ex08_specsub.py",
    "ex09_effects.py",
    "ex10_features.py",
    "ex11_mfcc.py",
]

VARIANT_MODULES = [
    "main.py",
    "audioio.py",
    "processing.py",
    "metrics.py",
    "plots.py",
    "synth.py",
    "audioutils.py",
    "README.md",
    "RESULT.md",
]


def check(condition: bool, message: str, errors: list[str]) -> None:
    """Записать сообщение об ошибке, если условие не выполнено."""
    if condition:
        print(f"OK: {message}")
    else:
        print(f"FAIL: {message}")
        errors.append(message)


def main() -> int:
    """Проверить наличие всех обязательных файлов и запуск --help."""
    errors: list[str] = []

    for name in EXAMPLES:
        check((ROOT / name).exists(), f"пример {name}", errors)

    for name in ["tone.wav", "chord.wav", "chirp.wav", "noisy.wav"]:
        check((ROOT / "sounds" / name).exists(), f"sounds/{name}", errors)

    for path in [
        ROOT / "results" / "example_logs.txt",
        ROOT / "results" / "example_verification.txt",
        ROOT / "results" / "example_results.json",
        ROOT / "audio" / "metronome_120bpm_16beats.wav",
        ROOT / "audio" / "SOURCES.md",
        ROOT / "requirements.txt",
        ROOT / "status.md",
        ROOT / "README.md",
        ROOT / "report.docx",
    ]:
        check(path.exists() and path.stat().st_size > 0, str(path.relative_to(ROOT)), errors)

    variant = ROOT / "lab19_variant07"
    for name in VARIANT_MODULES:
        check((variant / name).exists(), f"lab19_variant07/{name}", errors)

    for path in [
        variant / "results" / "summary.json",
        variant / "results" / "experiments.csv",
    ]:
        check(path.exists() and path.stat().st_size > 0, str(path.relative_to(ROOT)), errors)

    wavs = list((variant / "results").rglob("*.wav")) if (variant / "results").exists() else []
    pngs = list((variant / "results").rglob("*.png")) if (variant / "results").exists() else []
    check(len(wavs) >= 1, f"WAV в lab19_variant07/results ({len(wavs)})", errors)
    check(len(pngs) >= 3, f"PNG в lab19_variant07/results ({len(pngs)})", errors)

    if PYTHON.exists() and (variant / "main.py").exists():
        result = subprocess.run(
            [str(PYTHON), "main.py", "--help"],
            cwd=variant,
            capture_output=True,
            text=True,
            check=False,
        )
        check(result.returncode == 0, "main.py --help", errors)
        check("--input" in result.stdout and "--method" in result.stdout, "опции CLI в --help", errors)

    if errors:
        print(f"\nВсего ошибок: {len(errors)}")
        return 1
    print("\nПроверка пройдена.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
