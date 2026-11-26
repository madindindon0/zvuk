"""Проверка установленных библиотек для лабораторной работы №19."""

import importlib
import sys

NAMES = [
    "numpy",
    "scipy",
    "matplotlib",
    "soundfile",
    "librosa",
    "sounddevice",
]


def main() -> None:
    """Вывести версии доступных пакетов и сообщить об отсутствующих."""
    print("Python:", sys.version.split()[0])
    for name in NAMES:
        try:
            mod = importlib.import_module(name)
            print(f"{name}: {mod.__version__}")
        except (ImportError, OSError):
            print(f"{name}: НЕ УСТАНОВЛЕН")


if __name__ == "__main__":
    main()
