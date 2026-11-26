# Лабораторная работа №19 — цифровая обработка звука (вариант 7)

Репозиторий содержит полное выполнение лабораторной работы №19 по дисциплине
«Практикум по программированию».

## Состав

- `task.md` — исходные методические указания.
- `status.md` — актуальное состояние проекта, Decisions Log и инструкция для продолжения работы.
- `check_env.py` — проверка установленных библиотек.
- `requirements.txt` — зафиксированные версии зависимостей.
- `ex01_generate.py` … `ex11_mfcc.py` — одиннадцать примеров из главы 3 `task.md`.
- `sounds/` — синтетические сигналы, создаваемые `ex01_generate.py`.
- `audio/` — внешний открытый аудиофайл с эталоном (источник указан в `audio/SOURCES.md`).
- `results/` — логи, графики и проверка числовых результатов примеров.
- `lab19_variant07/` — модуль индивидуального задания (вариант 7): обнаружение атак и оценка темпа.
- `build_report.py` — генератор отчёта `report.docx`.
- `report.docx` — итоговый отчёт по требованиям раздела 4.2 `task.md`.

## Окружение

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python check_env.py
```

## Воспроизведение примеров

Примеры запускаются строго по порядку, так как первый создаёт тестовые WAV-файлы:

```bash
for f in ex01_generate.py ex02_read.py ex03_quant.py ex04_spectrum.py \
         ex05_spectrogram.py ex06_filters.py ex07_denoise.py ex08_specsub.py \
         ex09_effects.py ex10_features.py ex11_mfcc.py; do
    .venv/bin/python "$f"
done
```

Логи и графики сохраняются в `results/`.

## Индивидуальное задание (вариант 7)

```bash
cd lab19_variant07
../.venv/bin/python main.py --help
../.venv/bin/python main.py --method all --input ../audio/metronome_120bpm_16beats.wav --output results
```

Подробности — в `lab19_variant07/README.md` и `lab19_variant07/RESULT.md`.

## Отчёт

```bash
.venv/bin/python build_report.py
```

Результат: `report.docx`. Для проверки числа страниц:

```bash
libreoffice --headless --convert-to pdf --outdir . report.docx
pdfinfo report.pdf | grep Pages
```

## Git

Коммиты создаются локально. Push в `origin/main` выполняется только по отдельной команде пользователя.
