# Doctor Opus ECG Engine

Исследовательское приложение для ЭКГ. Это не медицинское изделие.

Два входа:

- снимок или текст разбирают Gemini 3.8 Flash и Opus 5.5;
- цифровую запись около 10 секунд и 12 отведений считает ECGFounder; по той же кривой снимается разметка зубцов: ЧСС, интервалы, смещение ST и внеочередные комплексы.

## Запуск

Нужен Python 3.11. Веса ECGFounder лежат локально в `models/ecgfounder/12_lead_ECGFounder.pth` и в git не входят.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. uvicorn backend.app:app --port 8765
```

Страница: [http://127.0.0.1:8765](http://127.0.0.1:8765).

Разбор снимка и текста требует `LLM_API_KEY` или `OPENROUTER_API_KEY`. Без ключа этот канал отвечает 503.

```bash
PYTHONPATH=. pytest
```
