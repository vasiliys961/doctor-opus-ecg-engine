HTTP-сервис: `app.py`.

Страницу отдаёт `GET /`. Снимок или текст идут в `POST /api/ecg/analyze`, бланк — в `POST /api/ecg/protocol`. Цифровой CSV считается в `POST /api/ecg/signal/csv`, заключение — в `POST /api/ecg/signal/conclusion`. Ансамбль 531 этим сервисом не вызывается.
