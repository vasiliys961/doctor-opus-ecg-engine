(function () {
  const LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
  const profiles = new Map();
  const DEVICES = [
    { id: "csv", title: "Уже есть файл CSV", action: "file", line: "Выберите файл CSV и нажмите «Посчитать»." },
    { id: "schiller", title: "Schiller Cardiovit", action: "file", line: "Запись хранит программа SEMA. Сохраните CSV и нажмите «Посчитать»." },
    { id: "ge", title: "GE MAC", action: "file", line: "Запись снимается на SD-карту или уходит в программу GE. На страницу нужен CSV." },
    { id: "philips", title: "Philips PageWriter", action: "file", line: "Запись уходит по сети или на флешку. Сохраните CSV и нажмите «Посчитать»." },
    { id: "edan", title: "Edan", action: "file", line: "Запись копируется по USB или по сети. На страницу нужен CSV, не PDF." },
    { id: "mindray", title: "Mindray BeneHeart", action: "file", line: "Запись копируется на флешку. На страницу нужен CSV, не PDF." },
    { id: "btl", title: "BTL-08", action: "file", line: "USB-кабель открывает на компьютере диск с записями. Нужен CSV, внутренний файл BTL не подойдёт." },
    { id: "neurosoft", title: "Нейрософт Поли-Спектр", action: "file", line: "USB и Bluetooth открываются в программе «Поли-Спектр». Оттуда сохраните CSV и выберите его здесь." },
    { id: "contec", title: "Contec", action: "file", line: "Сохраните CSV в программе аппарата и нажмите «Посчитать»." },
    { id: "nihon", title: "Nihon Kohden", action: "file", line: "Сохраните CSV из программы аппарата. PDF страница не читает." },
  ];

  function serialAvailable() {
    return typeof navigator !== "undefined" && navigator.serial != null;
  }

  function profileNames() {
    return Array.from(profiles.keys());
  }

  function devices() {
    return DEVICES.map((item) => ({ ...item }));
  }

  function deviceById(id) {
    return devices().find((item) => item.id === id) || devices()[0];
  }

  function registerDeviceProfile(profile) {
    if (!profile || typeof profile.model !== "string" || !profile.model.trim()) {
      throw new Error("У профиля нет названия модели.");
    }
    if (!Number.isInteger(profile.baudRate) || profile.baudRate <= 0) {
      throw new Error("У профиля нет скорости порта из описания этой модели.");
    }
    if (!(Number(profile.samplingRate) > 0)) {
      throw new Error("У профиля нет частоты дискретизации.");
    }
    if (typeof profile.start !== "function" || typeof profile.accept !== "function") {
      throw new Error("Профиль должен сам разбирать байты своей модели.");
    }
    if (profile.filters != null && !Array.isArray(profile.filters)) {
      throw new Error("Фильтр порта должен быть списком.");
    }
    const model = profile.model.trim();
    if (profiles.has(model)) {
      throw new Error("Эта модель уже подключена.");
    }
    profiles.set(model, profile);
  }

  function samplesToCsv(columns) {
    if (!Array.isArray(columns) || columns.length !== LEADS.length) {
      throw new Error("Нужны 12 отведений в порядке I, II, III, aVR, aVL, aVF, V1–V6.");
    }
    const width = Array.isArray(columns[0]) ? columns[0].length : 0;
    if (!width) throw new Error("В записи нет отсчётов.");
    const lines = [LEADS.join(",")];
    for (let time = 0; time < width; time += 1) {
      const cells = [];
      for (let lead = 0; lead < LEADS.length; lead += 1) {
        const column = columns[lead];
        if (!Array.isArray(column) || column.length !== width) {
          throw new Error("В каждом отведении должно быть одно и то же число отсчётов.");
        }
        const value = Number(column[time]);
        if (!Number.isFinite(value)) throw new Error("В отсчёте есть NaN или Inf.");
        cells.push(String(value));
      }
      lines.push(cells.join(","));
    }
    return `${lines.join("\n")}\n`;
  }

  function openOptions(profile) {
    const options = { baudRate: profile.baudRate };
    ["dataBits", "stopBits", "parity", "bufferSize", "flowControl"].forEach((key) => {
      if (profile[key] != null) options[key] = profile[key];
    });
    return options;
  }

  async function captureCableRecording(model) {
    if (!serialAvailable()) {
      throw new Error("В этом браузере кабель не открывается. Выберите файл CSV.");
    }
    const profile = profiles.get(model);
    if (!profile) {
      throw new Error("Модель аппарата не названа. Байтовый протокол не задан.");
    }
    const port = profile.filters
      ? await navigator.serial.requestPort({ filters: profile.filters })
      : await navigator.serial.requestPort();
    await port.open(openOptions(profile));
    const reader = port.readable.getReader();
    let state = profile.start();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value) continue;
        const step = profile.accept(state, value);
        if (!step || !Object.prototype.hasOwnProperty.call(step, "state")) {
          throw new Error("Профиль вернул пустой шаг разбора.");
        }
        state = step.state;
        if (!step.recording) continue;
        const samplingRate = Number(step.recording.samplingRate || profile.samplingRate);
        if (!(samplingRate > 0) || !step.recording.columns) {
          throw new Error("Профиль не вернул таблицу отсчётов и частоту.");
        }
        return { columns: step.recording.columns, samplingRate };
      }
      throw new Error("Порт закрылся раньше, чем собралась таблица отсчётов.");
    } finally {
      try { reader.releaseLock(); } catch (ignore) { /* порт уже отпущен */ }
      try { await port.close(); } catch (ignore) { /* порт уже закрыт */ }
    }
  }

  globalThis.cable = {
    LEADS,
    serialAvailable,
    profileNames,
    devices,
    deviceById,
    registerDeviceProfile,
    samplesToCsv,
    captureCableRecording,
  };
})();
