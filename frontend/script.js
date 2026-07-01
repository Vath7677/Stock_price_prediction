

// ---- Backend endpoints 

const PREDICT_FROM_FILE_URL = "http://127.0.0.1:8000/predict-from-file/";

const predictionRows = document.querySelector("#predictionRows");
const predictButton = document.querySelector("#predictButton");
const formPredictButton = document.querySelector("#formPredictButton");
const downloadButton = document.querySelector("#downloadButton");
const refreshButton = document.querySelector("#refreshButton");
const stockSymbol = document.querySelector("#stockSymbol");
const forecastDays = document.querySelector("#forecastDays");
const sequenceWindow = document.querySelector("#sequenceWindow");
const windowValue = document.querySelector("#windowValue");
const datasetFile = document.querySelector("#datasetFile");
const predictedPrice = document.querySelector("#predictedPrice");
const priceChange = document.querySelector("#priceChange");
const resultSummary = document.querySelector("#resultSummary");
const datasetMessage = document.querySelector("#datasetMessage");
const resultNote = document.querySelector("#resultNote");
const chartTitle = document.querySelector("#chartTitle");
const chartEmptyState = document.querySelector("#chartEmptyState");
const chartSvg = document.querySelector("#chartSvg");
const actualPath = document.querySelector("#actualPath");
const predictedPath = document.querySelector("#predictedPath");
const areaPath = document.querySelector("#areaPath");
const maeValue = document.querySelector("#maeValue");
const rmseValue = document.querySelector("#rmseValue");
const r2Value = document.querySelector("#r2Value");
const accuracyValue = document.querySelector("#accuracyValue");
const liveBadge = document.querySelector("#liveBadge");
const statusPulse = document.querySelector("#statusPulse");
const statusText = document.querySelector("#statusText");
const windowSizeStat = document.querySelector("#windowSizeStat");
const lastPredictionStat = document.querySelector("#lastPredictionStat");

let latestPredictionRows = [];
let isPredicting = false;

// ---- Small helpers --------------------------------------------------

function currency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(Number(value) || 0);
}

function setMessage(text, type = "") {
  datasetMessage.textContent = text;
  datasetMessage.className = `dataset-message ${type}`.trim();
}

function setStatus(state, label) {
  // state: "idle" | "busy" | "ready" | "error"
  statusPulse.className = `pulse ${state === "idle" ? "" : state}`.trim();
  statusText.textContent = label;

  liveBadge.textContent =
    state === "busy" ? "Running…" : state === "error" ? "Error" : state === "ready" ? "Live Preview" : "Idle";
  liveBadge.className = `badge ${state === "busy" ? "busy" : state === "error" ? "error" : state === "ready" ? "" : "idle"}`.trim();
}

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  formPredictButton.disabled = isLoading;
  refreshButton.disabled = isLoading;
}

// ---- Rendering: table -------------------------------------------------

function renderRows(rows) {
  latestPredictionRows = rows;

  if (!rows.length) {
    predictionRows.innerHTML = `<tr class="empty-row"><td colspan="4">No predictions yet.</td></tr>`;
    downloadButton.disabled = true;
    return;
  }

  const topFiveRows = rows.slice(0, 5);

  predictionRows.innerHTML = topFiveRows
    .map(
      (row) => `
        <tr>
          <td>${row.date}</td>
          <td>${currency(row.actual)}</td>
          <td>${currency(row.predicted)}</td>
          <td><span class="signal ${row.signal.toLowerCase()}">${row.signal}</span></td>
        </tr>
      `
    )
    .join("");

  downloadButton.disabled = false;
}

// ---- Rendering: chart ---------------------------------------------------

function makeLinePath(values) {
  const width = 580;
  const height = 190;
  const left = 20;
  const top = 35;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1);

  return values
    .map((value, index) => {
      const x = left + (index / Math.max(values.length - 1, 1)) * width;
      const y = top + height - ((value - min) / range) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function showChart(actualValues, predictedValues) {
  const actualLine = makeLinePath(actualValues);
  const predictedLine = makeLinePath(predictedValues);

  actualPath.setAttribute("d", actualLine);
  predictedPath.setAttribute("d", predictedLine);
  areaPath.setAttribute("d", `${actualLine} L600 245 L20 245 Z`);

  chartEmptyState.classList.add("hidden");
  chartSvg.classList.remove("hidden");
}

function clearChart() {
  actualPath.setAttribute("d", "");
  predictedPath.setAttribute("d", "");
  areaPath.setAttribute("d", "");
  chartSvg.classList.add("hidden");
  chartEmptyState.classList.remove("hidden");
  chartEmptyState.textContent = "Run a prediction to see the chart";
}

// ---- Full reset to the empty state (used on load and after an error) ----

function resetDashboard(message) {
  predictedPrice.textContent = "$0.00";

  priceChange.textContent = "";
  priceChange.classList.add("hidden");

  maeValue.textContent = "--";
  rmseValue.textContent = "--";
  r2Value.textContent = "--";
  accuracyValue.textContent = "--";

  chartTitle.textContent = "No data yet";
  clearChart();

  resultSummary.textContent =
    message || "No prediction yet. Choose a ticker or upload a CSV, then press Run Prediction.";
  resultNote.textContent = "Prediction rows will appear here after you run the forecast.";

  renderRows([]);
  setStatus("idle", "Waiting for input");
}

function showPredictionError(message) {
  setMessage(message, "error");
  setStatus("error", "Prediction failed");
  resultSummary.textContent = message;
}

// ---- Dashboard update: ticker-based /predict response --------------------

function updateDashboardFromTicker(data) {
  predictedPrice.textContent = currency(data.predicted_price);

  const change = Number(data.change_percent);
  if (!Number.isNaN(change)) {
    priceChange.textContent = `${change >= 0 ? "+" : ""}${change}%`;
    priceChange.classList.remove("hidden");
    priceChange.classList.toggle("negative", change < 0);
  } else {
    priceChange.classList.add("hidden");
  }

  maeValue.textContent = data.mae != null ? Number(data.mae).toFixed(2) : "--";
  rmseValue.textContent = data.rmse != null ? Number(data.rmse).toFixed(2) : "--";
  r2Value.textContent = data.r2 != null ? Number(data.r2).toFixed(2) : "--";
  accuracyValue.textContent = data.accuracy != null ? `${Number(data.accuracy).toFixed(0)}%` : "--";

  chartTitle.textContent = `${data.ticker} · Last 60 Days`;
  resultSummary.textContent = `${data.ticker} prediction result: next close ${currency(data.predicted_price)}.`;
  resultNote.textContent = "Live prediction from the FastAPI backend.";
  lastPredictionStat.textContent = data.ticker;

  const history = Array.isArray(data.history) ? data.history : [];
  const rows = history.map((item) => ({
    date: item.date,
    actual: item.close,
    predicted: item.close,
    signal: "Hold",
  }));

  renderRows(rows);

  if (rows.length > 1) {
    const closes = rows.map((row) => Number(row.actual));
    showChart(closes, closes);
  } else {
    clearChart();
    chartEmptyState.textContent = "Backend did not return enough history to draw a chart.";
  }
}

// ---- Dashboard update: uploaded-file /predict-from-file/ response --------
function updateDashboardFromFile(data) {
  if (predictedPrice) {
    predictedPrice.textContent = currency(data.predicted_price || 0);
  }
  if (priceChange) priceChange.classList.add("hidden"); 

  const maeEl = document.getElementById("mae-value") || maeValue;
  const rmseEl = document.getElementById("rmse-value") || rmseValue;
  const r2El = document.getElementById("r2-value") || r2Value;
  const accEl = document.getElementById("accuracy-value") || accuracyValue;

  if (maeEl) maeEl.textContent = data.mae != null ? Number(data.mae).toFixed(4) : "--";
  if (rmseEl) rmseEl.textContent = data.rmse != null ? Number(data.rmse).toFixed(4) : "--";
  if (r2El) r2El.textContent = data.r2 != null ? Number(data.r2).toFixed(4) : "--";
  if (accEl) accEl.textContent = data.accuracy != null ? `${Number(data.accuracy).toFixed(2)}%` : "--";

  if (chartTitle) chartTitle.textContent = `${data.filename} · Uploaded dataset`;
  if (resultSummary) resultSummary.textContent = `Prediction from "${data.filename}": next close ${currency(data.predicted_price || 0)}.`;

  const history = Array.isArray(data.history) ? data.history : [];
  if (history.length > 1) {
    const rows = history.map((item) => ({
      date: item.date,
      actual: item.close,
      predicted: item.close, 
      signal: "Hold",
    }));
    renderRows(rows);
    showChart(rows.map(r => Number(r.actual)), rows.map(r => Number(r.actual)));
  }
}


// ---- API calls ------------------------------------------------------

async function runTickerPrediction() {
  const ticker = stockSymbol.value.trim().toUpperCase();

  if (!ticker) {
    throw new Error("Enter a stock symbol or choose a CSV file first.");
  }

  stockSymbol.value = ticker;
  setMessage(`Fetching prediction for ${ticker}...`, "loading");
  setStatus("busy", `Predicting ${ticker}…`);

  const params = new URLSearchParams({
    ticker: ticker,
    start: "2000-01-01",
  });

  const response = await fetch(`${PREDICT_URL}?${params.toString()}`, { method: "GET" });

  if (!response.ok) {
    throw new Error(`Prediction failed (status ${response.status})`);
  }

  const data = await response.json();
  updateDashboardFromTicker(data);
  setMessage(`Prediction loaded for ${ticker}.`, "success");
  setStatus("ready", "Prediction ready");
}

async function runFilePrediction(file) {
  setMessage(`Running prediction using ${file.name}...`, "loading");
  setStatus("busy", `Predicting from ${file.name}…`);

  const formData = new FormData();
  formData.append("file", file);
  // Optional extras — harmless to send even if your FastAPI route
  // doesn't read them yet; wire them up in main.py if you want them used.
  formData.append("forecast_days", forecastDays.value);
  formData.append("sequence_window", sequenceWindow.value);

  const response = await fetch(PREDICT_FROM_FILE_URL, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Prediction failed (status ${response.status})`);
  }

  const data = await response.json();
  updateDashboardFromFile(data);
  setMessage(`Prediction loaded from ${data.filename}.`, "success");
  setStatus("ready", "Prediction ready");
}

// A file takes priority over a typed ticker, since that's the more specific input.
async function runPrediction() {
  if (isPredicting) return;

  const file = datasetFile.files[0];
  const ticker = stockSymbol.value.trim();

  if (!file && !ticker) {
    setMessage("Enter a stock symbol or choose a CSV file first.", "error");
    return;
  }

  isPredicting = true;
  setLoading(true);

  try {
    if (file) {
      await runFilePrediction(file);
    } else {
      await runTickerPrediction();
    }
  } catch (error) {
    showPredictionError(error.message);
  } finally {
    isPredicting = false;
    setLoading(false);
  }
}

function downloadResult() {
  if (!latestPredictionRows.length) return;

  const csv = [
    "Date,Actual,Predicted,Signal",
    ...latestPredictionRows.map((row) =>
      [row.date, Number(row.actual).toFixed(2), Number(row.predicted).toFixed(2), row.signal].join(",")
    ),
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const link = document.createElement("a");
  const symbol = (stockSymbol.value.trim() || "stock").toLowerCase();

  link.href = URL.createObjectURL(blob);
  link.download = `${symbol}_prediction_result.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

// ---- Wiring -----------------------------------------------------------

sequenceWindow.addEventListener("input", () => {
  windowValue.textContent = sequenceWindow.value;
});

datasetFile.addEventListener("change", () => {
  const file = datasetFile.files[0];
  if (file) {
    setMessage(`"${file.name}" selected. Press Run Prediction to use it.`, "");
  } else {
    setMessage(
      "Enter a stock symbol OR choose a CSV file, then press Run Prediction. A file takes priority if both are set.",
      ""
    );
  }
});

predictButton.addEventListener("click", runPrediction);
formPredictButton.addEventListener("click", runPrediction);
refreshButton.addEventListener("click", runPrediction);
downloadButton.addEventListener("click", downloadResult);

// Start with a genuinely empty dashboard — no placeholder numbers.
resetDashboard();