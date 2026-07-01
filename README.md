# Stock Price Predictor

A high-performance stock price analysis and prediction application powered by FastAPI and a machine learning pipeline.

## Features
- **FastAPI Backend**: Clean and efficient API endpoints for data processing and model inference.
- **Pipeline**: Modular architecture for data ingestion, preprocessing, and prediction using `pipeline.py` and `predictor.py`.
- **Vanilla HTML/CSS/JS Frontend**: Clean, modern web UI for visualizing stock data and model results.

---

## Setup and Usage

This project utilizes `uv` for efficient package management and environment handling.

### 1. Clone the Repository
```bash
git clone [https://github.com/Vath7677/Stock_price_prediction.git](https://github.com/Vath7677/Stock_price_prediction.git)
cd Stock_price_prediction
```

### 2. Set Up the Environment
Ensure you have uv installed. If not, you can install it via pip install uv. Then, sync the environment:
```bash
uv sync
```

### 3. Run the Application
To start the FastAPI backend, run the following command from the root directory:
```bash
# Activate the environment
source .venv/bin/activate  # On Windows, use: .venv\\Scripts\\activate
```

```bash
# Start the FastAPI development server
fastapi dev app/main.py
```

* **Start the Backend**:
  ```bash
  uv run uvicorn app.main:app --reload
  ```
  *(API: http://localhost:8000)*
  
* **Start the Static Web Server**:
  ```bash
  uv run python -m http.server 3000 --directory frontend
  ```
  *(Frontend: http://localhost:3000)*










