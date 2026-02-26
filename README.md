# End-to-End Credit Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Champion-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

An end-to-end Machine Learning pipeline that predicts loan default probability. Built with a focus on reproducibility, real-time API inference, and algorithmic fairness.

## Project Overview

This project processes over 30,000 credit records to predict default risk using the UCI Default of Credit Card Clients dataset. It features a champion-challenger model architecture, achieving a **0.76 ROC-AUC** score using XGBoost.

**Key Features:**

- **Automated ETL Pipeline:** Stratified splitting and standard scaling to prevent data leakage.
- **RESTful API:** Containerized FastAPI backend with <200ms latency.
- **Interactive UI:** Streamlit dashboard for stakeholder User Acceptance Testing (UAT).
- **MLOps Monitoring:** Automated checks for demographic fairness and data drift (Kolmogorov-Smirnov Test).

## Data Dictionary (Input Features)

The model requires an array of 23 numerical features representing a client's profile and 6-month financial history.

**Test the UI with this sample High-Risk applicant:**
`20000, 2, 2, 1, 24, 2, 2, -1, -1, -2, -2, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0`

### 1. Demographics & Profile (Indices 1-5)

- `LIMIT_BAL`: Amount of given credit (NT dollars)
- `SEX`: Gender (1 = male; 2 = female)
- `EDUCATION`: Education level (1 = grad school; 2 = university; 3 = high school; 4 = others)
- `MARRIAGE`: Marital status (1 = married; 2 = single; 3 = others)
- `AGE`: Age in years

### 2. Repayment History (Indices 6-11)

_History of past payment from April to September. (-1 = pay duly, 1 = one month delay, 2 = two month delay, etc.)_

- `PAY_0`: Repayment status in September
- `PAY_2` to `PAY_6`: Repayment status from August back to April

### 3. Bill Amounts (Indices 12-17)

_Amount of bill statement (NT dollars)._

- `BILL_AMT1`: Amount of bill statement in September
- `BILL_AMT2` to `BILL_AMT6`: Amount of bill statement from August back to April

### 4. Previous Payments (Indices 18-23)

_Amount of previous payment (NT dollars)._

- `PAY_AMT1`: Amount paid in September
- `PAY_AMT2` to `PAY_AMT6`: Amount paid from August back to April

## Repository Structure

```text
credit-risk-ml/
├── data/raw/             # Raw UCI dataset
├── data/processed/       # Cleaned, scaled matrices
├── models/               # Serialized .pkl models & scalers
├── src/                  # ML Pipeline (preprocess, train, evaluate, monitor)
├── api/                  # FastAPI backend
├── app_ui.py             # Streamlit frontend
├── Dockerfile            # Containerization configuration
└── requirements.txt      # Dependencies
```

## Quickstart

**1. Clone the repository**

```bash
git clone [https://github.com/YOUR_USERNAME/credit-risk-ml.git](https://github.com/YOUR_USERNAME/credit-risk-ml.git)
cd credit-risk-ml
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the ML Pipeline**

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/monitor.py    # Runs Fairness & Drift checks
```

**4. Start the Application**
Start the backend API:

```bash
uvicorn api.app:app --reload
```

Start the frontend UI (in a new terminal):

```bash
streamlit run app_ui.py
```

## 📊 Model Performance

- **Algorithm:** XGBoost Classifier
- **Primary Metric (ROC-AUC):** 0.7565
- **Fairness Metric:** Demographic Parity Difference < 0.05
