# ✈️ US Flight Delay Analytics Pipeline

An end-to-end big data pipeline built with **PySpark** and **Databricks Community Edition**, processing **5.8 million US domestic flights** to analyse delay patterns and predict flight delays using machine learning.

## 🔴 Live Dashboard
**[flight-delay-pipeline-satvik.streamlit.app](https://flight-delay-pipeline-satvik.streamlit.app)**

---

## 🏗️ Architecture — Medallion Architecture

Raw CSV Files (Databricks Volume)
↓
[Bronze Layer]   → Raw ingestion, saved as Parquet
↓
[Silver Layer]   → Cleaned, enriched, feature-engineered
↓
[Gold Layer]     → Business aggregations (airline, route, monthly, hourly)
↓
[ML Layer]       → PySpark MLlib Random Forest delay predictor
↓
[Streamlit Dashboard] → Live visualisation of insights and ML results

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Total flights processed | 5,714,008 |
| Overall delay rate | 17.91% |
| ML Model Accuracy | 94.44% |
| AUC Score | 0.961 |
| F1 Score | 0.9428 |
| Top delay predictor | Departure Delay (89.07% importance) |

---

## 📁 Project Structure

├── notebooks/
│   ├── 01_bronze_ingestion.ipynb       # Raw CSV → Bronze Parquet
│   ├── 02_silver_transformation.ipynb  # Clean, enrich, feature engineer
│   ├── 03_gold_aggregations.ipynb      # Business-level aggregations
│   ├── 04_ml_training.ipynb            # Random Forest delay predictor
│   └── 05_export_for_dashboard.ipynb   # Export CSVs for Streamlit
│
├── dashboard/
│   ├── app.py                          # Streamlit dashboard
│   ├── requirements.txt
│   ├── runtime.txt                     # Python 3.11
│   └── data/                           # Exported Gold + ML CSVs
│       ├── airline_delays.csv
│       ├── hourly_delays.csv
│       ├── ml_importance.csv
│       ├── ml_metrics.csv
│       ├── monthly_trend.csv
│       └── route_delays.csv
│
└── README.md

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **PySpark** | Distributed data processing across 5.8M rows |
| **Databricks Community Edition** | Managed Spark cluster and notebook environment |
| **Spark MLlib** | Distributed ML pipeline — StringIndexer + VectorAssembler + RandomForestClassifier |
| **Medallion Architecture** | Bronze/Silver/Gold layered data pipeline pattern |
| **Parquet** | Columnar storage format for all pipeline layers |
| **Streamlit** | Interactive dashboard deployment |
| **Plotly** | Interactive charts |
| **Python** | Core language |

---

## 🤖 ML Pipeline

The ML pipeline uses PySpark MLlib with three chained stages:

1. **StringIndexer** — converts AIRLINE carrier codes to numeric indices
2. **VectorAssembler** — combines features into a single vector column
3. **RandomForestClassifier** — 50 trees, max depth 10, trained on 4.5M rows

**Features used** (all known before departure — no data leakage):
- Month, Day of Week
- Airline
- Departure Delay
- Distance
- Scheduled Departure Hour
- Taxi Out Time

**Feature Importance:**
- DEPARTURE_DELAY: 89.07%
- TAXI_OUT: 8.75%
- SCHEDULED_DEPARTURE: 0.95%

---

## 📈 Key Insights

- **Spirit Airlines** has the highest delay rate at 28.79%; **Hawaiian Airlines** the lowest at 10.53%
- **June** is the worst month for delays (22.73%); **October** the best (11.85%)
- Delays compound through the day — 5am flights have 6.9% delay rate vs 25.5% at 8pm
- **New York → Columbus** is the most delayed route at 32.67%

---

## 📓 Notebooks

All five Databricks notebooks are exported with full outputs in the `notebooks/` folder. They can be imported directly into any Databricks workspace and re-run.

---

## Dataset

**2015 Flight Delays and Cancellations** — US Bureau of Transportation Statistics via Kaggle  
5,819,079 domestic US flights across 14 airlines and 322 airports.

---

## 👤 Author
**Satvik Pandey** — [GitHub](https://github.com/SatvikSPandey) | [LinkedIn](https://www.linkedin.com/in/satvikpandey-433555365)