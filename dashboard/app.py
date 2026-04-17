import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="US Flight Delay Analytics",
    page_icon="✈️",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    airline = pd.read_csv(f"{DATA_PATH}/airline_delays.csv")
    hourly = pd.read_csv(f"{DATA_PATH}/hourly_delays.csv")
    monthly = pd.read_csv(f"{DATA_PATH}/monthly_trend.csv")
    routes = pd.read_csv(f"{DATA_PATH}/route_delays.csv")
    metrics = pd.read_csv(f"{DATA_PATH}/ml_metrics.csv")
    importance = pd.read_csv(f"{DATA_PATH}/ml_importance.csv")
    return airline, hourly, monthly, routes, metrics, importance

airline_df, hourly_df, monthly_df, routes_df, metrics_df, importance_df = load_data()

# ============================================================
# HEADER
# ============================================================
st.title("✈️ US Flight Delay Analytics Pipeline")
st.markdown("""
**End-to-end PySpark + Databricks pipeline** processing **5.8 million flights** (2015 US domestic data).  
Built using Medallion Architecture (Bronze → Silver → Gold) with PySpark MLlib delay prediction model.
""")
st.divider()

# ============================================================
# SECTION 1 — OVERVIEW METRICS
# ============================================================
st.subheader("📊 Pipeline Overview")

total_flights = monthly_df["TOTAL_FLIGHTS"].sum()
overall_delay_rate = round(
    (airline_df["TOTAL_FLIGHTS"] * airline_df["DELAY_RATE_PCT"]).sum() /
    airline_df["TOTAL_FLIGHTS"].sum(), 2
)
avg_delay = round(
    (airline_df["TOTAL_FLIGHTS"] * airline_df["AVG_ARRIVAL_DELAY"]).sum() /
    airline_df["TOTAL_FLIGHTS"].sum(), 2
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flights Processed", f"{total_flights:,}")
col2.metric("Overall Delay Rate", f"{overall_delay_rate}%")
col3.metric("Avg Arrival Delay", f"{avg_delay} mins")
col4.metric("Airlines Analyzed", f"{len(airline_df)}")

st.divider()

# ============================================================
# SECTION 2 — AIRLINE PERFORMANCE
# ============================================================
st.subheader("🏢 Delay Rate by Airline")

airline_sorted = airline_df.sort_values("DELAY_RATE_PCT", ascending=True)

fig_airline = px.bar(
    airline_sorted,
    x="DELAY_RATE_PCT",
    y="AIRLINE_NAME",
    orientation="h",
    color="DELAY_RATE_PCT",
    color_continuous_scale="RdYlGn_r",
    labels={"DELAY_RATE_PCT": "Delay Rate (%)", "AIRLINE_NAME": "Airline"},
    text="DELAY_RATE_PCT"
)
fig_airline.update_traces(texttemplate="%{text}%", textposition="outside")
fig_airline.update_layout(height=500, coloraxis_showscale=False)
st.plotly_chart(fig_airline, use_container_width=True)

st.divider()

# ============================================================
# SECTION 3 — MONTHLY TREND
# ============================================================
st.subheader("📅 Monthly Delay Trend")

month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
               7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
monthly_df["MONTH_NAME"] = monthly_df["MONTH"].map(month_names)

fig_monthly = go.Figure()
fig_monthly.add_trace(go.Scatter(
    x=monthly_df["MONTH_NAME"],
    y=monthly_df["DELAY_RATE_PCT"],
    mode="lines+markers+text",
    text=monthly_df["DELAY_RATE_PCT"].astype(str) + "%",
    textposition="top center",
    line=dict(color="#e74c3c", width=3),
    marker=dict(size=10)
))
fig_monthly.update_layout(
    xaxis_title="Month",
    yaxis_title="Delay Rate (%)",
    height=400
)
st.plotly_chart(fig_monthly, use_container_width=True)

st.divider()

# ============================================================
# SECTION 4 — TIME OF DAY ANALYSIS
# ============================================================
st.subheader("🕐 Delay Rate by Departure Hour")

fig_hourly = go.Figure()
fig_hourly.add_trace(go.Scatter(
    x=hourly_df["DEPARTURE_HOUR"],
    y=hourly_df["DELAY_RATE_PCT"],
    mode="lines+markers",
    fill="tozeroy",
    line=dict(color="#3498db", width=3),
    marker=dict(size=8)
))
fig_hourly.update_layout(
    xaxis_title="Departure Hour (24h)",
    yaxis_title="Delay Rate (%)",
    height=400,
    xaxis=dict(tickmode="linear", tick0=0, dtick=1)
)
st.plotly_chart(fig_hourly, use_container_width=True)

st.divider()

# ============================================================
# SECTION 5 — TOP DELAYED ROUTES
# ============================================================
st.subheader("🗺️ Top 15 Most Delayed Routes")

routes_df["ROUTE"] = routes_df["ORIGIN_CITY"] + " → " + routes_df["DEST_CITY"]
top_routes = routes_df.nlargest(15, "DELAY_RATE_PCT").sort_values("DELAY_RATE_PCT", ascending=True)

fig_routes = px.bar(
    top_routes,
    x="DELAY_RATE_PCT",
    y="ROUTE",
    orientation="h",
    color="DELAY_RATE_PCT",
    color_continuous_scale="Reds",
    labels={"DELAY_RATE_PCT": "Delay Rate (%)", "ROUTE": "Route"},
    text="DELAY_RATE_PCT"
)
fig_routes.update_traces(texttemplate="%{text}%", textposition="outside")
fig_routes.update_layout(height=500, coloraxis_showscale=False)
st.plotly_chart(fig_routes, use_container_width=True)

st.divider()

# ============================================================
# SECTION 6 — ML MODEL RESULTS
# ============================================================
st.subheader("🤖 ML Model — Random Forest Delay Predictor")
st.markdown("Trained on **5.7 million flights** using PySpark MLlib with a 3-stage pipeline: StringIndexer → VectorAssembler → RandomForestClassifier (50 trees, max depth 10)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{round(metrics_df['ACCURACY'][0]*100, 2)}%")
col2.metric("AUC Score", f"{round(metrics_df['AUC'][0], 4)}")
col3.metric("F1 Score", f"{round(metrics_df['F1_SCORE'][0], 4)}")
col4.metric("Trees Trained", f"{int(metrics_df['NUM_TREES'][0])}")

st.markdown("#### Feature Importance")
importance_sorted = importance_df.sort_values("IMPORTANCE", ascending=True)

fig_importance = px.bar(
    importance_sorted,
    x="IMPORTANCE",
    y="FEATURE",
    orientation="h",
    color="IMPORTANCE",
    color_continuous_scale="Blues",
    labels={"IMPORTANCE": "Importance Score", "FEATURE": "Feature"},
    text=importance_sorted["IMPORTANCE"].round(4)
)
fig_importance.update_traces(textposition="outside")
fig_importance.update_layout(height=400, coloraxis_showscale=False)
st.plotly_chart(fig_importance, use_container_width=True)

st.divider()
st.caption("Built with PySpark + Databricks Community Edition | Medallion Architecture | PySpark MLlib | Streamlit")