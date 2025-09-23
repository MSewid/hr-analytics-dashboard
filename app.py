import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- Streamlit Config ---
st.set_page_config(
    page_title="HR Analytics Dashboard - Madre Integrated Engineering",
    page_icon="📊",
    layout="wide"
)

# --- Custom Theme (Ocean Style) ---
st.markdown("""
    <style>
    :root {
        --ocean-blue: #0b5fa5;
        --ocean-light: #6fb3ff;
        --ocean-muted: #e9f3ff;
        --ocean-grey: #f4f6f8;
    }
    .reportview-container, .main {
        background: linear-gradient(180deg, white 0%, var(--ocean-muted) 100%);
        color: #0f2340;
    }
    h1, h2, h3, h4 {
        color: var(--ocean-blue);
    }
    .stMetric {
        background: white;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(11,95,165,0.08);
    }
    .stPlotlyChart {
        background: white;
        padding: 8px;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(11,95,165,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_and_process_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_excel("hr_data.xlsx")

        df.columns = df.columns.str.strip()

        # Fix typo
        if "Date Receieved" in df.columns and "Date Received" not in df.columns:
            df.rename(columns={"Date Receieved": "Date Received"}, inplace=True)

        for col in ["Date Received", "Date Closed"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "Date Closed" in df.columns:
            df["Date Closed"] = df["Date Closed"].fillna(pd.Timestamp.now())

        if "Date Received" in df.columns and "Date Closed" in df.columns:
            df["Days Open"] = (df["Date Closed"] - df["Date Received"]).dt.days

        if "Date Received" in df.columns:
            df["Month"] = df["Date Received"].dt.to_period("M").dt.to_timestamp()
            df["Year"] = df["Date Received"].dt.year

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Metrics Calculation ---
def calculate_metrics(df):
    metrics = {}
    total_requirements = df["Req"].sum()
    total_filled = df[df["Status"] == "Filled"]["Mob"].sum()

    metrics["bandwidth"] = (total_filled / total_requirements * 100) if total_requirements > 0 else 0
    filled_positions = df[df["Status"] == "Filled"]
    metrics["time_to_fill"] = filled_positions["Days Open"].mean() if len(filled_positions) > 0 else 0
    open_positions = df[df["Status"] == "In Progress"]
    metrics["aging"] = open_positions["Days Open"].mean() if len(open_positions) > 0 else 0
    closed_positions = df[df["Status"].isin(["Filled", "Closed"])]
    filled_count = len(filled_positions)
    total_closed = len(closed_positions)
    metrics["conversion_rate"] = (filled_count / total_closed * 100) if total_closed > 0 else 0

    return metrics

# --- Forecasting Function ---
def forecast_requirements(monthly_req_df, months_ahead=6):
    if monthly_req_df is None or monthly_req_df.shape[0] < 3:
        return monthly_req_df

    df = monthly_req_df.copy().reset_index(drop=True)
    df = df.sort_values("Month")
    df["t"] = np.arange(len(df))

    coef = np.polyfit(df["t"], df["Req"], deg=1)
    slope, intercept = coef[0], coef[1]

    future_rows = []
    last_t = df["t"].iloc[-1]
    last_month = df["Month"].iloc[-1]
    for m in range(1, months_ahead + 1):
        t_future = last_t + m
        req_pred = max(0, intercept + slope * t_future)
        future_month = (last_month + pd.DateOffset(months=m)).replace(day=1)
        future_rows.append({"Month": future_month, "Req": req_pred, "Forecast": True})

    df["Forecast"] = False
    future_df = pd.DataFrame(future_rows)
    result = pd.concat([df[["Month", "Req", "Forecast"]], future_df], ignore_index=True)
    result = result.sort_values("Month").reset_index(drop=True)
    result["MA_3"] = result["Req"].rolling(3, min_periods=1).mean()
    return result

# --- Dashboard ---
def main():
    st.title("HR Analytics Dashboard - Madre Integrated Engineering")
    st.markdown("An interactive and actionable dashboard to monitor HR performance and client needs.")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    df = load_and_process_data(uploaded_file)

    if df is None:
        st.stop()

    # Sidebar filters
    st.sidebar.header("Filters")
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique())
        selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1 if years else 0)
        df = df[df["Year"] == selected_year]

    selected_hr = st.sidebar.multiselect("Select HR Representatives", df["HR"].dropna().unique(), default=df["HR"].dropna().unique())
    selected_status = st.sidebar.multiselect("Select Status", df["Status"].dropna().unique(), default=df["Status"].dropna().unique())

    df = df[(df["HR"].isin(selected_hr)) & (df["Status"].isin(selected_status))]

    if df.empty:
        st.warning("No data for selected filters.")
        st.stop()

    # --- Key HR Metrics Section ---
    st.header("Key HR Metrics")

    metrics = calculate_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bandwidth Utilization", f"{metrics['bandwidth']:.1f}%")
    col2.metric("Avg Time to Fill", f"{metrics['time_to_fill']:.0f} days")
    col3.metric("Aging of Open Req.", f"{metrics['aging']:.0f} days")
    col4.metric("Conversion Rate", f"{metrics['conversion_rate']:.1f}%")

    st.markdown("---")

    # --- Trend & Distribution Charts ---
    st.subheader("Recruitment Trends & Distributions")
    col1, col2 = st.columns(2)

    with col1:
        monthly = df.groupby("Month").agg({"Req": "sum", "Mob": "sum"}).reset_index()
        monthly["Utilization"] = monthly["Mob"] / monthly["Req"] * 100
        fig = px.line(monthly, x="Month", y="Utilization", markers=True, title="Bandwidth Utilization Over Time (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        filled = df[df["Status"] == "Filled"]
        if not filled.empty:
            fig = px.histogram(filled, x="Days Open", nbins=20, title="Distribution of Time to Fill (days)")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        open_reqs = df[df["Status"] == "In Progress"]
        if not open_reqs.empty:
            fig = px.box(open_reqs, y="Days Open", title="Aging of Open Requirements")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        conv = df.groupby("HR").apply(
            lambda x: (len(x[x["Status"] == "Filled"]) / len(x[x["Status"].isin(["Filled", "Closed"])])) * 100
            if len(x[x["Status"].isin(["Filled", "Closed"])]) > 0 else 0
        ).reset_index(name="Conversion Rate")
        fig = px.bar(conv, x="HR", y="Conversion Rate", text_auto=".1f", title="Conversion Rate by HR (%)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Frequent Requirements ---
    st.subheader("Frequent Requirements")
    freq = df.groupby("Designation")["Req"].sum().reset_index().sort_values(by="Req", ascending=False).head(10)
    fig = px.bar(freq, x="Designation", y="Req", text_auto=True, title="Top 10 Frequent Requirements")
    st.plotly_chart(fig, use_container_width=True)

    # --- Client Performance ---
    st.subheader("Client Performance Analysis")
    col1, col2 = st.columns(2)
    with col1:
        client_req = df.groupby("Client")["Req"].sum().reset_index().sort_values(by="Req", ascending=False).head(10)
        fig = px.bar(client_req, x="Client", y="Req", text_auto=True, title="Top 10 Clients by Requirements")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        client_aging = df.groupby("Client")["Days Open"].mean().reset_index().sort_values(by="Days Open", ascending=False).head(10)
        fig = px.bar(client_aging, x="Client", y="Days Open", text_auto=".0f", title="Clients with Longest Average Aging")
        st.plotly_chart(fig, use_container_width=True)

    # --- Forecasting ---
    st.subheader("Predictive Modeling: Forecast Future Requirements")
    monthly_req = df.groupby("Month").agg({"Req": "sum"}).reset_index()
    forecast_df = forecast_requirements(monthly_req, months_ahead=6)

    fig_fc = go.Figure()
    hist = forecast_df[forecast_df["Forecast"] == False]
    fut = forecast_df[forecast_df["Forecast"] == True]
    fig_fc.add_trace(go.Bar(x=hist["Month"], y=hist["Req"], name="Historical"))
    if not fut.empty:
        fig_fc.add_trace(go.Bar(x=fut["Month"], y=fut["Req"], name="Forecast"))
    fig_fc.add_trace(go.Scatter(x=forecast_df["Month"], y=forecast_df["MA_3"], name="3-Month MA", mode="lines+markers"))
    fig_fc.update_layout(title="Requirements Forecast", xaxis_title="Month", yaxis_title="Requirements", barmode="group")
    st.plotly_chart(fig_fc, use_container_width=True)

    # --- Raw Data ---
    st.markdown("---")
    st.subheader("Detailed Data")
    st.dataframe(df, use_container_width=True)

    # --- Export ---
    st.sidebar.markdown("---")
    st.sidebar.header("Export Data")
    csv = df.to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv, file_name=f"hr_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")


if __name__ == "__main__":
    main()
