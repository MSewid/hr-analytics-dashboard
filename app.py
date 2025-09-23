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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Professional Theme ---
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        color: #2c3e50;
    }
    h1, h2, h3, h4 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .stPlotlyChart {
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .sidebar .stSelectbox > div > div {
        background-color: white;
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
            try:
                df = pd.read_excel("hr_data.xlsx")
            except FileNotFoundError:
                st.error("No data file found. Please upload an Excel file.")
                return None

        # Clean column names
        df.columns = df.columns.str.strip()

        # Fix potential typo in date column
        if "Date Receieved" in df.columns and "Date Received" not in df.columns:
            df.rename(columns={"Date Receieved": "Date Received"}, inplace=True)

        # Convert date columns
        date_cols = ["Date Received", "Date Closed"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Handle missing Date Closed (for open positions)
        if "Date Closed" in df.columns:
            df["Date Closed"] = df["Date Closed"].fillna(pd.Timestamp.now())

        # Calculate Days Open
        if all(col in df.columns for col in ["Date Received", "Date Closed"]):
            df["Days Open"] = (df["Date Closed"] - df["Date Received"]).dt.days.clip(lower=0)
            df["Days Open"] = df["Days Open"].fillna(0)

        # Add time-based columns
        if "Date Received" in df.columns:
            df["Month"] = df["Date Received"].dt.to_period("M").dt.to_timestamp()
            df["Year"] = df["Date Received"].dt.year

        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# --- Metrics Calculation ---
def calculate_metrics(df):
    metrics = {}
    
    total_requirements = df["Req"].sum() if "Req" in df.columns else 0
    total_filled = df[df["Status"] == "Filled"]["Mob"].sum() if all(col in df.columns for col in ["Status", "Mob"]) else 0
    
    metrics["bandwidth"] = (total_filled / total_requirements * 100) if total_requirements > 0 else 0
    
    filled_positions = df[df["Status"] == "Filled"]
    metrics["time_to_fill"] = filled_positions["Days Open"].mean() if (len(filled_positions) > 0 and "Days Open" in df.columns) else 0
    
    open_positions = df[df["Status"] == "In Progress"]
    metrics["aging"] = open_positions["Days Open"].mean() if (len(open_positions) > 0 and "Days Open" in df.columns) else 0
    
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
        req_pred = max(0, round(intercept + slope * t_future))
        future_month = last_month + pd.DateOffset(months=m)
        future_rows.append({"Month": future_month, "Req": req_pred, "Forecast": True})

    df["Forecast"] = False
    future_df = pd.DataFrame(future_rows)
    result = pd.concat([df[["Month", "Req", "Forecast"]], future_df], ignore_index=True)
    result = result.sort_values("Month").reset_index(drop=True)
    result["MA_3"] = result["Req"].rolling(3, min_periods=1).mean()
    
    return result

# --- Main Dashboard ---
def main():
    # Header Section
    st.markdown("""
        <div class="logo-container">
            <img src="https://madre-me.com/wp-content/themes/theme/images/logo.jpg" 
                 alt="Madre Integrated Engineering Logo" 
                 style="max-height: 80px;">
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("HR Analytics Dashboard")
    st.markdown("**Madre Integrated Engineering**")
    st.markdown('</div>', unsafe_allow_html=True)

    # File Upload
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    df = load_and_process_data(uploaded_file)

    if df is None:
        st.stop()

    # Sidebar Filters
    st.sidebar.header("Filters")
    
    # Year filter
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique(), reverse=True)
        selected_year = st.sidebar.selectbox("Year", years, index=0 if years else 0)
        df = df[df["Year"] == selected_year]

    # HR Representatives filter
    if "HR" in df.columns:
        hr_options = sorted(df["HR"].dropna().unique())
        selected_hr = st.sidebar.multiselect("HR Representatives", hr_options, default=hr_options)
        if selected_hr:
            df = df[df["HR"].isin(selected_hr)]

    # Status filter
    if "Status" in df.columns:
        status_options = sorted(df["Status"].dropna().unique())
        selected_status = st.sidebar.multiselect("Status", status_options, default=status_options)
        if selected_status:
            df = df[df["Status"].isin(selected_status)]

    if df.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    # Key Metrics
    st.header("Key Metrics")
    metrics = calculate_metrics(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Bandwidth Utilization", f"{metrics['bandwidth']:.1f}%")

    with col2:
        st.metric("Average Time to Fill", f"{metrics['time_to_fill']:.0f} days")

    with col3:
        st.metric("Aging of Open Requirements", f"{metrics['aging']:.0f} days")

    with col4:
        st.metric("Conversion Rate", f"{metrics['conversion_rate']:.1f}%")

    st.markdown("---")

    # Charts
    st.header("Analytics")
    
    col1, col2 = st.columns(2)

    with col1:
        # Bandwidth over time
        if "Month" in df.columns and all(col in df.columns for col in ["Req", "Mob"]):
            monthly = df.groupby("Month").agg({"Req": "sum", "Mob": "sum"}).reset_index()
            monthly["Utilization"] = (monthly["Mob"] / monthly["Req"] * 100).fillna(0)
            
            fig = px.line(monthly, x="Month", y="Utilization", 
                         title="Bandwidth Utilization Over Time")
            fig.update_traces(line_color='#2c3e50')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Time to fill distribution
        filled = df[df["Status"] == "Filled"]
        if not filled.empty and "Days Open" in df.columns:
            fig = px.histogram(filled, x="Days Open", nbins=20, 
                             title="Time to Fill Distribution")
            fig.update_traces(marker_color='#2c3e50')
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Aging analysis
        open_reqs = df[df["Status"] == "In Progress"]
        if not open_reqs.empty and "Days Open" in df.columns:
            fig = px.box(open_reqs, y="Days Open", 
                        title="Aging of Open Requirements")
            fig.update_traces(marker_color='#2c3e50')
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        # HR Performance
        if all(col in df.columns for col in ["HR", "Status"]):
            conv = df.groupby("HR").apply(
                lambda x: (len(x[x["Status"] == "Filled"]) / len(x[x["Status"].isin(["Filled", "Closed"])]) * 100)
                if len(x[x["Status"].isin(["Filled", "Closed"])]) > 0 else 0
            ).reset_index(name="Conversion Rate")
            
            fig = px.bar(conv, x="HR", y="Conversion Rate", 
                        title="Conversion Rate by HR")
            fig.update_traces(marker_color='#2c3e50')
            st.plotly_chart(fig, use_container_width=True)

    # Frequent Requirements
    if all(col in df.columns for col in ["Designation", "Req"]):
        st.subheader("Most Frequent Requirements")
        freq = df.groupby("Designation")["Req"].sum().reset_index()
        freq = freq.sort_values(by="Req", ascending=False).head(10)
        
        fig = px.bar(freq, x="Designation", y="Req", 
                    title="Top 10 Most Requested Positions")
        fig.update_traces(marker_color='#2c3e50')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Client Analysis
    st.subheader("Client Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if all(col in df.columns for col in ["Client", "Req"]):
            client_req = df.groupby("Client")["Req"].sum().reset_index()
            client_req = client_req.sort_values(by="Req", ascending=False).head(10)
            
            fig = px.bar(client_req, x="Client", y="Req", 
                        title="Top Clients by Requirements")
            fig.update_traces(marker_color='#2c3e50')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if all(col in df.columns for col in ["Client", "Days Open"]):
            client_aging = df.groupby("Client")["Days Open"].mean().reset_index()
            client_aging = client_aging.sort_values(by="Days Open", ascending=False).head(10)
            
            fig = px.bar(client_aging, x="Client", y="Days Open", 
                        title="Average Processing Time by Client")
            fig.update_traces(marker_color='#6c757d')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # Forecasting
    st.subheader("Requirements Forecast")
    if "Month" in df.columns and "Req" in df.columns:
        monthly_req = df.groupby("Month").agg({"Req": "sum"}).reset_index()
        forecast_df = forecast_requirements(monthly_req, months_ahead=6)

        if forecast_df is not None:
            fig = go.Figure()
            
            hist = forecast_df[forecast_df["Forecast"] == False]
            fut = forecast_df[forecast_df["Forecast"] == True]
            
            fig.add_trace(go.Bar(x=hist["Month"], y=hist["Req"], 
                               name="Historical", marker_color='#2c3e50'))
            
            if not fut.empty:
                fig.add_trace(go.Bar(x=fut["Month"], y=fut["Req"], 
                                   name="Forecast", marker_color='#6c757d'))
            
            fig.update_layout(title="6-Month Requirements Forecast",
                            xaxis_title="Month", yaxis_title="Requirements",
                            barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    # Data Summary
    st.markdown("---")
    st.header("Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Status Breakdown")
        if "Status" in df.columns:
            status_summary = df["Status"].value_counts().reset_index()
            status_summary.columns = ["Status", "Count"]
            st.dataframe(status_summary, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("HR Performance")
        if "HR" in df.columns:
            hr_summary = df.groupby("HR").agg({
                "Req": "sum",
                "Status": lambda x: (x == "Filled").sum(),
                "Days Open": "mean"
            }).reset_index()
            hr_summary.columns = ["HR", "Total Requests", "Filled", "Avg Days"]
            st.dataframe(hr_summary, use_container_width=True, hide_index=True)

    # Export
    st.sidebar.markdown("---")
    st.sidebar.header("Export")
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "Download Data",
        csv_data,
        file_name=f"hr_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#6c757d; padding: 1rem;">'
        '<p>Madre Integrated Engineering</p>'
        '<p><a href="https://www.madre-me.com/" target="_blank" style="color: #2c3e50;">www.madre-me.com</a></p>'
        '</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
