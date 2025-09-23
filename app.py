# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# -----------------------
# Config & Branding
# -----------------------
st.set_page_config(
    page_title="Madre - HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MADRE_LOGO = "https://madre-me.com/wp-content/themes/theme/images/logo.jpg"
MADRE_WEBSITE = "https://www.madre-me.com/"

# Ocean theme CSS (white, blue, light blue, light grey)
st.markdown(
    """
    <style>
    :root {
        --ocean-blue: #0b5fa5;
        --ocean-light: #6fb3ff;
        --ocean-muted: #e9f3ff;
        --ocean-grey: #f4f6f8;
        --card-radius: 10px;
    }
    .reportview-container, .main {
        background: linear-gradient(180deg, white 0%, var(--ocean-muted) 100%);
        color: #0f2340;
    }
    .madre-header {
        display:flex;
        align-items:center;
        gap:16px;
    }
    .madre-header img {
        height:64px;
        border-radius:8px;
        box-shadow: 0 4px 14px rgba(11,95,165,0.12);
    }
    .madre-title {
        font-size:20px;
        font-weight:700;
        color: var(--ocean-blue);
    }
    .madre-sub {
        font-size:12px;
        color:#1f3a5a;
    }
    .card {
        background: white;
        padding: 14px;
        border-radius: var(--card-radius);
        box-shadow: 0 6px 18px rgba(25, 45, 80, 0.06);
        border: 1px solid rgba(11,95,165,0.06);
    }
    .top-center {
        display:flex;
        justify-content:center;
        align-items:center;
    }
    .kpi-label {
        font-size:12px;
        color:#2b4a6f;
        margin-bottom:4px;
    }
    .small-muted {
        font-size:12px;
        color:#667f9a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Utility functions
# -----------------------
@st.cache_data
def load_and_process_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_excel("hr_data.xlsx")

        df.columns = df.columns.str.strip()

        if "Date Receieved" in df.columns and "Date Received" not in df.columns:
            df.rename(columns={"Date Receieved": "Date Received"}, inplace=True)

        for date_col in ["Date Received", "Date Closed"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        if "Date Closed" in df.columns:
            df["Date Closed"] = df["Date Closed"].fillna(pd.Timestamp.now())

        if "Date Received" in df.columns and "Date Closed" in df.columns:
            df["Days Open"] = (df["Date Closed"] - df["Date Received"]).dt.days

        if "Date Received" in df.columns:
            df["Month"] = df["Date Received"].dt.to_period("M").dt.to_timestamp()
            df["Month_Str"] = df["Date Received"].dt.strftime("%Y-%m")
            df["Year"] = df["Date Received"].dt.year
            df["Week"] = df["Date Received"].dt.isocalendar().week

        for col in ["Req", "Mob", "Days Open"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "Status" in df.columns:
            df["Status"] = df["Status"].astype(str).str.strip()

        for c in ["HR", "Client", "Designation"]:
            if c not in df.columns:
                df[c] = "Unknown"

        return df
    except Exception:
        return None


def calculate_metrics(df):
    metrics = {}
    if df is None or df.shape[0] == 0:
        return {
            "bandwidth_pct": 0,
            "bandwidth_counts": (0, 0),
            "avg_time_to_fill": 0,
            "avg_aging_open": 0,
            "conversion_pct": 0,
        }

    total_requirements = df["Req"].sum()
    total_filled = df[df["Status"].str.lower() == "filled"]["Mob"].sum()

    metrics["bandwidth_pct"] = (total_filled / total_requirements * 100) if total_requirements > 0 else 0
    metrics["bandwidth_counts"] = (int(total_filled), int(total_requirements))

    filled_positions = df[df["Status"].str.lower() == "filled"]
    metrics["avg_time_to_fill"] = filled_positions["Days Open"].replace([np.inf, -np.inf], np.nan).mean() if not filled_positions.empty else 0

    open_positions = df[df["Status"].str.lower() == "in progress"]
    metrics["avg_aging_open"] = open_positions["Days Open"].replace([np.inf, -np.inf], np.nan).mean() if not open_positions.empty else 0

    closed_positions = df[df["Status"].str.lower().isin(["filled", "closed"])]
    filled_count = len(filled_positions)
    total_closed = len(closed_positions)
    metrics["conversion_pct"] = (filled_count / total_closed * 100) if total_closed > 0 else 0

    freq = df.groupby("Designation")["Req"].sum().reset_index().sort_values(by="Req", ascending=False).head(10)
    metrics["freq_requirements"] = freq

    return metrics


def forecast_requirements(monthly_req_df, months_ahead=6):
    if monthly_req_df is None or monthly_req_df.shape[0] < 3:
        monthly_req_df = monthly_req_df.copy()
        monthly_req_df["Forecast"] = False
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
        req_pred = intercept + slope * t_future
        req_pred = max(0, req_pred)
        future_month = (last_month + pd.DateOffset(months=m)).replace(day=1)
        future_rows.append({"Month": future_month, "Req": req_pred, "Forecast": True})

    df["Forecast"] = False
    future_df = pd.DataFrame(future_rows)
    result = pd.concat([df[["Month", "Req", "Forecast"]], future_df], ignore_index=True)
    result = result.sort_values("Month").reset_index(drop=True)
    result["MA_3"] = result["Req"].rolling(3, min_periods=1).mean()
    return result


# -----------------------
# App interface
# -----------------------
def main():
    header_col1, header_col2 = st.columns([1, 6])
    with header_col1:
        st.image(MADRE_LOGO, width=96)
    with header_col2:
        st.markdown(f"""
            <div class="madre-header">
                <div>
                    <div class="madre-title">Madre Integrated Engineering — HR Analytics Dashboard</div>
                    <div class="madre-sub">Website: <a href="{MADRE_WEBSITE}" target="_blank">{MADRE_WEBSITE}</a></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader("Upload an Excel file (default: hr_data.xlsx)", type=["xlsx", "xls"])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    df = load_and_process_data(st.session_state.uploaded_file)

    if df is None:
        st.error("Could not load data. Please upload a valid Excel file named 'hr_data.xlsx'.")
        st.stop()

    st.sidebar.header("Filters")
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique().tolist())
        selected_year = st.sidebar.selectbox("Select Year", years, index=len(years) - 1 if years else 0)
        df = df[df["Year"] == int(selected_year)]

    selected_hr = st.sidebar.multiselect("HR Representatives", options=sorted(df["HR"].dropna().unique().tolist()), default=sorted(df["HR"].dropna().unique().tolist()))
    selected_status = st.sidebar.multiselect("Status", options=sorted(df["Status"].dropna().unique().tolist()), default=sorted(df["Status"].dropna().unique().tolist()))
    selected_client = st.sidebar.multiselect("Clients", options=sorted(df["Client"].dropna().unique().tolist()), default=sorted(df["Client"].dropna().unique().tolist()))

    df_filtered = df[
        (df["HR"].isin(selected_hr)) &
        (df["Status"].isin(selected_status)) &
        (df["Client"].isin(selected_client))
    ].copy()

    if df_filtered.empty:
        st.warning("No data matches selected filters.")
        st.stop()

    metrics = calculate_metrics(df_filtered)

    st.markdown('<div class="top-center">', unsafe_allow_html=True)
    kcol1, kcol2, kcol3, kcol4 = st.columns([1,1,1,1])
    with kcol1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">Bandwidth Utilization</div><h2 style="color:var(--ocean-blue)">{metrics["bandwidth_pct"]:.1f}%</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-muted">Filled / Total = {metrics["bandwidth_counts"][0]}/{metrics["bandwidth_counts"][1]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kcol2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">Avg Time to Fill</div><h2 style="color:#0b5fa5">{metrics["avg_time_to_fill"]:.0f} days</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kcol3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">Aging of Open Requirements</div><h2 style="color:#0b5fa5">{metrics["avg_aging_open"]:.0f} days</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kcol4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">Conversion Rate</div><h2 style="color:#0b5fa5">{metrics["conversion_pct"]:.1f}%</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Recruitment Trends & Distributions")
    monthly_req = df_filtered.groupby("Month").agg({"Req": "sum", "Mob": "sum"}).reset_index()
    monthly_req = monthly_req.sort_values("Month").reset_index(drop=True)
    monthly_req["Utilization_pct"] = (monthly_req["Mob"] / monthly_req["Req"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    fig_trend = px.line(monthly_req, x="Month", y="Req", markers=True, title="Monthly Requirements")
    st.plotly_chart(fig_trend, use_container_width=True)

    fig_util = px.line(monthly_req, x="Month", y="Utilization_pct", markers=True, title="Bandwidth Utilization (%)")
    st.plotly_chart(fig_util, use_container_width=True)

    st.subheader("Forecasting")
    forecast_df = forecast_requirements(monthly_req[["Month", "Req"]], months_ahead=6)
    fig_fc = go.Figure()
    hist = forecast_df[forecast_df["Forecast"] == False]
    fut = forecast_df[forecast_df["Forecast"] == True]
    fig_fc.add_trace(go.Bar(x=hist["Month"], y=hist["Req"], name="Historical"))
    if not fut.empty:
        fig_fc.add_trace(go.Bar(x=fut["Month"], y=fut["Req"], name="Forecast"))
    fig_fc.add_trace(go.Scatter(x=forecast_df["Month"], y=forecast_df["MA_3"], name="3-Month MA", mode="lines+markers"))
    fig_fc.update_layout(title="Monthly Requirements Forecast", xaxis_title="Month", yaxis_title="Requirements", barmode="group")
    st.plotly_chart(fig_fc, use_container_width=True)

    st.subheader("Raw Data (Filtered)")
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True, height=360)

    csv_filtered = df_filtered.to_csv(index=False)
    st.download_button(label="Download Filtered Data (CSV)", data=csv_filtered, file_name=f"madre_hr_filtered_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    st.markdown(f'<div style="text-align:center; color:#56738a; padding-top:16px;">Powered for <b>Madre Integrated Engineering</b> — <a href="{MADRE_WEBSITE}" target="_blank">madre-me.com</a></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
