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

# Ocean theme CSS (white, blue, light blue, light grey) and centered layout improvements
st.markdown(
    f"""
    <style>
    :root {{
        --ocean-blue: #0b5fa5;      /* primary deep blue */
        --ocean-light: #6fb3ff;     /* light blue */
        --ocean-muted: #e9f3ff;     /* very light blue */
        --ocean-grey: #f4f6f8;      /* light grey */
        --card-radius: 10px;
    }}
    /* Page background */
    .reportview-container, .main {{
        background: linear-gradient(180deg, white 0%, var(--ocean-muted) 100%);
        color: #0f2340;
    }}
    /* Header / brand */
    .madre-header {{
        display:flex;
        align-items:center;
        gap:16px;
    }}
    .madre-header img {{
        height:64px;
        border-radius:8px;
        box-shadow: 0 4px 14px rgba(11,95,165,0.12);
    }}
    .madre-title {{
        font-size:20px;
        font-weight:700;
        color: var(--ocean-blue);
    }}
    .madre-sub {{
        font-size:12px;
        color:#1f3a5a;
    }}
    /* Card look for KPI & charts */
    .card {{
        background: white;
        padding: 14px;
        border-radius: var(--card-radius);
        box-shadow: 0 6px 18px rgba(25, 45, 80, 0.06);
        border: 1px solid rgba(11,95,165,0.06);
    }}
    /* centered top row */
    .top-center {{
        display:flex;
        justify-content:center;
        align-items:center;
    }}
    /* style for metric labels */
    .kpi-label {{
        font-size:12px;
        color:#2b4a6f;
        margin-bottom:4px;
    }}
    .small-muted {{
        font-size:12px;
        color:#667f9a;
    }}
    /* sidebar tweaks */
    .css-1d391kg { /* narrow down streamlit default top padding - may vary with streamlit version */
        padding-top: 0rem;
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
    """
    Load the HR Excel file (uploaded or default hr_data.xlsx),
    clean column names, parse dates, create derived fields.
    """
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_excel("hr_data.xlsx")

        # Normalize columns
        df.columns = df.columns.str.strip()

        # Fix common typo name
        if "Date Receieved" in df.columns and "Date Received" not in df.columns:
            df.rename(columns={"Date Receieved": "Date Received"}, inplace=True)

        # Parse dates
        for date_col in ["Date Received", "Date Closed"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Fill Date Closed (for open reqs) with now
        if "Date Closed" in df.columns:
            df["Date Closed"] = df["Date Closed"].fillna(pd.Timestamp.now())

        # Days Open calculation (use existing if present but recompute for consistency)
        if "Date Received" in df.columns and "Date Closed" in df.columns:
            df["Days Open"] = (df["Date Closed"] - df["Date Received"]).dt.days

        # Time features
        if "Date Received" in df.columns:
            df["Month"] = df["Date Received"].dt.to_period("M").dt.to_timestamp()
            df["Month_Str"] = df["Date Received"].dt.strftime("%Y-%m")
            df["Year"] = df["Date Received"].dt.year
            df["Week"] = df["Date Received"].dt.isocalendar().week

        # Fill or ensure numeric fields
        for col in ["Req", "Mob", "Days Open"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Standardize Status strings
        if "Status" in df.columns:
            df["Status"] = df["Status"].astype(str).str.strip()

        # Ensure HR, Client, Designation exist
        for c in ["HR", "Client", "Designation"]:
            if c not in df.columns:
                df[c] = "Unknown"

        return df

    except Exception as e:
        # Return None and let caller show error
        return None


def calculate_metrics(df):
    """Return dictionary with key HR metrics (bandwidth, time to fill, aging, conversion)."""
    metrics = {}
    # Safeguard: if df empty
    if df is None or df.shape[0] == 0:
        return {
            "bandwidth_pct": 0,
            "bandwidth_counts": (0, 0),
            "avg_time_to_fill": 0,
            "avg_aging_open": 0,
            "conversion_pct": 0,
        }

    total_requirements = df["Req"].sum()
    # 'Mob' indicates positions filled (based on your sample)
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

    # Frequent requirements
    freq = df.groupby("Designation")["Req"].sum().reset_index().sort_values(by="Req", ascending=False).head(10)
    metrics["freq_requirements"] = freq

    return metrics


def forecast_requirements(monthly_req_df, months_ahead=6):
    """
    Simple forecasting using linear regression on time index.
    monthly_req_df: DataFrame with columns ['Month' (datetime), 'Req'] sorted ascending.
    Returns DataFrame with historical + forecast, with 'Forecast' boolean flag.
    """
    if monthly_req_df is None or monthly_req_df.shape[0] < 3:
        # Not enough data to forecast - return original w/ no forecast rows
        monthly_req_df = monthly_req_df.copy()
        monthly_req_df["Forecast"] = False
        return monthly_req_df

    df = monthly_req_df.copy().reset_index(drop=True)
    df = df.sort_values("Month")
    # numeric time index
    df["t"] = np.arange(len(df))
    # Fit linear regression Req ~ t
    coef = np.polyfit(df["t"], df["Req"], deg=1)
    slope, intercept = coef[0], coef[1]

    # Build forecast rows
    future_rows = []
    last_t = df["t"].iloc[-1]
    last_month = df["Month"].iloc[-1]
    for m in range(1, months_ahead + 1):
        t_future = last_t + m
        req_pred = intercept + slope * t_future
        # ensure non-negative and round
        req_pred = max(0, req_pred)
        future_month = (last_month + pd.DateOffset(months=m)).replace(day=1)
        future_rows.append({"Month": future_month, "Req": req_pred, "Forecast": True})

    df["Forecast"] = False
    future_df = pd.DataFrame(future_rows)
    result = pd.concat([df[["Month", "Req", "Forecast"]], future_df], ignore_index=True)
    # Add simple moving average (3 months) baseline
    result = result.sort_values("Month").reset_index(drop=True)
    result["MA_3"] = result["Req"].rolling(3, min_periods=1).mean()
    return result


# -----------------------
# App interface
# -----------------------
def main():
    # Header with Madre branding centered
    header_col1, header_col2 = st.columns([1, 6])
    with header_col1:
        st.image(MADRE_LOGO, width=96)
    with header_col2:
        st.markdown(f"""
            <div class="madre-header">
                <div>
                    <div class="madre-title">Madre Integrated Engineering — HR Analytics</div>
                    <div class="madre-sub">Website: <a href="{MADRE_WEBSITE}" target="_blank">{MADRE_WEBSITE}</a></div>
                    <div class="small-muted">Actionable HR insights — Bandwidth, Time-to-Fill, Aging, Conversion, and Forecasts</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # File uploader (outside cached function)
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader("Upload an Excel file (hr_data.xlsx by default)", type=["xlsx", "xls"])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # Load data via cached function
    df = load_and_process_data(st.session_state.uploaded_file)

    if df is None:
        st.error("Could not load data. Please upload a valid Excel file named 'hr_data.xlsx' or use the uploader.")
        st.stop()

    # Basic data sanity info
    st.sidebar.header("Data Info & Filters")
    st.sidebar.info(f"Rows: {len(df):,} • Columns: {len(df.columns)}")

    # Year filter
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique().tolist())
        if not years:
            years = [df["Date Received"].dt.year.min()]
        selected_year = st.sidebar.selectbox("Select Year", years, index=len(years) - 1 if years else 0)
        df = df[df["Year"] == int(selected_year)]

    # HR / Status / Client filters
    selected_hr = st.sidebar.multiselect("HR Representatives", options=sorted(df["HR"].dropna().unique().tolist()), default=sorted(df["HR"].dropna().unique().tolist()))
    selected_status = st.sidebar.multiselect("Status", options=sorted(df["Status"].dropna().unique().tolist()), default=sorted(df["Status"].dropna().unique().tolist()))
    selected_client = st.sidebar.multiselect("Clients (optional)", options=sorted(df["Client"].dropna().unique().tolist()), default=sorted(df["Client"].dropna().unique().tolist()) )

    # Apply filters
    df_filtered = df[
        (df["HR"].isin(selected_hr)) &
        (df["Status"].isin(selected_status)) &
        (df["Client"].isin(selected_client))
    ].copy()

    if df_filtered.empty:
        st.warning("No data matches selected filters. Please change filters.")
        st.stop()

    # Compute metrics
    metrics = calculate_metrics(df_filtered)

    # Top metrics row (centered)
    st.markdown('<div class="top-center">', unsafe_allow_html=True)
    kcol1, kcol2, kcol3, kcol4 = st.columns([1,1,1,1])
    with kcol1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">Bandwidth Utilization</div><h2 style="color:var(--ocean-blue)">{metrics["bandwidth_pct"]:.1f}%</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-muted">Filled / Total = {metrics["bandwidth_counts"][0]}/{metrics["bandwidth_counts"][1]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kcol2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">Avg Time to Fill (filled reqs)</div><h2 style="color:#0b5fa5">{metrics["avg_time_to_fill"]:.0f} days</h2>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Lower is better — target & track over time</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kcol3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">Aging of Open Requirements</div><h2 style="color:#0b5fa5">{metrics["avg_aging_open"]:.0f} days</h2>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Focus on long-standing open reqs</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kcol4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-label">Conversion Rate (filled/closed)</div><h2 style="color:#0b5fa5">{metrics["conversion_pct"]:.1f}%</h2>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Higher = more successful closures</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------
    # Visualizations: Trends, Distributions, HR & Client performance
    # -----------------------
    st.subheader("Recruitment Trends & Distributions")

    # Prepare monthly requirements summary
    if "Month" in df_filtered.columns:
        monthly_req = df_filtered.groupby("Month").agg({"Req": "sum", "Mob": "sum"}).reset_index()
        monthly_req = monthly_req.sort_values("Month").reset_index(drop=True)
        monthly_req["Month_Str"] = monthly_req["Month"].dt.strftime("%Y-%m")
        monthly_req["Utilization_pct"] = (monthly_req["Mob"] / monthly_req["Req"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        # fallback: group by Month_Str if Month missing
        monthly_req = df_filtered.groupby("Month_Str").agg({"Req": "sum", "Mob": "sum"}).reset_index()
        monthly_req["Month"] = pd.to_datetime(monthly_req["Month_Str"] + "-01")
        monthly_req["Utilization_pct"] = (monthly_req["Mob"] / monthly_req["Req"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Columns for charts
    t1, t2 = st.columns(2)
    with t1:
        fig_trend = px.line(monthly_req, x="Month", y="Req", markers=True,
                            title="Monthly Requirements (historical)",
                            labels={"Req": "Total Requirements", "Month": "Month"})
        fig_trend.update_layout(plot_bgcolor="white", height=400)
        st.plotly_chart(fig_trend, use_container_width=True)

    with t2:
        fig_util = px.line(monthly_req, x="Month", y="Utilization_pct", markers=True,
                           title="Bandwidth Utilization (%) Over Time",
                           labels={"Utilization_pct": "Utilization (%)", "Month": "Month"})
        fig_util.update_layout(plot_bgcolor="white", height=400)
        st.plotly_chart(fig_util, use_container_width=True)

    # Distribution charts
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Time to Fill — Distribution (Filled positions)**")
        filled_df = df_filtered[df_filtered["Status"].str.lower() == "filled"]
        if not filled_df.empty:
            fig_hist = px.histogram(filled_df, x="Days Open", nbins=20, title="Time to Fill (days)")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No filled positions in the filtered data to show distribution.")

    with d2:
        st.markdown("**Aging of Open Requirements**")
        open_df = df_filtered[df_filtered["Status"].str.lower() == "in progress"]
        if not open_df.empty:
            fig_box = px.box(open_df, y="Days Open", title="Aging (Days Open) of Active Requirements")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No 'In Progress' positions in the filtered data.")

    st.markdown("---")

    # HR performance (conversion & TTF)
    st.subheader("HR Performance & Top Contributors")
    hr_perf = df_filtered.groupby("HR").agg(
        Total_Req=("Req", "sum"),
        Filled_Count=("Status", lambda x: (x.str.lower() == "filled").sum()),
        Avg_Days_Open=("Days Open", "mean")
    ).reset_index()
    # Fill rate per HR
    hr_perf["Fill_Rate_pct"] = (hr_perf["Filled_Count"] / hr_perf["Total_Req"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    hr1, hr2 = st.columns([2, 1])
    with hr1:
        fig_hr = px.scatter(hr_perf, x="Avg_Days_Open", y="Fill_Rate_pct",
                            size="Total_Req", text="HR",
                            title="HR: Fill Rate vs Average Time to Fill",
                            labels={"Avg_Days_Open": "Avg Days to Fill", "Fill_Rate_pct": "Fill Rate (%)"})
        fig_hr.update_traces(textposition="top center")
        st.plotly_chart(fig_hr, use_container_width=True)
    with hr2:
        st.markdown("**Top HR by Volume**")
        top_hr = hr_perf.sort_values("Total_Req", ascending=False).head(8)
        st.dataframe(top_hr[["HR", "Total_Req", "Fill_Rate_pct", "Avg_Days_Open"]].round(1), height=300)

    st.markdown("---")

    # Frequent requirements & Client performance
    st.subheader("Frequent Requirements & Client Performance")
    fr1, fr2 = st.columns(2)
    with fr1:
        freq = df_filtered.groupby("Designation")["Req"].sum().reset_index().sort_values("Req", ascending=False).head(12)
        fig_freq = px.bar(freq, x="Designation", y="Req", title="Top Frequent Requirements (by Req)", text_auto=True)
        st.plotly_chart(fig_freq, use_container_width=True)

    with fr2:
        client_summary = df_filtered.groupby("Client").agg(
            Total_Req=("Req", "sum"),
            Avg_Days_Open=("Days Open", "mean"),
            Filled_Count=("Status", lambda x: (x.str.lower() == "filled").sum())
        ).reset_index()
        client_summary["Fill_Rate_pct"] = (client_summary["Filled_Count"] / client_summary["Total_Req"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
        top_clients = client_summary.sort_values("Total_Req", ascending=False).head(12)
        fig_client = px.bar(top_clients, x="Client", y="Total_Req", title="Top Clients by Requirements", text_auto=True)
        st.plotly_chart(fig_client, use_container_width=True)

    st.markdown("---")

    # -----------------------
    # Forecasting section
    # -----------------------
    st.header("🔮 Requirements Forecast (simple trend + MA baseline)")
    st.markdown("A simple linear-trend forecast (next 6 months) is shown below alongside a 3-month moving average baseline. This is a lightweight forecast intended for quick decision-making and planning. For production forecasting, consider ARIMA/Prophet/ML models.")

    # Build monthly_req for forecasting: ensure Month col present
    monthly_forecast_input = monthly_req[["Month", "Req"]].rename(columns={"Req": "Req"}).dropna().reset_index(drop=True)
    forecast_df = forecast_requirements(monthly_forecast_input, months_ahead=6)

    # Plot historical + forecast
    fig_fc = go.Figure()
    hist = forecast_df[forecast_df["Forecast"] == False]
    fut = forecast_df[forecast_df["Forecast"] == True]

    fig_fc.add_trace(go.Bar(x=hist["Month"], y=hist["Req"], name="Historical Req", marker_color="#6fb3ff"))
    if not fut.empty:
        fig_fc.add_trace(go.Bar(x=fut["Month"], y=fut["Req"], name="Forecast Req (linear)", marker_color="#0b5fa5", opacity=0.85))
    fig_fc.add_trace(go.Scatter(x=forecast_df["Month"], y=forecast_df["MA_3"], name="3-month MA baseline", mode="lines+markers", line=dict(color="#0f2f57")))
    fig_fc.update_layout(title="Monthly Requirements — Historical + Forecast",
                         xaxis_title="Month", yaxis_title="Requirements", barmode="group", height=450, plot_bgcolor="white")
    st.plotly_chart(fig_fc, use_container_width=True)

    # Download forecast CSV
    csv_fc = forecast_df.to_csv(index=False)
    st.download_button(label="Download Forecast CSV", data=csv_fc, file_name=f"madre_requirements_forecast_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    st.markdown("---")

    # Raw data & export
    st.subheader("📋 Filtered Raw Data")
    st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True, height=360)

    csv_filtered = df_filtered.to_csv(index=False)
    st.download_button(label="Download Filtered Data (CSV)", data=csv_filtered, file_name=f"madre_hr_filtered_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    # Final notes / action items
    st.markdown(
        """
        **Actionable insights (examples):**
        - Focus recruiting resources on designations with high req & low fill rate.
        - Investigate HR reps with high avg days to fill to find process improvements.
        - Prioritize clients with consistently high aging — assign dedicated follow-ups.
        - Use the forecast to pre-plan sourcing and hiring capacity for the next 6 months.
        """
    )

    # Footer with company link
    st.markdown(f'<div style="text-align:center; color:#56738a; padding-top:16px;">Powered for <b>Madre Integrated Engineering</b> — <a href="{MADRE_WEBSITE}" target="_blank">madre-me.com</a></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
