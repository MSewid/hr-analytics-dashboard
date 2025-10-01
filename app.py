import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# --- Streamlit Config ---
st.set_page_config(
    page_title="HR Analytics Dashboard - Madre Integrated Engineering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Ocean Theme ---
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        color: #0f172a;
    }
    h1, h2, h3, h4 {
        color: #0077b6;
        font-family: sans-serif;
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
        background: linear-gradient(135deg, #f1f5f9 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #0077b6;
        box-shadow: 0 2px 4px rgba(0,119,182,0.1);
    }
    .stPlotlyChart {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    .sidebar .stSelectbox > div > div {
        background-color: #ffffff;
    }
    .sidebar {
        background-color: #f1f5f9;
    }
    .info-box {
        background: #e0f2fe;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0077b6;
        margin: 1rem 0;
    }
    .insight-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #0ea5e9;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_and_process_data(uploaded_file=None):
    """
    Load and preprocess HR data from Excel file.
    
    What this does:
    - Loads data from uploaded file or default file
    - Cleans column names and fixes typos
    - Converts dates to proper format
    - Calculates how long positions have been open
    - Adds time-based columns for analysis
    """
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            try:
                df = pd.read_excel("hr_data.xlsx")
            except FileNotFoundError:
                st.error("📁 No data file found. Please upload an Excel file.")
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
            df["Quarter"] = df["Date Received"].dt.to_period("Q").astype(str)
            df["Week"] = df["Date Received"].dt.isocalendar().week

        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# --- Metrics Calculation ---
def calculate_metrics(df):
    """
    Calculate key HR performance metrics.
    
    Metrics explained:
    - Bandwidth: % of required positions that are filled
    - Time to Fill: Average days to fill a position
    - Aging: Average days open positions have been waiting
    - Conversion Rate: % of closed positions that were successfully filled
    """
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
    
    # Additional metrics
    metrics["total_open"] = len(open_positions)
    metrics["total_filled"] = len(filled_positions)
    metrics["total_requirements"] = total_requirements
    
    return metrics

# --- Enhanced Forecasting Function ---
def advanced_forecast_requirements(monthly_req_df, months_ahead=6):
    """
    Advanced forecasting using multiple methods:
    - Linear Trend: Simple trend line
    - Moving Average: Smoothed prediction based on recent patterns
    - Seasonal Adjustment: Accounts for recurring patterns
    
    Returns forecast with confidence intervals to show uncertainty.
    """
    if monthly_req_df is None or monthly_req_df.shape[0] < 3:
        return monthly_req_df, None

    df = monthly_req_df.copy().reset_index(drop=True)
    df = df.sort_values("Month")
    df["t"] = np.arange(len(df))

    # Linear trend
    coef = np.polyfit(df["t"], df["Req"], deg=1)
    slope, intercept = coef[0], coef[1]
    df["Trend"] = intercept + slope * df["t"]

    # Calculate residuals and standard error
    residuals = df["Req"] - df["Trend"]
    std_error = np.std(residuals)

    # Moving average with exponential weighting
    df["MA_3"] = df["Req"].rolling(3, min_periods=1).mean()
    df["EMA"] = df["Req"].ewm(span=3, adjust=False).mean()

    # Generate future predictions
    future_rows = []
    last_t = df["t"].iloc[-1]
    last_month = df["Month"].iloc[-1]
    
    for m in range(1, months_ahead + 1):
        t_future = last_t + m
        
        # Linear trend prediction
        trend_pred = intercept + slope * t_future
        
        # Add seasonal component if enough data
        if len(df) >= 12:
            month_num = (last_month + pd.DateOffset(months=m)).month
            seasonal_factor = df.groupby(df["Month"].dt.month)["Req"].mean().get(month_num, 0)
            overall_mean = df["Req"].mean()
            seasonal_adj = seasonal_factor - overall_mean if overall_mean > 0 else 0
        else:
            seasonal_adj = 0
        
        # Combined forecast
        req_pred = max(0, round(trend_pred + seasonal_adj * 0.3))
        
        # Confidence intervals (95%)
        confidence_margin = 1.96 * std_error * np.sqrt(1 + 1/len(df))
        lower_bound = max(0, round(req_pred - confidence_margin))
        upper_bound = round(req_pred + confidence_margin)
        
        future_month = last_month + pd.DateOffset(months=m)
        future_rows.append({
            "Month": future_month,
            "Req": req_pred,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Forecast": True
        })

    df["Forecast"] = False
    df["Lower_Bound"] = df["Req"]
    df["Upper_Bound"] = df["Req"]
    
    future_df = pd.DataFrame(future_rows)
    result = pd.concat([df[["Month", "Req", "Forecast", "Lower_Bound", "Upper_Bound"]], future_df], ignore_index=True)
    result = result.sort_values("Month").reset_index(drop=True)
    
    # Calculate forecast accuracy metrics
    accuracy_metrics = {
        "trend_slope": slope,
        "std_error": std_error,
        "r_squared": 1 - (np.sum(residuals**2) / np.sum((df["Req"] - df["Req"].mean())**2)) if len(df) > 1 else 0
    }
    
    return result, accuracy_metrics

# --- Predictive Analytics ---
def predict_time_to_fill(df):
    """
    Predict expected time to fill for open positions based on historical data.
    Uses factors like designation, client, and current aging.
    """
    if df.empty or "Days Open" not in df.columns:
        return None
    
    filled = df[df["Status"] == "Filled"].copy()
    if len(filled) < 5:
        return None
    
    # Calculate average time by designation
    designation_avg = filled.groupby("Designation")["Days Open"].mean().to_dict()
    
    # Calculate average time by client
    client_avg = filled.groupby("Client")["Days Open"].mean().to_dict() if "Client" in df.columns else {}
    
    # Overall average
    overall_avg = filled["Days Open"].mean()
    
    return {
        "designation_avg": designation_avg,
        "client_avg": client_avg,
        "overall_avg": overall_avg,
        "median": filled["Days Open"].median(),
        "percentile_75": filled["Days Open"].quantile(0.75),
        "percentile_90": filled["Days Open"].quantile(0.90)
    }

# --- Risk Assessment ---
def assess_hiring_risks(df):
    """
    Identify positions at risk of delayed filling.
    A position is at risk if it's been open longer than the 75th percentile.
    """
    if df.empty or "Days Open" not in df.columns:
        return None
    
    filled = df[df["Status"] == "Filled"]
    open_positions = df[df["Status"] == "In Progress"]
    
    if len(filled) < 5 or len(open_positions) == 0:
        return None
    
    # Calculate risk threshold (75th percentile of filled positions)
    risk_threshold = filled["Days Open"].quantile(0.75)
    
    # Identify at-risk positions
    at_risk = open_positions[open_positions["Days Open"] > risk_threshold].copy()
    at_risk["Days_Over_Threshold"] = at_risk["Days Open"] - risk_threshold
    at_risk = at_risk.sort_values("Days_Over_Threshold", ascending=False)
    
    return {
        "at_risk_positions": at_risk,
        "risk_threshold": risk_threshold,
        "at_risk_count": len(at_risk),
        "total_open": len(open_positions),
        "risk_percentage": (len(at_risk) / len(open_positions) * 100) if len(open_positions) > 0 else 0
    }

# --- Trend Analysis ---
def analyze_trends(df):
    """
    Analyze hiring trends over time to identify patterns and changes.
    """
    if "Month" not in df.columns:
        return None
    
    monthly = df.groupby("Month").agg({
        "Req": "sum",
        "Days Open": "mean"
    }).reset_index()
    
    if len(monthly) < 3:
        return None
    
    # Calculate month-over-month changes
    monthly["Req_Change"] = monthly["Req"].pct_change() * 100
    monthly["Days_Change"] = monthly["Days Open"].pct_change() * 100
    
    # Identify trend direction
    recent_months = monthly.tail(3)
    req_trend = "increasing" if recent_months["Req"].is_monotonic_increasing else \
                "decreasing" if recent_months["Req"].is_monotonic_decreasing else "stable"
    
    days_trend = "increasing" if recent_months["Days Open"].is_monotonic_increasing else \
                 "decreasing" if recent_months["Days Open"].is_monotonic_decreasing else "stable"
    
    return {
        "monthly_data": monthly,
        "req_trend": req_trend,
        "days_trend": days_trend,
        "avg_req_change": monthly["Req_Change"].tail(3).mean(),
        "avg_days_change": monthly["Days_Change"].tail(3).mean()
    }

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
    st.title("📊 HR Analytics Dashboard")
    st.markdown("**Madre Integrated Engineering**")
    st.markdown("*Real-time insights into recruitment performance and predictions*")
    st.markdown('</div>', unsafe_allow_html=True)

    # Information Box
    with st.expander("ℹ️ How to Use This Dashboard", expanded=False):
        st.markdown("""
        **Welcome to the HR Analytics Dashboard!** This tool helps you understand and predict hiring patterns.
        
        **Key Features:**
        - 📈 **Key Metrics**: Overview of current recruitment performance
        - 🔮 **Predictive Analysis**: Forecast future hiring needs with confidence intervals
        - ⚠️ **Risk Assessment**: Identify positions at risk of delays
        - 📊 **Trend Analysis**: Understand patterns in your hiring process
        
        **Getting Started:**
        1. Upload your Excel file or use the default data
        2. Use filters in the sidebar to focus on specific time periods or HR reps
        3. Scroll through sections to explore different insights
        4. Hover over charts for detailed information
        5. Download processed data using the export button
        """)

    # File Upload
    uploaded_file = st.file_uploader("📁 Upload Excel file (optional)", type=["xlsx", "xls"])
    df = load_and_process_data(uploaded_file)

    if df is None:
        st.stop()

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")
    
    # Year filter
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique(), reverse=True)
        selected_year = st.sidebar.selectbox("📅 Year", years, index=0 if years else 0)
        df_filtered = df[df["Year"] == selected_year].copy()
    else:
        df_filtered = df.copy()

    # HR Representatives filter
    if "HR" in df.columns:
        hr_options = sorted(df_filtered["HR"].dropna().unique())
        selected_hr = st.sidebar.multiselect("👥 HR Representatives", hr_options, default=hr_options)
        if selected_hr:
            df_filtered = df_filtered[df_filtered["HR"].isin(selected_hr)]

    # Status filter
    if "Status" in df.columns:
        status_options = sorted(df_filtered["Status"].dropna().unique())
        selected_status = st.sidebar.multiselect("📋 Status", status_options, default=status_options)
        if selected_status:
            df_filtered = df_filtered[df_filtered["Status"].isin(selected_status)]

    if df_filtered.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
        st.stop()

    # Key Metrics
    st.header("📊 Key Performance Metrics")
    st.markdown("""
    <div class="info-box">
    <b>What these metrics mean:</b><br>
    • <b>Bandwidth Utilization</b>: Percentage of required positions that have been filled<br>
    • <b>Time to Fill</b>: Average number of days to fill a position<br>
    • <b>Aging</b>: Average days that open positions have been waiting<br>
    • <b>Conversion Rate</b>: Percentage of recruitment processes that successfully filled positions
    </div>
    """, unsafe_allow_html=True)
    
    metrics = calculate_metrics(df_filtered)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Bandwidth Utilization", f"{metrics['bandwidth']:.1f}%",
                 delta=f"{metrics['total_filled']}/{metrics['total_requirements']} filled")

    with col2:
        st.metric("Average Time to Fill", f"{metrics['time_to_fill']:.0f} days",
                 help="Lower is better - shows recruitment efficiency")

    with col3:
        st.metric("Aging of Open Requirements", f"{metrics['aging']:.0f} days",
                 delta=f"{metrics['total_open']} open",
                 help="Average days open positions have been waiting")

    with col4:
        st.metric("Conversion Rate", f"{metrics['conversion_rate']:.1f}%",
                 help="Percentage of recruitment processes successfully completed")

    st.markdown("---")

    # Predictive Analysis Section
    st.header("🔮 Predictive Analysis & Forecasting")
    
    # Requirements Forecast
    st.subheader("📈 Requirements Forecast (6 Months)")
    st.markdown("""
    <div class="info-box">
    <b>Understanding the Forecast:</b><br>
    This prediction uses historical patterns to estimate future hiring needs. The shaded area shows the confidence interval 
    (range where actual values are likely to fall). A wider range means more uncertainty in the prediction.
    </div>
    """, unsafe_allow_html=True)
    
    if "Month" in df.columns and "Req" in df.columns:
        monthly_req = df.groupby("Month").agg({"Req": "sum"}).reset_index()
        forecast_df, accuracy = advanced_forecast_requirements(monthly_req, months_ahead=6)

        if forecast_df is not None and accuracy is not None:
            fig = go.Figure()
            
            hist = forecast_df[forecast_df["Forecast"] == False]
            fut = forecast_df[forecast_df["Forecast"] == True]
            
            # Historical data
            fig.add_trace(go.Bar(
                x=hist["Month"], 
                y=hist["Req"], 
                name="Historical Data", 
                marker_color='#0077b6',
                text=hist["Req"],
                textposition='outside'
            ))
            
            # Forecast
            if not fut.empty:
                fig.add_trace(go.Bar(
                    x=fut["Month"], 
                    y=fut["Req"], 
                    name="Forecast", 
                    marker_color='#90e0ef',
                    text=fut["Req"],
                    textposition='outside'
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=fut["Month"].tolist() + fut["Month"].tolist()[::-1],
                    y=fut["Upper_Bound"].tolist() + fut["Lower_Bound"].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,119,182,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
            
            fig.update_layout(
                title="Monthly Requirements: Historical + 6-Month Forecast",
                xaxis_title="Month",
                yaxis_title="Number of Requirements",
                barmode="group",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast insights
            col1, col2, col3 = st.columns(3)
            with col1:
                trend_direction = "📈 Increasing" if accuracy["trend_slope"] > 0 else "📉 Decreasing"
                st.metric("Trend Direction", trend_direction)
            with col2:
                st.metric("Forecast Accuracy (R²)", f"{accuracy['r_squared']:.2f}",
                         help="Closer to 1.0 means more reliable predictions")
            with col3:
                avg_forecast = fut["Req"].mean() if not fut.empty else 0
                st.metric("Avg Monthly Forecast", f"{avg_forecast:.0f}")

    # Trend Analysis
    st.subheader("📊 Hiring Trends Analysis")
    trends = analyze_trends(df)
    
    if trends:
        col1, col2 = st.columns(2)
        
        with col1:
            trend_emoji = "📈" if trends["req_trend"] == "increasing" else "📉" if trends["req_trend"] == "decreasing" else "➡️"
            st.markdown(f"""
            <div class="insight-box">
            <b>{trend_emoji} Requirements Trend:</b> {trends["req_trend"].capitalize()}<br>
            <small>Average change in last 3 months: {trends["avg_req_change"]:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            trend_emoji = "⚠️" if trends["days_trend"] == "increasing" else "✅" if trends["days_trend"] == "decreasing" else "➡️"
            st.markdown(f"""
            <div class="insight-box">
            <b>{trend_emoji} Time-to-Fill Trend:</b> {trends["days_trend"].capitalize()}<br>
            <small>Average change in last 3 months: {trends["days_trend_change"]:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)

    # Risk Assessment
    st.subheader("⚠️ Risk Assessment: Positions at Risk")
    st.markdown("""
    <div class="info-box">
    <b>What is a "Risk Position"?</b><br>
    Positions that have been open longer than 75% of historical filled positions are flagged as "at risk". 
    These may need special attention to avoid further delays.
    </div>
    """, unsafe_allow_html=True)
    
    risk_data = assess_hiring_risks(df_filtered)
    
    if risk_data and risk_data["at_risk_count"] > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("At-Risk Positions", risk_data["at_risk_count"])
        with col2:
            st.metric("Total Open Positions", risk_data["total_open"])
        with col3:
            color = "🔴" if risk_data["risk_percentage"] > 30 else "🟡" if risk_data["risk_percentage"] > 15 else "🟢"
            st.metric(f"{color} Risk Level", f"{risk_data['risk_percentage']:.1f}%")
        
        st.markdown(f"**Risk Threshold:** Positions open for more than {risk_data['risk_threshold']:.0f} days")
        
        # Display at-risk positions
        if not risk_data["at_risk_positions"].empty:
            display_cols = ["Designation", "Client", "Days Open", "Days_Over_Threshold", "HR"]
            available_cols = [col for col in display_cols if col in risk_data["at_risk_positions"].columns]
            st.dataframe(
                risk_data["at_risk_positions"][available_cols].head(10),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.success("✅ No positions currently at risk! All open positions are within normal time frames.")

    # Time to Fill Predictions
    st.subheader("⏱️ Time-to-Fill Predictions")
    st.markdown("""
    <div class="info-box">
    <b>Predicted Time to Fill:</b><br>
    Based on historical data, these are typical timeframes for filling positions. Use these estimates for planning purposes.
    </div>
    """, unsafe_allow_html=True)
    
    ttf_predictions = predict_time_to_fill(df)
    
    if ttf_predictions:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average", f"{ttf_predictions['overall_avg']:.0f} days")
        with col2:
            st.metric("Median (50%)", f"{ttf_predictions['median']:.0f} days",
                     help="Half of positions fill faster than this")
        with col3:
            st.metric("75th Percentile", f"{ttf_predictions['percentile_75']:.0f} days",
                     help="75% of positions fill within this time")
        with col4:
            st.metric("90th Percentile", f"{ttf_predictions['percentile_90']:.0f} days",
                     help="90% of positions fill within this time")

    st.markdown("---")

    # Analytics Charts
    st.header("📈 Analytics & Visualizations")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bandwidth Utilization Over Time")
        if "Month" in df_filtered.columns and all(col in df_filtered.columns for col in ["Req", "Mob"]):
            monthly = df_filtered.groupby("Month").agg({"Req": "sum", "Mob": "sum"}).reset_index()
            monthly["Utilization"] = (monthly["Mob"] / monthly["Req"] * 100).fillna(0)
            
            fig = px.line(monthly, x="Month", y="Utilization",
                         title="Bandwidth Utilization Trend (%)")
            fig.update_traces(marker_color='#0077b6')
            fig.add_hline(y=conv["Conversion Rate"].mean(), line_dash="dash",
                         line_color="red", annotation_text="Average")
            st.plotly_chart(fig, use_container_width=True)
            
            best_performer = conv.loc[conv["Conversion Rate"].idxmax(), "HR"]
            st.markdown(f"**Top Performer:** {best_performer} ({conv['Conversion Rate'].max():.1f}%)")

    # Frequent Requirements
    st.subheader("🎯 Most Frequent Requirements")
    st.markdown("*Top 10 positions with highest demand*")
    
    if all(col in df_filtered.columns for col in ["Designation", "Req"]):
        freq = df_filtered.groupby("Designation")["Req"].sum().reset_index()
        freq = freq.sort_values(by="Req", ascending=False).head(10)
        
        fig = px.bar(freq, x="Designation", y="Req",
                    title="Top 10 Most Requested Positions",
                    text="Req")
        fig.update_traces(marker_color='#0077b6', textposition='outside')
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Client Analysis
    st.subheader("🏢 Client Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Requirements by Client**")
        if all(col in df_filtered.columns for col in ["Client", "Req"]):
            client_req = df_filtered.groupby("Client")["Req"].sum().reset_index()
            client_req = client_req.sort_values(by="Req", ascending=False).head(10)
            
            fig = px.bar(client_req, x="Client", y="Req",
                        title="Top 10 Clients by Requirements Volume",
                        text="Req")
            fig.update_traces(marker_color='#0077b6', textposition='outside')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Average Processing Time by Client**")
        if all(col in df_filtered.columns for col in ["Client", "Days Open"]):
            client_aging = df_filtered.groupby("Client")["Days Open"].mean().reset_index()
            client_aging = client_aging.sort_values(by="Days Open", ascending=False).head(10)
            
            fig = px.bar(client_aging, x="Client", y="Days Open",
                        title="Avg Days to Process by Client",
                        text=client_aging["Days Open"].round(0))
            fig.update_traces(marker_color='#64748b', textposition='outside')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # Seasonal Analysis
    if "Month" in df.columns and len(df) >= 12:
        st.subheader("📅 Seasonal Patterns")
        st.markdown("""
        <div class="info-box">
        <b>Understanding Seasonal Trends:</b><br>
        This chart shows if certain months consistently have higher or lower hiring activity.
        Use this to plan recruitment resources and anticipate busy periods.
        </div>
        """, unsafe_allow_html=True)
        
        monthly_pattern = df.groupby(df["Date Received"].dt.month)["Req"].mean().reset_index()
        monthly_pattern.columns = ["Month_Num", "Avg_Req"]
        monthly_pattern["Month_Name"] = monthly_pattern["Month_Num"].apply(
            lambda x: datetime(2024, x, 1).strftime("%B")
        )
        
        fig = px.bar(monthly_pattern, x="Month_Name", y="Avg_Req",
                    title="Average Requirements by Month (Historical Pattern)",
                    text=monthly_pattern["Avg_Req"].round(1))
        fig.update_traces(marker_color='#0077b6', textposition='outside')
        fig.update_layout(xaxis_title="Month", yaxis_title="Average Requirements")
        st.plotly_chart(fig, use_container_width=True)
        
        busiest_month = monthly_pattern.loc[monthly_pattern["Avg_Req"].idxmax(), "Month_Name"]
        slowest_month = monthly_pattern.loc[monthly_pattern["Avg_Req"].idxmin(), "Month_Name"]
        st.markdown(f"**📊 Insight:** Busiest month is typically **{busiest_month}**, "
                   f"while **{slowest_month}** tends to be slower.")

    # Conversion Funnel
    st.subheader("🔄 Recruitment Conversion Funnel")
    st.markdown("""
    <div class="info-box">
    <b>Conversion Funnel:</b><br>
    Shows the flow from initial requirements to filled positions. Each stage represents 
    how many positions moved through the recruitment process.
    </div>
    """, unsafe_allow_html=True)
    
    if "Status" in df_filtered.columns:
        funnel_data = {
            "Stage": ["Total Requirements", "Closed/Filled", "Successfully Filled"],
            "Count": [
                len(df_filtered),
                len(df_filtered[df_filtered["Status"].isin(["Filled", "Closed"])]),
                len(df_filtered[df_filtered["Status"] == "Filled"])
            ]
        }
        funnel_df = pd.DataFrame(funnel_data)
        
        fig = go.Figure(go.Funnel(
            y=funnel_df["Stage"],
            x=funnel_df["Count"],
            textposition="inside",
            textinfo="value+percent initial",
            marker={"color": ["#0077b6", "#0096c7", "#00b4d8"]}
        ))
        fig.update_layout(title="Recruitment Process Funnel", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Data Summary
    st.header("📋 Data Summary & Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Status Breakdown")
        if "Status" in df_filtered.columns:
            status_summary = df_filtered["Status"].value_counts().reset_index()
            status_summary.columns = ["Status", "Count"]
            status_summary["Percentage"] = (status_summary["Count"] / status_summary["Count"].sum() * 100).round(1)
            status_summary["Percentage"] = status_summary["Percentage"].astype(str) + "%"
            st.dataframe(status_summary, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("HR Performance Summary")
        if "HR" in df_filtered.columns:
            hr_summary = df_filtered.groupby("HR").agg({
                "Req": "sum",
                "Status": lambda x: (x == "Filled").sum(),
                "Days Open": "mean"
            }).reset_index()
            hr_summary.columns = ["HR", "Total Requests", "Filled", "Avg Days"]
            hr_summary["Avg Days"] = hr_summary["Avg Days"].round(1)
            hr_summary = hr_summary.sort_values("Total Requests", ascending=False)
            st.dataframe(hr_summary, use_container_width=True, hide_index=True)

    # Detailed Position Status
    with st.expander("📊 View Detailed Position Data", expanded=False):
        st.markdown("**Current Open Positions** (sorted by days open)")
        open_positions = df_filtered[df_filtered["Status"] == "In Progress"].copy()
        if not open_positions.empty:
            display_cols = ["Designation", "Client", "Req", "Days Open", "HR", "Date Received"]
            available_cols = [col for col in display_cols if col in open_positions.columns]
            open_positions_sorted = open_positions[available_cols].sort_values("Days Open", ascending=False)
            st.dataframe(open_positions_sorted, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions matching the current filters.")

    # Key Insights Summary
    st.markdown("---")
    st.header("💡 Key Insights & Recommendations")
    
    insights = []
    
    # Bandwidth insight
    if metrics["bandwidth"] < 70:
        insights.append(f"⚠️ **Low Bandwidth Alert:** Current utilization is {metrics['bandwidth']:.1f}%. "
                       f"Consider reviewing recruitment strategies to improve fill rates.")
    elif metrics["bandwidth"] > 90:
        insights.append(f"✅ **Excellent Bandwidth:** {metrics['bandwidth']:.1f}% utilization shows strong performance!")
    
    # Time to fill insight
    if ttf_predictions and metrics["time_to_fill"] > ttf_predictions["percentile_75"]:
        insights.append(f"⏰ **Time to Fill Above Average:** Current average ({metrics['time_to_fill']:.0f} days) "
                       f"exceeds the 75th percentile. Consider streamlining the process.")
    
    # Risk insight
    if risk_data and risk_data["risk_percentage"] > 25:
        insights.append(f"🔴 **High Risk Level:** {risk_data['risk_percentage']:.1f}% of open positions are at risk. "
                       f"Prioritize positions open for more than {risk_data['risk_threshold']:.0f} days.")
    
    # Trend insight
    if trends:
        if trends["req_trend"] == "increasing":
            insights.append(f"📈 **Growing Demand:** Requirements are trending upward. "
                           f"Consider increasing recruitment resources.")
        if trends["days_trend"] == "increasing":
            insights.append(f"📉 **Efficiency Declining:** Time-to-fill is increasing. "
                           f"Review bottlenecks in the hiring process.")
    
    # Forecast insight
    if accuracy and accuracy["trend_slope"] > 5:
        insights.append(f"🔮 **Strong Growth Forecast:** Projected requirements show significant upward trend. "
                       f"Plan for increased capacity in coming months.")
    
    if insights:
        for insight in insights:
            st.markdown(f"""
            <div class="insight-box">
            {insight}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ All metrics are within healthy ranges. Keep up the good work!")

    # Action Items
    st.subheader("✅ Recommended Action Items")
    action_items = []
    
    if risk_data and risk_data["at_risk_count"] > 0:
        action_items.append(f"Review {risk_data['at_risk_count']} at-risk positions and escalate as needed")
    
    if metrics["time_to_fill"] > 60:
        action_items.append("Conduct process review to identify time-to-fill bottlenecks")
    
    if trends and trends["req_trend"] == "increasing":
        action_items.append("Prepare recruitment resources for anticipated demand increase")
    
    action_items.append("Regular weekly review of open positions with HR team")
    action_items.append("Update job descriptions for high-demand positions")
    
    for i, item in enumerate(action_items, 1):
        st.markdown(f"{i}. {item}")

    st.markdown("---")

    # Export Section
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export Data")
    
    export_options = st.sidebar.radio(
        "Choose export format:",
        ["Filtered Data", "Summary Report", "At-Risk Positions"]
    )
    
    if export_options == "Filtered Data":
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        file_name = f"hr_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    elif export_options == "Summary Report":
        summary_df = pd.DataFrame([metrics])
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        file_name = f"hr_summary_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    else:
        if risk_data and not risk_data["at_risk_positions"].empty:
            csv_data = risk_data["at_risk_positions"].to_csv(index=False).encode('utf-8')
        else:
            csv_data = pd.DataFrame().to_csv(index=False).encode('utf-8')
        file_name = f"hr_at_risk_positions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    st.sidebar.download_button(
        "⬇️ Download Data",
        csv_data,
        file_name=file_name,
        mime="text/csv",
        help="Download the selected data as CSV file"
    )

    # Additional Info in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Dashboard Info")
    st.sidebar.info(f"""
    **Data Summary:**
    - Total Records: {len(df_filtered)}
    - Date Range: {df_filtered['Date Received'].min().strftime('%Y-%m-%d') if 'Date Received' in df_filtered.columns else 'N/A'} to {df_filtered['Date Received'].max().strftime('%Y-%m-%d') if 'Date Received' in df_filtered.columns else 'N/A'}
    - Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#6c757d; padding: 1rem;">'
        '<p><b>Madre Integrated Engineering</b></p>'
        '<p>HR Analytics Dashboard v2.0</p>'
        '<p><a href="https://www.madre-me.com/" target="_blank" style="color: #0077b6; text-decoration: none;">🌐 www.madre-me.com</a></p>'
        '<p style="font-size: 0.8rem; margin-top: 1rem;">Dashboard built with Streamlit | Data updates in real-time</p>'
        '</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()(line_color='#0077b6', line_width=3)
            fig.add_hline(y=monthly["Utilization"].mean(), line_dash="dash", 
                         line_color="red", annotation_text="Average")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            avg_util = monthly["Utilization"].mean()
            st.markdown(f"**Average Utilization:** {avg_util:.1f}%")

    with col2:
        st.subheader("Time to Fill Distribution")
        filled = df_filtered[df_filtered["Status"] == "Filled"]
        if not filled.empty and "Days Open" in df_filtered.columns:
            fig = px.histogram(filled, x="Days Open", nbins=20,
                             title="Distribution of Time to Fill (Days)")
            fig.update_traces(marker_color='#0077b6')
            fig.add_vline(x=filled["Days Open"].median(), line_dash="dash",
                         line_color="red", annotation_text="Median")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Median:** {filled['Days Open'].median():.0f} days | "
                       f"**Average:** {filled['Days Open'].mean():.0f} days")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Aging of Open Requirements")
        open_reqs = df_filtered[df_filtered["Status"] == "In Progress"]
        if not open_reqs.empty and "Days Open" in df_filtered.columns:
            fig = px.box(open_reqs, y="Days Open",
                        title="Aging Analysis (Box Plot)")
            fig.update_traces(marker_color='#0077b6', boxmean='sd')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Min:** {open_reqs['Days Open'].min():.0f} days | "
                       f"**Max:** {open_reqs['Days Open'].max():.0f} days | "
                       f"**Avg:** {open_reqs['Days Open'].mean():.0f} days")

    with col4:
        st.subheader("HR Performance Comparison")
        if all(col in df_filtered.columns for col in ["HR", "Status"]):
            conv = df_filtered.groupby("HR").apply(
                lambda x: (len(x[x["Status"] == "Filled"]) / len(x[x["Status"].isin(["Filled", "Closed"])]) * 100)
                if len(x[x["Status"].isin(["Filled", "Closed"])]) > 0 else 0
            ).reset_index(name="Conversion Rate")
            
            fig = px.bar(conv, x="HR", y="Conversion Rate",
                        title="Conversion Rate by HR Representative (%)")
            fig.update_traces
