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

# --- Custom Theme (Ocean Style) ---
st.markdown("""
    <style>
    :root {
        --ocean-blue: #0b5fa5;
        --ocean-light: #6fb3ff;
        --ocean-muted: #e9f3ff;
        --ocean-grey: #f4f6f8;
        --ocean-dark: #0f2340;
    }
    .stApp {
        background: linear-gradient(180deg, white 0%, var(--ocean-muted) 100%);
        color: var(--ocean-dark);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--ocean-blue);
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .stMetric {
        background: white;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(11,95,165,0.08);
        border-left: 4px solid var(--ocean-blue);
    }
    .stPlotlyChart {
        background: white;
        padding: 8px;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(11,95,165,0.05);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, white 0%, #f8fafc 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .sidebar .stSelectbox > div > div {
        background-color: white;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_and_process_data(uploaded_file=None):
    """Load and process the HR data with comprehensive error handling"""
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records from uploaded file")
        else:
            try:
                df = pd.read_excel("hr_data.xlsx")
                st.info(f"Loaded {len(df)} records from default file (hr_data.xlsx)")
            except FileNotFoundError:
                st.error("No data file found. Please upload an Excel file to continue.")
                return None

        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Display column information for debugging
        with st.expander("Data Information", expanded=False):
            st.write("**Columns found:**", list(df.columns))
            st.write("**Data shape:**", df.shape)
            st.write("**First few rows:**")
            st.dataframe(df.head(3))

        # Fix potential typo in date column
        if "Date Receieved" in df.columns and "Date Received" not in df.columns:
            df.rename(columns={"Date Receieved": "Date Received"}, inplace=True)
            st.info("Fixed column name: 'Date Receieved' → 'Date Received'")

        # Convert date columns with better error handling
        date_cols = ["Date Received", "Date Closed"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                invalid_dates = df[col].isna().sum()
                if invalid_dates > 0:
                    st.warning(f"Found {invalid_dates} invalid dates in column '{col}'")

        # Handle missing Date Closed (for open positions)
        if "Date Closed" in df.columns:
            open_positions = df["Date Closed"].isna().sum()
            df["Date Closed"] = df["Date Closed"].fillna(pd.Timestamp.now())
            if open_positions > 0:
                st.info(f"Filled {open_positions} missing 'Date Closed' values with current date")

        # Calculate Days Open with validation
        if all(col in df.columns for col in ["Date Received", "Date Closed"]):
            df["Days Open"] = (df["Date Closed"] - df["Date Received"]).dt.days.clip(lower=0)
            # Handle any remaining NaN values
            df["Days Open"] = df["Days Open"].fillna(0)

        # Add time-based columns
        if "Date Received" in df.columns:
            df["Month"] = df["Date Received"].dt.to_period("M").dt.to_timestamp()
            df["Year"] = df["Date Received"].dt.year
            df["Quarter"] = df["Date Received"].dt.quarter
            df["Week"] = df["Date Received"].dt.isocalendar().week

        # Data quality checks
        required_cols = ["Client", "Designation", "Req", "Status", "HR"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None

        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please check your file format and column names")
        return None

# --- Enhanced Metrics Calculation ---
def calculate_metrics(df):
    """Calculate comprehensive HR metrics with error handling"""
    metrics = {}
    
    try:
        total_requirements = df["Req"].sum() if "Req" in df.columns else 0
        total_filled = df[df["Status"] == "Filled"]["Mob"].sum() if all(col in df.columns for col in ["Status", "Mob"]) else 0
        
        # Bandwidth utilization
        metrics["bandwidth"] = (total_filled / total_requirements * 100) if total_requirements > 0 else 0
        
        # Time to fill for filled positions only
        filled_positions = df[df["Status"] == "Filled"]
        metrics["time_to_fill"] = filled_positions["Days Open"].mean() if (len(filled_positions) > 0 and "Days Open" in df.columns) else 0
        
        # Aging of open positions
        open_positions = df[df["Status"] == "In Progress"]
        metrics["aging"] = open_positions["Days Open"].mean() if (len(open_positions) > 0 and "Days Open" in df.columns) else 0
        
        # Conversion rate
        closed_positions = df[df["Status"].isin(["Filled", "Closed"])]
        filled_count = len(filled_positions)
        total_closed = len(closed_positions)
        metrics["conversion_rate"] = (filled_count / total_closed * 100) if total_closed > 0 else 0
        
        # Additional metrics
        metrics["total_requirements"] = total_requirements
        metrics["total_filled"] = total_filled
        metrics["total_open"] = len(open_positions)
        metrics["total_closed"] = len(df[df["Status"] == "Closed"])
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {key: 0 for key in ["bandwidth", "time_to_fill", "aging", "conversion_rate"]}

# --- Enhanced Forecasting Function ---
def forecast_requirements(monthly_req_df, months_ahead=6):
    """Enhanced forecasting with trend analysis"""
    if monthly_req_df is None or monthly_req_df.shape[0] < 3:
        return monthly_req_df

    df = monthly_req_df.copy().reset_index(drop=True)
    df = df.sort_values("Month")
    df["t"] = np.arange(len(df))

    try:
        # Linear trend
        coef = np.polyfit(df["t"], df["Req"], deg=1)
        slope, intercept = coef[0], coef[1]

        # Generate forecasts
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
        
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return monthly_req_df

# --- Enhanced Visualization Functions ---
def create_enhanced_bandwidth_chart(df):
    """Create enhanced bandwidth utilization chart"""
    if "Month" in df.columns and all(col in df.columns for col in ["Req", "Mob"]):
        monthly = df.groupby("Month").agg({
            "Req": "sum", 
            "Mob": "sum",
            "Status": lambda x: (x == "Filled").sum()
        }).reset_index()
        monthly["Utilization"] = (monthly["Status"] / monthly["Req"] * 100).fillna(0)
        
        fig = px.line(monthly, x="Month", y="Utilization", markers=True, 
                     title="Bandwidth Utilization Over Time (%)",
                     color_discrete_sequence=['#0b5fa5'])
        fig.update_layout(yaxis_range=[0, 100])
        return fig
    return None

def create_performance_heatmap(df):
    """Create HR performance heatmap"""
    if all(col in df.columns for col in ["HR", "Month", "Status"]):
        pivot_data = df.groupby(["HR", df["Month"].dt.strftime("%Y-%m")])["Status"].apply(
            lambda x: (x == "Filled").sum()
        ).reset_index()
        pivot_table = pivot_data.pivot(index="HR", columns="Month", values="Status").fillna(0)
        
        fig = px.imshow(pivot_table, 
                       title="Monthly Performance Heatmap (Filled Positions)",
                       color_continuous_scale="Blues",
                       aspect="auto")
        return fig
    return None

# --- Main Dashboard ---
def main():
    # Logo and Header Section
    st.markdown("""
        <div class="logo-container">
            <img src="https://madre-me.com/wp-content/themes/theme/images/logo.jpg" 
                 alt="Madre Integrated Engineering Logo" 
                 style="max-height: 100px; margin-bottom: 1rem;">
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">', unsafe_allow_html=True)
    st.title("HR Analytics Dashboard")
    st.markdown("**Madre Integrated Engineering**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<p class="subtitle">Real-time monitoring and analysis of HR recruitment performance</p>', unsafe_allow_html=True)

    # File Upload Section
    st.markdown("### Data Source")
    uploaded_file = st.file_uploader(
        "Upload your Excel file", 
        type=["xlsx", "xls"],
        help="Upload your HR data Excel file to begin analysis"
    )
    
    df = load_and_process_data(uploaded_file)

    if df is None:
        st.stop()

    # Data Summary
    total_records = len(df)
    date_range = f"{df['Date Received'].min().strftime('%Y-%m-%d')} to {df['Date Received'].max().strftime('%Y-%m-%d')}" if 'Date Received' in df.columns else "N/A"
    
    st.info(f"📊 **Data Overview:** {total_records:,} records | Date range: {date_range}")

    # Sidebar Filters
    st.sidebar.markdown("## 🔍 Filters & Controls")
    
    # Year filter
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique(), reverse=True)
        selected_year = st.sidebar.selectbox("📅 Select Year", years, index=0 if years else 0)
        df = df[df["Year"] == selected_year]

    # HR Representatives filter
    if "HR" in df.columns:
        hr_options = sorted(df["HR"].dropna().unique())
        selected_hr = st.sidebar.multiselect(
            "👥 HR Representatives", 
            hr_options, 
            default=hr_options,
            help="Select one or more HR representatives"
        )
        if selected_hr:
            df = df[df["HR"].isin(selected_hr)]

    # Status filter
    if "Status" in df.columns:
        status_options = sorted(df["Status"].dropna().unique())
        selected_status = st.sidebar.multiselect(
            "📋 Status", 
            status_options, 
            default=status_options,
            help="Filter by requirement status"
        )
        if selected_status:
            df = df[df["Status"].isin(selected_status)]

    # Client filter
    if "Client" in df.columns:
        client_options = ["All"] + sorted(df["Client"].dropna().unique())
        selected_client = st.sidebar.selectbox("🏢 Client", client_options, index=0)
        if selected_client != "All":
            df = df[df["Client"] == selected_client]

    if df.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
        st.stop()

    # Display filtered data info
    st.sidebar.success(f"✅ {len(df):,} records after filtering")

    # Key HR Metrics Section
    st.markdown("---")
    st.header("🎯 Key Performance Indicators")

    metrics = calculate_metrics(df)

    # Create metric columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Bandwidth Utilization", 
            f"{metrics['bandwidth']:.1f}%",
            delta=f"{metrics.get('total_filled', 0)}/{metrics.get('total_requirements', 0)} filled",
            help="Percentage of requirements filled vs total requirements"
        )

    with col2:
        st.metric(
            "Avg Time to Fill", 
            f"{metrics['time_to_fill']:.0f} days",
            delta="Filled positions only",
            help="Average number of days to fill positions"
        )

    with col3:
        st.metric(
            "Aging of Open Requirements", 
            f"{metrics['aging']:.0f} days",
            delta=f"{metrics.get('total_open', 0)} open positions",
            help="Average age of currently open positions"
        )

    with col4:
        st.metric(
            "Conversion Rate", 
            f"{metrics['conversion_rate']:.1f}%",
            delta="Filled vs Total Closed",
            help="Percentage of closed positions that were successfully filled"
        )

    # Detailed Analytics Section
    st.markdown("---")
    st.header("📈 Detailed Analytics")

    # Trends and Distributions
    st.subheader("Recruitment Trends & Distributions")
    
    col1, col2 = st.columns(2)

    with col1:
        bandwidth_chart = create_enhanced_bandwidth_chart(df)
        if bandwidth_chart:
            st.plotly_chart(bandwidth_chart, use_container_width=True)

    with col2:
        # Time to fill distribution
        filled = df[df["Status"] == "Filled"]
        if not filled.empty and "Days Open" in df.columns:
            fig = px.histogram(
                filled, x="Days Open", nbins=20, 
                title="Distribution of Time to Fill (days)",
                color_discrete_sequence=['#0b5fa5']
            )
            fig.add_vline(x=filled["Days Open"].mean(), line_dash="dash", 
                         annotation_text=f"Avg: {filled['Days Open'].mean():.0f} days")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Aging analysis
        open_reqs = df[df["Status"] == "In Progress"]
        if not open_reqs.empty and "Days Open" in df.columns:
            fig = px.box(
                open_reqs, y="Days Open", 
                title="Aging Analysis of Open Requirements",
                color_discrete_sequence=['#0b5fa5']
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        # HR Performance comparison
        if all(col in df.columns for col in ["HR", "Status"]):
            hr_performance = df.groupby("HR").agg({
                "Status": [
                    lambda x: (x == "Filled").sum(),
                    lambda x: (x == "Closed").sum() + (x == "Filled").sum(),
                    "count"
                ],
                "Days Open": "mean"
            }).reset_index()
            
            hr_performance.columns = ["HR", "Filled", "Total_Closed", "Total_Reqs", "Avg_Days"]
            hr_performance["Conversion_Rate"] = (hr_performance["Filled"] / hr_performance["Total_Closed"] * 100).fillna(0)
            
            fig = px.bar(
                hr_performance, x="HR", y="Conversion_Rate", 
                title="Conversion Rate by HR Representative (%)",
                color="Conversion_Rate",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Performance Heatmap
    heatmap = create_performance_heatmap(df)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)

    # Frequent Requirements Analysis
    st.subheader("🔄 Frequent Requirements Analysis")
    if all(col in df.columns for col in ["Designation", "Req"]):
        freq_analysis = df.groupby("Designation").agg({
            "Req": "sum",
            "Status": lambda x: (x == "Filled").sum(),
            "Days Open": "mean"
        }).reset_index()
        freq_analysis.columns = ["Designation", "Total_Req", "Total_Filled", "Avg_Days"]
        freq_analysis["Fill_Rate"] = (freq_analysis["Total_Filled"] / freq_analysis["Total_Req"] * 100).fillna(0)
        freq_analysis = freq_analysis.sort_values("Total_Req", ascending=False).head(15)
        
        fig = px.scatter(
            freq_analysis, x="Avg_Days", y="Fill_Rate", 
            size="Total_Req", hover_data=["Designation"],
            title="Requirements Analysis: Fill Rate vs Average Time",
            color="Total_Req",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Client Performance Analysis
    st.subheader("🏢 Client Performance Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if all(col in df.columns for col in ["Client", "Req"]):
            client_req = df.groupby("Client")["Req"].sum().reset_index()
            client_req = client_req.sort_values(by="Req", ascending=False).head(10)
            fig = px.bar(
                client_req, x="Client", y="Req", 
                title="Top 10 Clients by Total Requirements",
                color="Req",
                color_continuous_scale="Blues"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if all(col in df.columns for col in ["Client", "Days Open"]):
            client_aging = df.groupby("Client")["Days Open"].mean().reset_index()
            client_aging = client_aging.sort_values(by="Days Open", ascending=False).head(10)
            fig = px.bar(
                client_aging, x="Client", y="Days Open", 
                title="Clients with Longest Average Processing Time",
                color="Days Open",
                color_continuous_scale="Reds"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # Forecasting Section
    st.subheader("🔮 Requirements Forecasting")
    if "Month" in df.columns and "Req" in df.columns:
        monthly_req = df.groupby("Month").agg({"Req": "sum"}).reset_index()
        forecast_months = st.slider("Forecast period (months)", 3, 12, 6)
        forecast_df = forecast_requirements(monthly_req, months_ahead=forecast_months)

        if forecast_df is not None:
            fig_fc = go.Figure()
            hist = forecast_df[forecast_df["Forecast"] == False]
            fut = forecast_df[forecast_df["Forecast"] == True]
            
            fig_fc.add_trace(go.Bar(
                x=hist["Month"], y=hist["Req"], 
                name="Historical", marker_color='lightblue'
            ))
            
            if not fut.empty:
                fig_fc.add_trace(go.Bar(
                    x=fut["Month"], y=fut["Req"], 
                    name="Forecast", marker_color='orange'
                ))
            
            fig_fc.add_trace(go.Scatter(
                x=forecast_df["Month"], y=forecast_df["MA_3"], 
                name="3-Month Moving Average", 
                mode="lines+markers", line=dict(color='green', width=3)
            ))
            
            fig_fc.update_layout(
                title=f"Requirements Forecast - Next {forecast_months} Months",
                xaxis_title="Month", yaxis_title="Requirements", 
                barmode="group", height=500
            )
            st.plotly_chart(fig_fc, use_container_width=True)

    # Data Table Section
    st.markdown("---")
    st.header("📊 Data Tables")
    
    # Summary Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Status Summary")
        if "Status" in df.columns:
            status_summary = df["Status"].value_counts().reset_index()
            status_summary.columns = ["Status", "Count"]
            status_summary["Percentage"] = (status_summary["Count"] / status_summary["Count"].sum() * 100).round(1)
            st.dataframe(status_summary, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("HR Performance Summary")
        if "HR" in df.columns:
            hr_summary = df.groupby("HR").agg({
                "Req": "sum",
                "Status": lambda x: (x == "Filled").sum(),
                "Days Open": "mean"
            }).reset_index()
            hr_summary.columns = ["HR", "Total Req", "Filled", "Avg Days"]
            hr_summary["Fill Rate %"] = (hr_summary["Filled"] / hr_summary["Total Req"] * 100).round(1)
            st.dataframe(hr_summary, use_container_width=True, hide_index=True)

    # Raw Data
    with st.expander("🔍 View Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True, height=400)

    # Export Section
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export Options")
    
    # Generate export data
    export_df = df.copy()
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        "📊 Download Filtered Data (CSV)",
        csv_data,
        file_name=f"hr_analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        help="Download the currently filtered data as CSV"
    )
    
    # Generate summary report
    summary_data = {
        "Metric": ["Total Requirements", "Filled Positions", "Open Positions", "Closed Positions", 
                  "Bandwidth Utilization (%)", "Avg Time to Fill (days)", "Conversion Rate (%)"],
        "Value": [
            metrics.get('total_requirements', 0),
            metrics.get('total_filled', 0),
            metrics.get('total_open', 0),
            metrics.get('total_closed', 0),
            f"{metrics['bandwidth']:.1f}",
            f"{metrics['time_to_fill']:.0f}",
            f"{metrics['conversion_rate']:.1f}"
        ]
    }
    summary_csv = pd.DataFrame(summary_data).to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        "📈 Download Summary Report (CSV)",
        summary_csv,
        file_name=f"hr_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        help="Download key metrics summary"
    )

    # Footer
    st.markdown("---")
    MADRE_WEBSITE = "https://www.madre-me.com/"
    st.markdown(
        f'<div style="text-align:center; color:#56738a; padding: 2rem 0; font-size: 0.9em;">'
        f'<p><strong>HR Analytics Dashboard</strong></p>'
        f'<p>Powered by <strong>Madre Integrated Engineering</strong></p>'
        f'<p><a href="{MADRE_WEBSITE}" target="_blank" style="color: #0b5fa5; text-decoration: none;">🌐 madre-me.com</a></p>'
        f'<p style="font-size: 0.8em; margin-top: 1rem;">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>'
        f'</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
