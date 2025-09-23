import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the HR data"""
    try:
        # Try to load from uploaded file first
        uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            # Load from default file if it exists
            try:
                df = pd.read_excel('hr_data.xlsx')  # Put your Excel file here
            except FileNotFoundError:
                st.error("Please upload an Excel file or place 'hr_data.xlsx' in the same directory")
                return None
    
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Convert date columns - handle different date formats
        date_columns = ['Date Receieved', 'Date Received', 'Date Closed']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if 'Date Received' not in df.columns and col == 'Date Receieved':
                    df['Date Received'] = df[col]
        
        # Handle missing Date Closed for open positions
        df['Date Closed'] = df['Date Closed'].fillna(pd.Timestamp.now())
        
        # Calculate Days Open if not present or recalculate
        if 'Date Received' in df.columns:
            df['Calculated Days Open'] = (df['Date Closed'] - df['Date Received']).dt.days
            # Use existing Days Open if available, otherwise use calculated
            if 'Days Open' not in df.columns:
                df['Days Open'] = df['Calculated Days Open']
        
        # Calculate additional metrics
        if 'Date Received' in df.columns:
            df['Month'] = df['Date Received'].dt.strftime('%Y-%m')
            df['Year'] = df['Date Received'].dt.year
            df['Week'] = df['Date Received'].dt.isocalendar().week
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_metrics(df):
    """Calculate key HR metrics"""
    # Filter for 2025 data
    df_2025 = df[df['Date Received'].dt.year == 2025]
    
    metrics = {}
    
    # 1. Bandwidth (Total requirements vs. filled positions)
    total_requirements = df_2025['Req'].sum()
    total_filled = df_2025[df_2025['Status'] == 'Filled']['Mob'].sum()
    metrics['bandwidth_utilization'] = (total_filled / total_requirements * 100) if total_requirements > 0 else 0
    
    # 2. Time to Fill (Average days to close filled positions)
    filled_positions = df_2025[df_2025['Status'] == 'Filled']
    metrics['avg_time_to_fill'] = filled_positions['Days Open'].mean() if len(filled_positions) > 0 else 0
    
    # 3. Aging of Requirements (Current open positions)
    open_positions = df_2025[df_2025['Status'] == 'In Progress']
    metrics['avg_aging'] = open_positions['Days Open'].mean() if len(open_positions) > 0 else 0
    
    # 4. Conversion Rate (Filled vs. Total closed)
    closed_positions = df_2025[df_2025['Status'].isin(['Filled', 'Closed'])]
    filled_count = len(df_2025[df_2025['Status'] == 'Filled'])
    total_closed = len(closed_positions)
    metrics['conversion_rate'] = (filled_count / total_closed * 100) if total_closed > 0 else 0
    
    # 5. Additional metrics
    metrics['total_requirements'] = total_requirements
    metrics['total_filled'] = total_filled
    metrics['total_open'] = len(open_positions)
    metrics['total_closed_unfilled'] = len(df_2025[df_2025['Status'] == 'Closed'])
    
    return metrics

def create_bandwidth_chart(df):
    """Create bandwidth utilization chart"""
    monthly_data = df.groupby('Month').agg({
        'Req': 'sum',
        'Mob': lambda x: (df.loc[x.index, 'Status'] == 'Filled').sum()
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Requirements', x=monthly_data['Month'], y=monthly_data['Req'], marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Filled', x=monthly_data['Month'], y=monthly_data['Mob'], marker_color='darkblue'))
    
    fig.update_layout(
        title='Monthly Bandwidth: Requirements vs. Filled Positions',
        xaxis_title='Month',
        yaxis_title='Number of Positions',
        barmode='group',
        height=400
    )
    return fig

def create_time_to_fill_chart(df):
    """Create time to fill analysis chart"""
    filled_df = df[df['Status'] == 'Filled'].copy()
    
    # Group by month and calculate average time to fill
    monthly_ttf = filled_df.groupby('Month')['Days Open'].mean().reset_index()
    
    fig = px.line(monthly_ttf, x='Month', y='Days Open', 
                  title='Average Time to Fill by Month',
                  markers=True)
    fig.update_layout(height=400, yaxis_title='Days')
    return fig

def create_aging_analysis(df):
    """Create aging analysis for open positions"""
    open_positions = df[df['Status'] == 'In Progress'].copy()
    
    # Create aging buckets
    bins = [0, 30, 60, 90, 120, float('inf')]
    labels = ['0-30 days', '31-60 days', '61-90 days', '91-120 days', '120+ days']
    open_positions['Aging_Bucket'] = pd.cut(open_positions['Days Open'], bins=bins, labels=labels, right=False)
    
    aging_counts = open_positions['Aging_Bucket'].value_counts().reset_index()
    aging_counts.columns = ['Aging_Bucket', 'Count']
    
    fig = px.pie(aging_counts, values='Count', names='Aging_Bucket',
                 title='Aging Analysis of Open Positions')
    fig.update_layout(height=400)
    return fig

def create_hr_performance(df):
    """Create HR performance comparison"""
    hr_performance = df.groupby('HR').agg({
        'Req': 'sum',
        'Days Open': 'mean',
        'Status': lambda x: (x == 'Filled').sum()
    }).reset_index()
    hr_performance.columns = ['HR', 'Total_Requirements', 'Avg_Days_Open', 'Total_Filled']
    hr_performance['Fill_Rate'] = (hr_performance['Total_Filled'] / hr_performance['Total_Requirements'] * 100).round(2)
    
    fig = px.scatter(hr_performance, x='Avg_Days_Open', y='Fill_Rate', 
                     size='Total_Requirements', hover_data=['HR'],
                     title='HR Performance: Fill Rate vs. Average Time to Fill')
    fig.update_layout(height=400, xaxis_title='Average Days to Fill', yaxis_title='Fill Rate (%)')
    return fig

def create_frequent_requirements(df):
    """Analyze most frequent job requirements"""
    freq_req = df.groupby('Designation').agg({
        'Req': 'sum',
        'Status': lambda x: (x == 'Filled').sum()
    }).reset_index()
    freq_req.columns = ['Designation', 'Total_Requirements', 'Total_Filled']
    freq_req['Fill_Rate'] = (freq_req['Total_Filled'] / freq_req['Total_Requirements'] * 100).round(2)
    freq_req = freq_req.sort_values('Total_Requirements', ascending=False).head(10)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(name='Total Requirements', x=freq_req['Designation'], y=freq_req['Total_Requirements']),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(name='Fill Rate %', x=freq_req['Designation'], y=freq_req['Fill_Rate'], 
                   mode='lines+markers', line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Job Designation")
    fig.update_yaxes(title_text="Number of Requirements", secondary_y=False)
    fig.update_yaxes(title_text="Fill Rate (%)", secondary_y=True)
    fig.update_layout(title_text="Top 10 Most Frequent Requirements", height=500)
    
    return fig

def main():
    st.title("📊 HR Analytics Dashboard - 2025")
    st.markdown("---")
    
    # Load data
    df = load_and_process_data()
    
    if df is None:
        st.stop()
    
    # Show data info
    st.sidebar.info(f"Data loaded: {len(df)} records")
    
    metrics = calculate_metrics(df)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Year filter
    if 'Year' in df.columns:
        years = sorted(df['Year'].unique())
        selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1 if years else 0)
        df = df[df['Year'] == selected_year]
    
    selected_hr = st.sidebar.multiselect(
        "Select HR Representatives",
        options=df['HR'].unique(),
        default=df['HR'].unique()
    )
    
    selected_status = st.sidebar.multiselect(
        "Select Status",
        options=df['Status'].unique(),
        default=df['Status'].unique()
    )
    
    # Filter data
    filtered_df = df[
        (df['HR'].isin(selected_hr)) & 
        (df['Status'].isin(selected_status))
    ]
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters.")
        return
    
    # Recalculate metrics for filtered data
    metrics = calculate_metrics(filtered_df)
    
    # Key Metrics Row
    st.header("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Bandwidth Utilization",
            value=f"{metrics['bandwidth_utilization']:.1f}%",
            delta=f"{metrics['total_filled']}/{metrics['total_requirements']} filled"
        )
    
    with col2:
        st.metric(
            label="Avg Time to Fill",
            value=f"{metrics['avg_time_to_fill']:.0f} days",
            delta="Filled positions only"
        )
    
    with col3:
        st.metric(
            label="Conversion Rate",
            value=f"{metrics['conversion_rate']:.1f}%",
            delta="Filled vs. Total Closed"
        )
    
    with col4:
        st.metric(
            label="Open Positions Aging",
            value=f"{metrics['avg_aging']:.0f} days",
            delta=f"{metrics['total_open']} open positions"
        )
    
    st.markdown("---")
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_bandwidth_chart(filtered_df), use_container_width=True)
        st.plotly_chart(create_aging_analysis(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_time_to_fill_chart(filtered_df), use_container_width=True)
        st.plotly_chart(create_hr_performance(filtered_df), use_container_width=True)
    
    # Full width chart
    st.plotly_chart(create_frequent_requirements(filtered_df), use_container_width=True)
    
    # Data Table
    st.header("📋 Detailed Data")
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Status Summary")
        status_summary = filtered_df['Status'].value_counts().reset_index()
        status_summary.columns = ['Status', 'Count']
        st.dataframe(status_summary, use_container_width=True)
    
    with col2:
        st.subheader("Top Clients")
        client_summary = filtered_df.groupby('Client')['Req'].sum().sort_values(ascending=False).head(10).reset_index()
        client_summary.columns = ['Client', 'Total Requirements']
        st.dataframe(client_summary, use_container_width=True)
    
    # Raw data (expandable)
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)
    
    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Export Data")
    
    if st.sidebar.button("Generate CSV Download"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"hr_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()