"""
ELECTRICITY DEMAND FORECASTING & PEAK RISK DETECTION - STREMLIT APP
====================================================================
Fixed version - compatible with all matplotlib versions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing matplotlib with fallback
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib not available. Some visualizations will use Plotly only.")

# Page configuration
st.set_page_config(
    page_title="Electricity Load Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #1a1a1a 0%, #333333 100%);
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SAFE PKL LOADING FUNCTION
# ============================================================================

def safe_pickle_load(file_path):
    """Safely load pickle file with error handling"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load {file_path}: {str(e)}")
        return None

# ============================================================================
# LOAD PKL MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_pkl_files():
    """Load all PKL files from the models and output directories"""
    models = {}
    
    # Search paths
    search_paths = [
        '.',
        './models',
        './output',
        '../models',
        '../output',
        'D:/electricity_project'
    ]
    
    # Files to look for
    target_files = [
        'all_results.pkl',
        'predictions.pkl',
        'residuals.pkl',
        'peak_analysis.pkl',
        'models/forecaster/xgb_model.pkl',
        'models/forecaster/scaler.pkl',
        'models/classifier/peak_classifier.pkl',
        'models/all_features.pkl'
    ]
    
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue
            
        for file_pattern in target_files:
            full_path = os.path.join(base_path, file_pattern)
            if os.path.exists(full_path):
                key_name = file_pattern.replace('.pkl', '').replace('/', '_')
                models[key_name] = safe_pickle_load(full_path)
    
    return models


@st.cache_data
def load_csv_data():
    csv_data = {}

    base_dir = os.path.dirname(__file__)

    search_paths = [
        base_dir,
        os.path.join(base_dir, "output")
    ]

    target_files = [
        'test_predictions.csv',
        'next_24h_forecast.csv',
        'monthly_aggregation.csv',
        'power_system_load_cleaned.csv'
    ]

    for path in search_paths:
        for file in target_files:
            full_path = os.path.join(path, file)

            if os.path.exists(full_path):
                df = pd.read_csv(full_path)

                for col in ['datetime', 'timestamp', 'date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        df = df.dropna(subset=[col])

                csv_data[file.replace('.csv','')] = df

    return csv_data
# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Electricity Load Forecasting & Peak Risk Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading PKL models and data..."):
        models = load_pkl_files()
        csv_data = load_csv_data()
    
    # Display loaded files
    if models or csv_data:
        st.success(f"✅ Loaded {len(models)} PKL files and {len(csv_data)} CSV files")
    else:
        st.error("No data files found. Please ensure PKL files are in the correct directory.")
        st.info("""
        Expected files:
        - all_results.pkl
        - predictions.pkl
        - residuals.pkl
        - peak_analysis.pkl
        - test_predictions.csv
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/electrical.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", 
             "📈 Forecast Analysis", 
             "⚠️ Peak Risk Detection",
             "📊 Model Performance"]
        )
        
        st.markdown("---")
        
        # File status
        with st.expander("📦 Loaded Files", expanded=False):
            if models:
                st.write("**PKL Files:**")
                for key in models.keys():
                    st.success(f"✓ {key}")
            
            if csv_data:
                st.write("**CSV Files:**")
                for key in csv_data.keys():
                    st.info(f"📊 {key}")
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard(models, csv_data)
    elif page == "📈 Forecast Analysis":
        show_forecast_analysis(models, csv_data)
    elif page == "⚠️ Peak Risk Detection":
        show_peak_analysis(models, csv_data)
    elif page == "📊 Model Performance":
        show_model_performance(models, csv_data)

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

def show_dashboard(models, csv_data):
    st.header("📊 Dashboard Overview")
    
    # Extract metrics from results
    metrics = extract_metrics(models, csv_data)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mae = metrics.get('mae', 'N/A')
        st.markdown(f"""
        <div class="metric-card">
            <h3>MAE</h3>
            <h2>{mae if mae == 'N/A' else f'{mae:.2f} MW'}</h2>
            <p>Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rmse = metrics.get('rmse', 'N/A')
        st.markdown(f"""
        <div class="metric-card">
            <h3>RMSE</h3>
            <h2>{rmse if rmse == 'N/A' else f'{rmse:.2f} MW'}</h2>
            <p>Root Mean Square Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        f1 = metrics.get('f1', 'N/A')
        st.markdown(f"""
        <div class="success-card">
            <h3>F1 Score</h3>
            <h2>{f1 if f1 == 'N/A' else f'{f1:.3f}'}</h2>
            <p>Peak Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        peak_pct = metrics.get('peak_percentage', 0)
        st.markdown(f"""
        <div class="warning-card">
            <h3>Peak Hours</h3>
            <h2>{peak_pct:.1f}%</h2>
            <p>of total time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Actual vs Predicted Load")
        plot_actual_vs_predicted(csv_data)
    
    with col2:
        st.subheader("⚠️ Peak Risk Distribution")
        plot_peak_distribution(csv_data)
    
    # Next 24 hours
    st.markdown("---")
    st.subheader("⏰ Next 24 Hours Forecast")
    plot_24h_forecast(csv_data)

def extract_metrics(models, csv_data):
    """Extract metrics from loaded data"""
    metrics = {
        'mae': 'N/A',
        'rmse': 'N/A',
        'f1': 'N/A',
        'peak_percentage': 0
    }
    
    # Try from all_results.pkl
    if 'all_results' in models:
        results = models['all_results']
        if isinstance(results, dict):
            if 'regression_metrics' in results:
                metrics['mae'] = results['regression_metrics'].get('MAE', 'N/A')
                metrics['rmse'] = results['regression_metrics'].get('RMSE', 'N/A')
            
            if 'classification_metrics' in results:
                metrics['f1'] = results['classification_metrics'].get('F1-Score', 'N/A')
            
            if 'peak_patterns' in results:
                metrics['peak_percentage'] = results['peak_patterns'].get('peak_percentage', 0)
    
    # Try from test_predictions
    elif 'test_predictions' in csv_data:
        df = csv_data['test_predictions']
        if 'actual_load' in df.columns and 'predicted_load' in df.columns:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            metrics['mae'] = mean_absolute_error(df['actual_load'], df['predicted_load'])
            metrics['rmse'] = np.sqrt(mean_squared_error(df['actual_load'], df['predicted_load']))
        
        if 'peak_actual' in df.columns and 'peak_predicted' in df.columns:
            from sklearn.metrics import f1_score
            try:
                metrics['f1'] = f1_score(df['peak_actual'], df['peak_predicted'])
            except:
                pass
            
            metrics['peak_percentage'] = (df['peak_actual'].sum() / len(df)) * 100
    
    return metrics

def plot_actual_vs_predicted(csv_data):
    """Plot actual vs predicted using Plotly"""
    if 'test_predictions' not in csv_data:
        st.info("No test predictions data available")
        return
    
    df = csv_data['test_predictions'].tail(168)  # Last week
    
    fig = go.Figure()
    
    x_data = df['datetime'] if 'datetime' in df.columns else df.index
    
    if 'actual_load' in df.columns:
        fig.add_trace(go.Scatter(
            x=x_data,
            y=df['actual_load'],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
    
    if 'predicted_load' in df.columns:
        fig.add_trace(go.Scatter(
            x=x_data,
            y=df['predicted_load'],
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_peak_distribution(csv_data):
    """Plot peak risk distribution"""
    if 'test_predictions' not in csv_data:
        st.info("No test predictions data available")
        return
    
    df = csv_data['test_predictions']
    
    if 'peak_probability' in df.columns:
        # Create risk categories
        risk_levels = pd.cut(
            df['peak_probability'] * 100,
            bins=[0, 30, 70, 100],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        risk_counts = risk_levels.value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=.3,
            marker_colors=['green', 'orange', 'red']
        )])
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No peak probability data available")

def plot_24h_forecast(csv_data):
    """Plot 24-hour forecast"""
    if 'next_24h' in csv_data:
        df_24h = csv_data['next_24h']
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Load Forecast', 'Peak Risk'),
            row_heights=[0.7, 0.3]
        )
        
        # Load forecast
        if 'predicted_load' in df_24h.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_24h['hour'] if 'hour' in df_24h.columns else df_24h.index,
                    y=df_24h['predicted_load'],
                    mode='lines+markers',
                    name='Load',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # Peak risk
        if 'peak_risk' in df_24h.columns:
            colors = ['green' if x == 'Low' else 'orange' if x == 'Medium' else 'red' 
                     for x in df_24h.get('risk_level', ['Low']*len(df_24h))]
            
            fig.add_trace(
                go.Bar(
                    x=df_24h['hour'] if 'hour' in df_24h.columns else df_24h.index,
                    y=df_24h['peak_risk'],
                    name='Peak Risk',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=500, hovermode='x unified')
        fig.update_xaxes(title_text="Hour", row=2, col=1)
        fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
        fig.update_yaxes(title_text="Risk (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 24-hour forecast data available")

# ============================================================================
# FORECAST ANALYSIS PAGE
# ============================================================================

def show_forecast_analysis(models, csv_data):
    st.header("📈 Detailed Forecast Analysis")
    
    if 'test_predictions' in csv_data:
        df = csv_data['test_predictions']
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.slider(
                "Select Number of Hours to Display",
                min_value=24,
                max_value=min(720, len(df)),
                value=168,
                step=24
            )
        
        with col2:
            chart_type = st.selectbox(
                "Chart Type",
                ["Line Chart", "Scatter Plot"]
            )
        
        # Filter data
        df_display = df.tail(date_range)
        
        # Main forecast chart
        fig = go.Figure()
        
        x_data = df_display['datetime'] if 'datetime' in df_display.columns else df_display.index
        
        if 'actual_load' in df_display.columns:
            mode = 'lines' if chart_type == "Line Chart" else 'markers'
            fig.add_trace(go.Scatter(
                x=x_data,
                y=df_display['actual_load'],
                mode=mode,
                name='Actual',
                line=dict(color='blue', width=2) if chart_type == "Line Chart" else None,
                marker=dict(size=4) if chart_type == "Scatter Plot" else None
            ))
        
        if 'predicted_load' in df_display.columns:
            mode = 'lines' if chart_type == "Line Chart" else 'markers'
            fig.add_trace(go.Scatter(
                x=x_data,
                y=df_display['predicted_load'],
                mode=mode,
                name='Predicted',
                line=dict(color='red', width=2, dash='dash') if chart_type == "Line Chart" else None,
                marker=dict(size=4, color='red') if chart_type == "Scatter Plot" else None
            ))
        
        fig.update_layout(
            height=500,
            title="Actual vs Predicted Load Over Time",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        if 'actual_load' in df.columns and 'predicted_load' in df.columns:
            st.subheader("📊 Error Analysis")
            
            df['error'] = df['actual_load'] - df['predicted_load']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Error distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df['error'],
                    nbinsx=50,
                    marker_color='purple',
                    opacity=0.7
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    height=300,
                    title="Error Distribution",
                    xaxis_title="Error (MW)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Error vs Predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['predicted_load'],
                    y=df['error'],
                    mode='markers',
                    marker=dict(
                        color=df['error'],
                        colorscale='RdBu',
                        size=5,
                        showscale=True,
                        colorbar=dict(title="Error")
                    )
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    height=300,
                    title="Error vs Predicted Value",
                    xaxis_title="Predicted Load (MW)",
                    yaxis_title="Error (MW)"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No test predictions data available")

# ============================================================================
# PEAK ANALYSIS PAGE
# ============================================================================

def show_peak_analysis(models, csv_data):
    st.header("⚠️ Peak Risk Detection Analysis")
    
    if 'test_predictions' in csv_data:
        df = csv_data['test_predictions']
        
        # Peak metrics
        if 'peak_actual' in df.columns and 'peak_predicted' in df.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                actual_peaks = df['peak_actual'].sum()
                st.metric("Actual Peak Hours", f"{actual_peaks}")
            
            with col2:
                pred_peaks = df['peak_predicted'].sum()
                st.metric("Predicted Peak Hours", f"{pred_peaks}")
            
            with col3:
                from sklearn.metrics import precision_score, recall_score, f1_score
                try:
                    precision = precision_score(df['peak_actual'], df['peak_predicted'])
                    recall = recall_score(df['peak_actual'], df['peak_predicted'])
                    f1 = f1_score(df['peak_actual'], df['peak_predicted'])
                    
                    st.metric("Precision", f"{precision:.3f}")
                    st.metric("Recall", f"{recall:.3f}")
                    st.metric("F1 Score", f"{f1:.3f}")
                except:
                    st.warning("Could not calculate metrics")
        
        st.markdown("---")
        
        # Peak probability timeline
        if 'peak_probability' in df.columns:
            st.subheader("📈 Peak Probability Timeline")
            
            hours_to_show = st.slider("Hours to display", 24, min(336, len(df)), 168)
            df_prob = df.tail(hours_to_show)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Load
            if 'actual_load' in df_prob.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_prob['datetime'] if 'datetime' in df_prob.columns else df_prob.index,
                        y=df_prob['actual_load'],
                        name="Load (MW)",
                        line=dict(color='blue')
                    ),
                    secondary_y=False
                )
            
            # Peak probability
            fig.add_trace(
                go.Scatter(
                    x=df_prob['datetime'] if 'datetime' in df_prob.columns else df_prob.index,
                    y=df_prob['peak_probability'] * 100,
                    name="Peak Risk %",
                    line=dict(color='red', dash='dash'),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.1)'
                ),
                secondary_y=True
            )
            
            # Add threshold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, secondary_y=True)
            fig.add_hline(y=30, line_dash="dash", line_color="orange", opacity=0.5, secondary_y=True)
            
            fig.update_layout(
                height=400,
                title="Load vs Peak Risk Probability",
                hovermode='x unified'
            )
            fig.update_yaxes(title_text="Load (MW)", secondary_y=False)
            fig.update_yaxes(title_text="Risk Probability (%)", secondary_y=True, range=[0, 100])
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No test predictions data available for peak analysis")

# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================

def show_model_performance(models, csv_data):
    st.header("📊 Model Performance Metrics")
    
    # Check for all_results.pkl
    if 'all_results' in models:
        results = models['all_results']
        
        if isinstance(results, dict):
            # Regression metrics
            if 'regression_metrics' in results:
                st.subheader("🎯 Regression Performance (Load Forecasting)")
                metrics = results['regression_metrics']
                
                cols = st.columns(4)
                for i, (name, value) in enumerate(metrics.items()):
                    if name != 'Confusion_Matrix' and isinstance(value, (int, float)):
                        with cols[i % 4]:
                            st.metric(name, f"{value:.4f}")
            
            # Classification metrics
            if 'classification_metrics' in results:
                st.subheader("🎯 Classification Performance (Peak Detection)")
                metrics = results['classification_metrics']
                
                cols = st.columns(4)
                for i, (name, value) in enumerate(metrics.items()):
                    if name != 'Confusion_Matrix' and isinstance(value, (int, float)):
                        with cols[i % 4]:
                            st.metric(name, f"{value:.4f}")
    
    # Feature importance from CSV or models
    if 'test_predictions' in csv_data:
        df = csv_data['test_predictions']
        
        # Calculate feature correlations as proxy for importance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'actual_load' in numeric_cols:
            correlations = df[numeric_cols].corr()['actual_load'].abs().sort_values(ascending=False)
            correlations = correlations[correlations.index != 'actual_load'].head(10)
            
            st.subheader("🔑 Feature Correlation with Target")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=correlations.index,
                x=correlations.values,
                orientation='h',
                marker=dict(
                    color=correlations.values,
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                height=400,
                title="Top 10 Features by Correlation with Load",
                xaxis_title="Absolute Correlation",
                yaxis_title="Feature"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()