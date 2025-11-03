import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))

if src_path not in sys.path:
    sys.path.append(src_path)

# Optional: Debug print
print("SRC Path:", src_path)

try:
    from model_trainer import ModelTrainer
    from feature_engineer import FeatureEngineer
    from data_collector import StockDataCollector

except ImportError as e:
    import streamlit as st
    st.error(f"Error importing modules: {e}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="QuantumTrade Pro - Advanced Market Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme with neon accents
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Custom header with neon effect */
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00ffff, #ff00ff, #ffff00, #00ffff);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: neonGlow 3s ease-in-out infinite;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
    }
    
    @keyframes neonGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #00ffff;
    }
    
    /* Metric cards with glassmorphism effect */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 255, 255, 0.2);
    }
    
    /* Prediction boxes with cyberpunk styling */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        font-family: 'Exo 2', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .prediction-box:hover::before {
        left: 100%;
    }
    
    .prediction-up {
        background: linear-gradient(135deg, #00ff88, #00cc66);
        border: 2px solid #00ff88;
        color: #000000;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    .prediction-down {
        background: linear-gradient(135deg, #ff3366, #cc1144);
        border: 2px solid #ff3366;
        color: #ffffff;
        box-shadow: 0 0 20px rgba(255, 51, 102, 0.3);
    }
    
    .prediction-stable {
        background: linear-gradient(135deg, #ffaa00, #ff8800);
        border: 2px solid #ffaa00;
        color: #000000;
        box-shadow: 0 0 20px rgba(255, 170, 0, 0.3);
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 255, 255, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid #00ffff;
        border-radius: 10px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        margin: 0 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        color: #000000;
        font-weight: bold;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #ff00ff, #00ffff);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration if models aren't available."""
    # Create sample data for demo purposes
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    sample_data = []

    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        base_price = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300}[symbol]
        prices = []
        current_price = base_price

        for _ in dates:
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + change)
            prices.append(current_price)

        for i, date in enumerate(dates):
            sample_data.append({
                'Date': date,
                'Symbol': symbol,
                'Close': prices[i],
                'Volume': np.random.randint(1000000, 10000000)
            })

    return pd.DataFrame(sample_data)


@st.cache_resource
def load_models():
    """Load trained models and components."""
    try:
        trainer = ModelTrainer()
        trainer.load_models("models")

        engineer = FeatureEngineer()

        return trainer, engineer, True
    except Exception as e:
        st.warning(f"Could not load models: {e}")
        return None, None, False


def get_prediction_color_and_text(prediction, probabilities=None):
    """Get color and text for prediction display."""
    class_names = ['📉 DOWN', '📊 STABLE', '📈 UP']
    colors = ['prediction-down', 'prediction-stable', 'prediction-up']

    pred_text = class_names[prediction]
    pred_color = colors[prediction]

    if probabilities is not None:
        confidence = probabilities[prediction] * 100
        pred_text += f" ({confidence:.1f}% confidence)"

    return pred_color, pred_text


def create_price_chart(df, symbol):
    """Create an interactive price chart."""
    symbol_data = df[df['Symbol'] == symbol].copy()
    symbol_data = symbol_data.sort_values('Date')

    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=symbol_data['Date'],
        y=symbol_data['Close'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='#1f77b4', width=2)
    ))

    # Add volume bars (secondary y-axis)
    if 'Volume' in symbol_data.columns:
        fig.add_trace(go.Bar(
            x=symbol_data['Date'],
            y=symbol_data['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3,
            marker_color='lightblue'
        ))

    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price Trend',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=500
    )

    return fig


def create_prediction_gauge(prediction, probabilities):
    """Create a gauge chart for prediction confidence."""
    confidence = probabilities[prediction] * 100

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_feature_importance_chart(importance_df):
    """Create feature importance chart."""
    if importance_df.empty:
        return None

    top_features = importance_df.head(15)

    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 15 Most Important Features',
        labels={'Importance': 'Feature Importance', 'Feature': 'Technical Indicators'}
    )

    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def main():
    """Main Streamlit application."""

    # Header with new branding
    st.markdown('<h1 class="main-header">⚡ QUANTUMTRADE PRO ⚡</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #00ffff; font-family: \'Exo 2\', sans-serif; margin-bottom: 2rem;">ADVANCED MARKET INTELLIGENCE PLATFORM</h3>', unsafe_allow_html=True)

    # Load models
    trainer, engineer, models_loaded = load_models()

    if not models_loaded:
        st.error("🚨 **Models not found!** Please train the models first by running:")
        st.code("python model_trainer.py")
        st.info("For demo purposes, using sample data below...")

        # Show sample visualization
        sample_df = load_sample_data()
        st.subheader("📊 Sample Data Visualization")

        symbol = st.selectbox("Select Stock Symbol", ['AAPL', 'GOOGL', 'MSFT'])
        fig = create_price_chart(sample_df, symbol)
        st.plotly_chart(fig, use_container_width=True)

        return

    # Sidebar for controls with new styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #00ffff, #ff00ff); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: #000; text-align: center; margin: 0; font-family: 'Orbitron', monospace;">⚡ QUANTUM CONTROL ⚡</h3>
    </div>
    """, unsafe_allow_html=True)

    # Stock selection
    available_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    selected_symbol = st.sidebar.selectbox(
        "📈 Select Stock Symbol",
        available_symbols,
        index=0
    )

    # Date range selection
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)  # Last 6 months

    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )

    # Prediction threshold
    threshold = st.sidebar.slider(
        "🎯 Prediction Threshold (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Minimum price change to classify as Up/Down"
    ) / 100

    # Real-time data toggle
    use_live_data = st.sidebar.checkbox(
        "📡 Use Live Data",
        value=True,
        help="Fetch real-time data from Yahoo Finance"
    )

    # Add new tabs for different analysis modes
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 QUANTUM ANALYSIS", "📊 PORTFOLIO INSIGHTS", "⚡ RISK ASSESSMENT", "🔮 PREDICTION ENGINE"])

    # Tab 1: Quantum Analysis (Main Analysis)
    with tab1:
        st.markdown("""
        <div style="background: rgba(0, 255, 255, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid #00ffff; margin-bottom: 2rem;">
            <h3 style="color: #00ffff; text-align: center; margin: 0;">🚀 QUANTUM MARKET ANALYSIS</h3>
            <p style="text-align: center; margin: 0.5rem 0;">Advanced AI-powered market intelligence with real-time predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"⚡ {selected_symbol} QUANTUM ANALYSIS")

        # Fetch and display data
        try:
            if use_live_data:
                # Fetch live data
                collector = StockDataCollector([selected_symbol])
                raw_data = collector.fetch_stock_data(period="5y", interval="1d")

                if len(date_range) == 2:
                    start_date, end_date = date_range
                    raw_data = raw_data[
                        (raw_data['Date'].dt.date >= start_date) & 
                        (raw_data['Date'].dt.date <= end_date)
                    ]

                # Create price chart
                fig = create_price_chart(raw_data, selected_symbol)
                st.plotly_chart(fig, use_container_width=True)

                # Get latest data for prediction
                if len(raw_data) > 0:
                    # Engineer features for the latest data
                    try:
                        featured_data = engineer.process_stock_data(raw_data, threshold=threshold)

                        if len(featured_data) > 0:
                            # Get the most recent data point
                            latest_data = featured_data.iloc[-1:].copy()

                            # Prepare features for prediction
                            feature_cols = trainer.feature_names
                            X_latest = latest_data[feature_cols].values

                            # Make prediction
                            predictions, probabilities = trainer.predict(X_latest)

                            # Display prediction in sidebar
                            with col2:
                                st.subheader("Market Prediction")

                                pred_color, pred_text = get_prediction_color_and_text(
                                    predictions[0], probabilities[0] if probabilities is not None else None
                                )

                                st.markdown(f'<div class="prediction-box {pred_color}">{pred_text}</div>', 
                                          unsafe_allow_html=True)

                                # Show prediction probabilities
                                if probabilities is not None:
                                    st.subheader("📊 Class Probabilities")
                                    prob_df = pd.DataFrame({
                                        'Direction': ['📉 Down', '📊 Stable', '📈 Up'],
                                        'Probability': probabilities[0] * 100
                                    })

                                    fig_prob = px.bar(
                                        prob_df,
                                        x='Direction',
                                        y='Probability',
                                        color='Direction',
                                        color_discrete_map={
                                            '📉 Down': '#dc3545',
                                            '📊 Stable': '#ffc107',
                                            '📈 Up': '#28a745'
                                        }
                                    )
                                    fig_prob.update_layout(height=300, showlegend=False)
                                    st.plotly_chart(fig_prob, use_container_width=True)

                                # Confidence gauge
                                if probabilities is not None:
                                    gauge_fig = create_prediction_gauge(predictions[0], probabilities[0])
                                    st.plotly_chart(gauge_fig, use_container_width=True)

                            # Technical indicators summary
                            st.subheader("📋 Latest Technical Indicators")

                            # Select key indicators to display
                            key_indicators = ['Close', 'SMA_20', 'RSI_14', 'MACD', 'Volume_Ratio']
                            available_indicators = [col for col in key_indicators if col in latest_data.columns]

                            if available_indicators:
                                indicator_data = latest_data[available_indicators].iloc[0]

                                cols = st.columns(len(available_indicators))
                                for i, (indicator, value) in enumerate(indicator_data.items()):
                                    with cols[i]:
                                        st.metric(
                                            label=indicator.replace('_', ' '),
                                            value=f"{value:.3f}" if not pd.isna(value) else "N/A"
                                        )

                        else:
                            st.warning("Not enough data for feature engineering. Need more historical data.")

                    except Exception as e:
                        st.error(f"Error processing features: {e}")
                        st.info("This might happen with limited data. Try selecting a longer date range.")

                else:
                    st.warning("No data available for the selected date range.")

            else:
                st.info("Live data is disabled. Enable it in the sidebar to see real-time analysis.")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.info("Please check your internet connection and try again.")

    # Tab 2: Portfolio Insights
    with tab2:
        st.markdown("""
        <div style="background: rgba(255, 0, 255, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid #ff00ff; margin-bottom: 2rem;">
            <h3 style="color: #ff00ff; text-align: center; margin: 0;">📊 PORTFOLIO INSIGHTS</h3>
            <p style="text-align: center; margin: 0.5rem 0;">Comprehensive portfolio analysis and optimization recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Portfolio Performance")
            # Create sample portfolio data
            portfolio_data = {
                'Stock': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
                'Shares': [100, 50, 75, 25, 30],
                'Current Price': [175.43, 142.56, 378.85, 155.12, 248.50],
                'Value': [17543, 7128, 28414, 3878, 7455],
                'Daily Change %': [2.3, -1.2, 0.8, 3.1, -0.5]
            }
            portfolio_df = pd.DataFrame(portfolio_data)
            portfolio_df['Total Value'] = portfolio_df['Value'].sum()
            portfolio_df['Weight %'] = (portfolio_df['Value'] / portfolio_df['Total Value'] * 100).round(2)
            
            st.dataframe(portfolio_df, use_container_width=True)
            
        with col2:
            st.subheader("📈 Portfolio Metrics")
            total_value = portfolio_df['Value'].sum()
            daily_pnl = (portfolio_df['Value'] * portfolio_df['Daily Change %'] / 100).sum()
            
            st.metric("💰 Total Portfolio Value", f"${total_value:,.2f}")
            st.metric("📊 Daily P&L", f"${daily_pnl:,.2f}", delta=f"{daily_pnl/total_value*100:.2f}%")
            st.metric("🎯 Diversification Score", "8.5/10")
            st.metric("⚡ Risk Level", "MODERATE")
    
    # Tab 3: Risk Assessment
    with tab3:
        st.markdown("""
        <div style="background: rgba(255, 170, 0, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid #ffaa00; margin-bottom: 2rem;">
            <h3 style="color: #ffaa00; text-align: center; margin: 0;">⚡ RISK ASSESSMENT</h3>
            <p style="text-align: center; margin: 0.5rem 0;">Advanced risk analysis and volatility assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Risk Metrics")
            risk_metrics = {
                'Metric': ['Beta', 'Sharpe Ratio', 'Max Drawdown', 'Volatility (30d)', 'VaR (95%)', 'CVaR (95%)'],
                'Value': ['1.23', '0.85', '-12.3%', '18.7%', '-2.1%', '-3.4%'],
                'Status': ['HIGH', 'GOOD', 'ACCEPTABLE', 'MODERATE', 'LOW', 'LOW']
            }
            risk_df = pd.DataFrame(risk_metrics)
            st.dataframe(risk_df, use_container_width=True)
            
        with col2:
            st.subheader("📊 Risk Visualization")
            # Create a risk gauge chart
            risk_score = 7.2  # Sample risk score out of 10
            
            fig_risk = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                delta = {'reference': 5},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgreen"},
                        {'range': [3, 7], 'color': "yellow"},
                        {'range': [7, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 8
                    }
                }
            ))
            
            fig_risk.update_layout(height=400)
            st.plotly_chart(fig_risk, use_container_width=True)
    
    # Tab 4: Prediction Engine
    with tab4:
        st.markdown("""
        <div style="background: rgba(0, 255, 136, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid #00ff88; margin-bottom: 2rem;">
            <h3 style="color: #00ff88; text-align: center; margin: 0;">🔮 PREDICTION ENGINE</h3>
            <p style="text-align: center; margin: 0.5rem 0;">Advanced AI predictions and market forecasting</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Prediction Confidence")
            # Sample prediction data
            predictions = {
                'Timeframe': ['1 Day', '1 Week', '1 Month', '3 Months'],
                'Prediction': ['UP', 'STABLE', 'UP', 'DOWN'],
                'Confidence': [78, 65, 72, 68],
                'Price Target': ['$180.50', '$182.30', '$185.75', '$170.20']
            }
            pred_df = pd.DataFrame(predictions)
            st.dataframe(pred_df, use_container_width=True)
            
        with col2:
            st.subheader("📈 Prediction Trends")
            # Create a prediction trend chart
            days = list(range(1, 31))
            confidence_trend = [65 + np.random.normal(0, 5) for _ in days]
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=days,
                y=confidence_trend,
                mode='lines+markers',
                name='Confidence Trend',
                line=dict(color='#00ffff', width=3),
                marker=dict(size=6)
            ))
            
            fig_trend.update_layout(
                title="30-Day Prediction Confidence Trend",
                xaxis_title="Days Ahead",
                yaxis_title="Confidence %",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        st.subheader("🔧 Model Insights")

        # Model performance metrics
        results = trainer.results[trainer.best_model_name]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Model Accuracy", f"{results['accuracy']:.1%}")
        with col2:
            st.metric("📊 F1 Score", f"{results['f1_score']:.3f}")
        with col3:
            st.metric("⏱️ Training Time", f"{results['training_time']:.1f}s")
        with col4:
            st.metric("🏆 Best Model", trainer.best_model_name)

        # Feature importance chart
        importance_df = results.get('feature_importance', pd.DataFrame())
        if not importance_df.empty:
            fig_importance = create_feature_importance_chart(importance_df)
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)

    # Footer with new branding
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #00ffff; font-family: "Exo 2", sans-serif;'>
        <h4>⚡ QUANTUMTRADE PRO - ADVANCED MARKET INTELLIGENCE ⚡</h4>
        <p style='color: #ffffff;'>⚠️ <strong>Disclaimer:</strong> This platform is for educational and research purposes only. 
        Not financial advice. Always consult with qualified financial professionals before making investment decisions.</p>
        <p style='color: #ff00ff; font-size: 0.9rem;'>Powered by Quantum AI Technology | Version 2.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
