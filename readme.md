# AI Market Trend Analysis

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.2-red.svg)

**⚡ Advanced AI-powered stock market trend analysis and prediction platform ⚡**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Technologies](#-technologies)

</div>

---

## 📋 Overview

AI Market Trend Analysis is a comprehensive machine learning platform designed to predict stock market trends using advanced technical analysis and ensemble learning methods. The project provides an end-to-end pipeline from data collection to interactive visualization, making it ideal for financial analysis, research, and educational purposes.

**⚠️ Disclaimer**: This project is for **educational and research purposes only**. It is not financial advice. Always consult with qualified financial professionals before making investment decisions.

---

## ✨ Features

### 📊 Data Collection & Processing
- **Automated Data Retrieval**: Fetches historical stock data from Yahoo Finance API
- **Multi-Stock Support**: Analyze multiple stocks simultaneously (AAPL, GOOGL, MSFT, AMZN, TSLA)
- **Flexible Time Periods**: Configurable historical data ranges (1 year to 5 years)
- **Data Quality Checks**: Built-in validation and cleaning procedures

### 🔧 Feature Engineering
- **40+ Technical Indicators**: Comprehensive technical analysis features including:
  - **Moving Averages**: SMA (5, 10, 20, 50, 200), EMA (12, 26, 50)
  - **Momentum Indicators**: RSI, ROC, MACD, Stochastic Oscillator
  - **Volatility Measures**: Bollinger Bands, ATR, Historical Volatility
  - **Volume Indicators**: OBV, VPT, PVT, Volume Ratios
  - **Price Action**: Gap analysis, daily ranges, support/resistance levels

### 🤖 Machine Learning Models
- **Ensemble Approach**: Multiple state-of-the-art algorithms:
  - **Random Forest**: Robust ensemble method with tree-based learning
  - **Logistic Regression**: Linear baseline model for comparison
  - **XGBoost**: Gradient boosting for high-performance predictions
  - **LightGBM**: Fast gradient boosting with tree-based learning
- **Model Selection**: Automatic best model identification based on performance metrics
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 🎨 Interactive Dashboard
- **Streamlit UI**: Modern, responsive web interface with dark theme
- **Real-time Analysis**: Live data fetching and prediction updates
- **Visual Analytics**:
  - Interactive price charts with Plotly
  - Technical indicator overlays
  - Model performance visualizations
  - Feature importance rankings
- **Multi-Tab Interface**:
  - **Quantum Analysis**: Main market analysis and predictions
  - **Portfolio Insights**: Portfolio performance tracking
  - **Risk Assessment**: Volatility and risk metrics
  - **Prediction Engine**: AI forecast engine

### 📈 Analytics & Insights
- **Prediction Confidence**: Probability-based forecast confidence levels
- **Feature Importance**: Understand which indicators drive predictions
- **Model Comparison**: Side-by-side performance metrics
- **Historical Analysis**: Backtesting capabilities

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/shivu77/AI-for-Market-Analysis.git
cd AI-for-Market-Analysis
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Setup Script
```bash
python setup.py
```

This automated setup will:
- Create necessary directories
- Download historical stock data
- Engineer features
- Train machine learning models

---

## 💻 Usage

### Quick Start

#### 1. Data Collection
```bash
python src/data_collector.py
```
Collects 5 years of historical data for major tech stocks and saves to `data/raw/stock_data.csv`

#### 2. Feature Engineering
```bash
python src/feature_engineer.py
```
Creates 40+ technical indicators and saves to `data/features/stock_features.csv`

#### 3. Model Training
```bash
python src/model_trainer.py
```
Trains multiple ML models, compares performance, and saves best model to `models/`

#### 4. Launch Dashboard
```bash
streamlit run streamlit_app/app.py
```
Opens interactive dashboard at `http://localhost:8501`

### Interactive Dashboard Features

1. **Select Stock**: Choose from AAPL, GOOGL, MSFT, AMZN, TSLA
2. **Set Date Range**: Analyze specific time periods
3. **Adjust Threshold**: Configure prediction sensitivity
4. **View Predictions**: Get AI-powered market forecasts
5. **Explore Analytics**: Dive into technical indicators and model insights

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```
Explore data with pre-built analysis notebooks

---

## 📁 Project Structure

```
AI-for-Market-Analysis/
│
├── 📂 data/                          # Data storage
│   ├── raw/                          # Raw stock data
│   │   └── stock_data.csv
│   └── features/                     # Processed features
│       └── stock_features.csv
│
├── 📂 src/                           # Source code
│   ├── data_collector.py             # Fetch stock data
│   ├── feature_engineer.py           # Create technical indicators
│   └── model_trainer.py              # Train ML models
│
├── 📂 models/                        # Trained models
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── scaler.pkl
│   ├── model_metadata.json
│   └── feature_importance.csv
│
├── 📂 notebooks/                     # Jupyter notebooks
│   └── 01_data_exploration.ipynb
│
├── 📂 streamlit_app/                 # Dashboard app
│   └── app.py
│
├── 📂 .devcontainer/                 # Dev container config
│   └── devcontainer.json
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 setup.py                       # Automated setup script
└── 📄 README.md                      # This file
```

---

## 🔬 Technologies

### Core Libraries
- **Pandas** (≥2.2.0): Data manipulation and analysis
- **NumPy** (≥1.26.0): Numerical computing
- **Scikit-learn** (≥1.4.2): Machine learning framework

### Data & API
- **yfinance** (≥0.2.30): Yahoo Finance API integration
- **pandas-ta**: Technical analysis indicators

### Machine Learning
- **XGBoost** (≥2.0.3): Gradient boosting framework
- **LightGBM** (≥4.3.0): Microsoft's gradient boosting

### Visualization & UI
- **Streamlit** (≥1.28.2): Interactive web dashboard
- **Plotly** (≥6.0.0): Interactive visualizations
- **Matplotlib** (≥3.8.4): Static plotting
- **Seaborn** (≥0.13.2): Statistical visualizations

### Utilities
- **Joblib** (≥1.4.0): Model serialization
- **Rich** (≥13.7.0): Enhanced terminal output

---

## 📊 Model Performance

The platform compares multiple models and selects the best performer based on:

- **Accuracy**: Overall prediction correctness
- **F1-Score**: Balanced precision-recall metric
- **Cross-Validation**: Robust performance estimation
- **Training Time**: Computational efficiency

Typical performance metrics:
- Random Forest: ~85-90% accuracy
- XGBoost: ~87-92% accuracy
- LightGBM: ~88-93% accuracy (often best)

---

## 🔮 Future Enhancements

- [ ] Additional asset classes (crypto, forex, commodities)
- [ ] Advanced deep learning models (LSTM, GRU, Transformers)
- [ ] Real-time streaming predictions
- [ ] Portfolio optimization algorithms
- [ ] Backtesting framework
- [ ] API endpoints for programmatic access
- [ ] Multi-strategy trading signals
- [ ] Sentiment analysis integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Shivu77**
- GitHub: [@shivu77](https://github.com/shivu77)

---

## 🙏 Acknowledgments

- Yahoo Finance for providing free stock market data
- Open-source community for excellent ML libraries
- Streamlit team for the amazing dashboard framework

---

## 📞 Support

For questions, issues, or suggestions, please:
- Open an issue on GitHub
- Check existing documentation
- Review the code comments

---

<div align="center">

**⭐ If you find this project useful, please give it a star! ⭐**

Made with ❤️ for the financial analysis community

**Remember**: This is for educational purposes only, not financial advice!

</div>