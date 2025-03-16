# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

company_to_ticker = {
    'google': 'GOOGL',
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'amazon': 'AMZN',
    'facebook': 'META',
    'tesla': 'TSLA',
    'netflix': 'NFLX',
    'nvidia': 'NVDA',
    'intel': 'INTC',
    'ibm': 'IBM',
    'twitter': 'TWTR',
    'snapchat': 'SNAP',
    'uber': 'UBER',
    'lyft': 'LYFT',
    'paypal': 'PYPL',
    'adobe': 'ADBE',
    'salesforce': 'CRM',
    'shopify': 'SHOP',
    'spotify': 'SPOT',
    'zoom': 'ZM',
    'square': 'SQ',
    'pinterest': 'PINS',
    'slack': 'WORK',
    'dropbox': 'DBX',
    'airbnb': 'ABNB',
    'doordash': 'DASH',
    'robinhood': 'HOOD',
    'coinbase': 'COIN',
    'palantir': 'PLTR',
    'snowflake': 'SNOW',
    'roku': 'ROKU',
    'etsy': 'ETSY',
    'wayfair': 'W',
    'chewy': 'CHWY',
    'zillow': 'Z',
    'redfin': 'RDFN',
    'opendoor': 'OPEN',
    'beyond meat': 'BYND',
    'peloton': 'PTON',
    'draftkings': 'DKNG',
    'roblox': 'RBLX',
    'unity': 'U',
    'affirm': 'AFRM',
    'lucid': 'LCID',
    'rivian': 'RIVN',
    'fisker': 'FSR',
    'nio': 'NIO',
    'xpeng': 'XPEV',
    'li auto': 'LI',
    'baidu': 'BIDU',
    'alibaba': 'BABA',
    'jd.com': 'JD',
    'pinduoduo': 'PDD',
    'tencent': 'TCEHY',
    'meituan': 'MPNGY',
    'bilibili': 'BILI',
    'netease': 'NTES',
    'trip.com': 'TCOM',
    'weibo': 'WB',
    'iqiyi': 'IQ',
    'huya': 'HUYA',
    'douyu': 'DOYU',
    'kuaishou': 'KUAISHOU',
    'tiktok': 'TIKTOK',
    'wechat': 'WECHAT',
    'qq': 'QQ',
    'baidu': 'BIDU',
    'alibaba': 'BABA',
    'jd': 'JD',
    'pinduoduo': 'PDD',
    'tencent': 'TCEHY',
    'meituan': 'MPNGY',
    'bilibili': 'BILI',
    'netease': 'NTES',
    'trip': 'TCOM',
    'weibo': 'WB',
    'iqiyi': 'IQ',
    'huya': 'HUYA',
    'douyu': 'DOYU',
    'kuaishou': 'KUAISHOU',
    'tiktok': 'TIKTOK',
    'wechat': 'WECHAT',
    'qq': 'QQ',
    # Add more mappings as needed
}

def fetch_data(ticker, period='1y'):
    """Fetch historical stock data from Yahoo Finance."""
    ticker = company_to_ticker.get(ticker.lower(), ticker)  # Look up the ticker symbol
    try:
        stock_data = yf.Ticker(ticker)
        historical_data = stock_data.history(period=period)
        if historical_data.empty:
            return None
        return historical_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def preprocess_data(df):
    """Clean and preprocess the data."""
    df = df.dropna().copy()  # Drop missing values
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date format
    df.sort_values('Date', inplace=True)  # Sort data by Date
    return df

def plot_data(df):
    """Save and plot the stock closing prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.title('Stock Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.savefig('static/plot_closing_price.png')
    plt.close()

def plot_decomposition(df):
    """Perform and save seasonal decomposition plots."""
    decomposition = seasonal_decompose(df["Close"], model="additive", period=30)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    decomposition.trend.plot(ax=axes[0], title='Trend', color='red')
    decomposition.seasonal.plot(ax=axes[1], title='Seasonality', color='green')
    decomposition.resid.plot(ax=axes[2], title='Residuals', color='gray')

    plt.tight_layout()
    plt.savefig('static/plot_decomposition.png')
    plt.close()

def plot_acf_pacf(df):
    """Save ACF and PACF plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(df["Close"], lags=50, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(df["Close"], lags=50, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout()
    plt.savefig('static/plot_acf_pacf.png')
    plt.close()

def check_stationarity(df):
    """Perform the Augmented Dickey-Fuller test."""
    result = adfuller(df['Close'].dropna())
    return result[1]  # Return p-value

def train_models(df):
    """Train ARIMA, SARIMA, and GARCH models."""
    models = {}

    # ARIMA Model
    arima_model = ARIMA(df['Close'], order=(1, 1, 1))
    models['ARIMA'] = arima_model.fit()

    # SARIMA Model
    sarima_model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    models['SARIMA'] = sarima_model.fit()

    # GARCH Model
    garch_model = arch_model(df['Close'], vol='Garch', p=1, q=1)
    models['GARCH'] = garch_model.fit(disp="off")

    return models

def forecast_future(df, models, steps=5):
    """Forecast future values using trained models."""
    forecast_values = {}
    conf_int = {}

    for model_name, model in models.items():
        if model_name in ['ARIMA', 'SARIMA']:
            forecast = model.get_forecast(steps=steps)
            forecast_values[model_name] = forecast.predicted_mean
            conf_int[model_name] = forecast.conf_int(alpha=0.05)  # Set confidence level to 95%
        elif model_name == 'GARCH':
            garch_forecast = model.forecast(horizon=steps)
            forecast_values['GARCH'] = garch_forecast.mean.iloc[-1].values  # Extract volatility forecast

    return forecast_values, conf_int

def plot_forecast(df, forecast_values, conf_int, n_days):
    """Plot actual vs. predicted values with confidence intervals."""
    plt.figure(figsize=(12, 6))

    # Plot actual data
    plt.plot(df['Date'], df['Close'], label="Actual Prices", color="blue")

    # Generate future dates
    future_dates = pd.date_range(df['Date'].iloc[-1], periods=n_days + 1, freq="D")[1:]

    # Plot ARIMA forecast with confidence intervals
    if "ARIMA" in forecast_values:
        plt.plot(future_dates, forecast_values["ARIMA"], label="ARIMA Forecast", linestyle="--", color="red")
        plt.fill_between(
            future_dates,
            conf_int["ARIMA"].iloc[:, 0],  # Lower bound
            conf_int["ARIMA"].iloc[:, 1],  # Upper bound
            color="red", alpha=0.2, label="ARIMA Confidence Interval"
        )

    # Plot SARIMA forecast with confidence intervals
    if "SARIMA" in forecast_values:
        plt.plot(future_dates, forecast_values["SARIMA"], label="SARIMA Forecast", linestyle="--", color="green")
        plt.fill_between(
            future_dates,
            conf_int["SARIMA"].iloc[:, 0],  # Lower bound
            conf_int["SARIMA"].iloc[:, 1],  # Upper bound
            color="green", alpha=0.2, label="SARIMA Confidence Interval"
        )

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Forecast for Next {n_days} Days with Confidence Intervals")
    plt.legend()
    plt.grid()
    plt.savefig("static/plot_forecast.png")
    plt.close()