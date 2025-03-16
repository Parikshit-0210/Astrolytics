from flask import Flask, render_template, request
from utils import fetch_data, preprocess_data, plot_data, plot_decomposition, plot_acf_pacf, check_stationarity, \
    plot_forecast, train_models, forecast_future
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    ticker = request.form['ticker']
    period = request.form['period']
    n_days = int(request.form['n_days'])
    model_name = request.form['model']

    df = fetch_data(ticker, period)
    if df is None:
        return render_template('error.html', message="Error fetching data.")
    df = preprocess_data(df)

    plot_data(df)
    plot_decomposition(df)
    plot_acf_pacf(df)

    p_value = check_stationarity(df)

    model_results = train_models(df)
    if model_results is None:
        return render_template('error.html', message="Error training models.")

    forecast_values, conf_int = forecast_future(df, model_results, steps=n_days)

    # Compute evaluation metrics
    actual_values = df['Close'].iloc[-n_days:].values.flatten()
    forecasted_values = forecast_values[model_name].values

    mse = mean_squared_error(actual_values, forecasted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, forecasted_values)

    # Generate updated forecast plot
    plot_forecast(df, forecast_values, conf_int, n_days)

    results = {
        'ticker': ticker,
        'p_value': p_value,
        'n_days': n_days,
        'plots': [
            'static/plot_closing_price.png',
            'static/plot_decomposition.png',
            'static/plot_acf_pacf.png',
            'static/plot_forecast.png'
        ],
        'forecast_values': forecasted_values.tolist(),
        'metrics': {'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    }

    return render_template('results.html', results=results)
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
