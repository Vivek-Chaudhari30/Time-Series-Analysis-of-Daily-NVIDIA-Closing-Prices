import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import shapiro
from arch import arch_model

# Streamlit page setup
st.set_page_config(page_title="NVIDIA Stock Analysis", layout="wide")
st.title("NVIDIA Stock Time Series Analysis")

# 1. Retrieve NVIDIA stock data
st.header("1. Download NVIDIA Stock Data")
start_date = st.date_input("Start Date", value=pd.to_datetime('2005-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('2025-01-01'))

if start_date >= end_date:
    st.error("Error: End date must fall after start date.")
else:
    data = yf.download('NVDA', start=start_date, end=end_date)
    close_prices = data['Close']
    st.line_chart(close_prices)

    # 2. Plot closing prices
    st.header("2. Closing Prices")
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(close_prices, color='black')
    ax1.set_title('NVIDIA Corporation Closing Prices')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    st.pyplot(fig1)

    # 3. Decompose time series
    st.header("3. Time Series Decomposition")
    try:
        decomp = seasonal_decompose(close_prices, model='multiplicative', period=30)
        fig2 = decomp.plot()
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Decomposition Error: {e}")

    # 4. Stationarity Tests
    st.header("4. Stationarity Tests on Original Data")
    result_adf = adfuller(close_prices)
    result_kpss = kpss(close_prices, regression='c', nlags="auto")

    st.subheader("ADF Test")
    st.write(f"ADF Statistic: {result_adf[0]:.4f}")
    st.write(f"p-value: {result_adf[1]:.4f}")

    st.subheader("KPSS Test")
    st.write(f"KPSS Statistic: {result_kpss[0]:.4f}")
    st.write(f"p-value: {result_kpss[1]:.4f}")

    # 5. First Differencing
    st.header("5. First Differencing")
    diff_close = close_prices.diff().dropna()
    fig3, ax3 = plt.subplots(figsize=(10,6))
    ax3.plot(diff_close, color='black')
    ax3.set_title('First Differenced Closing Prices')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Differenced Price')
    st.pyplot(fig3)

    # Log-differencing (optional)
    st.header("5b. Log-Differencing (Optional)")
    log_close = np.log(close_prices)
    log_diff_close = log_close.diff().dropna()

    result_adf_log = adfuller(log_diff_close)
    result_kpss_log = kpss(log_diff_close, regression='c', nlags="auto")

    st.subheader("ADF Test (Log Differenced)")
    st.write(f"ADF Statistic: {result_adf_log[0]:.4f}")
    st.write(f"p-value: {result_adf_log[1]:.4f}")

    st.subheader("KPSS Test (Log Differenced)")
    st.write(f"KPSS Statistic: {result_kpss_log[0]:.4f}")
    st.write(f"p-value: {result_kpss_log[1]:.4f}")

    # 6. Plot ACF and PACF
    st.header("6. ACF and PACF Plots")
    fig4, ax4 = plt.subplots()
    plot_acf(diff_close, lags=40, ax=ax4)
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    plot_pacf(diff_close, lags=40, ax=ax5)
    st.pyplot(fig5)

    # 7. ARIMA Model Fitting
    st.header("7. ARIMA Model Fitting")
    order_list = [(1,1,0), (0,1,1), (1,1,1), (2,1,1), (1,1,2), (2,1,2)]
    best_aic = np.inf
    best_order = None
    best_model = None

    for order in order_list:
        try:
            model = ARIMA(close_prices, order=order)
            model_fit = model.fit()
            st.write(f"ARIMA{order} - AIC:{model_fit.aic:.2f}")
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_model = model_fit
        except Exception as e:
            st.write(f"ARIMA{order} - Error: {e}")

    if best_model:
        st.success(f"Best ARIMA Order: {best_order} with AIC: {best_aic:.2f}")
        st.subheader("Forecast Plot")
        forecast = best_model.get_forecast(steps=30)
        forecast_index = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), periods=30)
        forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

        fig6, ax6 = plt.subplots(figsize=(10,6))
        ax6.plot(close_prices, label='Observed')
        ax6.plot(forecast_series, label='Forecast', color='red')
        ax6.legend()
        st.pyplot(fig6)
    else:
        st.warning("No ARIMA model could be fitted.")
