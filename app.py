
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import kpss, adfuller
import warnings
from datetime import datetime

import io


st.set_page_config(page_title="NVIDIA Time Series Analysis", layout="wide")

st.title("📊 Time Series Analysis of NVIDIA (NVDA) Daily Closing Prices")
st.markdown("### By **Vivek Chaudhari** – 202201294")
st.markdown("---")

warnings.filterwarnings("ignore")
#!/usr/bin/env python
# coding: utf-8

# # SC475 - Time Series Analysis  
# 
# ## 📊 Topic: Time Series Analysis of Daily NVIDIA Closing Prices  
# 
# - By **VIVEK CHAUDHARI** – 202201294
# 

# #1. Importing Necessary Libraries and Loading Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller

# Define the NVIDIA ticker symbol
nvd = yf.Ticker("NVDA")

# Fetch historical data from 2014 to 2025
nvd_data = nvd.history(start="2014-01-01", end="2025-03-09", interval="1d")

# Remove timezone information
nvd_data.index = nvd_data.index.tz_localize(None)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(nvd_data.index, nvd_data['Close'], label='Close Price', color='black')

# Formatting
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.title('NVIDIA Close Price Over Time')
plt.legend()
plt.grid(True)
st.pyplot(plt.gcf())


# #2. Data Exploration

df = nvd_data
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())
import io

df.head()

df.tail()

df.isnull().sum()

df.shape

df.describe()


# Calculating rolling averages
#

df['rolling_month'] = df['Close'].rolling(window=30).mean()
df['rolling_quarter'] = df['Close'].rolling(window=90).mean()
df['rolling_year'] = df['Close'].rolling(window=365).mean()

# Create a figure with a 2x2 subplot layout
fig = plt.figure(figsize=[15, 10])
plt.suptitle('NVIDIA Closing Prices with Rolling Averages', fontsize=22)

# Subplot 1: Original Close Price
plt.subplot(221)
plt.plot(df.index, df['Close'], label='Original Close Price', color='black')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.title('NVIDIA Closing Prices')

# Subplot 2: One-Month Rolling Average
plt.subplot(222)
plt.plot(df.index, df['rolling_month'], label='One-Month Rolling Average', color='blue')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.title('One-Month Rolling Average')

# Subplot 3: Quarter-Year Rolling Average
plt.subplot(223)
plt.plot(df.index, df['rolling_quarter'], label='Quarter-Year Rolling Average', color='green')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.title('Quarter-Year Rolling Average')

# Subplot 4: One-Year Rolling Average
plt.subplot(224)
plt.plot(df.index, df['rolling_year'], label='One-Year Rolling Average', color='red')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.title('One-Year Rolling Average')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust spacing to fit suptitle
st.pyplot(plt.gcf())


# # Resample to Monthly Data
# 
# We want to use monthly average (or last-day-of-month close) to reduce noise and produce a more stable series.

# Resample to monthly and take the mean of the daily close prices
df_monthly = df['Close'].resample('ME').mean()

# Plot the monthly closing prices
plt.figure(figsize=(12,6))
plt.plot(df_monthly, label='Monthly Average Close', color='orange')
plt.title('NVIDIA Closing Price (Monthly)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot(plt.gcf())


# # Stationarity Checks
# 
# Time series models such as ARIMA assume stationarity. We’ll use:
# 
# 1. **Augmented Dickey-Fuller (ADF) test**
# 
# If the p-value in ADF is less than significance level (commonly 0.05), we typically say the series is stationary.
# 
# 2. **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test**
# 
# In the KPSS test, if the p-value is high, the series is more likely stationary (the null hypothesis for KPSS is stationarity).

from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series, title=''):
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Statistic','p-value','# Lags Used','Number of Observations Used']
    out = dict(zip(labels, result[0:4]))
    for key, val in out.items():
        print(f"   {key}: {val}")
    for key,val in result[4].items():
        print(f"   Critical Value {key}: {val}")
    print("")

def kpss_test(series, **kw):
    print('KPSS Test:')
    statistic, p_value, n_lags, critical_values = kpss(series.dropna(), **kw)
    print(f'   Test Statistic: {statistic}')
    print(f'   p-value: {p_value}')
    print(f'   # Lags: {n_lags}')
    for key, val in critical_values.items():
        print(f'   Critical Value {key}: {val}')
    print("")

# Apply ADF and KPSS
adf_test(df_monthly, title='Monthly NVIDIA Close')
kpss_test(df_monthly, regression='c')


# If ADF p-value > 0.05, it indicates non-stationarity.
# 
# If KPSS p-value < 0.05, it also indicates non-stationarity.
# 

# # Transformation
# 
# ARIMA often handles data more gracefully when it’s transformed to reduce heteroskedasticity (variance changing over time) and large fluctuations:

import numpy as np

df_monthly_log = np.log(df_monthly)
plt.figure(figsize=(12,6))
plt.plot(df_monthly_log, label='Log of Monthly Close', color='green')
plt.title('Log-Transformed Monthly Close')
plt.legend()
st.pyplot(plt.gcf())


# # First Differencing
# 
# After log transformation, we often apply differencing to remove trends and make the series stationary.

df_monthly_log_diff = df_monthly_log.diff().dropna()

# Plot differenced log
plt.figure(figsize=(12,6))
plt.plot(df_monthly_log_diff, label='Log Diff', color='red')
plt.title('Log-Differenced Monthly Close')
plt.legend()
st.pyplot(plt.gcf())

adf_test(df_monthly_log_diff, title='Log-Diff Monthly NVIDIA Close')
kpss_test(df_monthly_log_diff, regression='c')


# # Plot ACF & PACF
# 
# We use the AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF) plots to get initial guesses for (p,d,q) for the ARIMA model

import statsmodels.api as sm

fig, axes = plt.subplots(1, 2, figsize=(16,4))
sm.graphics.tsa.plot_acf(df_monthly_log_diff, lags=20, ax=axes[0])
sm.graphics.tsa.plot_pacf(df_monthly_log_diff, lags=20, ax=axes[1])
st.pyplot(plt.gcf())


# # Train-Test Split
# 
# Although there are many ways (like rolling forecasts), a simple approach is:
# 
# Split the monthly data into a training set (e.g., from 2014 up to, say, end of 2023).
# 
# Leave the last portion of data (e.g., 2024 to 2025) as a test set to assess performance.

# Example: split in 2023-01
split_date = '2023-01-01'
train = df_monthly_log[:split_date]
test = df_monthly_log[split_date:]

print(train.shape, test.shape)


# # Build the ARIMA Model

import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA

# We'll guess p=1, d=1, q=1 for demonstration.
# Adjust based on your ACF/PACF analysis.
p, d, q = 1, 1, 1

arima_model = ARIMA(train, order=(p, d, q))
arima_result = arima_model.fit()
print(arima_result.summary())

residuals = arima_result.resid
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(residuals)
axes[0].set_title('Residuals')
sm.graphics.tsa.plot_acf(residuals, lags=20, ax=axes[1])
st.pyplot(plt.gcf())


# # Forecast

forecast_steps = len(test)
forecast_result = arima_result.get_forecast(steps=forecast_steps)

# Predicted mean in log space
forecast_log = forecast_result.predicted_mean

# Confidence intervals in log space
ci = forecast_result.conf_int()

# Convert log-forecast back to original scale
forecast = np.exp(forecast_log)
ci_exp = np.exp(ci)

# Build a DataFrame to compare
forecast_index = test.index
forecast_df = pd.DataFrame({
    'Forecast': forecast.values,
    'Lower_CI': ci_exp.iloc[:, 0].values,
    'Upper_CI': ci_exp.iloc[:, 1].values
}, index=forecast_index)

# Combine actual (test) & forecast
combined_df = pd.concat([df_monthly.loc[test.index], forecast_df], axis=1)

plt.figure(figsize=(12,6))
plt.plot(df_monthly, label='Original Series', color='blue')
plt.plot(combined_df['Forecast'], label='Forecast', color='red')
plt.fill_between(
    combined_df.index,
    combined_df['Lower_CI'],
    combined_df['Upper_CI'],
    color='pink', alpha=0.3
)
plt.title('NVIDIA Monthly Close Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
st.pyplot(plt.gcf())

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller, kpss
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima.model import ARIMA
# from scipy.stats import shapiro
# from arch import arch_model

# # Streamlit page setup
# st.set_page_config(page_title="NVIDIA Stock Analysis", layout="wide")
# st.title("NVIDIA Stock Time Series Analysis")

# # 1. Retrieve NVIDIA stock data
# st.header("1. Download NVIDIA Stock Data")
# start_date = st.date_input("Start Date", value=pd.to_datetime('2005-01-01'))
# end_date = st.date_input("End Date", value=pd.to_datetime('2025-01-01'))

# if start_date >= end_date:
#     st.error("Error: End date must fall after start date.")
# else:
#     data = yf.download('NVDA', start=start_date, end=end_date)
#     close_prices = data['Close']
#     st.line_chart(close_prices)

#     # 2. Plot closing prices
#     st.header("2. Closing Prices")
#     fig1, ax1 = plt.subplots(figsize=(10,6))
#     ax1.plot(close_prices, color='black')
#     ax1.set_title('NVIDIA Corporation Closing Prices')
#     ax1.set_xlabel('Date')
#     ax1.set_ylabel('Price')
#     st.pyplot(fig1)

#     # 3. Decompose time series
#     st.header("3. Time Series Decomposition")
#     try:
#         decomp = seasonal_decompose(close_prices, model='multiplicative', period=30)
#         fig2 = decomp.plot()
#         st.pyplot(fig2)
#     except Exception as e:
#         st.error(f"Decomposition Error: {e}")

#     # 4. Stationarity Tests
#     st.header("4. Stationarity Tests on Original Data")
#     result_adf = adfuller(close_prices)
#     result_kpss = kpss(close_prices, regression='c', nlags="auto")

#     st.subheader("ADF Test")
#     st.write(f"ADF Statistic: {result_adf[0]:.4f}")
#     st.write(f"p-value: {result_adf[1]:.4f}")

#     st.subheader("KPSS Test")
#     st.write(f"KPSS Statistic: {result_kpss[0]:.4f}")
#     st.write(f"p-value: {result_kpss[1]:.4f}")

#     # 5. First Differencing
#     st.header("5. First Differencing")
#     diff_close = close_prices.diff().dropna()
#     fig3, ax3 = plt.subplots(figsize=(10,6))
#     ax3.plot(diff_close, color='black')
#     ax3.set_title('First Differenced Closing Prices')
#     ax3.set_xlabel('Date')
#     ax3.set_ylabel('Differenced Price')
#     st.pyplot(fig3)

#     # Log-differencing (optional)
#     st.header("5b. Log-Differencing (Optional)")
#     log_close = np.log(close_prices)
#     log_diff_close = log_close.diff().dropna()

#     result_adf_log = adfuller(log_diff_close)
#     result_kpss_log = kpss(log_diff_close, regression='c', nlags="auto")

#     st.subheader("ADF Test (Log Differenced)")
#     st.write(f"ADF Statistic: {result_adf_log[0]:.4f}")
#     st.write(f"p-value: {result_adf_log[1]:.4f}")

#     st.subheader("KPSS Test (Log Differenced)")
#     st.write(f"KPSS Statistic: {result_kpss_log[0]:.4f}")
#     st.write(f"p-value: {result_kpss_log[1]:.4f}")

#     # 6. Plot ACF and PACF
#     st.header("6. ACF and PACF Plots")
#     fig4, ax4 = plt.subplots()
#     plot_acf(diff_close, lags=40, ax=ax4)
#     st.pyplot(fig4)

#     fig5, ax5 = plt.subplots()
#     plot_pacf(diff_close, lags=40, ax=ax5)
#     st.pyplot(fig5)

#     # 7. ARIMA Model Fitting
#     st.header("7. ARIMA Model Fitting")
#     order_list = [(1,1,0), (0,1,1), (1,1,1), (2,1,1), (1,1,2), (2,1,2)]
#     best_aic = np.inf
#     best_order = None
#     best_model = None

#     for order in order_list:
#         try:
#             model = ARIMA(close_prices, order=order)
#             model_fit = model.fit()
#             st.write(f"ARIMA{order} - AIC:{model_fit.aic:.2f}")
#             if model_fit.aic < best_aic:
#                 best_aic = model_fit.aic
#                 best_order = order
#                 best_model = model_fit
#         except Exception as e:
#             st.write(f"ARIMA{order} - Error: {e}")

#     if best_model:
#         st.success(f"Best ARIMA Order: {best_order} with AIC: {best_aic:.2f}")
#         st.subheader("Forecast Plot")
#         forecast = best_model.get_forecast(steps=30)
#         forecast_index = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), periods=30)
#         forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

#         fig6, ax6 = plt.subplots(figsize=(10,6))
#         ax6.plot(close_prices, label='Observed')
#         ax6.plot(forecast_series, label='Forecast', color='red')
#         ax6.legend()
#         st.pyplot(fig6)
#     else:
#         st.warning("No ARIMA model could be fitted.")
