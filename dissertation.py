#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:49:21 2024

@author: maddierush
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import r2_score 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss
from scipy import stats
from matplotlib.ticker import MaxNLocator
import mplfinance as mpf
import yfinance as yf

def preparing(df):
    df[df == 0] = np.nan
    df = df.dropna(axis=0, how='all')
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])
    df['SP500'] = pd.to_numeric(df['SP500'], errors='coerce')
    df = df.dropna(subset=['SP500'])
    df = df.sort_values('DATE')
    return df

def lineplot(data, title):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='DATE', y='SP500', errorbar=None)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=13))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=13))
    plt.xticks(rotation=45)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def boxplot(data, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=data['SP500'])
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
    
def plot_sma(df, title):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='DATE', y='SP500', data=df, label='S&P 500 Close Price', color='blue', alpha=0.5)
    sns.lineplot(x='DATE', y='SMA_50', data=df, label='50-Day SMA', color='orange', alpha=0.7)
    sns.lineplot(x='DATE', y='SMA_200', data=df, label='200-Day SMA', color='green', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def plot_ema(df, title):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='DATE', y='SP500', data=df, label='S&P 500 Close Price', color='blue', alpha=0.5)
    sns.lineplot(x='DATE', y='EMA_50', data=df, label='50-Day EMA', color='orange', alpha=0.7)
    sns.lineplot(x='DATE', y='EMA_200', data=df, label='200-Day EMA', color='green', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def stationary_check(time_series):
    kpss_stat, p_value, lags, critical_values = kpss(time_series, regression='c')
    print(f'KPSS Statistic: {kpss_stat}')
    print(f'p-value: {p_value}')
    print(f'Lags used: {lags}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    alpha = 0.05
    if p_value < alpha:
        print("Result: The time series is not stationary (reject the null hypothesis).")
        return False
    else:
        print("Result: The time series is stationary (fail to reject the null hypothesis).")
        return True

    
def differencing(series, label, order):
    diff_series = series.diff(periods=order).dropna()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    sns.lineplot(x=series.index, y=series, label='Original Series')  # Use sns.lineplot for the original series
    plt.title(f'Original Time Series {label}')
    plt.xlabel('Date')
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    sns.lineplot(x=diff_series.index, y=diff_series, color='orange', label=f'{order}-order Difference')  # sns.lineplot for differenced series
    plt.title(f'{order}-order Differenced Time Series {label}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
    
    return diff_series
  
def plot_acf_pacf(series, lags=40):
    from statsmodels.tsa.stattools import acf, pacf
    acf_values = acf(series, nlags=lags)
    pacf_values = pacf(series, nlags=lags)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    sns.lineplot(x=range(len(acf_values)), y=acf_values, marker="o", label='ACF')
    plt.axhline(0, linestyle='--', color='gray')  # Zero line
    plt.title('Auto-Correlation Function (ACF)')
    plt.xlabel('Lags')
    plt.ylabel('ACF')

    plt.subplot(122)
    sns.lineplot(x=range(len(pacf_values)), y=pacf_values, marker="o", color='orange', label='PACF')
    plt.axhline(0, linestyle='--', color='gray')  # Zero line
    plt.title('Partial Auto-Correlation Function (PACF)')
    plt.xlabel('Lags')
    plt.ylabel('PACF')

    plt.tight_layout()
    plt.show()


def parameters(data):
    model = pm.auto_arima(data, start_p=1, start_q=1,
                          max_p=5, max_q=5, seasonal=False,
                          stepwise=True, trace=True,
                          error_action='ignore', suppress_warnings=True)
    print(model.summary())
    return model.order

def model_fit(data, order):
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    print(fitted_model.summary())
    return fitted_model

def arima_pipeline(title, df_compare, data):
    forecast_start_date = pd.to_datetime('2023-10-11')
    forecast_end_date = pd.to_datetime('2024-05-05')
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='B')  # Business days
    n = len(forecast_dates)

    difference_count = 0
    data_diff = data.diff().dropna()
    difference_count += 1
    print(f"Differencing applied {difference_count} time(s).")

    stationary = stationary_check(data_diff)
    if not stationary:
        data_diff = data_diff.diff().dropna()
        difference_count += 1
        print(f"Second differencing applied. Total: {difference_count} time(s).")
        
    print("Final Stationarity Check:")
    stationary_check(data_diff)

    order = parameters(data_diff)
    print('The orders are:', order)
    
    fitted_model = model_fit(data_diff, order)
    forecasted_diff = fitted_model.forecast(steps=n)
    forecasted = forecasted_diff.cumsum() + data.iloc[-1]
    forecast_df = pd.DataFrame({
        'DATE': forecast_dates,
        'Forecasted_SP500': forecasted
    })
    
    mean_forecast = np.mean(forecasted)
    std_dev = np.std(forecasted, ddof =1)
    n = len(forecasted)
    se = std_dev/np.sqrt(n)
    confidence_level = 0.95
    degrees_freedom = n -1
    critical_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error = critical_value * se
    
    lower_bound = mean_forecast - margin_of_error
    upper_bound = mean_forecast + margin_of_error
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=forecast_df['DATE'], y=forecast_df['Forecasted_SP500'], label='Forecasted Data', color='red')
    plt.fill_between(forecast_df['DATE'], forecasted - margin_of_error, forecasted + margin_of_error, 
                 color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.title('Forecasted data using ' + title)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=13))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=13))
    plt.xticks(rotation=45)
    plt.show()

# Plot 2: Actual data with forecast
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=forecast_df['DATE'], y=forecast_df['Forecasted_SP500'], label='Forecast', color='red', linewidth=0.5)
    sns.lineplot(x=data.index, y=data, label='Original Data', color='blue', linewidth=1)
    plt.fill_between(forecast_df['DATE'], forecasted - margin_of_error, forecasted + margin_of_error, 
                 color='purple', alpha=0.2, label='95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.title('Actual Past Data with the addition of the forecasted data using ' + title)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=13))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=13))
    plt.xticks(rotation=45)
    plt.show()

# Plot 3: Actual vs forecasted comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df_compare['DATE'], y=df_compare['SP500'], label='Actual Data')
    sns.lineplot(x=forecast_df['DATE'], y=forecast_df['Forecasted_SP500'], label='Forecasted Data', color='red')
    plt.fill_between(forecast_df['DATE'], forecasted - margin_of_error, forecasted + margin_of_error, 
                 color='purple', alpha=0.2, label='95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.title('Actual data compared to forecasted data using ' + title)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=13))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=13))
    plt.xticks(rotation=45)
    plt.show()

    return fitted_model, forecast_df


    

def calculate_errors(df_compare, actual_column, forecast_columns):

    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        epsilon = np.finfo(np.float64).eps
        y_true = np.where(y_true == 0, epsilon, y_true)
  
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100                             
    
    error_metrics = {}

    for forecast_column in forecast_columns:
        y_true = df_compare[actual_column].values  # Actual values
        y_pred = df_compare[forecast_column].values  # Forecasted values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        error_metrics[forecast_column] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    return error_metrics

def Rsquared(df_compare, actual_column, forecast_column):
    y_true = df_compare[actual_column].values
    y_pred = df_compare[forecast_column].values
    score = r2_score(y_true, y_pred)
    return score

def theils_u(actual, forecast):
    numerator = np.sqrt(np.mean(np.square(actual - forecast)))
    denominator = np.sqrt(np.mean(np.square(actual[1:] - actual[:-1])))
    return numerator / denominator

def dm_test(actual, forecast1, forecast2, h=1, crit="MSE"):
    def loss_func(actual, forecast, crit="MSE"):
        if crit == "MSE":
            return (actual - forecast) ** 2
        elif crit == "MAE":
            return np.abs(actual - forecast)
        else:
            raise ValueError("Criterion must be 'MSE' or 'MAE'")
    
    d1 = loss_func(actual, forecast1, crit)
    d2 = loss_func(actual, forecast2, crit)
    
    d = d1 - d2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    dm_stat = mean_d / np.sqrt((var_d / len(d)))

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=len(d)-1))
    
    return dm_stat, p_value

def advanced_analysis(df_compare, actual_column, forecast_columns):
    result = {}
    
    # Theil's U
    for forecast_column in forecast_columns:
        u_stat = theils_u(df_compare[actual_column].values, df_compare[forecast_column].values)
        result[f'{forecast_column}_Theils_U'] = u_stat
        print(f"Theil's U for {forecast_column}: {u_stat}")

    # Diebold-Mariano test between the two forecasts
    if len(forecast_columns) == 2:
        forecast1 = df_compare[forecast_columns[0]].values
        forecast2 = df_compare[forecast_columns[1]].values
        actual = df_compare[actual_column].values
        
        dm_stat, p_value = dm_test(actual, forecast1, forecast2)
        result['DM_test_stat'] = dm_stat
        result['DM_test_p_value'] = p_value
        print(f"DM test statistic: {dm_stat}, p-value: {p_value}")
    
    return result

# EXPLORATORY ANALYSIS
    
df_whole = pd.read_csv('/Users/maddierush/Desktop/MASTERS/DISSERTATION/SP500 5.csv')
df_whole = preparing(df_whole)
lineplot(df_whole, 'Average S&P 500 Close Price from 16/10/2019 to 10/10/2023')

df_whole['SMA_50'] = df_whole['SP500'].rolling(window=50).mean()
df_whole['SMA_200'] = df_whole['SP500'].rolling(window=200).mean()
plot_sma(df_whole, 'Average S&P 500 Close Price from 16/10/2019 to 10/10/2023')

df_whole['EMA_50'] = df_whole['SP500'].ewm(span=50, adjust=False).mean()
df_whole['EMA_200'] = df_whole['SP500'].ewm(span=200, adjust=False).mean()
plot_ema(df_whole, 'Average S&P 500 Close Price with 50-day and 200-day EMA from 16/10/2019 to 10/10/2023')

#CANDLESTICK 
ticker = '^GSPC'
df = yf.download(ticker, start='2019-10-16', end='2023-10-10')
df.index.name = 'Date'

my_style = mpf.make_mpf_style(base_mpf_style='yahoo',
                              gridstyle='-.',
                              gridcolor='lightgray',
                              y_on_right=False)  # Default position of y-axis
kwargs = dict(type='candle', 
              style=my_style,
              ylabel='Price (USD)', 
              datetime_format='%Y-%b',  # Clearer date format
              xrotation=45,  # Rotate x-axis labels
              volume=False,  # Hide volume if not needed
              tight_layout=True)  # Optimize spacing
mpf.plot(df, **kwargs)

df_covid = pd.read_csv('/Users/maddierush/Desktop/MASTERS/DISSERTATION/SP500.csv')
df_covid = preparing(df_covid)

lineplot(df_covid, 'Average S&P 500 Close Price from 11/03/2020 to 05/05/2023')
#boxplot(df_covid, 'Variation in S&P 500 Close prices from 11/03/2020 to 05/05/2023')

df_precovid = pd.read_csv('/Users/maddierush/Desktop/MASTERS/DISSERTATION/SP500 2.csv')
df_precovid = preparing(df_precovid)    

#lineplot(df_precovid, 'Average S&P 500 Close price from 16/10/2019 to 11/03/2020')
#boxplot(df_precovid, 'Variation in S&P 500 Close prices from 16/10/2019 to 11/03/2020')

df_postcovid = pd.read_csv('/Users/maddierush/Desktop/MASTERS/DISSERTATION/SP500 3.csv')
df_postcovid = preparing(df_postcovid)

#lineplot(df_postcovid, 'Average S&P 500 Close Price from 05/05/2023 to 10/10/2023')
#boxplot(df_postcovid, 'Variation in S&P 500 Close prices from 05/05/2023 to 10/10/2023')

df_compare = pd.read_csv('/Users/maddierush/Desktop/MASTERS/DISSERTATION/SP500 4.csv')
df_compare = preparing(df_compare)

lineplot(df_compare, 'Average S&P 500 Close Price from 10/10/2023 to 05/03/2024')

df_whole['DATE'] = pd.to_datetime(df_whole['DATE'])
df_whole = df_whole.set_index('DATE')

df_combined = pd.concat([df_precovid, df_postcovid])

df_combined['DATE'] = pd.to_datetime(df_combined['DATE'])

df_combined = df_combined.set_index('DATE').sort_index()

lineplot(df_combined.reset_index(), 'Combined Pre and Post COVID Data')

diff_whole = differencing(df_whole['SP500'], '(Full Sample)', order=1)
diff_combined = differencing(df_combined['SP500'], '(Sub Sample)', order=1)

stationary_check(diff_whole)
stationary_check(diff_combined)

plot_acf_pacf(diff_whole)
order1 = parameters(diff_whole)
plot_acf_pacf(diff_combined)
order2 = parameters(diff_combined)

ts_whole = df_whole['SP500']
fitted_model_whole, forecasted_whole_df = arima_pipeline('the Full Sample', df_compare, ts_whole)

combined_ts = df_combined['SP500']
fitted_model_combined, forecasted_combined_df = arima_pipeline('the Sub Sample', df_compare, combined_ts)

df_compare = df_compare.set_index('DATE')  
forecasted_whole_df = forecasted_whole_df.set_index('DATE')
forecasted_combined_df = forecasted_combined_df.set_index('DATE')

df_compare['Forecast_Whole'] = forecasted_whole_df['Forecasted_SP500']
df_compare['Forecast_Combined'] = forecasted_combined_df['Forecasted_SP500']

forecast_columns = ['Forecast_Whole', 'Forecast_Combined']
error_metrics = calculate_errors(df_compare, actual_column='SP500', forecast_columns=forecast_columns)

df_compare = df_compare.dropna(subset=['SP500', 'Forecast_Whole', 'Forecast_Combined'])

df_compare['SP500'] = pd.to_numeric(df_compare['SP500'], errors='coerce')
df_compare['Forecast_Whole'] = pd.to_numeric(df_compare['Forecast_Whole'], errors='coerce')
df_compare['Forecast_Combined'] = pd.to_numeric(df_compare['Forecast_Combined'], errors='coerce')

error_metrics = calculate_errors(df_compare, actual_column='SP500', forecast_columns=['Forecast_Whole', 'Forecast_Combined'])

print("Error Metrics:")
for forecast_column, errors in error_metrics.items():
    print(f"{forecast_column}: MAE = {errors['MAE']}, RMSE = {errors['RMSE']}, MAPE={errors['MAPE']}")

score_whole = Rsquared(df_compare, actual_column='SP500', forecast_column='Forecast_Whole')
print(f"R-squared for Forecast_Whole: {score_whole}")

score_combined = Rsquared(df_compare, actual_column='SP500', forecast_column='Forecast_Combined')
print(f"R-squared for Forecast_Combined: {score_combined}") 


advanced_metrics = advanced_analysis(df_compare, actual_column='SP500', forecast_columns=['Forecast_Whole', 'Forecast_Combined'])

