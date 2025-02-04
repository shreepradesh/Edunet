import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define stock symbols and time period
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
start_date = '2020-01-01'
end_date = '2024-01-01'

# Fetch historical data
data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
daily_returns = data.pct_change()

# Plot stock price trends
plt.figure(figsize=(12, 6))
data.plot(title='Stock Price Trends', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend(stocks)
plt.show()

# Calculate moving averages (50-day and 200-day)
moving_avg_50 = data.rolling(window=50).mean()
moving_avg_200 = data.rolling(window=200).mean()

# Plot moving averages
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['AAPL'], label='AAPL Price', alpha=0.6)
plt.plot(data.index, moving_avg_50['AAPL'], label='50-day MA', linestyle='dashed')
plt.plot(data.index, moving_avg_200['AAPL'], label='200-day MA', linestyle='dashed')
plt.title('AAPL Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate stock volatility (Standard Deviation of returns)
volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
print("Stock Volatility:")
print(volatility)

# Correlation matrix of stock returns
correlation_matrix = daily_returns.corr()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Stock Correlation Matrix')
plt.show()
