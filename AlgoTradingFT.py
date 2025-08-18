import time, hmac, hashlib, requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

API_KEY = ""
SECRET = "secret"

BASE_URL = "https://api.coinex.com"
#with open('C:/Users/degod/Dev/python-coinex/key_data.txt', 'r') as file:
         #api_key = file.read().strip()
         #secret = file.read().strip()
         #print(f" Key is {api_key} and secret is {secret}")
#         print("teste")

API_KLINE_PATH = "/v2/futures/kline"

def sign_request(method, path, body=""):
    ts = str(int(time.time() * 1000))
    payload = method + path + body + ts
    signature = hmac.new(bytes(SECRET, 'latin-1'),
                         msg=bytes(payload, 'latin-1'),
                         digestmod=hashlib.sha256).hexdigest().lower()
    return ts, signature

def get_futures_ticker(symbol):
    path = f"/v2/futures/ticker?market={symbol}"
    ts, sig = sign_request("GET", path)
    headers = {
        "X-COINEX-KEY": API_KEY,
        "X-COINEX-SIGN": sig,
        "X-COINEX-TIMESTAMP": ts,
        "Content-Type": "application/json; charset=utf-8"
    }
    resp = requests.get(BASE_URL + path, headers=headers)
    j = resp.json()
    return j if j.get("code") == 0 else {"error": j}

def get_futures_data(market: str, days: int = 60):
    url = BASE_URL + API_KLINE_PATH
    params = {
        "market": market,
        "period": "1day",  # daily data
        "limit": days
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("code") != 0:
        raise Exception(f"Error fetching {market} data: {data.get('message')}")
    
    df = pd.DataFrame(data['data']).dropna()
    ts = df['created_at'] / 1000
    df['time'] = pd.to_datetime(ts, unit='s')

    return df

# Define measurement function H (dynamic, depends on x_t)
def update_H(x_t):
    return np.array([[x_t, 1]])  # y_t = slope * x_t + intercept + noise

def run_kalman_filter_momentum(y, x,periods):
    kf = KalmanFilter(dim_x=2, dim_z=1)  # State: [slope, intercept], Measurement: y
    # Define state transition matrix (random walk for slope and intercept)
    kf.F = np.array([[1, 0],  # Slope stays constant (random walk)
                     [0, 1]]) # Intercept stays constant (random walk)
    
    # Initial state and covariance
    kf.x = np.array([0.5, 0])  # Initial guess: slope = 0.5, intercept = 0
    kf.P *= 1  # Initial uncertainty
    kf.R = 0.008057805602701749    # Measurement noise variance
    kf.Q = np.eye(2) * 1e-5  # Process noise (small for slow evolution)

    # Step 3: Run Kalman Filter and compute dynamic spread
    spreads = []
    for t in range(periods):
        kf.H = update_H(x[t])  # Update measurement matrix with current x_t
        kf.predict()
        kf.update(y[t])
        slope, intercept = kf.x
        spread = y[t] - (slope * x[t] + intercept)  # Dynamic spread
        spreads.append(spread)


    return slope, intercept, spreads

def check_cointegration(symbolY,symbolX,periods):
    asset1_prices = get_futures_data(symbolY,days=periods)
    asset2_prices = get_futures_data(symbolX,days=periods)

    asset1_prices['close'] = asset1_prices['close'].astype(float)
    asset2_prices['close'] = asset2_prices['close'].astype(float)

    correlation = asset1_prices['close'].corr(asset2_prices['close'])
    dates = asset1_prices['time']
    asset1_prices = np.array(asset1_prices['close'])
    asset2_prices = np.array(asset2_prices['close'])
     # Log-transform the prices
    log_asset1 = np.log(asset1_prices)
    log_asset2 = np.log(asset2_prices)
    
    data = pd.DataFrame({'LogPrice1': log_asset1, 'LogPrice2': log_asset2}, index=dates)

    # Run the Kalman Filter
    slope, intercept, spreads = run_kalman_filter_momentum(log_asset1, log_asset2,periods)

    # Convert spreads to a pandas Series for easier manipulation
    dates = np.array(data.index)
    spreads = pd.Series(spreads, index=dates)
    slope = pd.Series(slope, index=dates)
    
    mean_spreads = spreads.mean()
    std_spreads = spreads.std()
    zscore_spread = (spreads - mean_spreads) / std_spreads

    # Step 4: Compute z-scores of the spread
    mean_spread = spreads.rolling(window=60, min_periods=1).mean()
    std_spread = spreads.rolling(window=60, min_periods=1).std()
    z_scores_rolling = (spreads - mean_spread) / std_spread

    # Convert `dynamic_spread` to a pandas Series
    spread_series = pd.Series(zscore_spread, name="dynamic_spread")
    # Calculate the lagged spread and the difference
    spread_lagged = spread_series.shift(1).fillna(0)  # Lagged spread (y_{t-1})
    spread_delta = spread_series - spread_lagged  # Change in spread (Delta y_t)
    
    # Reshape the data for linear regression
    X = spread_lagged.values.reshape(-1, 1)  # Independent variable
    y = spread_delta.values  # Dependent variable

    # Perform linear regression: y = kappa * X + noise
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    kappa = -model.coef_[0]  # Speed of mean reversion (kappa)

    # Calculate the half-life
    half_life = np.log(2) / kappa


    adf_result = adfuller(spreads)
    p_value = adf_result[1]
    coint_t = adf_result[0]
    critical_value = adf_result[4]['5%']
    t_check = coint_t < critical_value
    coint_flag = p_value < 0.05 and t_check

    return z_scores_rolling,coint_flag,correlation,half_life 
     

def main():
    PERIODS = 252

    major_pairs_y = ["BTCUSDT","ETHUSDT","ADAUSDT","AVAXUSDT"]
    major_pairs_x = ["XRPUSDT","LTCUSDT","SOLUSDT","LINKUSDT"]

    for i in range(len(major_pairs_y)):
        for j in range(len(major_pairs_x)):
             print(f"Checking symbols {major_pairs_y[i]} and {major_pairs_x[j]}")
             #z_scores_rolling = check_cointegration(major_pairs_y[i],major_pairs_x[j],PERIODS)
             data_1 = get_futures_data(major_pairs_y[i],days=PERIODS)
             data_2 = get_futures_data(major_pairs_x[j],days=PERIODS)
             price1 = np.array(data_1['close'].astype(float))
             price2 = np.array(data_2['close'].astype(float))
             dates = np.array(data_1['time'])
             price_data = pd.DataFrame({'Price1': price1,'Price2': price2}, index=dates)
             plot_series(price_data,major_pairs_y[i],major_pairs_x[j])       

def plot_series(price_data,symbol_y,symbol_x):    
    price_data['Return1'] = price_data['Price1'].pct_change().cumsum()
    price_data['Return2'] = price_data['Price2'].pct_change().cumsum()

    z_scores_rolling,coint,correlation,half_life = check_cointegration(symbol_y,symbol_x,252)
    
    if coint and half_life < 3:
        print(f"Is cointegrated ? {coint} half life is {half_life} z score is {z_scores_rolling[-1]} correlation is {correlation}")
        plot_asset_spreads(price_data,z_scores_rolling,symbol_y,symbol_x)

def plot_asset_spreads(price_data,z_scores,symbol_y,symbol_x):

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(price_data.index, price_data['Return1'], label='Cumulative returns BTCUSDT', color='red')
    plt.plot(price_data.index, price_data['Return2'], label='Cumulative returns ETHUSDT', color='blue')
    plt.title(f'Pair Trade Cumulative Returns of {symbol_y} and {symbol_x}')
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Plot the original and smoothed Z-scores
    plt.subplot(2, 1, 2)
    plt.plot(price_data.index,z_scores, label='Rolling Z-scores', color='green', linewidth=1)
    plt.axhline(0, color='black')
    plt.axhline(1, color='blue', linestyle='--')
    plt.axhline(2, color='green', linestyle='--', label='+2 Std Dev')
    plt.axhline(-1, color='red', linestyle='--')
    plt.axhline(-2, color='green', linestyle='--', label='-2 Std Dev')
    plt.xlabel('Date')
    plt.ylabel('Z-score')
    plt.title(f'Z-scores for assets {symbol_y} and {symbol_x} ')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(-4.0, 4.0)
    plt.show()    

if __name__ == "__main__":
    main()