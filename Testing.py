# -*- coding: utf-8 -*-
import hashlib
import json
import time
import hmac
import os
from urllib.parse import urlparse, urlencode
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from filterpy.kalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller

API_KLINE_PATH = "/futures/kline"

def load_api_keys():
    # Path to the file in Windows home folder
    home_dir = os.path.expanduser("~")
    key_file = os.path.join(home_dir, "Dev\python-coinex\key_data")

    api_key = None
    secret = None

    # Read file line by line
    with open(key_file, "r") as f:
        for line in f:
            if line.startswith("API_KEY="):
                api_key = line.strip().split("=", 1)[1]
            elif line.startswith("SECRET="):
                secret = line.strip().split("=", 1)[1]

    if not api_key or not secret:
        raise ValueError("API_KEY or SECRET not found in file")
    api_key = api_key.strip('"')
    secret = secret.strip('"')
    
    return api_key, secret

class RequestsClient(object):
    HEADERS = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "X-COINEX-KEY": "",
        "X-COINEX-SIGN": "",
        "X-COINEX-TIMESTAMP": "",
    }

    def __init__(self):
        self.access_id, self.secret_key = load_api_keys()
        self.url = "https://api.coinex.com/v2"
        self.headers = self.HEADERS.copy()

    # Generate your signature string
    def gen_sign(self, method, request_path, body, timestamp):
        print(f"Generating signature for method: {self.secret_key}")
        prepared_str = f"{method}{request_path}{body}{timestamp}"
        signature = hmac.new(
            bytes(self.secret_key, 'latin-1'), 
            msg=bytes(prepared_str, 'latin-1'), 
            digestmod=hashlib.sha256
        ).hexdigest().lower()
        return signature

    def get_common_headers(self, signed_str, timestamp):
        print(f"Loaded API Key: {self.access_id} ")
        headers = self.HEADERS.copy()
        headers["X-COINEX-KEY"] = self.access_id
        headers["X-COINEX-SIGN"] = signed_str
        headers["X-COINEX-TIMESTAMP"] = timestamp
        headers["Content-Type"] = "application/json; charset=utf-8"
        return headers

    def request(self, method, url, params={}, data=""):
        req = urlparse(url)
        request_path = req.path

        timestamp = str(int(time.time() * 1000))
        if method.upper() == "GET":
            # If params exist, query string needs to be added to the request path
            if params:
                for item in params:
                    if params[item] is None:
                        del params[item]
                        continue
                request_path = request_path + "?" + urlencode(params)

            signed_str = self.gen_sign(
                method, request_path, body="", timestamp=timestamp
            )
            response = requests.get(
                url,
                params=params,
                headers=self.get_common_headers(signed_str, timestamp),
            )

        else:
            signed_str = self.gen_sign(
                method, request_path, body=data, timestamp=timestamp
            )
            response = requests.post(
                url, data, headers=self.get_common_headers(signed_str, timestamp)
            )

        if response.status_code != 200:
            raise ValueError(response.text)
        return response


request_client = RequestsClient()


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

    return z_scores_rolling,coint_flag,correlation,half_life,slope 

def verify_pairs():
    PERIODS = 252
    major_pairs_y = ["BTCUSDT","ETHUSDT","ADAUSDT","AVAXUSDT"]
    major_pairs_x = ["XRPUSDT","LTCUSDT","SOLUSDT","LINKUSDT"]

    for i in range(len(major_pairs_y)):
        for j in range(len(major_pairs_x)):
            print(f"Processing pair: {major_pairs_y[i]} and {major_pairs_x[j]}")
            #z_scores_rolling = check_cointegration(major_pairs_y[i],major_pairs_x[j],PERIODS)
            data_1 = get_futures_data(major_pairs_y[i],days=PERIODS)
            data_2 = get_futures_data(major_pairs_x[j],days=PERIODS)
            price1 = np.array(data_1['close'].astype(float))
            price2 = np.array(data_2['close'].astype(float))
            dates = np.array(data_1['time'])
            price_data = pd.DataFrame({'Price1': price1,'Price2': price2}, index=dates)
            
            z_scores_rolling,coint,correlation,half_life,hedge_ratio = check_cointegration(major_pairs_y[i],major_pairs_x[j],252)
            z_score = z_scores_rolling[-1]
            hedge_ratio = hedge_ratio[-1]
            if coint and half_life < 10 and z_score > 0.60:
                print(f"Is cointegrated ? {coint} half life is {half_life} z score is {z_score} correlation is {correlation} hedge ratio is {hedge_ratio}")
                return z_score,correlation,hedge_ratio,major_pairs_y[i],major_pairs_x[j]
                #plot_asset_spreads(price_data,z_scores_rolling,symbol_y,symbol_x)
    return None, None, None, None, None

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
    plt.show()  

def get_spot_market():
    request_path = "/spot/market"
    params = {"market": "BTCUSDT"}
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        params=params,
    )
    return response

def get_futures_ticker(symbol):
    request_path = "/futures/ticker"
    params = {"market": symbol}
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        params=params,
    )

    return response

def get_futures_data(market: str, days: int = 60):
    request_path = "/futures/kline"
    params = {
        "market": market,
        "period": "1day",  # daily data
        "limit": days
    }
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        params=params,
    )

    data = response.json()
    
    if data.get("code") != 0:
        raise Exception(f"Error fetching {market} data: {data.get('message')}")
    
    df = pd.DataFrame(data['data']).dropna()
    ts = df['created_at'] / 1000
    df['time'] = pd.to_datetime(ts, unit='s')

    return df

def get_spot_balance():
    request_path = "/assets/spot/balance"
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
    )
    return response

def get_open_positions():
    request_path = "/futures/pending-position"
    params = {"market_type": "FUTURES"}
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        params=params,
    )
    return response

def get_futures_balance():
    request_path = "/assets/futures/balance"
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
    )
    return response

def get_deposit_address():
    request_path = "/assets/deposit-address"
    params = {"ccy": "USDT", "chain": "CSC"}

    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        params=params,
    )
    return response


def get_min_amount_futures(market):
    request_path = "/futures/market"
    params = {"market": market}
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        params=params,
    )
    return response

def send_order_futures_market(market,side,volume):
    request_path = "/futures/order"
    data = {
        "market": market,
        "market_type": "FUTURES",
        "side": side,
        "type": "market",
        "amount": float(volume),
        "client_id": "degodoy",
        "is_hide": True,
    }
    data = json.dumps(data)
    response = request_client.request(
        "POST",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        data=data,
    )
    return response
def get_futures_price(market):
    request_path = "/futures/ticker"
    params = {"market": market}
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        params=params,
    )
    return response

def calculate_volume(symbol_y,symbol_x,hedge_ratio):
    
    asset_price_y = get_futures_price(symbol_y).json()['data'][0]['mark_price']
    asset_price_x = get_futures_price(symbol_x).json()['data'][0]['mark_price']
    asset_price_y = float(asset_price_y)
    asset_price_x = float(asset_price_x)
    max_volume_usd = get_futures_balance().json()  # Maximum volume in USD to be allocated
    print(f"Max volume in USD: {max_volume_usd} for symbols {symbol_y} and {symbol_x}")
    max_volume_usd = float(max_volume_usd['data'][0]['available'])

    print(f"Balance {max_volume_usd} and hedge ratio {hedge_ratio} for symbols {symbol_y} and {symbol_x}")
    volume_y = (max_volume_usd/(1 + hedge_ratio))/asset_price_y
    volume_x = (max_volume_usd - volume_y)/asset_price_x
    
    return volume_y, volume_x


def close_futures_position(market):
    request_path = "/futures/close-position"
    data = {
        "market": market,
        "market_type": "FUTURES",
        "type": "market",
        "client_id": "degodoy",
        "is_hide": True,
    }
    data = json.dumps(data)
    response = request_client.request(
        "POST",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        data=data,
    )
    return response


def run_code():
    response = get_min_amount_futures("ADAUSDT").json()
    print(f"Response for minimum amount futures: {response['data'][0]['min_amount']}")
    PERIODS = 252
    data_1 = get_futures_data("ADAUSDT",days=PERIODS)
    data_2 = get_futures_data("XRPUSDT",days=PERIODS)
    price1 = np.array(data_1['close'].astype(float))
    price2 = np.array(data_2['close'].astype(float))
    dates = np.array(data_1['time'])
    
    price_data = pd.DataFrame({'Price1': price1,'Price2': price2}, index=dates)
    price_data['Return1'] = price_data['Price1'].pct_change().cumsum()
    price_data['Return2'] = price_data['Price2'].pct_change().cumsum()
    z_score,correlation,hedge_ratio,symbol_y,symbol_x = check_cointegration("ADAUSDT","XRPUSDT",252)
    #plot_asset_spreads(price_data, z_score, symbol_y, symbol_x)
    asset_volume = 50  # Total volume to be allocated between the two assets
    while True:
         asset_symbol = "ADAUSDT"
         response_1 = get_futures_price(asset_symbol).json()
         last = float(response_1['data'][0]['last'])
         index_price = float(response_1['data'][0]['index_price'])
         mark_price = float(response_1['data'][0]['mark_price'])
         print(f"Futures price for {asset_symbol}: last {last} and index price {index_price} and mark price {mark_price}")
         asset_y = "ADAUSDT"
         asset_x = "XRPUSDT"
         volume_y,volume_x = calculate_volume(asset_y,asset_x,hedge_ratio)
         print(f"Calculated volume for {asset_y}: {volume_y} and for {asset_x}: {volume_x}")
         time.sleep(10)

    while False:

            z_score,correlation,hedge_ratio,symbol_y,symbol_x = verify_pairs()
            print(f"Z score is {z_score} correlation is {correlation} hedge ratio is {hedge_ratio} for symbols {symbol_y} and {symbol_x}")
            if symbol_y is not None and symbol_x is not None:
                if correlation < 0.0 and hedge_ratio > 1.0:
                    volume_x = asset_volume*hedge_ratio
                    volume_y = 2*asset_volume - volume_x 
                    print(f"Volume {volume_y} for symbol {symbol_y} and volume {volume_x} for symbol {symbol_x}")
                    if z_score > 0.0:
                        send_order_futures_market(symbol_y, "sell", volume_y)
                        send_order_futures_market(symbol_x, "sell", volume_x)
                    elif z_score < 0.0:
                        send_order_futures_market(symbol_y, "buy", volume_y)
                        send_order_futures_market(symbol_x, "buy", volume_x)
                elif correlation > 0.0 and hedge_ratio < 1.0:
                    volume_y = asset_volume
                    volume_x = volume_y*hedge_ratio
                    print(f"Volume {volume_y} for symbol {symbol_y} and volume {volume_x} for symbol {symbol_x}")
                    if z_score > 0.0:
                        print(f"Placing orders for {symbol_y} and {symbol_x} with hedge ratio {hedge_ratio}")
                        resp_y = send_order_futures_market(symbol_y, "sell", volume_y)
                        resp_x = send_order_futures_market(symbol_x, "buy", volume_x)
                        print(f"Response for {symbol_y}: {resp_y.json()}")
                        print(f"Response for {symbol_x}: {resp_x.json()}")
                    elif z_score < 0.0: 
                        send_order_futures_market(symbol_y, "buy", volume_y)
                        send_order_futures_market(symbol_x, "sell", volume_x)
 



            #futures_order_response = put_buy_futures_market("DOGEUSDT", 50).json()
            #print(f"Response spot {futures_order_response['data']}")
            unrealized_pnl = 0.0
            positions = get_open_positions().json()
            total_positions = len(positions['data'])
            print(f"Open positions: {len(positions['data'])}")
            if total_positions > 0:
                unrealized_pnl = float(positions['data'][0]['unrealized_pnl'])

            if unrealized_pnl > 0.003:
                print("Profit detected, exiting...")
                close_response = close_futures_position("DOGEUSDT").json()
                print(f"Close position response: {close_response['data']}")
            time.sleep(5)
            #clear_console()'


if __name__ == "__main__":
    run_code()