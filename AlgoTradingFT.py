import time, hmac, hashlib, requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

API_KEY = "key"
SECRET = "secret"

BASE_URL = "https://api.coinex.com"
with open('C:/Users/degod/Dev/python-coinex/key_data.txt', 'r') as file:
         api_key = file.read().strip()
         secret = file.read().strip()
         print(f" Key is {api_key} and secret is {secret}")


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

def main():
    data_1 = get_futures_data("BTCUSDT",days=252)
    data_2 = get_futures_data("ETHUSDT",days=252)
    

    price1 = np.array(data_1['close'].astype(float))
    price2 = np.array(data_2['close'].astype(float))
    dates = np.array(data_1['time'])
    #print(f"Type of price {type(price1)}")  

    price_data = pd.DataFrame({'Price1': price1,'Price2': price2}, index=dates)
    #price_data.to_csv("btc_price_data.csv")
    #price_data['Price1'] = pd.to_numeric(price_data['Price1'])
    price_data['Return1'] = price_data['Price1'].pct_change().cumsum()
    price_data['Return2'] = price_data['Price2'].pct_change().cumsum()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(price_data.index, price_data['Return1'], label='Cumulative returns BTCUSDT', color='red')
    plt.plot(price_data.index, price_data['Return2'], label='Cumulative returns ETHUSDT', color='blue')
    plt.title("Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    main()