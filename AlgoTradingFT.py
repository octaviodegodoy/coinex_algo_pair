import time, hmac, hashlib, requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_URL = "https://api.coinex.com"
API_KEY = "key"
SECRET = "secret"

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

def get_futures_daily_closes(market: str, days: int = 60):
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

    price1 = np.array(df['close'])
    dates = np.array(df['time'])

    price_data = pd.DataFrame({'Price1': price1}, index=dates)
    price_data['Price1'] = pd.to_numeric(price_data['Price1'])
    price_data['Return1'] = price_data['Price1'].pct_change().cumsum()

    print(f"Prices {price_data['Price1']}")
    price_data['Return1'] = price_data['Price1']

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(price_data.index, price_data['Price1'], marker="o", linestyle="-")
    plt.title("Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    
    
    closes = []
    for candle in data["data"]:
        ts = candle['created_at']
        closes.append((ts, candle['close']))
        #cleprint(f"Close price is {closes}")
    return closes

def main():
    assets = ["BTCUSDT", "ETHUSDT"]
    get_futures_daily_closes("BTCUSDT",days=60)


    #for symbol in assets:
        #closes = get_futures_daily_closes(symbol, days=60)
        #data = get_futures_ticker(symbol)
        #print(f"{symbol}: {data['data'][0]['close']} and {closes}")

if __name__ == "__main__":
    main()