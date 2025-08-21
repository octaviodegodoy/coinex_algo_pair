# -*- coding: utf-8 -*-
import hashlib
import json
import time
import hmac
import os
from urllib.parse import urlparse, urlencode

import requests

access_id = ""  # Replace with your access id
secret_key = ""  # Replace with your secret key

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
        prepared_str = f"{method}{request_path}{body}{timestamp}"
        signature = hmac.new(
            bytes(self.secret_key, 'latin-1'), 
            msg=bytes(prepared_str, 'latin-1'), 
            digestmod=hashlib.sha256
        ).hexdigest().lower()
        return signature

    def get_common_headers(self, signed_str, timestamp):
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

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_spot_market():
    request_path = "/spot/market"
    params = {"market": "BTCUSDT"}
    response = request_client.request(
        "GET",
        "{url}{request_path}".format(url=request_client.url, request_path=request_path),
        params=params,
    )
    return response


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


def put_buy_futures_market(market, volume):
    request_path = "/futures/order"
    data = {
        "market": market,
        "market_type": "FUTURES",
        "side": "buy",
        "type": "market",
        "amount": volume,
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
    try:
        while True:            
            
            futures_response = get_futures_balance().json()
            if len(futures_response['data']) > 0:
                print(f"Futures unrealized pnl {futures_response['data'][0]['unrealized_pnl']}")
            else:
                print("No futures data available")

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
            elif unrealized_pnl < -0.50 and total_positions < 2:
                futures_order_response = put_buy_futures_market("DOGEUSDT", 50).json()
                print(f"Opened new position: {futures_order_response['data']}")
            time.sleep(5)
            #clear_console()

    except Exception as e:
        print("Error:" + str(e))
        time.sleep(3)
        run_code()


if __name__ == "__main__":
    run_code()