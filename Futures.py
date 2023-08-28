import time
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from binance import Client
from sklearn.preprocessing import StandardScaler
import os

lines = []
variables = {}
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[0:19])


for line in lines:
    if '=' in line:
        key, value = line.strip().split(' = ')
        variables[key.strip()] = value.strip()

api_key = variables['api_key']
api_secret = variables['api_secret']
model = tf.keras.models.load_model(variables["model_path"])
# Создайте экземпляр клиента Binance
client = Client(api_key, api_secret)

# Функция для получения последних 21 свечей
def get_recent_candles():
    candles = client.get_klines(symbol=f'{variables["coin1"]}{variables["coin2"]}', interval=eval(f'Client.KLINE_INTERVAL_{variables["clines_time"]}'), limit=(int(variables["block_size"])) +1)
    closed_candles = candles[:-1]
    return closed_candles


# Функция для обработки свечей и предсказания
def process_candles(data):
    data = pd.DataFrame(data,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                 'quote_asset_volume',
                                 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)

    # Проверяем и преобразуем столбцы с числовыми значениями
    numeric_columns = ['open', 'high', 'low', 'close']
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # DX (Directional Movement Index)
    data['plus_dm'] = data['high'].diff()
    data['minus_dm'] = data['low'].diff()
    data['plus_dm'] = data['plus_dm'].fillna(0)  # Заменяем пропущенные значения нулями
    data['minus_dm'] = data['minus_dm'].fillna(0)  # Заменяем пропущенные значения нулями
    data['plus_dm'] = data['plus_dm'].where(data['plus_dm'] > 0, 0)  # Заменяем отрицательные значения нулями
    data['minus_dm'] = abs(data['minus_dm'].where(data['minus_dm'] < 0, 0))  # Заменяем положительные значения нулями
    data['tr'] = data[['high', 'low']].diff(axis=1).max(axis=1)
    data['plus_di'] = 100 * (data['plus_dm'].rolling(window=14).sum() / data['tr'].rolling(window=14).sum())
    data['minus_di'] = 100 * (data['minus_dm'].rolling(window=14).sum() / data['tr'].rolling(window=14).sum())
    data['dx'] = 100 * (abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di']))

    # MOM (Momentum)
    data['mom'] = data['close'].pct_change(periods=10) * 100

    # Функция для вычисления RSI
    def compute_rsi(prices, window=14):
        deltas = np.diff(prices)
        seed = deltas[:window + 1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100. / (1. + rs)

        for i in range(window, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    # Функция для вычисления Bollinger Bands
    def compute_bollinger_bands(prices, window=20, num_std=2):
        rolling_mean = np.convolve(prices, np.ones(window) / window, mode='valid')
        rolling_std = np.std(prices, axis=0)

        upper_band = rolling_mean + (num_std * rolling_std)
        lower_band = rolling_mean - (num_std * rolling_std)

        return upper_band, lower_band

    # Получение массива с индикатором RSI
    rsi = compute_rsi(data['close'].values)
    upper_band, lower_band = compute_bollinger_bands(data['close'].values)
    # Конвертация в DataFrame
    rsi_df = pd.DataFrame(rsi, columns=['rsi'])
    bollinger_bands_df = pd.DataFrame({'upper_band': upper_band, 'lower_band': lower_band})

    # Добавление столбца с RSI в DataFrame data
    data = pd.concat([data, rsi_df], axis=1)
    data = pd.concat([data, bollinger_bands_df], axis=1)

    # CCI (Commodity Channel Index)
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    mean_price = typical_price.rolling(window=20).mean()
    mean_deviation = abs(typical_price - mean_price).rolling(window=20).mean()
    data['cci'] = (typical_price - mean_price) / (mean_deviation * 0.015)

    block_size = int(variables["block_size"])

    # Стандартизируем данные

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data[np.isnan(scaled_data)] = 0

    train_data_blocks = []
    train_data_block = scaled_data[:block_size]
    train_data_blocks.append(train_data_block)

    train_data = np.array(train_data_blocks)
    train_data[np.isnan(train_data)] = 0
    # Получение предсказаний от модели
    predictions = model.predict(train_data)

    # Вывод реальных значений и предсказаний
    for j in range(len(predictions)):
        max = np.argmax(predictions[j])
        predictions[j] = np.zeros_like(predictions[j])
        predictions[j][max] = 1
    return predictions


def orders():
    open_orders = client.futures_get_open_orders()
    if len(open_orders) > 0:
        for order in open_orders:
            print('--------------------')
            print("Символ:", order["symbol"])
            print("Тип ордера:", order["side"])
            print("Количество:", order["origQty"])
            print("Цена:", order["price"])
            print("--------------------")
    else:
        print("Нет открытых ордеров")

def create_position(side, quantity, stop_loss_price):
    if side == "long":
        order = client.futures_create_order(
            symbol=f'{variables["coin1"].upper()}{variables["coin2"].upper()}',
            side=Client.SIDE_BUY,
            type=Client.FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
        )
        stop_loss_order("sell", stop_loss_price, quantity)
    elif side == "short":
        order = client.futures_create_order(
            symbol=f'{variables["coin1"].upper()}{variables["coin2"].upper()}',
            side=Client.SIDE_SELL,
            type=Client.FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
        )
        stop_loss_order("buy", stop_loss_price, quantity)

def stop_loss_order(side, stop_price, amount):
    if side == "buy":
        order = client.futures_create_order(
            symbol=f'{variables["coin1"].upper()}{variables["coin2"].upper()}',
            side=Client.SIDE_BUY,
            type=Client.FUTURE_ORDER_TYPE_STOP_MARKET,
            quantity=amount,
            stopPrice=stop_price
        )
    elif side == "sell":
        order = client.futures_create_order(
            symbol=f'{variables["coin1"].upper()}{variables["coin2"].upper()}',
            side=Client.SIDE_SELL,
            type=Client.FUTURE_ORDER_TYPE_STOP_MARKET,
            quantity=amount,
            stopPrice=stop_price
        )

def close_positions():
    symbol = f'{variables["coin1"].upper()}{variables["coin2"].upper()}'

    position_info = client.futures_position_information(symbol=symbol)

    for position in position_info:
        if float(position["positionAmt"]) != 0:
            side = Client.SIDE_SELL if float(position["positionAmt"]) > 0 else Client.SIDE_BUY
            quantity = abs(float(position["positionAmt"]))
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type=Client.FUTURE_ORDER_TYPE_MARKET,
                quantity=quantity
            )

def run_script():
    firsttime = 1
    while True:
        print("Текущее время:", datetime.datetime.now())
        close = client.futures_cancel_all_open_orders(symbol=variables["coin1"] + variables["coin2"])
        close_positions()
        candles = get_recent_candles()
        prediction = process_candles(candles)

        first_coin_balance_info = next((item for item in client.futures_account_balance() if item["asset"] == variables["coin2"]), None)
        second_coin = round(float(first_coin_balance_info["balance"]), 2)
        info = client.futures_symbol_ticker(symbol=f'{variables["coin1"].upper()}{variables["coin2"].upper()}')
        print(info)
        print(f'{variables["coin2"]} balance: ', second_coin)
        print(f'{variables["coin1"]} price: ', info["price"])
        if firsttime == 0:
            if (predictions == "up" and float(previous_price) < float(info["price"])) or (predictions == "down" and float(previous_price) > float(info["price"])):
                with open("profit.txt", 'a') as file:
                    file.write(f"Prediction was right. Balance now {second_coin}" + '\n')
            else:
                with open("profit.txt", 'a') as file:
                    file.write(f"Prediction was wrong. Balance now {second_coin}" + '\n')

        previous_price = info["price"]

        amount = round((second_coin / float(info["price"])) * 2,0)

        if np.array_equal(prediction[0], [1, 0]):
            print("Предсказание: цена поднимается")
            create_position("long", amount, round(float(info["price"]) - round(float(variables["average_down_shadow"]), 2), 2))
            predictions = "up"
        else:
            print("Предсказание: цена упадёт")
            create_position("short",  amount, round(float(info["price"]) + float(variables["average_upper_shadow"]), 5))
            predictions = "down"
        orders()

        current_time = int(time.time())
        time_for_cycle_in_minutes = int(variables["time_for_cycle_in_minutes"])

        next_run_time = ((current_time // (60 * time_for_cycle_in_minutes)) + 1) * (60 * time_for_cycle_in_minutes)
        rounded_time = next_run_time - (next_run_time % (60 * time_for_cycle_in_minutes))
        sleep_time = rounded_time - current_time

        if sleep_time > 0:
            time.sleep(sleep_time)
        firsttime = 0
# Запуск скрипта
run_script()

print(' ')
print("---------------------------------------------------------------------------------------------------------------")
print(' ')
