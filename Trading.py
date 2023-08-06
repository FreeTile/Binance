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
    lines.extend(file.readlines()[1:3])
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[12:13])

for line in lines:
    key, value = line.strip().split(' = ')
    variables[key.strip()] = value.strip()

api_key = variables['api_key']
api_secret = variables['api_secret']
model = tf.keras.models.load_model(variables['model'])
# Создайте экземпляр клиента Binance
client = Client(api_key, api_secret)


# Функция для получения последних 21 свечей
def get_recent_candles():
    candles = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_5MINUTE, limit=21)
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

    block_size = 20

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
    open_orders = client.get_open_orders()
    if len(open_orders) > 0:
        for order in open_orders:
            print("Символ:", order["symbol"])
            print("Тип ордера:", order["side"])
            print("Количество:", order["origQty"])
            print("Цена:", order["price"])
            print("--------------------")
    else:
        print("Нет открытых ордеров")

def close_all_orders():
    open_orders = client.get_open_orders()
    for order in open_orders:
        symbol = order['symbol']
        order_id = order['orderId']
        result = client.cancel_order(symbol=symbol, orderId=order_id)
        if result['status'] == "CANCELED":
            print(f"Ордер {order_id} на символ {symbol} успешно закрыт.")

def clear_console():
    # Очищаем консоль в зависимости от операционной системы
    os.system('cls' if os.name == 'nt' else 'clear')

sell_amount = 0
buy_amount = 0

def stop_loss_order(side, stop_price, amount):
    if side == "buy":
        order = client.create_order(
            symbol='BTCUSDT',
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=amount,
            stopPrice=stop_price,
            price=stop_price + 5,
        )
    elif side == "sell":
        order = client.create_order(
            symbol='BTCUSDT',
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_STOP_LOSS_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=amount,
            stopPrice=stop_price,
            price=stop_price - 5,
        )

def buy(amount):
    print("Покупка на сумму ", amount)
    order = client.order_market_buy(
        symbol='BTCUSDT',
        quoteOrderQty=amount,
    )

def sell(amount, info):
    print("Продажа на сумму ", round(amount * float(info["price"]),2))
    order = client.order_market_sell(
        symbol='BTCUSDT',
        quantity=amount,
    )

def run_script():
    global sell_amount, buy_amount
    while True:
        print("Текущее время:", datetime.datetime.now())
        close_all_orders()
        candles = get_recent_candles()
        prediction = process_candles(candles)
        clear_console()

        BTC_info = client.get_asset_balance(asset='BTC')
        USDT_info = client.get_asset_balance(asset='USDT')
        BTC = round(float(BTC_info["free"]), 5)
        USDT = round(float(USDT_info["free"]), 2)
        print('BTC balance: ', BTC)
        print('USDT balance: ', USDT)

        info = client.get_avg_price(symbol='BTCUSDT')
        print("BTC price: ", info["price"])

        balance = round(float(USDT_info["free"]) + float(BTC_info["free"]) * float(info["price"]), 2)
        print("Примерный баланс в USDT", balance)

        with open("profit.txt", 'a') as file:
            file.write(str(balance) + '\n')

        if np.array_equal(prediction[0], [1, 0]):
            print("Предсказание: цена поднимается")
            if sell_amount > 0 and (sell_amount * float(info["price"])) > USDT+0.5:
                sell_amount = round(sell_amount * float(info["price"]), 2)
                buy(sell_amount)
            sell_amount = 0
            amount = round(max(USDT / 10, 10.1), 2)
            if USDT > 10.15:
                buy(amount)
                stop_loss_order("sell", round(float(info["price"]) - 20, 2), round(amount / float(info["price"]), 5))
                buy_amount += amount
        else:
            print("Предсказание: цена упадёт")
            if buy_amount > 0 and buy_amount > (BTC * float(info["price"]) + 0.5):
                buy_amount = round(buy_amount / float(info["price"]), 5)
                sell(buy_amount, info)
            buy_amount = 0
            amount = round(max(BTC / 10, 10.1 / float(info["price"])), 5)
            if BTC > (10.15 / float(info["price"])):
                sell(amount, info)
                stop_loss_order("buy", round(float(info["price"]) + 20, 2), amount)
                sell_amount += amount
        orders()

        current_time = int(time.time())
        next_run_time = ((current_time // 300) + 1) * 300  # Округление до следующего времени, кратного 5 минутам
        sleep_time = next_run_time - current_time
        time.sleep(sleep_time)

# Запуск скрипта
run_script()
