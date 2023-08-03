import time
import tensorflow as tf
import numpy as np
import pandas as pd
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from binance.enums import ORDER_TYPE_STOP_LOSS_LIMIT
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


def clear_console():
    # Очищаем консоль в зависимости от операционной системы
    os.system('cls' if os.name == 'nt' else 'clear')


sell_amount = 0
buy_amount = 0


# Функция для запуска скрипта каждые 5 минут
def run_script():
    global sell_amount, buy_amount
    return_balance()
    while True:
        candles = get_recent_candles()
        prediction = process_candles(candles)
        clear_console()
        BTC = client.get_asset_balance(asset='BTC')
        USDT = client.get_asset_balance(asset='USDT')
        print('BTC balance: ', BTC["free"])
        print('USDT balance: ', USDT["free"])
        info = client.get_avg_price(symbol='BTCUSDT')
        orders()
        print("BTC price: ", info["price"])
        if np.array_equal(prediction[0], [1, 0]):
            buy(float(USDT["free"]), info)
        else:
            sell(float(BTC["free"]), info)
        current_time = int(time.time())
        next_run_time = ((current_time // 300) + 1) * 300  # Округление до следующего времени, кратного 5 минутам
        sleep_time = next_run_time - current_time
        time.sleep(sleep_time)


def return_balance():
    BTC = client.get_asset_balance(asset='BTC')
    if float(BTC["free"]) < 1:
        amount = 1 - float(BTC["free"])
        amount = round(amount, 6)
        order = client.order_market_buy(
            symbol='BTCUSDT',
            quantity=amount
        )
    elif float(BTC["free"]) > 1:
        amount = float(BTC["free"]) - 1
        amount = round(amount, 6)
        order = client.order_market_sell(
            symbol='BTCUSDT',
            quantity=amount
        )


def buy(balance_USDT, info):
    global sell_amount, buy_amount
    if sell_amount > 0:
        sell_amount = round(sell_amount, 6)
        order = client.order_market_buy(
            symbol='BTCUSDT',
            quantity=sell_amount
        )
        sell_amount = 0
    # Расчет суммы покупки (1% от баланса USDT)
    amount = balance_USDT * 0.1 * 0.5
    amount = max(amount, 11)
    amount = round(amount, 6)
    buy_amount += amount
    stop_loss_price = round((float(info["price"])-10.5), 6) # Рассчитываем цену Stop Loss
    # Отправка рыночной заявки на покупку
    order = client.order_market_buy(
        symbol='BTCUSDT',
        quoteOrderQty=amount,
    )
    order = client.create_order(
        symbol='BTCUSDT',
        side=Client.SIDE_SELL,
        type=Client.ORDER_TYPE_STOP_LOSS,
        quoteOrderQty=amount,
        stopPrice = stop_loss_price
    )


def sell(balance_BTC, info):
    global sell_amount, buy_amount
    if buy_amount > 0:
        buy_amount = round(buy_amount, 6)
        order = client.order_market_sell(
            symbol='BTCUSDT',
            quoteOrderQty=buy_amount
        )
        buy_amount = 0
    # Расчет суммы продажи (1% от баланса BTC)
    amount = balance_BTC * 0.1 * 0.8
    amount = max((11 / float(info["price"])), amount)
    amount = round(amount, 6)
    sell_amount += amount
    stop_loss_price = round((float(info["price"])+10), 6)
    # Отправка рыночной заявки на продажу
    order = client.order_market_sell(
        symbol='BTCUSDT',
        quantity=amount,
    )
    order = client.create_order(
        symbol='BTCUSDT',
        side=Client.SIDE_BUY,
        type=Client.ORDER_TYPE_STOP_LOSS,
        quantity=amount,
        stopPrice=stop_loss_price
    )


# Запуск скрипта
run_script()
