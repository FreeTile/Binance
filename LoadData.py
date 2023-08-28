# Импортирование необходимых библиотек
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from binance.client import Client

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
block_size = int(variables['block_size'])


# Получение доступа к API биржи Binance
client = Client(api_key, api_secret)
bars = client.get_historical_klines(f'{variables["coin1"]}{variables["coin2"]}', eval(f'Client.KLINE_INTERVAL_{variables["clines_time"]}'), f"{variables['date']} days ago UTC")

# Подготовка данных для обучения нейросети
data = pd.DataFrame(bars,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
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

train_labels = []
for i in range(len(data) - block_size):
    prev_close = data.iloc[i + block_size - 1]['close']
    curr_close = data.iloc[i + block_size]['close']

    if prev_close < curr_close:
        train_label = [1, 0]
    else:
        train_label = [0, 1]

    train_labels.append(train_label)

train_labels = np.array(train_labels)

# Стандартизируем данные
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


train_data_blocks = []
for i in range(len(data) - block_size):
    train_data_block = scaled_data[i:i + block_size]
    train_data_blocks.append(train_data_block)

train_data = np.array(train_data_blocks)
train_data[np.isnan(train_data)] = 0

np.save(f'Data/train_data_{variables["coin1"]}{variables["coin2"]}_{variables["clines_time"]}.npy', train_data)
np.save(f'Data/train_labels_{variables["coin1"]}{variables["coin2"]}_{variables["clines_time"]}.npy', train_labels)

print(f"Датасет за последние {variables['date']} дней успешно загружен и обработан")

print(' ')
print("---------------------------------------------------------------------------------------------------------------")
print(' ')

