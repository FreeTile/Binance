import pandas as pd
from binance.client import Client

lines = []
variables = {}
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[1:5])
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[7:10])

for line in lines:
    key, value = line.strip().split(' = ')
    variables[key.strip()] = value.strip()

api_key = variables['api_key']
api_secret = variables['api_secret']

client = Client(api_key, api_secret)
bars = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, variables['date'])

data = pd.DataFrame(bars,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                             'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                             'ignore'])
data = data.drop(['volume', 'close_time', 'quote_asset_volume',
                             'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                             'ignore'], axis=1)
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

numeric_columns = ['open', 'high', 'low', 'close']
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
heads=0
legs=0
for i in range(len(data)):
    if data.iloc[i]['open'] > data.iloc[i]['close']:
        heads += data.iloc[i]['high'] - data.iloc[i]['open']
        legs += data.iloc[i]['close'] - data.iloc[i]['low']
    else:
        heads += data.iloc[i]['high'] - data.iloc[i]['close']
        legs += data.iloc[i]['open'] - data.iloc[i]['low']

print('Среднее головы свечи: ', heads/len(data))
print('Среднее ноги свечи: ', legs/len(data))