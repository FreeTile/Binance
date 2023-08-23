from binance import Client

print("Получаю пары с Binance")

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

# Создание экземпляра клиента Binance
client = Client(api_key, api_secret)

info = client.get_all_tickers()

pairs = [pair['symbol'] for pair in info]

with open('pairs.txt', 'w') as file:
    file.write('\n'.join(pairs))

print('Пары для торговли успешно записаны в файл pairs.txt')

print(' ')
print("---------------------------------------------------------------------------------------------------------------")
print(' ')