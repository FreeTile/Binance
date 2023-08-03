    # Импортирование необходимых библиотек
import pickle
import numpy as np
import tensorflow as tf
from binance.client import Client

lines = []
variables = {}
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[15:17])

for line in lines:
    key, value = line.strip().split(' = ')
    variables[key.strip()] = value.strip()

api_key = variables['api_key']
api_secret = variables['api_secret']
block_size = int(variables['block_size'])

# Получение доступа к API биржи Binance
client = Client(api_key, api_secret)
train_data = np.load('Data/train_data.npy')
train_labels = np.load('Data/train_labels.npy')

population = 2


# Создание модели на основе индивидуума
def create_model_from_individual(individual):
    inputs = tf.keras.Input(shape=(20, 22))
    print(individual)
    if isinstance(individual[0], tf.keras.layers.Dense):
        x = tf.keras.layers.Flatten()(inputs)  # Преобразование в одномерный формат
        x = tf.keras.layers.Dense(units=20 * 22)(x)  # Преобразование размерности
        x = tf.keras.layers.Reshape((20, 22))(x)  # Изменение размерности перед LSTM
    else:
        x = inputs

    for i, layer in enumerate(individual[:-1]):
        x = layer(x)

    x = tf.keras.layers.Flatten()(x)  # Преобразование в одномерный формат

    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


with open(variables['from'], 'rb') as f:
    best_individual = pickle.load(f)
    best_individual = best_individual[0]
model = create_model_from_individual(best_individual)
# Компиляция и обучение модели на тренировочных данных
batch_size = best_individual[-1]['batch_size']
epochs = best_individual[-1]['epochs']
validation_split = best_individual[-1].get('validation_split', 0.3)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
model.save(variables['save'])

model = tf.keras.models.load_model(variables['save'])
up_count = 0
down_count = 0


for j in range(len(train_labels)):
    if np.array_equal(train_labels[j], [1, 0]):
        up_count += 1
    elif np.array_equal(train_labels[j], [0, 1]):
        down_count += 1


up = 0
down = 0

# Получение предсказаний от модели
predictions = model.predict(train_data)
original_predictions = np.copy(predictions)

# Вывод реальных значений и предсказаний
for j in range(len(predictions)):
    max = np.argmax(predictions[j])
    predictions[j] = np.zeros_like(predictions[j])
    predictions[j][max] = 1

# Подсчет количества правильных предсказаний
for j in range(len(train_labels)):
    if np.array_equal(train_labels[j], predictions[j]):
        if train_labels[j][0] == 1:
            up += 1
        elif train_labels[j][1] == 1:
            down += 1

# Вычисление взвешенного среднего
total_count = len(train_labels)
accuracy_up = up / up_count if up_count != 0 else 0
accuracy_down = down / down_count if down_count != 0 else 0
weighted_average = (up_count / total_count) * accuracy_up + (
            down_count / total_count) * accuracy_down

print(f"Количество правильных предсказаний подъёма%: {up}/{up_count}")
print(f"Точность предсказания повышения: {accuracy_up * 100}%")
print(f"Количество правильных предсказаний падения: {down}/{down_count}")
print(f"Точность предсказания понижения: {accuracy_down * 100}%")
print(f"Средняя точность предсказания: {weighted_average * 100}%")
