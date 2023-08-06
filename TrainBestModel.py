# Импортирование необходимых библиотек
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client
from sklearn.preprocessing import StandardScaler

lines = []
variables = {}
epochs = 25
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[15:17])
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[1:5])

for line in lines:
    key, value = line.strip().split(' = ')
    variables[key.strip()] = value.strip()
block_size = variables['block_size']
api_key = variables['api_key']
api_secret = variables['api_secret']
block_size = int(variables['block_size'])

# Получение доступа к API биржи Binance
train_data = np.load('Data/train_data.npy')
train_labels = np.load('Data/train_labels.npy')

def data_generator(train_data, train_labels, batch_size):
    num_samples = train_data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # Перемешиваем индексы

    while True:
        for start_index in range(0, num_samples, batch_size):
            end_index = min(start_index + batch_size, num_samples)
            batch_indices = indices[start_index:end_index]
            yield train_data[batch_indices], train_labels[batch_indices]


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
batch_size = best_individual[-1]['batch_size']
train_data_generator = data_generator(train_data, train_labels, batch_size)
model = create_model_from_individual(best_individual)
# Компиляция и обучение модели на тренировочных данных
validation_split = best_individual[-1].get('validation_split', 0.3)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
model.fit(train_data_generator, batch_size=batch_size, epochs=epochs, steps_per_epoch=len(train_data) // batch_size,
          validation_steps=len(train_data) // batch_size * validation_split)
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
