# Импортирование необходимых библиотек
import random

from keras.src.layers import Attention
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

lines = []
variables = {}
with open('config.txt', 'r') as file:
    lines.extend(file.readlines()[0:19])

for line in lines:
    if '=' in line:
        key, value = line.strip().split(' = ')
        variables[key.strip()] = value.strip()

population_size = int(variables['population_size'])
num_generations = int(variables['num_generations'])
mutation_rate = round(float(variables['mutation_rate']), 1)
variables["block_size"] = int(variables["block_size"])
train_data = np.load(f'Data/train_data_{variables["coin1"]}{variables["coin2"]}_{variables["clines_time"]}.npy')
train_labels = np.load(f'Data/train_labels_{variables["coin1"]}{variables["coin2"]}_{variables["clines_time"]}.npy')


# Генетический алгоритм
# Гиперпараметры генетического алгоритма

# Создание начальной популяции
def generate_population(population_size):
    population = []
    for _ in range(population_size):
        individual = generate_individual()
        population.append(individual)
    return population


# Генерация случайного индивидуума
def generate_individual():
    individual = []

    num_layers = random.randint(5, 15)

    for index in range(num_layers):
        if index == 0 and random.random() < 0.5:
            attention_units = random.randint(16, 256)
            attention_layer = Attention(units=attention_units)
            individual.append(attention_layer)

        layer_type = random.choice(['dense', 'lstm'])

        if layer_type == 'dense':
            units = random.randint(16, 256)
            activation = random.choice(['relu', 'sigmoid', 'tanh'])
            layer = Dense(units=units, activation=activation)
            individual.append(layer)

            if random.random() < 0.5:
                dropout_rate = random.uniform(0.1, 0.5)
                dropout_layer = Dropout(rate=dropout_rate)
                individual.append(dropout_layer)

        elif layer_type == 'lstm':
            units = random.randint(16, 256)
            activation = random.choice(['tanh', 'sigmoid', 'relu'])
            return_sequences = True
            layer = LSTM(units=units, activation=activation, return_sequences=return_sequences)
            individual.append(layer)

            if random.random() < 0.5:
                dropout_rate = random.uniform(0.1, 0.5)
                dropout_layer = Dropout(rate=dropout_rate)
                individual.append(dropout_layer)

    batch_size = random.randint(1024, 8192)
    validation_split = random.uniform(0.2, 0.5)
    epochs = 20

    set_lstm_return_sequences(individual)  # Перемещение перед выводом слоев

    individual.append({'batch_size': batch_size, 'validation_split': validation_split, 'epochs': epochs})

    return individual


# Оценка приспособленности популяции
def evaluate_population(population):
    fitness_scores = []
    for individual in population:
        model = create_model_from_individual(individual)
        # Компиляция и обучение модели на тренировочных данных
        batch_size = individual[-1]['batch_size']
        epochs = individual[-1]['epochs']
        validation_split = individual[-1].get('validation_split', 0.3)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

        fitness = evaluate_model(model, validation_split)
        fitness_scores.append(fitness)
    return fitness_scores


# Создание модели на основе индивидуума
def create_model_from_individual(individual):
    inputs = tf.keras.Input(shape=(variables["block_size"], 22))

    if isinstance(individual[0], tf.keras.layers.Dense):
        x = tf.keras.layers.Flatten()(inputs)  # Преобразование в одномерный формат
        x = tf.keras.layers.Dense(units=variables["block_size"] * 22)(x)  # Преобразование размерности
        x = tf.keras.layers.Reshape((variables["block_size"], 22))(x)  # Изменение размерности перед LSTM
    else:
        x = inputs

    for i, layer in enumerate(individual[:-1]):
        x = layer(x)

    x = tf.keras.layers.Flatten()(x)  # Преобразование в одномерный формат

    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# Оценка модели (приспособленность индивидуума)
def evaluate_model(model, validation_split):
    # Разделение данных на обучающую и валидационную выборки
    train_data_1, val_data, train_labels_1, val_labels = train_test_split(train_data, train_labels,
                                                                          test_size=validation_split,
                                                                          random_state=random.randint(1, 100))

    train_loss, train_accuracy = model.evaluate(train_data_1, train_labels_1, verbose=0)
    val_loss, val_accuracy = model.evaluate(val_data, val_labels, verbose=0)

    # Разница между точностью и потерями
    diff_accuracy = 1 - (train_loss + val_loss) / 2

    if diff_accuracy < 0:
        return 0.01
    else:
        return diff_accuracy


# Выбор родителей для скрещивания
def select_parents(population, fitness_scores):
    parents = []
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]

    for _ in range(len(population) // 2):
        parent1_idx = roulette_wheel_selection(probabilities)

        # Поиск второго родителя с ненулевой вероятностью
        parent2_idx = None
        while parent2_idx is None or parent2_idx == parent1_idx:
            parent2_idx = roulette_wheel_selection(probabilities)

        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        parents.append((parent1, parent2))

        probabilities[parent1_idx] = 0.0
        probabilities[parent2_idx] = 0.0
    print('select_parents')
    return parents


def crossover(parents, population_size):
    offspring = []

    for parent1, parent2 in parents:
        child1 = crossover_individuals(parent1, parent2)
        child2 = crossover_individuals(parent2, parent1)

        offspring.append(child1)
        offspring.append(child2)

    if len(offspring) > population_size:
        offspring = offspring[:population_size]
    print('crossover')
    return offspring


def mutate(offspring, mutation_rate):
    mutated_offspring = []

    for individual in offspring:
        mutated_individual = mutate_individual(individual, mutation_rate)

        print('Layers for mutated individual:')
        for layer in mutated_individual:
            print(layer)
        print('-----------------------------')

        mutated_offspring.append(mutated_individual)

    print('mutate')
    return mutated_offspring


def get_best_individual(population, fitness_scores):
    best_idx = np.argmax(fitness_scores)
    best_individual = population[best_idx]
    print('get_best_individual')
    return best_individual


def roulette_wheel_selection(probabilities):
    r = random.random()
    cumulative_prob = 0.0

    for i, prob in enumerate(probabilities):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return i

    print('roulette_wheel_selection')
    # Если не был выбран ни один родитель, вернуть случайный индекс
    return random.randint(0, len(probabilities) - 1)


def crossover_individuals(parent1, parent2):
    min_length = min(len(parent1), len(parent2))
    crossover_point = random.randint(0, min_length - 1)

    child = parent1[:crossover_point] + parent2[crossover_point:]
    set_lstm_return_sequences(child)
    print('crossover_individuals')
    return child


def generate_random_layer():
    layer_type = random.choice(['dense', 'lstm'])

    if layer_type == 'dense':
        units = random.randint(32, 256)
        activation = random.choice(['relu', 'sigmoid', 'tanh'])
        layer = Dense(units=units, activation=activation)
    elif layer_type == 'lstm':
        units = random.randint(32, 256)
        activation = random.choice(['tanh', 'sigmoid', 'relu'])
        return_sequences = True
        layer = LSTM(units=units, activation=activation, return_sequences=return_sequences)

    return layer


def set_lstm_return_sequences(individual):
    lstm_indices = [i for i, layer in enumerate(individual) if isinstance(layer, LSTM)]
    if len(lstm_indices) <= 0:
        return individual

    for i, layer_index in enumerate(lstm_indices):
        if i == len(lstm_indices) - 1:
            individual[layer_index].return_sequences = False
        else:
            individual[layer_index].return_sequences = True

    return individual


def mutate_individual(individual, mutation_rate):
    mutated_individual = individual.copy()

    batch_size_mutated = False
    epochs_mutated = False
    val_mutated = False

    for i, layer in enumerate(mutated_individual):
        if isinstance(layer, dict):
            if random.random() < mutation_rate:
                if 'batch_size' in layer and not batch_size_mutated and not any(
                        isinstance(l, dict) and 'batch_size' in l for l in mutated_individual):
                    new_batch_size = layer['batch_size'] + random.randint(-64, 64)
                    new_batch_size = max(64, min(1024, new_batch_size))
                    layer['batch_size'] = new_batch_size
                    batch_size_mutated = True

                elif 'epochs' in layer and not epochs_mutated and not any(isinstance(l, dict) and 'epochs' in l for l in
                                                                          mutated_individual):
                    layer['epochs'] = 15
                    epochs_mutated = True

                elif 'validation_split' in layer and not val_mutated and not any(
                        isinstance(l, dict) and 'validation_split' in l for l in mutated_individual):
                    new_val_split = layer['validation_split'] + random.uniform(-0.1, 0.1)
                    new_val_split = max(0.1, min(0.7, new_val_split))
                    layer['validation_split'] = new_val_split
                    val_mutated = True

    for i, layer in enumerate(mutated_individual):
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                new_layer = generate_random_layer()

                if i < len(mutated_individual) - 1 and not isinstance(mutated_individual[i + 1], Dropout):
                    mutated_individual.insert(i + 1, new_layer)
            elif len(mutated_individual) > 1:
                if isinstance(layer, Dropout):
                    mutated_individual.pop(i)
                elif isinstance(layer, Attention):
                    continue  # Пропускаем удаление слоя внимания
                elif i < len(mutated_individual) - 1 and isinstance(mutated_individual[i + 1], Dropout):
                    mutated_individual.pop(i + 1)
                    if len(mutated_individual) > 0 and isinstance(mutated_individual[-1], Dropout):
                        mutated_individual.pop(i - 1)
                else:
                    mutated_individual.pop(i)

    for i, layer in enumerate(mutated_individual):
        if isinstance(layer, Dense):
            units = random.randint(32, 256)
            activation = random.choice(['relu', 'sigmoid', 'tanh'])
            mutated_layer = Dense(units=units, activation=activation)
        elif isinstance(layer, LSTM):
            units = random.randint(32, 256)
            activation = random.choice(['tanh', 'sigmoid', 'relu'])
            return_sequences = layer.return_sequences
            mutated_layer = LSTM(units=units, activation=activation, return_sequences=return_sequences)
        elif isinstance(layer, Dropout):
            rate = random.uniform(0.1, 0.5)
            mutated_layer = Dropout(rate=rate)
        else:
            mutated_layer = layer

        mutated_individual[i] = mutated_layer

    set_lstm_return_sequences(mutated_individual)
    return mutated_individual


# Генетический алгоритм
def genetic_algorithm(population_size, num_generations, mutation_rate):
    population = [generate_individual() for _ in range(population_size)]
    best_fitness_scores = []
    i = 0;

    for generation in range(num_generations):
        fitness_scores = evaluate_population(population)
        best_individual = get_best_individual(population, fitness_scores)
        best_fitness_scores.append(max(fitness_scores))

        print(f"Generation {generation + 1}, Best Fitness: {max(fitness_scores)}, Best Individual: {best_individual}")
        np.save(f'best_individual_{variables["coin1"]}{variables["coin2"]}_{variables["clines_time"]}.npy', best_individual)
        parents = select_parents(population, fitness_scores)
        offspring = crossover(parents, population_size)
        mutated_offspring = mutate(offspring, mutation_rate)

        population = mutated_offspring  # Обновление списка population

    best_individual = get_best_individual(population, fitness_scores)
    best_fitness_scores.append(max(fitness_scores))

    print(f"Generation {num_generations}, Best Fitness: {max(fitness_scores)}, Best Individual: {best_individual}")

    return best_individual, best_fitness_scores


# Применение генетического алгоритма к модели
best_individual = genetic_algorithm(population_size, num_generations, mutation_rate)

# Сохранение лучших данных
np.save(f'best_individual_{variables["coin1"]}{variables["coin2"]}_{variables["clines_time"]}.npy', best_individual)

print(' ')
print("---------------------------------------------------------------------------------------------------------------")
print(' ')
