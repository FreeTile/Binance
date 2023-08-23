import tkinter as tk
import traceback
from tkinter import ttk, filedialog
from tkinter import messagebox
import subprocess
import threading
from PIL import ImageTk, Image
from binance import Client

client = Client()
def load_dataset():
    load_dataset_button.config(state=tk.DISABLED)  # Делаем кнопку неактивной
    process = subprocess.Popen(["python", "LoadData.py"])
    process.wait()

    load_dataset_button.config(state=tk.NORMAL)  # Включаем кнопку после выполнения скрипта

def train_model():
    train_model_button.config(state=tk.DISABLED)  # Делаем кнопку неактивной
    process = subprocess.Popen(["python", "TrainBestModel.py"])
    process.wait()

    train_model_button.config(state=tk.NORMAL)  # Включаем кнопку после выполнения скрипта

def calculate_averages():
    calculate_averages_button.config(state=tk.DISABLED)  # Делаем кнопку неактивной
    process = subprocess.Popen(["python", "Average_shadows.py"])
    process.wait()

    calculate_averages_button.config(state=tk.NORMAL)  # Включаем кнопку после выполнения скрипт


def open_gen_algorithm_window():
    # Функция для проверки корректности заполнения полей и активации кнопки
    def validate_fields(*args):
        population_size = population_size_entry.get()
        num_generations = num_generations_entry.get()
        mutation_rate = mutation_rate_entry.get()

        if population_size.isnumeric() and num_generations.isnumeric() and 0 <= float(mutation_rate) <= 1:
            gen_algorithm_button.config(state="normal")
        else:
            gen_algorithm_button.config(state="disabled")

    # Функция для сохранения значений полей в файл config.txt
    def save_config():
        config["population_size"] = population_size_entry.get()
        config["num_generations"] = num_generations_entry.get()
        config["mutation_rate"] = mutation_rate_entry.get()

        with open("config.txt", "w") as file:
            for key, value in config.items():
                file.write(f"{key} = {value}\n")

    # Функция для выполнения генетического алгоритма
    def run_gen_algorithm():
        try:
            gen_algorithm_button.config(state=tk.DISABLED)  # Делаем кнопку неактивной
            process = subprocess.Popen(["python", "Genetic_Algorithm.py"])
            process.wait()

            if process.returncode == 0:
                messagebox.showinfo("Выполнено", "Генетический алгоритм выполнен")
            else:
                messagebox.showerror("Ошибка", "Произошла ошибка при выполнении генетического алгоритма")
        except Exception as e:
            error_message = f"Произошла ошибка: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Ошибка", error_message)

        gen_algorithm_button.config(state=tk.NORMAL)  # Включаем кнопку после выполнения скрипта
        gen_algorithm_window.destroy()

    # Создание нового окна
    gen_algorithm_window = tk.Toplevel(root)
    gen_algorithm_window.title("Генетический алгоритм")

    # Загрузка значений из файла config.txt
    config = read_config()

    # Создание и размещение элементов в окне
    population_size_label = tk.Label(gen_algorithm_window, text="Размер популяции:")
    population_size_entry = tk.Entry(gen_algorithm_window)
    population_size_entry.insert(0, config.get("population_size", "20"))
    population_size_entry.bind("<KeyRelease>", validate_fields)

    num_generations_label = tk.Label(gen_algorithm_window, text="Количество поколений:")
    num_generations_entry = tk.Entry(gen_algorithm_window)
    num_generations_entry.insert(0, config.get("num_generations", "20"))
    num_generations_entry.bind("<KeyRelease>", validate_fields)

    mutation_rate_label = tk.Label(gen_algorithm_window, text="Коэффициент мутации:")
    mutation_rate_entry = tk.Entry(gen_algorithm_window)
    mutation_rate_entry.insert(0, config.get("mutation_rate", "0.3"))

    gen_algorithm_button = tk.Button(gen_algorithm_window, text="Запустить генетический алгоритм",
                                     state="disabled", command=run_gen_algorithm)

    save_button = tk.Button(gen_algorithm_window, text="Сохранить изменения", command=save_config)

    population_size_label.pack()
    population_size_entry.pack()
    num_generations_label.pack()
    num_generations_entry.pack()
    mutation_rate_label.pack()
    mutation_rate_entry.pack()
    gen_algorithm_button.pack()
    save_button.pack()

    # Проверка полей и активация кнопки при изменении значений
    validate_fields()

    # Закрытие окна генетического алгоритма
    def close_gen_algorithm_window():
        gen_algorithm_window.destroy()

    gen_algorithm_window.protocol("WM_DELETE_WINDOW", close_gen_algorithm_window)


def read_config():
    config = {}
    with open('config.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=")
                config[key.strip()] = value.strip()
    return config

def save_to_config(key:str, value:str):

    config[key] = value

    with open("config.txt", "w") as file:
        for key, value in config.items():
            file.write(f"{key} = {value}\n")


def open_api_settings_window():
    def save_api_settings():
        api_key = api_key_entry.get()
        api_secret = api_secret_entry.get()

        config["api_key"] = api_key
        config["api_secret"] = api_secret

        with open("config.txt", "w") as file:
            for key, value in config.items():
                file.write(f"{key} = {value}\n")

        api_settings_window.destroy()

    def show_api_settings():
        messagebox.showinfo("API настройки",
                            f"API ключ: {config.get('api_key')}\nAPI секрет: {config.get('api_secret')}")

    api_settings_window = tk.Toplevel(root)
    api_settings_window.title("API настройки")
    api_settings_window.geometry("400x200")

    api_key_label = tk.Label(api_settings_window, text="API ключ:")
    api_key_entry = tk.Entry(api_settings_window, width=30, show="*")
    api_key_entry.insert(0, config.get("api_key", ""))
    api_key_entry.focus_set()

    api_secret_label = tk.Label(api_settings_window, text="API секрет:")
    api_secret_entry = tk.Entry(api_settings_window, width=30, show="*")
    api_secret_entry.insert(0, config.get("api_secret", ""))

    save_button = tk.Button(api_settings_window, text="Сохранить", command=save_api_settings)
    cancel_button = tk.Button(api_settings_window, text="Отмена", command=api_settings_window.destroy)

    api_key_label.pack()
    api_key_entry.pack()
    api_secret_label.pack()
    api_secret_entry.pack()
    save_button.pack(pady=10)
    cancel_button.pack(pady=10)

    api_settings_window.transient(root)
    api_settings_window.grab_set()
    api_settings_window.geometry("+%d+%d" % (root.winfo_screenwidth() / 2 - 150, root.winfo_screenheight() / 2 - 75))
    root.wait_window(api_settings_window)


def choose_best_individual_path():
    previous_value = best_individual_entry.get()
    file_path = filedialog.askopenfilename(initialdir="./individuals", title="Выберите лучшего индивидуума для тренировки", filetypes=[("PKL Files", "*.pkl")])
    if file_path:
        best_individual_entry.delete(0, tk.END)
        best_individual_entry.insert(0, file_path)
        save_to_config("best_individual_path", best_individual_entry.get())
    else:
        best_individual_entry.delete(0, tk.END)
        best_individual_entry.insert(0, previous_value)


def choose_model_path():
    previous_value = model_path_entry.get()
    folder_path = filedialog.askopenfilename(initialdir="./models", title="Выберите модель для торговли", filetypes=[("Keras Files", "*.keras")])
    if folder_path:
        model_path_entry.delete(0, tk.END)
        model_path_entry.insert(0, folder_path)
        save_to_config("model_path", model_path_entry.get())
    else:
        model_path_entry.delete(0, tk.END)
        model_path_entry.insert(0, previous_value)

def refresh_pairs():
    # Update the pairs from the txt file
    with open("pairs.txt", "r") as file:
        pairs = file.read().splitlines()
    pair_combobox["values"] = pairs

def update_pair_config(event, сlient=client):
    selected_pair = pair_combobox.get()
    info = сlient.get_symbol_info(selected_pair)
    save_to_config("coin1", info["baseAsset"])
    save_to_config("coin2", info["quoteAsset"])

def update_block_size_config(event):
    block_size = block_size_combobox.get()
    save_to_config("block_size", block_size)

def update_dataset_years_config(event):
    dataset_years = dataset_years_combobox.get()
    save_to_config("date", dataset_years)
def update_candle_time_config(event):
    candle_time = candle_time_combobox.get()
    if candle_time == "1MINUTE":
        time_for_cycle_in_minutes = "1"
    elif candle_time == "5MINUTE":
        time_for_cycle_in_minutes = "5"
    elif candle_time == "15MINUTE":
        time_for_cycle_in_minutes = "15"
    elif candle_time == "30MINUTE":
        time_for_cycle_in_minutes = "30"
    elif candle_time == "1HOUR":
        time_for_cycle_in_minutes = "60"
    elif candle_time == "2HOUR":
        time_for_cycle_in_minutes = "120"
    elif candle_time == "4HOUR":
        time_for_cycle_in_minutes = "240"
    elif candle_time == "6HOUR":
        time_for_cycle_in_minutes = "360"
    elif candle_time == "8HOUR":
        time_for_cycle_in_minutes = "480"
    elif candle_time == "12HOUR":
        time_for_cycle_in_minutes = "720"
    elif candle_time == "1DAY":
        time_for_cycle_in_minutes = "1440"
    elif candle_time == "3DAY":
        time_for_cycle_in_minutes = "4320"
    elif candle_time == "1WEEK":
        time_for_cycle_in_minutes = "10080"
    elif candle_time == "1MONTH":
        time_for_cycle_in_minutes = "43200"
    # Добавьте другие значения в соответствии с выбранными параметрами в candle_time_combobox
    else:
        time_for_cycle_in_minutes = ""

    save_to_config("time_for_cycle_in_minutes", time_for_cycle_in_minutes)
    save_to_config("clines_time", candle_time_combobox.get())

config = read_config()

root = tk.Tk()
root.geometry("1200x450")  # Изменение размера окна
root.title("Slurper")
root.resizable(False, False)  # Запрет изменения размера окна

# Создание фонового изображения
image = Image.open("slurper.png")
image = image.resize((500, 300), Image.LANCZOS)
photo = ImageTk.PhotoImage(image)
panel = tk.Label(root, image=photo)
panel.pack(side="left", fill="both", expand="yes")

# Поля для лучшего индивидуума и модели
fields_frame = tk.Frame(root, width=300)
fields_frame.pack(side="right", fill="y", padx=20, pady=20)

best_individual_label = tk.Label(fields_frame, text="Путь для лучшего индивидуума:")
best_individual_entry = tk.Entry(fields_frame)
best_individual_entry.insert(0, config.get("best_individual_path", ""))
best_individual_label.pack(anchor="w")
best_individual_entry.pack(anchor="w")

best_individual_button = ttk.Button(fields_frame, text="Выбрать", command=choose_best_individual_path)
best_individual_button.pack(anchor="w", pady=5)

model_path_label = tk.Label(fields_frame, text="Путь для модели:")
model_path_entry = tk.Entry(fields_frame)
model_path_entry.insert(0, config.get("model_path", ""))
model_path_label.pack(anchor="w")
model_path_entry.pack(anchor="w")

model_path_button = ttk.Button(fields_frame, text="Выбрать", command=choose_model_path)
model_path_button.pack(anchor="w", pady=5)

# Поле с выпадающим списком для выбора пары для торговли
pair_label = tk.Label(fields_frame, text="Выберите пару:")
pair_combobox = ttk.Combobox(fields_frame, values=[], state="normal", postcommand=refresh_pairs)
pair_label.pack(anchor="w")
pair_combobox.pack(anchor="w")

refresh_pairs_button = ttk.Button(fields_frame, text="Обновить", command=refresh_pairs)
refresh_pairs_button.pack(anchor="w")

# Поле для времени свечей
candle_time_label = tk.Label(fields_frame, text="Время свечей:")
candle_time_combobox = ttk.Combobox(fields_frame,
                                    values=["1MINUTE", "5MINUTE", "15MINUTE", "30MINUTE", "1HOUR", "2HOUR", "4HOUR",
                                            "6HOUR", "8HOUR", "12HOUR", "1DAY", "3DAY", "1WEEK", "1MONTH"])
candle_time_label.pack(anchor="w")
candle_time_combobox.pack(anchor="w")

pair_combobox.set(config["coin1"] + config["coin2"])
candle_time_combobox.set(config.get("clines_time", ""))
candle_time_combobox.set(config["clines_time"])

pair_combobox.bind("<<ComboboxSelected>>", update_pair_config)
candle_time_combobox.bind("<<ComboboxSelected>>", update_candle_time_config)

block_size_label = tk.Label(fields_frame, text="Количество последних свечей:")
block_size_combobox = ttk.Combobox(fields_frame, values=["5", "10", "15", "20", "25", "30"], state="readonly")
block_size_combobox.set(config.get("block_size", ""))
block_size_label.pack(anchor="w")
block_size_combobox.pack(anchor="w")

block_size_combobox.bind("<<ComboboxSelected>>", update_block_size_config)

dataset_years_label = tk.Label(fields_frame, text="Выберите количество дней для сбора датасета:")
dataset_years_combobox = ttk.Combobox(fields_frame, values=["360", "720", "1080", "1800"], state="readonly")
dataset_years_label.pack(anchor="w")
dataset_years_combobox.pack(anchor="w")

dataset_years_combobox.set(config.get("date", ""))
dataset_years_combobox.bind("<<ComboboxSelected>>", update_dataset_years_config)

# Создание кнопок с отступами и стилями
button_frame = tk.Frame(root)
button_frame.pack(side="right", pady=10)

style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.map("TButton", foreground=[('pressed', 'red'), ('active', 'blue')],
          background=[('pressed', '!disabled', 'black'), ('active', 'white')])

load_dataset_button = ttk.Button(button_frame, text="Загрузить датасет", command=load_dataset)
train_model_button = ttk.Button(button_frame, text="Тренировка модели", command=train_model)
calculate_averages_button = ttk.Button(button_frame, text="Вычислить средние тени свечей", command=calculate_averages)
open_gen_algorithm_button = ttk.Button(button_frame, text="Запустить генетический алгоритм",
                                       command=open_gen_algorithm_window)
save_api_button = ttk.Button(button_frame, text="API ключ и секрет", command=open_api_settings_window)

load_dataset_button.pack(side="top", pady=5)
train_model_button.pack(side="top", pady=5)
calculate_averages_button.pack(side="top", pady=5)
open_gen_algorithm_button.pack(side="top", pady=5)
save_api_button.pack(side="top", pady=5)

root.mainloop()
