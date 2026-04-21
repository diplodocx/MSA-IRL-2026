import glob
import os
import pandas as pd

# Найти последний (алфавитно) CSV-файл в папке expert
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = sorted(glob.glob(os.path.join(script_dir, "*.csv")))
csv_files = [f for f in csv_files if os.path.basename(f) != "processed_expert.csv"]

if not csv_files:
    raise FileNotFoundError("Нет CSV-файлов в папке expert")

input_path = csv_files[-1]
print(f"Обрабатываю: {input_path}")

df = pd.read_csv(input_path)

# Сортировка по timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Нумерация строк начиная с 1
df.insert(0, "step", range(1, len(df) + 1))

# Добавить поля следующей заявки (shift на -1)
df["next_highway1_len"] = df["highway1_len"].shift(-1)
df["next_drain_delay_h1"] = df["drain_delay_h1"].shift(-1)
df["next_can_take2"] = df["can_take2"].shift(-1)

# Удалить последнюю строку (нет следующей заявки)
df = df.iloc[:-1]

# Привести типы: next_can_take2 должен быть int
df["next_can_take2"] = df["next_can_take2"].astype(int)
df["next_highway1_len"] = df["next_highway1_len"].astype(int)

output_path = os.path.join(script_dir, "processed_expert.csv")
df.to_csv(output_path, index=False)
print(f"Сохранено: {output_path}")
print(df.head())
