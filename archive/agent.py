#!/usr/bin/env python3
"""
MaxEnt IRL Agent — обучение агента методом максимальной энтропии
(Maximum Entropy Inverse Reinforcement Learning, Ziebart et al., 2008)

Датасет:  expert/processed_expert.csv
Таргет:   can_take1  (бинарное действие агента)
Фичи:     все столбцы кроме step и timestamp
"""

import os
import csv
import json
import numpy as np

# ─────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../expert/processed_expert.csv")
WEIGHTS_PATH = os.path.join(BASE_DIR, "maxent_weights.npz")
META_PATH    = os.path.join(BASE_DIR, "maxent_meta.json")

TARGET_COL  = "can_take1"          # таргет-переменная (бинарное действие)
SKIP_COLS   = {"step", "timestamp"}

LR          = 0.1      # скорость обучения
N_EPOCHS    = 3000     # число итераций градиентного подъёма
L2_LAMBDA   = 0.001    # коэффициент L2-регуляризации
LOG_EVERY   = 500      # печатать лог каждые N эпох


# ─────────────────────────────────────────────────────
# Загрузка датасета (без pandas)
# ─────────────────────────────────────────────────────
def load_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames
    return rows, cols


print(f"Загрузка датасета: {DATASET_PATH}")
rows, all_cols = load_csv(DATASET_PATH)

feature_cols = [c for c in all_cols if c not in SKIP_COLS]
print(f"Фичи  ({len(feature_cols)}): {feature_cols}")
print(f"Таргет: {TARGET_COL}")
print(f"Строк:  {len(rows)}\n")

X = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)
y = np.array([int(r[TARGET_COL]) for r in rows], dtype=np.float64)

# Стандартизация фич (z-score)
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8      # защита от деления на ноль
X_norm = (X - X_mean) / X_std


# ─────────────────────────────────────────────────────
# MaxEnt IRL — математическая основа
#
# Функция вознаграждения:   R(s) = θ · φ(s)
# Политика (softmax/sigmoid): π(a=1|s) = σ(R(s))
#
# Градиент log-правдоподобия эксперта:
#   ∂L/∂θ = E_expert[φ(s)] − E_π[φ(s)]
#          = (1/N) Σ_i (y_i − σ(θ·φ(s_i))) · φ(s_i)
#
# Это в точности эквивалентно MLE-логистической регрессии,
# что является частным случаем MaxEnt IRL для бинарных действий.
# ─────────────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def log_likelihood(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    p = sigmoid(X @ theta)
    eps = 1e-12
    return float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def accuracy(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    return float(((sigmoid(X @ theta) >= 0.5) == y).mean())


def train_maxent_irl(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    n_epochs: int,
    l2_lambda: float,
) -> np.ndarray:
    n, d = X.shape
    theta = np.zeros(d)

    # Ожидаемые фичи эксперта: μ_E = среднее φ(s) по состояниям с действием 1
    expert_mask     = y == 1
    expert_feat_exp = X[expert_mask].mean(axis=0) if expert_mask.any() else np.zeros(d)

    print("Epoch     | log-likelihood | accuracy")
    print("-" * 40)

    for epoch in range(n_epochs):
        probs = sigmoid(X @ theta)          # π(a=1|s) по каждому состоянию

        # Ожидаемые фичи текущей политики: μ_π = Σ π(a=1|s)·φ(s) / N
        policy_feat_exp = (probs[:, None] * X).mean(axis=0)

        # Градиент MaxEnt IRL + L2-регуляризация
        grad   = expert_feat_exp - policy_feat_exp - l2_lambda * theta
        theta += lr * grad

        if (epoch + 1) % LOG_EVERY == 0:
            ll  = log_likelihood(X, y, theta)
            acc = accuracy(X, y, theta)
            print(f"{epoch+1:6d}    | {ll:14.4f} | {acc:.4f}")

    return theta


# ─────────────────────────────────────────────────────
# Обучение
# ─────────────────────────────────────────────────────
theta = train_maxent_irl(X_norm, y, LR, N_EPOCHS, L2_LAMBDA)

final_ll  = log_likelihood(X_norm, y, theta)
final_acc = accuracy(X_norm, y, theta)
print(f"\nИтог: log-likelihood={final_ll:.4f}, accuracy={final_acc:.4f}")

# Веса в интерпретируемом виде
print("\nВеса признаков (reward weights θ):")
for name, w in sorted(zip(feature_cols, theta), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:30s}: {w:+.6f}")


# ─────────────────────────────────────────────────────
# Сохранение модели
# ─────────────────────────────────────────────────────
np.savez(
    WEIGHTS_PATH,
    theta  = theta,
    X_mean = X_mean,
    X_std  = X_std,
)

meta = {
    "feature_cols": feature_cols,
    "target_col":   TARGET_COL,
    "accuracy":     final_acc,
    "log_likelihood": final_ll,
    "n_epochs":     N_EPOCHS,
    "lr":           LR,
    "l2_lambda":    L2_LAMBDA,
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nВеса сохранены:  {WEIGHTS_PATH}")
print(f"Метаданные:      {META_PATH}")


# ─────────────────────────────────────────────────────
# Интерфейс для использования обученной модели
# ─────────────────────────────────────────────────────
def load_model(weights_path: str = WEIGHTS_PATH, meta_path: str = META_PATH):
    """Загрузить обученную модель из файлов."""
    data  = np.load(weights_path)
    theta = data["theta"]
    mean  = data["X_mean"]
    std   = data["X_std"]
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return theta, mean, std, meta


def predict(state: dict, theta: np.ndarray, mean: np.ndarray,
            std: np.ndarray, feature_cols: list) -> tuple[int, float]:
    """
    Предсказать действие агента по вектору состояния.

    :param state: словарь {имя_фичи: значение}
    :return: (action: int, probability: float)
    """
    phi   = np.array([float(state[c]) for c in feature_cols])
    phi_n = (phi - mean) / std
    prob  = float(sigmoid(phi_n @ theta))
    return int(prob >= 0.5), prob


# ─────────────────────────────────────────────────────
# Быстрая проверка загрузки
# ─────────────────────────────────────────────────────
print("\n--- Проверка загрузки модели ---")
theta2, mean2, std2, meta2 = load_model()
sample_state = {c: float(X[0, i]) for i, c in enumerate(feature_cols)}
action, prob = predict(sample_state, theta2, mean2, std2, meta2["feature_cols"])
print(f"Пример (строка 0): action={action}, prob={prob:.4f}, истинное={int(y[0])}")
