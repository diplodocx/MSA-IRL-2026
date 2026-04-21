#!/usr/bin/env python3
"""
MCE IRL  (Maximum Causal Entropy Inverse Reinforcement Learning)
Обучение агента с использованием библиотеки imitation.

Установка зависимостей:
    pip install imitation torch gymnasium numpy

Датасет:  expert/processed_expert.csv
Таргет:   can_take1  (бинарное действие агента)
Фичи:     все столбцы кроме step и timestamp
"""

import os
import sys
import csv
import json
import numpy as np
import torch


import gymnasium as gym
from gymnasium import spaces

from imitation.algorithms.mce_irl import MCEIRL
from imitation.rewards.reward_nets import BasicRewardNet


# ─────────────────────────────────────────────────────
# Реализация TabularModelEnv (не входит в новые версии imitation)
# ─────────────────────────────────────────────────────
class TabularModelEnv(gym.Env):
    """
    Конечный табличный MDP с известной динамикой переходов.
    Нужен для MCE IRL: хранит transition_matrix, observation_matrix,
    initial_state_dist и horizon — всё, что MCEIRL читает напрямую.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,   # (S, A, S)
        observation_matrix: np.ndarray,  # (S, obs_dim)
        reward_matrix: np.ndarray,       # (S,)
        horizon: int,
        initial_state_dist: np.ndarray,  # (S,)
    ):
        super().__init__()
        self.transition_matrix   = transition_matrix
        self.observation_matrix  = observation_matrix
        self.reward_matrix       = reward_matrix
        self.horizon             = horizon
        self.initial_state_dist  = initial_state_dist

        S, A, _ = transition_matrix.shape
        obs_dim  = observation_matrix.shape[1]
        self.n_states  = S
        self.n_actions = A
        self.state_dim  = S   # нужно MCEIRL
        self.action_dim = A   # нужно MCEIRL
        self.state_space = spaces.Discrete(S)  # нужно MCEIRL

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(A)

        self._state: int = 0
        self._t: int = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = int(
            np.random.choice(self.n_states, p=self.initial_state_dist)
        )
        self._t = 0
        return self.observation_matrix[self._state].astype(np.float32), {}

    def step(self, action):
        prob       = self.transition_matrix[self._state, int(action)]
        next_state = int(np.random.choice(self.n_states, p=prob))
        reward     = float(self.reward_matrix[self._state])
        self._state = next_state
        self._t    += 1
        terminated  = self._t >= self.horizon
        obs         = self.observation_matrix[self._state].astype(np.float32)
        return obs, reward, terminated, False, {}

# ─────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET   = os.path.join(BASE_DIR, "..", "expert", "processed_expert.csv")
MODEL_PT  = os.path.join(BASE_DIR, "mce_irl_reward.pt")
META_JSON = os.path.join(BASE_DIR, "mce_irl_meta.json")

TARGET    = "can_take1"              # таргет-переменная (действие)
SKIP      = {"step", "timestamp"}    # столбцы, которые не являются фичами
MAX_BINS  = 4                        # макс. бинов при дискретизации фич
N_ITER    = 100                      # итераций MCE IRL
LR        = 0.01                     # learning rate
DISCOUNT  = 0.99                     # коэффициент дисконтирования


# ─────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────
def load_csv(path: str):
    """Загрузить CSV без pandas."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames


def discretize(X: np.ndarray, max_bins: int):
    """
    Дискретизировать фичи: бинарные / малокардинальные остаются
    как есть, непрерывные разбиваются на перцентильные бины.
    Возвращает массив индексов состояний и метаданные бинов.
    """
    n, d = X.shape
    digits = np.zeros((n, d), dtype=int)
    bin_info: list[dict] = []

    for j in range(d):
        uq = np.unique(X[:, j])
        if len(uq) <= max_bins:
            # Натуральная дискретизация
            mapping = {v: i for i, v in enumerate(uq)}
            digits[:, j] = np.array([mapping[v] for v in X[:, j]])
            bin_info.append({"kind": "natural", "values": uq.tolist()})
        else:
            # Перцентильные бины
            edges = np.unique(
                np.percentile(X[:, j], np.linspace(0, 100, max_bins + 1))
            )
            nb = len(edges) - 1
            digits[:, j] = np.clip(
                np.digitize(X[:, j], edges[1:-1]), 0, nb - 1
            )
            bin_info.append({"kind": "percentile", "edges": edges.tolist()})

    sizes = [
        len(b["values"]) if b["kind"] == "natural" else len(b["edges"]) - 1
        for b in bin_info
    ]
    mults = np.cumprod([1] + sizes[:-1]).astype(int)
    indices = (digits * mults).sum(axis=1).astype(int)
    n_states = int(np.prod(sizes))
    return indices, n_states, bin_info, sizes


def build_mdp(state_ids, actions, X, n_states, n_actions):
    """
    Построить матрицу переходов T(s, a, s') и матрицу наблюдений
    (среднее φ(s) по всем посещениям состояния s).
    """
    T = np.full((n_states, n_actions, n_states), 1e-10)
    obs_sum = np.zeros((n_states, X.shape[1]))
    obs_cnt = np.zeros(n_states)

    for t in range(len(state_ids) - 1):
        T[state_ids[t], actions[t], state_ids[t + 1]] += 1.0
    for i, s in enumerate(state_ids):
        obs_sum[s] += X[i]
        obs_cnt[s] += 1

    # Нормализация T → вероятности
    for s in range(n_states):
        for a in range(n_actions):
            T[s, a] /= T[s, a].sum()

    # Среднее наблюдение для каждого состояния
    mask = obs_cnt > 0
    obs_sum[mask] /= obs_cnt[mask, None]
    return T, obs_sum


def compute_soft_policy(T, reward_vec, horizon, discount):
    """
    Обратная индукция → soft-оптимальная политика π(a|s)
    (maximum causal entropy).
    """
    S, A, _ = T.shape
    V = np.zeros(S)
    pi = np.zeros((S, A))
    for _ in range(horizon):
        Q = np.zeros((S, A))
        for a in range(A):
            Q[:, a] = reward_vec + discount * T[:, a, :] @ V
        Q_max = Q.max(axis=1, keepdims=True)
        exp_Q = np.exp(np.clip(Q - Q_max, -50, 50))
        pi = exp_Q / exp_Q.sum(axis=1, keepdims=True)
        V = Q_max.squeeze() + np.log(exp_Q.sum(axis=1))
    return pi


# ─────────────────────────────────────────────────────
# 1. Загрузка данных
# ─────────────────────────────────────────────────────
print("═══ Загрузка данных ═══")
rows, cols = load_csv(DATASET)
feat_cols = [c for c in cols if c not in SKIP and c != TARGET]

X = np.array([[float(r[c]) for c in feat_cols] for r in rows])
y = np.array([int(r[TARGET]) for r in rows])

print(f"Фичи ({len(feat_cols)}): {feat_cols}")
print(f"Таргет: {TARGET}   (0→{(y==0).sum()},  1→{(y==1).sum()})")
print(f"Строк:  {len(y)}")


# ─────────────────────────────────────────────────────
# 2. Дискретизация → табличный MDP
# ─────────────────────────────────────────────────────
print("\n═══ Дискретизация ═══")
state_ids, S, bin_info, sizes = discretize(X, MAX_BINS)
A = 2                            # бинарное действие (0 / 1)
print(f"Бины на фичу: {sizes}  →  {S} состояний,  {A} действия")

T, obs_mat = build_mdp(state_ids, y, X, S, A)

# Начальное распределение состояний
p0 = np.zeros(S)
p0[state_ids[0]] = 1.0
horizon = min(len(y), 200)


# ─────────────────────────────────────────────────────
# 3. Создание TabularModelEnv
# ─────────────────────────────────────────────────────
env = TabularModelEnv(
    transition_matrix=T,
    observation_matrix=obs_mat,
    reward_matrix=np.zeros(S),       # вознаграждение будет выучено
    horizon=horizon,
    initial_state_dist=p0,
)


# ─────────────────────────────────────────────────────
# 4. Occupancy measure эксперта
# ─────────────────────────────────────────────────────
expert_om = np.bincount(state_ids, minlength=S).astype(float)
expert_om /= expert_om.sum()


# ─────────────────────────────────────────────────────
# 5. MCE IRL — обучение
# ─────────────────────────────────────────────────────
print(f"\n═══ MCE IRL ({N_ITER} итераций) ═══")

reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    use_state=True,
    use_action=False,          # R(s) — reward зависит только от состояния
    use_next_state=False,
    use_done=False,
    hid_sizes=(64,),
)

mce = MCEIRL(
    demonstrations=expert_om,
    env=env,
    reward_net=reward_net,
    optimizer_kwargs={"lr": LR},
    discount=DISCOUNT,
    rng=np.random.default_rng(42),
)
mce.train(max_iter=N_ITER)
print("Обучение завершено.")


# ─────────────────────────────────────────────────────
# 6. Извлечение политики π(a|s) из выученного reward
# ─────────────────────────────────────────────────────
with torch.no_grad():
    obs_th = torch.as_tensor(obs_mat, dtype=torch.float32)
    acts_th = torch.zeros(S, dtype=torch.long)
    nobs_th = torch.zeros_like(obs_th)
    done_th = torch.zeros(S, dtype=torch.bool)
    reward_vec = reward_net(obs_th, acts_th, nobs_th, done_th).numpy()

pi = compute_soft_policy(T, reward_vec, horizon, DISCOUNT)


# ─────────────────────────────────────────────────────
# 7. Оценка на обучающих данных
# ─────────────────────────────────────────────────────
preds = (pi[state_ids, 1] >= 0.5).astype(int)
acc = float((preds == y).mean())
print(f"\nAccuracy (train): {acc:.4f}")


# ─────────────────────────────────────────────────────
# 8. Сохранение модели
# ─────────────────────────────────────────────────────
torch.save(
    {
        "reward_net": reward_net.state_dict(),
        "policy": pi,                   # π(a|s), shape (S, A)
        "reward_vec": reward_vec,        # R(s),   shape (S,)
        "obs_mat": obs_mat,
        "T": T,
        "p0": p0,
        "bin_info": bin_info,
        "sizes": sizes,
        "feat_cols": feat_cols,
        "target": TARGET,
        "horizon": horizon,
        "discount": DISCOUNT,
    },
    MODEL_PT,
)

with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "feat_cols": feat_cols,
            "target": TARGET,
            "accuracy": acc,
            "n_states": S,
            "n_actions": A,
            "horizon": horizon,
            "n_iter": N_ITER,
            "lr": LR,
            "discount": DISCOUNT,
            "max_bins": MAX_BINS,
            "bin_sizes": sizes,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"\nМодель сохранена:  {MODEL_PT}")
print(f"Метаданные:        {META_JSON}")


# ═════════════════════════════════════════════════════
# Функции для повторного использования модели
# ═════════════════════════════════════════════════════
def load_model(path=MODEL_PT, meta_path=META_JSON):
    """Загрузить сохранённую модель и метаданные."""
    ckpt = torch.load(path, weights_only=False)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return ckpt, meta


def _discretize_one(x: np.ndarray, bin_info: list, sizes: list) -> int:
    """Дискретизировать один вектор фич → индекс состояния."""
    idx = 0
    mult = 1
    for j, (val, info) in enumerate(zip(x, bin_info)):
        if info["kind"] == "natural":
            vals = np.array(info["values"])
            b = int(np.argmin(np.abs(vals - val)))
        else:
            edges = info["edges"]
            b = int(np.clip(np.digitize(val, edges[1:-1]), 0, len(edges) - 2))
        idx += b * mult
        mult *= sizes[j]
    return idx


def predict(features: dict, ckpt: dict) -> tuple[int, float]:
    """
    Предсказать действие по словарю признаков.

    :param features: {имя_фичи: числовое_значение, ...}
    :param ckpt:     чекпоинт, загруженный через load_model()
    :return:         (action, probability_of_action_1)
    """
    x = np.array([float(features[c]) for c in ckpt["feat_cols"]])
    sid = _discretize_one(x, ckpt["bin_info"], ckpt["sizes"])
    prob1 = float(ckpt["policy"][sid, 1])
    return int(prob1 >= 0.5), prob1


# ─── Быстрая проверка загрузки ───
if __name__ == "__main__":
    print("\n═══ Проверка загрузки модели ═══")
    ckpt, meta = load_model()
    sample = {c: float(X[0, i]) for i, c in enumerate(feat_cols)}
    a, p = predict(sample, ckpt)
    print(f"Пример (строка 0): action={a}, prob={p:.4f}, "
          f"истинное={int(y[0])}")
