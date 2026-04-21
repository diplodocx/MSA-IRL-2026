#!/usr/bin/env python3
"""
Основная симуляция многоагентной дорожной среды.

can_take1 вычисляется автоматически моделью MCE IRL (prototype/mce_irl_reward.pt).
На вход модели подаются: highway1_len, drain_delay_h1, can_take2 (текущие значения).
can_take2 вычисляется автоматически: 1 если len(highway2) < threshold, иначе 0.

Правила выбора трассы:
  can_take1=0, can_take2=0 → машина ждёт
  can_take1=1, can_take2=1 → случайный выбор 50/50
  can_take1=1, can_take2=0 → highway1
  can_take1=0, can_take2=1 → highway2
"""

import os
import sys
import json
import random
import threading
import time
from collections import deque

import numpy as np
import torch

from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ─────────────────────────────────────────────────────
# Пути к файлам модели
# ─────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PT  = os.path.join(BASE_DIR, "..", "prototype", "mce_irl_reward.pt")
META_JSON = os.path.join(BASE_DIR, "..", "prototype", "mce_irl_meta.json")


# ─────────────────────────────────────────────────────
# Загрузка MCE IRL модели
# ─────────────────────────────────────────────────────
def _load_model():
    if not os.path.exists(MODEL_PT) or not os.path.exists(META_JSON):
        print(
            "ПРЕДУПРЕЖДЕНИЕ: файлы модели не найдены.\n"
            f"  {MODEL_PT}\n  {META_JSON}\n"
            "Запустите prototype/agent.py для обучения модели.\n"
            "До тех пор can_take1 = 1 (по умолчанию)."
        )
        return None

    ckpt = torch.load(MODEL_PT, weights_only=False)
    with open(META_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"Модель загружена. Accuracy: {meta.get('accuracy', '?'):.4f}")
    return ckpt


def _discretize_one(x: np.ndarray, bin_info: list, sizes: list) -> int:
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


def model_predict(ckpt, highway1_len: int, drain_delay_h1: float, can_take2: int):
    """
    Предсказать can_take1 на основе текущего состояния.
    Для next_* фич используются текущие значения (лучшая доступная аппроксимация).
    Возвращает (action: int, prob: float).
    """
    if ckpt is None:
        return 1, 1.0   # fallback если модель не загружена

    feat_cols = ckpt["feat_cols"]
    features = {
        "highway1_len":       float(highway1_len),
        "drain_delay_h1":     float(drain_delay_h1),
        "can_take2":          float(can_take2),
        "next_highway1_len":  float(highway1_len),
        "next_drain_delay_h1": float(drain_delay_h1),
        "next_can_take2":     float(can_take2),
    }

    x = np.array([features[c] for c in feat_cols])
    sid = _discretize_one(x, ckpt["bin_info"], ckpt["sizes"])
    prob1 = float(ckpt["policy"][sid, 1])
    return int(prob1 >= 0.5), prob1


MODEL_CKPT = _load_model()


# ─────────────────────────────────────────────────────
# Flask / SocketIO
# ─────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'multiagent-sim-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# ─────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────
MAIN_LINE_CAPACITY = 12
HIGHWAY_CAPACITY   = 10
HIGHWAY2_THRESHOLD = 5

# ─────────────────────────────────────────────────────
# Очереди
# ─────────────────────────────────────────────────────
main_line = deque()
highway1  = deque()
highway2  = deque()

# ─────────────────────────────────────────────────────
# Параметры симуляции
# ─────────────────────────────────────────────────────
sim_cfg = {
    'running':        False,
    'can_take1':      1,       # вычисляется моделью (read-only)
    'can_take1_prob': 1.0,     # уверенность модели
    'threshold':      HIGHWAY2_THRESHOLD,
    'main_delay':     1.0,
    'transfer_delay': 0.7,
    'drain_delay_h1': 2.0,
    'drain_delay_h2': 2.0,
}

lock = threading.Lock()


# ─────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────
def get_can_take2():
    return 1 if len(highway2) < sim_cfg['threshold'] else 0


def padded(q, capacity):
    lst = list(q)
    return [0] * (capacity - len(lst)) + lst


def broadcast():
    with lock:
        ct2 = get_can_take2()
        state = {
            'main_line': padded(main_line, MAIN_LINE_CAPACITY),
            'highway1':  padded(highway1,  HIGHWAY_CAPACITY),
            'highway2':  padded(highway2,  HIGHWAY_CAPACITY),
            'counts': {
                'main_line': len(main_line),
                'highway1':  len(highway1),
                'highway2':  len(highway2),
            },
            'cfg':           dict(sim_cfg),
            'can_take2':     ct2,
            'threshold':     sim_cfg['threshold'],
            'cap': {
                'main_line': MAIN_LINE_CAPACITY,
                'highway':   HIGHWAY_CAPACITY,
            },
        }
    socketio.emit('state', state)


# ─────────────────────────────────────────────────────
# Основной цикл симуляции
# ─────────────────────────────────────────────────────
def sim_loop():
    t_add      = 0.0
    t_transfer = 0.0
    t_drain_h1 = 0.0
    t_drain_h2 = 0.0

    while sim_cfg['running']:
        now     = time.time()
        changed = False

        with lock:
            cfg = dict(sim_cfg)

        # 1. Добавить машину в main_line
        if now - t_add >= cfg['main_delay']:
            t_add = now
            with lock:
                if len(main_line) < MAIN_LINE_CAPACITY:
                    main_line.append(1)
                    changed = True

        # 2. Перевод машины из main_line → highway
        if now - t_transfer >= cfg['transfer_delay']:
            t_transfer = now
            with lock:
                ct2 = get_can_take2()

                # Вычислить can_take1 через модель
                ct1, prob = model_predict(
                    MODEL_CKPT,
                    highway1_len    = len(highway1),
                    drain_delay_h1  = cfg['drain_delay_h1'],
                    can_take2       = ct2,
                )
                sim_cfg['can_take1']      = ct1
                sim_cfg['can_take1_prob'] = round(prob, 3)

                if main_line:
                    if ct1 == 0 and ct2 == 0:
                        pass  # ждём

                    elif ct1 == 1 and ct2 == 1:
                        if random.random() < 0.5:
                            if len(highway1) < HIGHWAY_CAPACITY:
                                main_line.popleft()
                                highway1.append(1)
                                changed = True
                        else:
                            if len(highway2) < HIGHWAY_CAPACITY:
                                main_line.popleft()
                                highway2.append(1)
                                changed = True

                    elif ct1 == 1 and ct2 == 0:
                        if len(highway1) < HIGHWAY_CAPACITY:
                            main_line.popleft()
                            highway1.append(1)
                            changed = True

                    elif ct1 == 0 and ct2 == 1:
                        if len(highway2) < HIGHWAY_CAPACITY:
                            main_line.popleft()
                            highway2.append(1)
                            changed = True

                changed = True  # can_take1 мог измениться — обновить UI

        # 3. Выезд из highway1
        if now - t_drain_h1 >= cfg['drain_delay_h1']:
            t_drain_h1 = now
            with lock:
                if highway1:
                    highway1.popleft()
                    changed = True

        # 4. Выезд из highway2
        if now - t_drain_h2 >= cfg['drain_delay_h2']:
            t_drain_h2 = now
            with lock:
                if highway2:
                    highway2.popleft()
                    changed = True

        if changed:
            broadcast()

        time.sleep(0.05)


# ─────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)


# ─────────────────────────────────────────────────────
# Socket.IO events
# ─────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    broadcast()


@socketio.on('start')
def on_start():
    global _thread
    with lock:
        if sim_cfg['running']:
            return
        sim_cfg['running'] = True
    _thread = threading.Thread(target=sim_loop, daemon=True)
    _thread.start()
    broadcast()


@socketio.on('stop')
def on_stop():
    with lock:
        sim_cfg['running'] = False
    broadcast()


@socketio.on('reset')
def on_reset():
    with lock:
        sim_cfg['running'] = False
        main_line.clear()
        highway1.clear()
        highway2.clear()
    broadcast()


@socketio.on('set_param')
def on_set_param(data):
    key = data.get('key')
    val = data.get('val')
    # can_take1 управляется только моделью
    if key not in sim_cfg or key in ('can_take1', 'can_take1_prob', 'running'):
        return
    with lock:
        if key == 'threshold':
            sim_cfg[key] = int(val)
        else:
            sim_cfg[key] = float(val)
    broadcast()


# ─────────────────────────────────────────────────────
# HTML + CSS + JS
# ─────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MAS — симуляция с MCE IRL</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:      #0f0f1a;
  --surface: #1a1a2e;
  --border:  #2a2a45;
  --text:    #c8d0e0;
  --dim:     #5a6070;
  --main:    #4cc9f0;
  --h1:      #f72585;
  --h2:      #7bed9f;
  --rnd:     #f4a261;
  --empty:   #252540;
  --blocked: #f72585;
  --model:   #a78bfa;
}

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 28px 20px;
  gap: 20px;
}

h1 { font-size: 1.2rem; font-weight: 600; letter-spacing: .5px; color: var(--main); }
h1 small { color: var(--model); font-size: .8rem; font-weight: 400; margin-left: 8px; }

.scene {
  display: flex;
  align-items: center;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 28px 20px;
  width: 100%;
  max-width: 1060px;
  overflow-x: auto;
}

.main-block { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.highways   { display: flex; flex-direction: column; gap: 18px; justify-content: center; }

.junction {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; width: 40px; flex-shrink: 0;
  gap: 6px; padding: 0 4px; position: relative;
}
.junction::before {
  content: ''; position: absolute; left: 50%; top: 0; bottom: 0; width: 2px;
  background: linear-gradient(to bottom, var(--h1), var(--h2));
  opacity: .35; transform: translateX(-50%);
}
.j-arrow { font-size: 1.1rem; line-height: 1; z-index: 1; }

.lane-row   { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.lane-label { font-size: .72rem; color: var(--dim); width: 68px; flex-shrink: 0; }

.bar {
  display: flex; gap: 3px; padding: 4px;
  background: #12122a; border-radius: 8px;
  border: 1px solid var(--border);
  transition: border-color .3s, box-shadow .3s;
}
.bar.blocked {
  border-color: var(--blocked);
  box-shadow: 0 0 12px #f7258540;
}

.cell {
  width: 30px; height: 26px; border-radius: 5px;
  transition: background-color .2s, box-shadow .2s;
  background: var(--empty); border: 1px solid #1e1e38;
}
.cell.car-main { background: var(--main); box-shadow: 0 0 7px var(--main); }
.cell.car-h1   { background: var(--h1);   box-shadow: 0 0 7px var(--h1);   }
.cell.car-h2   { background: var(--h2);   box-shadow: 0 0 7px var(--h2);   }

.counters {
  display: flex; gap: 28px; font-size: .82rem;
  color: var(--dim); flex-wrap: wrap; justify-content: center;
}
.counters b { color: var(--text); }

.badge {
  padding: 3px 12px; border-radius: 99px;
  font-size: .75rem; font-weight: 700; letter-spacing: .5px;
  border: 1px solid currentColor;
}
.badge.run  { color: var(--main); background: #4cc9f018; }
.badge.stop { color: var(--h1);   background: #f7258518; }

.controls {
  display: flex; flex-wrap: wrap; gap: 18px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 20px 24px;
  width: 100%; max-width: 1060px; align-items: flex-end;
}

.ctrl-group { display: flex; flex-direction: column; gap: 6px; min-width: 160px; }
.ctrl-group label { font-size: .72rem; color: var(--dim); line-height: 1.3; }
.ctrl-group .val  { font-size: .9rem; font-weight: 700; color: var(--text); }
input[type=range] { width: 100%; accent-color: var(--main); cursor: pointer; }

.btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
button {
  padding: 8px 18px; border: none; border-radius: 8px;
  font-size: .85rem; font-weight: 700; cursor: pointer; transition: opacity .15s;
}
button:hover { opacity: .8; }
#btn-start  { background: var(--main); color: #000; }
#btn-stop   { background: var(--h1);   color: #fff; }
#btn-reset  { background: #333;        color: var(--text); }
#btn-random { background: var(--rnd);  color: #000; }

/* Индикаторы флагов */
.flag-panel {
  display: flex; align-items: center; gap: 16px;
  flex-wrap: wrap;
}
.flag-item {
  display: flex; align-items: center; gap: 8px;
  font-size: .82rem; color: var(--dim);
}
.flag-dot {
  width: 14px; height: 14px; border-radius: 50%;
  border: 2px solid var(--dim);
  transition: background .3s, border-color .3s;
}
.flag-dot.on  { background: var(--h2);     border-color: var(--h2);   }
.flag-dot.off { background: var(--blocked); border-color: var(--blocked); }
.flag-dot.model-on  { background: var(--model); border-color: var(--model); }
.flag-dot.model-off { background: #444;         border-color: #666;         }

.model-tag {
  font-size: .65rem; color: var(--model);
  border: 1px solid var(--model); border-radius: 4px;
  padding: 1px 5px; margin-left: 4px; vertical-align: middle;
}

.accent-h1 { accent-color: var(--h1); }
.accent-h2 { accent-color: var(--h2); }
.label-h1  { color: var(--h1) !important; }
.label-h2  { color: var(--h2) !important; }
</style>
</head>
<body>

<h1>MAS — многоагентная дорожная среда <small>MCE IRL</small></h1>

<!-- Road scene -->
<div class="scene">
  <div class="main-block">
    <div class="lane-label" style="text-align:right; color:var(--main)">Main<br>Line</div>
    <div class="bar" id="bar-main"></div>
  </div>
  <div class="junction">
    <span class="j-arrow" style="color:var(--h1)">→</span>
    <span class="j-arrow" style="color:var(--dim)">⋮</span>
    <span class="j-arrow" style="color:var(--h2)">→</span>
  </div>
  <div class="highways">
    <div class="lane-row">
      <div class="lane-label" style="color:var(--h1)">Highway 1</div>
      <div class="bar" id="bar-h1"></div>
    </div>
    <div class="lane-row">
      <div class="lane-label" style="color:var(--h2)">Highway 2</div>
      <div class="bar" id="bar-h2"></div>
    </div>
  </div>
</div>

<!-- Counters -->
<div class="counters">
  <div>Main Line: <b><span id="cnt-m">0</span></b> / <span id="cap-m">12</span></div>
  <div>Highway 1: <b><span id="cnt-h1">0</span></b> / <span id="cap-h">10</span></div>
  <div>Highway 2: <b><span id="cnt-h2">0</span></b> / <span id="cap-h2">10</span></div>
  <div>Статус: <span id="badge" class="badge stop">СТОП</span></div>
</div>

<!-- Controls -->
<div class="controls">

  <!-- Кнопки управления -->
  <div class="ctrl-group">
    <label>Управление</label>
    <div class="btn-row">
      <button id="btn-start"  onclick="sock.emit('start')">▶ Старт</button>
      <button id="btn-stop"   onclick="sock.emit('stop')">■ Стоп</button>
      <button id="btn-reset"  onclick="sock.emit('reset')">↺ Сброс</button>
      <button id="btn-random" onclick="randomize()">⚄ Random</button>
    </div>
  </div>

  <!-- Флаги маршрутизации (только индикаторы, без кнопок управления) -->
  <div class="ctrl-group">
    <label>Флаги маршрутизации</label>
    <div class="flag-panel">

      <!-- can_take1: решение модели -->
      <div class="flag-item">
        <div class="flag-dot model-off" id="ct1-dot"></div>
        <span>
          can_take1: <b id="ct1-val">—</b>
          <span class="model-tag">MCE IRL</span>
        </span>
      </div>
      <div style="font-size:.68rem; color:var(--model)">
        p = <span id="ct1-prob">—</span>
      </div>

      <!-- can_take2: автоматический -->
      <div class="flag-item">
        <div class="flag-dot on" id="ct2-dot"></div>
        <span>can_take2: <b id="ct2-val">1</b></span>
      </div>

    </div>

    <!-- Порог highway2 -->
    <div style="display:flex; align-items:center; gap:8px; margin-top:6px">
      <span style="font-size:.72rem; color:var(--dim)">
        порог highway2: <b id="thresh-val">5</b>
      </span>
      <input id="sl-threshold" type="range" min="1" max="10" step="1" value="5"
             style="width:120px; accent-color:var(--h2)"
             oninput="setParam('threshold', this.value);
                      id('thresh-val').textContent = this.value">
    </div>
  </div>

  <!-- Слайдеры скоростей -->
  <div class="ctrl-group">
    <label>main_delay — пауза добавления <span class="val" id="v-main-delay">1.0 с</span></label>
    <input id="sl-main-delay" type="range" min="0.2" max="4" step="0.1" value="1.0"
           oninput="setParam('main_delay', this.value);
                    id('v-main-delay').textContent = fmt(this.value)">
  </div>

  <div class="ctrl-group">
    <label>transfer_delay — пауза перехода <span class="val" id="v-transfer-delay">0.7 с</span></label>
    <input id="sl-transfer-delay" type="range" min="0.1" max="4" step="0.1" value="0.7"
           oninput="setParam('transfer_delay', this.value);
                    id('v-transfer-delay').textContent = fmt(this.value)">
  </div>

  <div class="ctrl-group">
    <label class="label-h1">drain_delay_h1 — выезд из Highway 1 <span class="val" id="v-drain-h1">2.0 с</span></label>
    <input id="sl-drain-h1" class="accent-h1" type="range" min="0.1" max="8" step="0.1" value="2.0"
           oninput="setParam('drain_delay_h1', this.value);
                    id('v-drain-h1').textContent = fmt(this.value)">
  </div>

  <div class="ctrl-group">
    <label class="label-h2">drain_delay_h2 — выезд из Highway 2 <span class="val" id="v-drain-h2">2.0 с</span></label>
    <input id="sl-drain-h2" class="accent-h2" type="range" min="0.1" max="8" step="0.1" value="2.0"
           oninput="setParam('drain_delay_h2', this.value);
                    id('v-drain-h2').textContent = fmt(this.value)">
  </div>

</div>

<script>
const sock = io();
const id   = s => document.getElementById(s);
const fmt  = v => parseFloat(v).toFixed(1) + ' с';

function makeCells(containerId, cells, carClass) {
  const bar = id(containerId);
  while (bar.children.length < cells.length) bar.appendChild(document.createElement('div'));
  while (bar.children.length > cells.length) bar.removeChild(bar.lastChild);
  cells.forEach((v, i) => { bar.children[i].className = 'cell ' + (v ? carClass : ''); });
}

sock.on('state', s => {
  makeCells('bar-main', s.main_line, 'car-main');
  makeCells('bar-h1',   s.highway1,  'car-h1');
  makeCells('bar-h2',   s.highway2,  'car-h2');

  id('cnt-m').textContent  = s.counts.main_line;
  id('cnt-h1').textContent = s.counts.highway1;
  id('cnt-h2').textContent = s.counts.highway2;
  id('cap-m').textContent  = s.cap.main_line;
  id('cap-h').textContent  = s.cap.highway;
  id('cap-h2').textContent = s.cap.highway;

  const run = s.cfg.running;
  id('badge').textContent = run ? 'РАБОТАЕТ' : 'СТОП';
  id('badge').className   = 'badge ' + (run ? 'run' : 'stop');

  // can_take1 — индикатор решения модели
  const ct1 = s.cfg.can_take1;
  id('ct1-val').textContent  = ct1;
  id('ct1-prob').textContent = (s.cfg.can_take1_prob * 100).toFixed(1) + '%';
  id('ct1-dot').className    = 'flag-dot ' + (ct1 ? 'model-on' : 'model-off');

  // can_take2 — автоматический
  const ct2 = s.can_take2;
  id('ct2-val').textContent = ct2;
  id('ct2-dot').className   = 'flag-dot ' + (ct2 ? 'on' : 'off');
  id('bar-h2').classList.toggle('blocked', ct2 === 0);

  // Слайдеры
  id('v-main-delay').textContent     = fmt(s.cfg.main_delay);
  id('v-transfer-delay').textContent = fmt(s.cfg.transfer_delay);
  id('v-drain-h1').textContent       = fmt(s.cfg.drain_delay_h1);
  id('v-drain-h2').textContent       = fmt(s.cfg.drain_delay_h2);

  id('sl-main-delay').value     = s.cfg.main_delay;
  id('sl-transfer-delay').value = s.cfg.transfer_delay;
  id('sl-drain-h1').value       = s.cfg.drain_delay_h1;
  id('sl-drain-h2').value       = s.cfg.drain_delay_h2;
  id('sl-threshold').value      = s.cfg.threshold;
  id('thresh-val').textContent  = s.cfg.threshold;
});

function setParam(key, val) { sock.emit('set_param', { key, val }); }

function randomize() {
  const sliders = [
    { key: 'main_delay',     sid: 'sl-main-delay',     vid: 'v-main-delay',     min: 0.2, max: 4.0, step: 0.1, fmt },
    { key: 'transfer_delay', sid: 'sl-transfer-delay', vid: 'v-transfer-delay', min: 0.1, max: 4.0, step: 0.1, fmt },
    { key: 'drain_delay_h1', sid: 'sl-drain-h1',       vid: 'v-drain-h1',       min: 0.1, max: 8.0, step: 0.1, fmt },
    { key: 'drain_delay_h2', sid: 'sl-drain-h2',       vid: 'v-drain-h2',       min: 0.1, max: 8.0, step: 0.1, fmt },
    { key: 'threshold',      sid: 'sl-threshold',      vid: 'thresh-val',       min: 1,   max: 10,  step: 1,
      fmt: v => String(Math.round(v)) },
  ];
  sliders.forEach(s => {
    const steps = Math.round((s.max - s.min) / s.step);
    const val   = Math.round((s.min + Math.floor(Math.random() * (steps + 1)) * s.step) * 10) / 10;
    id(s.sid).value       = val;
    id(s.vid).textContent = s.fmt(val);
    setParam(s.key, val);
  });
}
</script>
</body>
</html>
"""

_thread = None

if __name__ == '__main__':
    print('Сервер запущен: http://localhost:5001')
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
