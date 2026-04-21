#!/usr/bin/env python3
"""
Прототип многоагентной дорожной среды — агентная версия (mas_simulation).

Маршрутизация из main_line в highway1/highway2 определяется двумя флагами:
  can_take1 — устанавливается пользователем (0 или 1)
  can_take2 — вычисляется автоматически: 1 если len(highway2) < THRESHOLD, иначе 0

Правила выбора трассы:
  can_take1=0, can_take2=0 → машина ждёт (не переводится)
  can_take1=1, can_take2=1 → случайный выбор 50/50
  can_take1=1, can_take2=0 → highway1
  can_take1=0, can_take2=1 → highway2

Если can_take2=0, очередь highway2 подсвечивается красным на визуализации.
"""

import random
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import threading
import time
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'multiagent-sim-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# ── Константы ─────────────────────────────────────────────────────────────────
MAIN_LINE_CAPACITY  = 12
HIGHWAY_CAPACITY    = 10
HIGHWAY2_THRESHOLD  = 5   # значение по умолчанию; далее управляется через sim_cfg

# ── Очереди ───────────────────────────────────────────────────────────────────
main_line = deque()
highway1  = deque()
highway2  = deque()

# ── Параметры симуляции ───────────────────────────────────────────────────────
sim_cfg = {
    'running':        False,
    'can_take1':      1,              # 0 или 1 — управляется пользователем
    'threshold':      HIGHWAY2_THRESHOLD,  # порог для can_take2 — управляется ползунком
    'main_delay':     1.0,   # с — пауза между добавлением машин в main_line
    'transfer_delay': 0.7,   # с — пауза между попытками перевода
    'drain_delay_h1': 2.0,   # с — пауза между выходом машин из highway1
    'drain_delay_h2': 2.0,   # с — пауза между выходом машин из highway2
}

lock = threading.Lock()


# ── Вспомогательные функции ───────────────────────────────────────────────────

def can_take2():
    """Возвращает 1 если highway2 не заполнена до порога, иначе 0.
    Вызывать при удержании lock.
    """
    return 1 if len(highway2) < sim_cfg['threshold'] else 0


def padded(q, capacity):
    """Начало очереди (старейшая машина) — справа, конец (новейшая) — слева."""
    lst = list(q)
    return [0] * (capacity - len(lst)) + lst


def broadcast():
    with lock:
        ct2 = can_take2()
        state = {
            'main_line': padded(main_line, MAIN_LINE_CAPACITY),
            'highway1':  padded(highway1,  HIGHWAY_CAPACITY),
            'highway2':  padded(highway2,  HIGHWAY_CAPACITY),
            'counts': {
                'main_line': len(main_line),
                'highway1':  len(highway1),
                'highway2':  len(highway2),
            },
            'cfg':       dict(sim_cfg),
            'can_take2': ct2,
            'threshold': HIGHWAY2_THRESHOLD,
            'cap': {
                'main_line': MAIN_LINE_CAPACITY,
                'highway':   HIGHWAY_CAPACITY,
            },
        }
    socketio.emit('state', state)


# ── Основной цикл симуляции ───────────────────────────────────────────────────

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

        # 1. Добавить машину в конец main_line
        if now - t_add >= cfg['main_delay']:
            t_add = now
            with lock:
                if len(main_line) < MAIN_LINE_CAPACITY:
                    main_line.append(1)
                    changed = True

        # 2. Перевести машину из main_line в highway по правилам маршрутизации
        if now - t_transfer >= cfg['transfer_delay']:
            t_transfer = now
            with lock:
                ct1 = cfg['can_take1']
                ct2 = can_take2()

                if main_line:
                    if ct1 == 0 and ct2 == 0:
                        pass  # ждём — ни одна очередь не принимает

                    elif ct1 == 1 and ct2 == 1:
                        # случайный выбор 50/50
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

        # 3. Выезд машин из highway1
        if now - t_drain_h1 >= cfg['drain_delay_h1']:
            t_drain_h1 = now
            with lock:
                if highway1:
                    highway1.popleft()
                    changed = True

        # 4. Выезд машин из highway2
        if now - t_drain_h2 >= cfg['drain_delay_h2']:
            t_drain_h2 = now
            with lock:
                if highway2:
                    highway2.popleft()
                    changed = True

        if changed:
            broadcast()

        time.sleep(0.05)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


# ── Socket.IO events ──────────────────────────────────────────────────────────

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
    if key not in sim_cfg:
        return
    with lock:
        if key in ('can_take1', 'threshold'):
            sim_cfg[key] = int(val)
        else:
            sim_cfg[key] = float(val)
    broadcast()


# ── HTML + CSS + JS ───────────────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MAS — агентная среда</title>
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

/* ─── Road scene ─── */
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

.main-block {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.highways {
  display: flex;
  flex-direction: column;
  gap: 18px;
  justify-content: center;
}

.junction {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 40px;
  flex-shrink: 0;
  gap: 6px;
  padding: 0 4px;
  position: relative;
}
.junction::before {
  content: '';
  position: absolute;
  left: 50%; top: 0; bottom: 0;
  width: 2px;
  background: linear-gradient(to bottom, var(--h1), var(--h2));
  opacity: .35;
  transform: translateX(-50%);
}
.j-arrow { font-size: 1.1rem; line-height: 1; z-index: 1; }

.lane-row {
  display: flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}

.lane-label {
  font-size: .72rem;
  color: var(--dim);
  width: 68px;
  flex-shrink: 0;
}

.bar {
  display: flex;
  gap: 3px;
  padding: 4px;
  background: #12122a;
  border-radius: 8px;
  border: 1px solid var(--border);
  transition: border-color .3s, box-shadow .3s;
}
.bar.blocked {
  border-color: var(--blocked);
  box-shadow: 0 0 12px #f7258540;
}

.cell {
  width: 30px;
  height: 26px;
  border-radius: 5px;
  transition: background-color .2s ease, box-shadow .2s ease;
  background: var(--empty);
  border: 1px solid #1e1e38;
}
.cell.car-main { background: var(--main); box-shadow: 0 0 7px var(--main); }
.cell.car-h1   { background: var(--h1);   box-shadow: 0 0 7px var(--h1);   }
.cell.car-h2   { background: var(--h2);   box-shadow: 0 0 7px var(--h2);   }

/* ─── Counters ─── */
.counters {
  display: flex;
  gap: 28px;
  font-size: .82rem;
  color: var(--dim);
  flex-wrap: wrap;
  justify-content: center;
}
.counters b { color: var(--text); }

/* ─── Status badge ─── */
.badge {
  padding: 3px 12px; border-radius: 99px;
  font-size: .75rem; font-weight: 700; letter-spacing: .5px;
  border: 1px solid currentColor;
}
.badge.run  { color: var(--main); background: #4cc9f018; }
.badge.stop { color: var(--h1);   background: #f7258518; }

/* ─── Controls ─── */
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 24px;
  width: 100%;
  max-width: 1060px;
  align-items: flex-end;
}

.ctrl-group {
  display: flex; flex-direction: column; gap: 6px; min-width: 160px;
}
.ctrl-group label { font-size: .72rem; color: var(--dim); line-height: 1.3; }
.ctrl-group .val  { font-size: .9rem; font-weight: 700; color: var(--text); }
input[type=range] { width: 100%; accent-color: var(--main); cursor: pointer; }

.btn-row { display: flex; gap: 8px; flex-wrap: wrap; }
button {
  padding: 8px 18px; border: none; border-radius: 8px;
  font-size: .85rem; font-weight: 700; cursor: pointer; transition: opacity .15s;
}
button:hover { opacity: .8; }
#btn-start { background: var(--main); color: #000; }
#btn-stop  { background: var(--h1);   color: #fff; }
#btn-reset  { background: #333;        color: var(--text); }
#btn-random { background: var(--rnd);  color: #000; }

/* can_take1 toggle */
#btn-ct1 { min-width: 130px; }
#btn-ct1.on  { background: #4ade80; color: #000; }
#btn-ct1.off { background: #555;    color: var(--text); }

/* can_take2 indicator */
.ct2-indicator {
  display: flex; align-items: center; gap: 8px;
  font-size: .82rem; color: var(--dim);
}
.ct2-dot {
  width: 14px; height: 14px; border-radius: 50%;
  border: 2px solid var(--dim);
  transition: background .3s, border-color .3s;
}
.ct2-dot.on  { background: var(--h2); border-color: var(--h2); }
.ct2-dot.off { background: var(--blocked); border-color: var(--blocked); }

.accent-h1 { accent-color: var(--h1); }
.accent-h2 { accent-color: var(--h2); }
.label-h1  { color: var(--h1) !important; }
.label-h2  { color: var(--h2) !important; }
</style>
</head>
<body>

<h1>MAS — многоагентная дорожная среда</h1>

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

  <!-- Управление -->
  <div class="ctrl-group">
    <label>Управление</label>
    <div class="btn-row">
      <button id="btn-start"  onclick="sock.emit('start')">▶ Старт</button>
      <button id="btn-stop"   onclick="sock.emit('stop')">■ Стоп</button>
      <button id="btn-reset"  onclick="sock.emit('reset')">↺ Сброс</button>
      <button id="btn-random" onclick="randomize()">⚄ Random</button>
    </div>
  </div>

  <!-- can_take1 и can_take2 -->
  <div class="ctrl-group">
    <label>Флаги маршрутизации</label>
    <div class="btn-row" style="align-items:center; gap:12px">
      <button id="btn-ct1" class="on" onclick="toggleCt1()">can_take1: 1</button>
      <div class="ct2-indicator">
        <div class="ct2-dot on" id="ct2-dot"></div>
        <span>can_take2: <b id="ct2-val">1</b></span>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:8px; margin-top:4px">
      <span style="font-size:.72rem; color:var(--dim)">порог highway2: <b id="thresh-val">5</b></span>
      <input id="sl-threshold" type="range" min="1" max="10" step="1" value="5"
             style="width:120px; accent-color:var(--h2)"
             oninput="setParam('threshold', this.value);
                      id('thresh-val').textContent = this.value">
    </div>
    <div style="display:none">
    </div>
  </div>

  <!-- Слайдеры скоростей -->
  <div class="ctrl-group">
    <label>
      main_delay — пауза добавления
      <span class="val" id="v-main-delay">1.0 с</span>
    </label>
    <input id="sl-main-delay" type="range" min="0.2" max="4" step="0.1" value="1.0"
           oninput="setParam('main_delay', this.value);
                    id('v-main-delay').textContent = fmt(this.value)">
  </div>

  <div class="ctrl-group">
    <label>
      transfer_delay — пауза перехода
      <span class="val" id="v-transfer-delay">0.7 с</span>
    </label>
    <input id="sl-transfer-delay" type="range" min="0.1" max="4" step="0.1" value="0.7"
           oninput="setParam('transfer_delay', this.value);
                    id('v-transfer-delay').textContent = fmt(this.value)">
  </div>

  <div class="ctrl-group">
    <label class="label-h1">
      drain_delay_h1 — выезд из Highway 1
      <span class="val" id="v-drain-h1">2.0 с</span>
    </label>
    <input id="sl-drain-h1" class="accent-h1" type="range" min="0.1" max="8" step="0.1" value="2.0"
           oninput="setParam('drain_delay_h1', this.value);
                    id('v-drain-h1').textContent = fmt(this.value)">
  </div>

  <div class="ctrl-group">
    <label class="label-h2">
      drain_delay_h2 — выезд из Highway 2
      <span class="val" id="v-drain-h2">2.0 с</span>
    </label>
    <input id="sl-drain-h2" class="accent-h2" type="range" min="0.1" max="8" step="0.1" value="2.0"
           oninput="setParam('drain_delay_h2', this.value);
                    id('v-drain-h2').textContent = fmt(this.value)">
  </div>

</div>

<script>
const sock = io();
const id  = s => document.getElementById(s);
const fmt = v => parseFloat(v).toFixed(1) + ' с';

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
  const badge = id('badge');
  badge.textContent = run ? 'РАБОТАЕТ' : 'СТОП';
  badge.className   = 'badge ' + (run ? 'run' : 'stop');

  // can_take1 кнопка
  const ct1 = s.cfg.can_take1;
  const btn = id('btn-ct1');
  btn.textContent = 'can_take1: ' + ct1;
  btn.className   = ct1 ? 'on' : 'off';

  // can_take2 индикатор + подсветка highway2
  const ct2 = s.can_take2;
  id('ct2-val').textContent = ct2;
  id('thresh-val').textContent = s.threshold;
  id('ct2-dot').className = 'ct2-dot ' + (ct2 ? 'on' : 'off');
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
    { key: 'main_delay',     sid: 'sl-main-delay',     vid: 'v-main-delay',     min: 0.2, max: 4.0, step: 0.1,  fmt: fmt },
    { key: 'transfer_delay', sid: 'sl-transfer-delay', vid: 'v-transfer-delay', min: 0.1, max: 4.0, step: 0.1,  fmt: fmt },
    { key: 'drain_delay_h1', sid: 'sl-drain-h1',       vid: 'v-drain-h1',       min: 0.1, max: 8.0, step: 0.1,  fmt: fmt },
    { key: 'drain_delay_h2', sid: 'sl-drain-h2',       vid: 'v-drain-h2',       min: 0.1, max: 8.0, step: 0.1,  fmt: fmt },
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

function toggleCt1() {
  const cur = parseInt(id('btn-ct1').textContent.slice(-1));
  setParam('can_take1', cur === 1 ? 0 : 1);
}
</script>
</body>
</html>
"""

_thread = None

if __name__ == '__main__':
    print('Сервер запущен: http://localhost:5003')
    socketio.run(app, host='0.0.0.0', port=5003, debug=False, allow_unsafe_werkzeug=True)
