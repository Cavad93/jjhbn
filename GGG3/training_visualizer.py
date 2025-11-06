# -*- coding: utf-8 -*-
"""
training_visualizer.py — Живая визуализация процесса обучения экспертов и МЕТА

Этот модуль создает интерактивный HTML dashboard, который обновляется в реальном времени
и показывает:
- Сходимость CEM/CMA-ES оптимизации МЕТА
- Метрики качества каждого эксперта
- Дерево взаимодействия между экспертами и МЕТА
- Анимированные графики обучения
- Эволюционный прогресс (от бактерии до человека)
"""
from __future__ import annotations
import os
import json
import time
import traceback
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import threading

@dataclass
class ExpertMetrics:
    """Метрики одного эксперта"""
    name: str
    accuracy: float
    n_samples: int
    cv_accuracy: Optional[float]
    cv_ci_lower: Optional[float]
    cv_ci_upper: Optional[float]
    mode: str  # "SHADOW" или "ACTIVE"
    last_update: float

@dataclass
class MetaTrainingStep:
    """Один шаг обучения МЕТА"""
    phase: int
    iteration: int
    best_loss: float
    median_loss: float
    sigma: float
    timestamp: float

class TrainingVisualizer:
    """
    Живая визуализация процесса обучения

    Собирает метрики от экспертов и МЕТА, сохраняет в JSON,
    генерирует HTML dashboard с графиками
    """
    def __init__(self, output_dir: str = "training_viz"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Thread-safe блокировка для защиты от race conditions
        self.lock = threading.RLock()  # RLock позволяет повторный захват из того же потока

        # Хранилища метрик с ограничением размера
        self.expert_metrics: Dict[str, deque] = {
            "XGB": deque(maxlen=1000),
            "RF": deque(maxlen=1000),
            "ARF": deque(maxlen=1000),
            "NN": deque(maxlen=1000)
        }

        self.meta_training_history: Dict[int, deque] = {
            i: deque(maxlen=500) for i in range(6)  # 6 фаз
        }

        # Файлы для обмена данными с HTML
        self.data_file = os.path.join(output_dir, "training_data.json")
        self.html_file = os.path.join(output_dir, "dashboard.html")

        # Генерируем HTML при инициализации
        self._generate_html_dashboard()

        # Создаем пустой JSON файл
        if not os.path.exists(self.data_file):
            self._save_data()

        print(f"[TrainingVisualizer] Initialized: {output_dir}")

    def record_expert_metrics(
        self,
        expert_name: str,
        accuracy: float,
        n_samples: int,
        cv_accuracy: Optional[float] = None,
        cv_ci_lower: Optional[float] = None,
        cv_ci_upper: Optional[float] = None,
        mode: str = "SHADOW"
    ):
        """Записывает метрики эксперта"""
        with self.lock:
            metric = ExpertMetrics(
                name=expert_name,
                accuracy=accuracy,
                n_samples=n_samples,
                cv_accuracy=cv_accuracy,
                cv_ci_lower=cv_ci_lower,
                cv_ci_upper=cv_ci_upper,
                mode=mode,
                last_update=time.time()
            )

            if expert_name in self.expert_metrics:
                self.expert_metrics[expert_name].append(metric)

            # Обновляем эволюцию (опционально)
            try:
                from evolution_tracker import get_evolution_tracker
                evo = get_evolution_tracker(self.output_dir)
                stage = evo.update_expert(expert_name, accuracy)

                # Если был значительный прогресс - отправляем в Telegram
                if stage is not None:
                    self._send_evolution_notification(expert_name, stage, accuracy)
            except ImportError:
                pass  # evolution_tracker не установлен
            except Exception as e:
                print(f"[TrainingVisualizer] Evolution tracking error: {e}")

            self._save_data()

    def record_meta_training_step(
        self,
        phase: int,
        iteration: int,
        best_loss: float,
        median_loss: float,
        sigma: float = 1.0
    ):
        """Записывает один шаг обучения МЕТА (CEM/CMA-ES)"""
        with self.lock:
            step = MetaTrainingStep(
                phase=phase,
                iteration=iteration,
                best_loss=best_loss,
                median_loss=median_loss,
                sigma=sigma,
                timestamp=time.time()
            )

            if phase not in self.meta_training_history:
                self.meta_training_history[phase] = deque(maxlen=500)

            self.meta_training_history[phase].append(step)

            self._save_data()

    def _send_evolution_notification(self, expert_name: str, stage, accuracy: float):
        """Отправляет уведомление об эволюции в Telegram"""
        try:
            # Импортируем функцию отправки
            from meta_report import send_telegram_text

            # Пытаемся получить конфиг
            try:
                from bnbusdrt6 import MLConfig
                cfg = MLConfig()
            except Exception:
                return

            if not hasattr(cfg, 'tg_bot_token') or not cfg.tg_bot_token:
                return

            if not hasattr(cfg, 'tg_chat_id') or not cfg.tg_chat_id:
                return

            message = f"""
🧬 <b>ЭВОЛЮЦИЯ: {expert_name}</b>

{stage.emoji} <b>Уровень {stage.level}/100</b>
<b>Стадия:</b> {stage.name}
<i>{stage.description}</i>

📊 <b>Accuracy:</b> {accuracy*100:.2f}%

{"🟢 АКТИВЕН" if accuracy > 0.65 else "⚪ SHADOW режим"}
"""

            send_telegram_text(cfg.tg_bot_token, cfg.tg_chat_id, message)
        except Exception as e:
            print(f"[TrainingVisualizer] Failed to send Telegram notification: {e}")

    def _save_data(self):
        """
        Сохраняет данные в JSON для HTML dashboard.
        Thread-safe версия с блокировкой и защитой от race conditions.
        """
        # Используем блокировку для thread-safety
        with self.lock:
            try:
                # Создаем копию данных под блокировкой для избежания изменений во время сериализации
                data_snapshot = {
                    "expert_metrics": {
                        name: [asdict(m) for m in list(metrics)]  # list() создает копию
                        for name, metrics in self.expert_metrics.items()
                    },
                    "meta_training": {
                        str(phase): [asdict(s) for s in list(steps)]  # list() создает копию
                        for phase, steps in self.meta_training_history.items()
                    },
                    "last_update": time.time()
                }

            except Exception as e:
                print(f"[TrainingVisualizer] Failed to prepare data snapshot: {e}")
                traceback.print_exc()
                return

        # Записываем вне блокировки чтобы не держать lock долго при I/O операциях
        try:
            # Используем временный файл для атомарной записи
            tmp_file = self.data_file + f".tmp.{os.getpid()}.{threading.get_ident()}"

            # Записываем во временный файл
            with open(tmp_file, "w", encoding="utf-8") as f:
                # Используем компактную сериализацию для больших объемов данных
                if len(json.dumps(data_snapshot)) > 1_000_000:  # Если больше 1MB
                    json.dump(data_snapshot, f, ensure_ascii=False, separators=(',', ':'))
                else:
                    json.dump(data_snapshot, f, ensure_ascii=False, indent=2)

            # Проверяем что файл записался корректно
            with open(tmp_file, "r", encoding="utf-8") as f:
                json.load(f)  # Проверка что JSON валидный

            # Атомарная замена основного файла
            # На Windows os.replace может не работать если файл открыт, используем fallback
            try:
                os.replace(tmp_file, self.data_file)
            except OSError:
                # Fallback для Windows
                if os.path.exists(self.data_file):
                    backup = self.data_file + ".backup"
                    if os.path.exists(backup):
                        os.remove(backup)
                    os.rename(self.data_file, backup)
                os.rename(tmp_file, self.data_file)

            # Подсчитываем статистику
            total_expert_points = sum(len(m) for m in data_snapshot["expert_metrics"].values())
            total_meta_points = sum(len(s) for s in data_snapshot["meta_training"].values())

            # Размер файла для мониторинга
            file_size_mb = os.path.getsize(self.data_file) / (1024 * 1024)

            # Предупреждение если файл слишком большой
            if file_size_mb > 10:
                print(f"[TrainingVisualizer] WARNING: Data file is getting large ({file_size_mb:.2f} MB). "
                    f"Consider cleaning old data.")

        except json.JSONDecodeError as e:
            print(f"[TrainingVisualizer] JSON validation failed: {e}")
            # Удаляем поврежденный временный файл
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception as e:
            print(f"[TrainingVisualizer] Failed to save data: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            # Очистка временного файла при ошибке
            if 'tmp_file' in locals() and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass

    def _generate_html_dashboard(self):
        """Генерирует HTML dashboard с живыми графиками"""
        html_content = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Training Dashboard - Experts & META</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .status-bar {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-around;
            align-items: center;
        }

        .status-item {
            text-align: center;
        }

        .status-value {
            font-size: 2em;
            font-weight: bold;
            color: #4ade80;
        }

        .status-label {
            font-size: 0.9em;
            opacity: 0.8;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }

        @media (max-width: 1200px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }

        .card h2 {
            margin-bottom: 15px;
            font-size: 1.5em;
        }

        .expert-tree {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }

        .experts-row {
            display: flex;
            justify-content: space-around;
            width: 100%;
            margin-bottom: 40px;
        }

        .expert-node {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: 3px solid #fff;
            border-radius: 50%;
            width: 120px;
            height: 120px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .expert-node.active {
            border-color: #4ade80;
            box-shadow: 0 0 30px rgba(74, 222, 128, 0.6);
        }

        .expert-name {
            font-weight: bold;
            font-size: 1.2em;
        }

        .expert-acc {
            font-size: 0.9em;
            margin-top: 5px;
        }

        .meta-node {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border: 4px solid #fff;
            border-radius: 20px;
            width: 200px;
            height: 100px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 20px rgba(0,0,0,0.4);
            animation: glow 2s infinite alternate;
        }

        @keyframes glow {
            from { box-shadow: 0 0 20px rgba(245, 87, 108, 0.5); }
            to { box-shadow: 0 0 40px rgba(245, 87, 108, 0.9); }
        }

        .connection {
            width: 2px;
            height: 40px;
            background: linear-gradient(to bottom, transparent, #fff);
            margin: 0 auto;
        }

        .refresh-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(74, 222, 128, 0.9);
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            animation: fadeIn 0.5s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Training Dashboard - Experts & META</h1>

        <div class="status-bar" id="statusBar">
            <div class="status-item">
                <div class="status-value" id="totalSamples">0</div>
                <div class="status-label">Total Samples</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="avgAccuracy">0%</div>
                <div class="status-label">Avg Accuracy</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="metaPhase">-</div>
                <div class="status-label">META Phase</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="lastUpdate">-</div>
                <div class="status-label">Last Update</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <h2>🌳 Expert Network Structure</h2>
            <div class="expert-tree" id="expertTree">
                <div class="experts-row">
                    <div class="expert-node" id="node-XGB">
                        <div class="expert-name">XGB</div>
                        <div class="expert-acc">-</div>
                    </div>
                    <div class="expert-node" id="node-RF">
                        <div class="expert-name">RF</div>
                        <div class="expert-acc">-</div>
                    </div>
                    <div class="expert-node" id="node-ARF">
                        <div class="expert-name">ARF</div>
                        <div class="expert-acc">-</div>
                    </div>
                    <div class="expert-node" id="node-NN">
                        <div class="expert-name">NN</div>
                        <div class="expert-acc">-</div>
                    </div>
                </div>
                <div class="connection"></div>
                <div class="meta-node">
                    <div class="expert-name">META</div>
                    <div class="expert-acc" id="metaAcc">CEM/CMA-ES</div>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📊 Experts Accuracy Over Time</h2>
                <div id="expertsPlot" style="height: 400px;"></div>
            </div>

            <div class="card">
                <h2>🔥 META CEM/CMA-ES Convergence</h2>
                <div id="metaPlot" style="height: 400px;"></div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📈 XGB Training Progress</h2>
                <div id="xgbPlot" style="height: 300px;"></div>
            </div>

            <div class="card">
                <h2>📈 RF Training Progress</h2>
                <div id="rfPlot" style="height: 300px;"></div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>📈 ARF Training Progress</h2>
                <div id="arfPlot" style="height: 300px;"></div>
            </div>

            <div class="card">
                <h2>📈 NN Training Progress</h2>
                <div id="nnPlot" style="height: 300px;"></div>
            </div>
        </div>
    </div>

    <div class="refresh-indicator" id="refreshIndicator" style="display: none;">
        ✨ Updated!
    </div>

    <script>
        let lastDataTimestamp = 0;

        async function loadData() {
            try {
                const response = await fetch('training_data.json?' + Date.now());
                const data = await response.json();

                if (data.last_update > lastDataTimestamp) {
                    lastDataTimestamp = data.last_update;
                    updateDashboard(data);
                    showRefreshIndicator();
                }
            } catch (error) {
                console.error('Failed to load data:', error);
            }
        }

        function showRefreshIndicator() {
            const indicator = document.getElementById('refreshIndicator');
            indicator.style.display = 'block';
            setTimeout(() => {
                indicator.style.display = 'none';
            }, 2000);
        }

        function updateDashboard(data) {
            updateStatusBar(data);
            updateExpertTree(data);
            updateExpertsPlot(data);
            updateMetaPlot(data);
            updateIndividualPlots(data);
        }

        function updateStatusBar(data) {
            let totalSamples = 0;
            let totalAcc = 0;
            let expertCount = 0;

            for (const [name, metrics] of Object.entries(data.expert_metrics)) {
                if (metrics.length > 0) {
                    const latest = metrics[metrics.length - 1];
                    totalSamples += latest.n_samples;
                    totalAcc += latest.accuracy;
                    expertCount++;
                }
            }

            document.getElementById('totalSamples').textContent = totalSamples;
            document.getElementById('avgAccuracy').textContent =
                expertCount > 0 ? (totalAcc / expertCount * 100).toFixed(1) + '%' : '0%';

            const phases = Object.keys(data.meta_training);
            document.getElementById('metaPhase').textContent =
                phases.length > 0 ? 'Phase ' + phases[phases.length - 1] : '-';

            const lastUpdate = new Date(data.last_update * 1000);
            document.getElementById('lastUpdate').textContent =
                lastUpdate.toLocaleTimeString();
        }

        function updateExpertTree(data) {
            for (const [name, metrics] of Object.entries(data.expert_metrics)) {
                if (metrics.length > 0) {
                    const latest = metrics[metrics.length - 1];
                    const node = document.getElementById('node-' + name);
                    if (node) {
                        const accDiv = node.querySelector('.expert-acc');
                        accDiv.textContent = (latest.accuracy * 100).toFixed(1) + '%';

                        if (latest.mode === 'ACTIVE') {
                            node.classList.add('active');
                        } else {
                            node.classList.remove('active');
                        }
                    }
                }
            }
        }

        function updateExpertsPlot(data) {
            const traces = [];

            for (const [name, metrics] of Object.entries(data.expert_metrics)) {
                if (metrics.length > 0) {
                    const x = metrics.map((m, i) => i);
                    const y = metrics.map(m => m.accuracy * 100);

                    traces.push({
                        x: x,
                        y: y,
                        name: name,
                        mode: 'lines',
                        line: { width: 2 }
                    });
                }
            }

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(255,255,255,0.05)',
                font: { color: '#fff' },
                xaxis: { title: 'Training Step', gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: { title: 'Accuracy (%)', gridcolor: 'rgba(255,255,255,0.1)' },
                margin: { l: 50, r: 20, t: 20, b: 50 }
            };

            Plotly.newPlot('expertsPlot', traces, layout, { responsive: true });
        }

        function updateMetaPlot(data) {
            const traces = [];

            for (const [phase, steps] of Object.entries(data.meta_training)) {
                if (steps.length > 0) {
                    const x = steps.map(s => s.iteration);
                    const y_best = steps.map(s => s.best_loss);
                    const y_median = steps.map(s => s.median_loss);

                    traces.push({
                        x: x,
                        y: y_best,
                        name: `Phase ${phase} (Best)`,
                        mode: 'lines',
                        line: { width: 2 }
                    });

                    traces.push({
                        x: x,
                        y: y_median,
                        name: `Phase ${phase} (Median)`,
                        mode: 'lines',
                        line: { width: 1, dash: 'dash' }
                    });
                }
            }

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(255,255,255,0.05)',
                font: { color: '#fff' },
                xaxis: { title: 'Iteration', gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: {
                    title: 'Loss (log scale)',
                    type: 'log',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                margin: { l: 50, r: 20, t: 20, b: 50 }
            };

            Plotly.newPlot('metaPlot', traces, layout, { responsive: true });
        }

        function updateIndividualPlots(data) {
            for (const [name, metrics] of Object.entries(data.expert_metrics)) {
                if (metrics.length > 0) {
                    const x = metrics.map((m, i) => i);
                    const y = metrics.map(m => m.accuracy * 100);

                    const traces = [{
                        x: x,
                        y: y,
                        name: 'Accuracy',
                        mode: 'lines+markers',
                        line: { width: 2 }
                    }];

                    const layout = {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(255,255,255,0.05)',
                        font: { color: '#fff' },
                        xaxis: { title: 'Step', gridcolor: 'rgba(255,255,255,0.1)' },
                        yaxis: {
                            title: 'Accuracy (%)',
                            gridcolor: 'rgba(255,255,255,0.1)'
                        },
                        margin: { l: 50, r: 20, t: 20, b: 50 }
                    };

                    const plotId = name.toLowerCase() + 'Plot';
                    Plotly.newPlot(plotId, traces, layout, { responsive: true });
                }
            }
        }

        setInterval(loadData, 2000);
        loadData();
    </script>
</body>
</html>"""

        try:
            with open(self.html_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"[TrainingVisualizer] HTML dashboard created: {self.html_file}")
        except Exception as e:
            print(f"[TrainingVisualizer] Failed to create HTML: {e}")

# Глобальное хранилище визуализаторов (по одному на директорию)
_visualizers: Dict[str, TrainingVisualizer] = {}

def get_visualizer(output_dir: str = "training_viz") -> TrainingVisualizer:
    """Получить визуализатор для конкретной директории"""
    global _visualizers
    if output_dir not in _visualizers:
        _visualizers[output_dir] = TrainingVisualizer(output_dir)
        abs_path = os.path.abspath(_visualizers[output_dir].html_file)
        print("=" * 60)
        print(f"📊 Training Dashboard initialized!")
        print(f"📁 HTML: file://{abs_path}")
        print(f"📁 JSON: {_visualizers[output_dir].data_file}")
        print(f"💡 For best results, run: cd {output_dir} && python -m http.server 8000")
        print("=" * 60)
    return _visualizers[output_dir]
