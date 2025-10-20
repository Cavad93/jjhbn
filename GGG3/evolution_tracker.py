# -*- coding: utf-8 -*-
"""
evolution_tracker.py — Эволюционная визуализация прогресса обучения

Каждый эксперт и META представлены как организм, эволюционирующий от простейших
форм жизни до человека по мере улучшения метрик.
"""
from __future__ import annotations
import json
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class EvolutionStage:
    """Одна стадия эволюции"""
    level: int  # 0-100
    name: str
    emoji: str
    description: str
    min_accuracy: float
    max_accuracy: float

# 100-уровневая эволюционная шкала
EVOLUTION_STAGES = [
    # 0-10: Происхождение жизни
    EvolutionStage(0, "Химический суп", "🧪", "Хаос, случайные предсказания", 0.0, 0.35),
    EvolutionStage(1, "РНК молекула", "🧬", "Первые паттерны", 0.35, 0.38),
    EvolutionStage(2, "Прокариот", "🦠", "Простейшая клетка", 0.38, 0.40),
    EvolutionStage(3, "Бактерия", "🦠", "Базовое обучение начато", 0.40, 0.42),
    EvolutionStage(4, "Цианобактерия", "🟢", "Фотосинтез знаний", 0.42, 0.44),
    EvolutionStage(5, "Архея", "🔴", "Экстремальная адаптация", 0.44, 0.45),
    EvolutionStage(6, "Эукариот", "🔵", "Ядро появилось", 0.45, 0.46),
    EvolutionStage(7, "Амеба", "🫧", "Движение к цели", 0.46, 0.47),
    EvolutionStage(8, "Инфузория", "🦠", "Организованная структура", 0.47, 0.48),
    EvolutionStage(9, "Водоросль", "🌿", "Колониальная жизнь", 0.48, 0.49),
    EvolutionStage(10, "Планария", "🪱", "Первая нервная система!", 0.49, 0.50),
    
    # 11-20: Многоклеточная жизнь
    EvolutionStage(11, "Губка", "🧽", "Многоклеточный организм", 0.50, 0.51),
    EvolutionStage(12, "Коралл", "🪸", "Симбиоз моделей", 0.51, 0.52),
    EvolutionStage(13, "Медуза", "🪼", "Плавающая в данных", 0.52, 0.53),
    EvolutionStage(14, "Плоский червь", "🪱", "Двусторонняя симметрия", 0.53, 0.54),
    EvolutionStage(15, "Круглый червь", "🐛", "Сегментация", 0.54, 0.55),
    EvolutionStage(16, "Кольчатый червь", "🪱", "Специализация сегментов", 0.55, 0.56),
    EvolutionStage(17, "Моллюск", "🐚", "Защитная оболочка", 0.56, 0.57),
    EvolutionStage(18, "Осьминог", "🐙", "Высокий интеллект!", 0.57, 0.58),
    EvolutionStage(19, "Морская звезда", "⭐", "Радиальная архитектура", 0.58, 0.59),
    EvolutionStage(20, "Морской еж", "🦔", "Иглокожий защитник", 0.59, 0.60),
    
    # 21-30: Хордовые - Рыбы
    EvolutionStage(21, "Ланцетник", "🐟", "Хорда появилась", 0.60, 0.61),
    EvolutionStage(22, "Минога", "🐍", "Первый позвоночник", 0.61, 0.62),
    EvolutionStage(23, "Акула", "🦈", "Хищное обучение", 0.62, 0.63),
    EvolutionStage(24, "Скат", "◇", "Электрическое чутьё", 0.63, 0.64),
    EvolutionStage(25, "Костная рыба", "🐠", "Прочный скелет", 0.64, 0.65),
    EvolutionStage(26, "Лосось", "🐟", "Миграция к цели", 0.65, 0.66),
    EvolutionStage(27, "Угорь", "🐍", "Гибкая стратегия", 0.66, 0.67),
    EvolutionStage(28, "Двоякодышащая", "🫁", "Дыхание на суше", 0.67, 0.68),
    EvolutionStage(29, "Кистепёрая рыба", "🦴", "Выход на сушу!", 0.68, 0.69),
    EvolutionStage(30, "Тиктаалик", "🐊", "Переходная форма", 0.69, 0.70),
    
    # 31-40: Земноводные
    EvolutionStage(31, "Ихтиостега", "🦎", "Первый тетрапод", 0.70, 0.71),
    EvolutionStage(32, "Головастик", "🐸", "Метаморфоза начата", 0.71, 0.72),
    EvolutionStage(33, "Лягушка", "🐸", "Амфибия освоена", 0.72, 0.73),
    EvolutionStage(34, "Жаба", "🐸", "Наземная адаптация", 0.73, 0.74),
    EvolutionStage(35, "Саламандра", "🦎", "Регенерация ошибок", 0.74, 0.75),
    EvolutionStage(36, "Тритон", "🦎", "Двойная жизнь", 0.75, 0.76),
    EvolutionStage(37, "Червяга", "🪱", "Роющий алгоритм", 0.76, 0.77),
    EvolutionStage(38, "Лабиринтодонт", "🐊", "Зубы лабиринта", 0.77, 0.78),
    EvolutionStage(39, "Сеймурия", "🦎", "Почти рептилия", 0.78, 0.79),
    EvolutionStage(40, "Диадект", "🦎", "Переход к рептилиям", 0.79, 0.80),
    
    # 41-50: Рептилии
    EvolutionStage(41, "Гилономус", "🦎", "Первая рептилия!", 0.80, 0.81),
    EvolutionStage(42, "Ящерица", "🦎", "Чешуйчатый охотник", 0.81, 0.82),
    EvolutionStage(43, "Змея", "🐍", "Безногий хищник", 0.82, 0.83),
    EvolutionStage(44, "Черепаха", "🐢", "Панцирная защита", 0.83, 0.84),
    EvolutionStage(45, "Крокодил", "🐊", "Древний хищник", 0.84, 0.85),
    EvolutionStage(46, "Птерозавр", "🦇", "Полёт освоен", 0.85, 0.86),
    EvolutionStage(47, "Велоцираптор", "🦖", "Умный хищник", 0.86, 0.87),
    EvolutionStage(48, "Трицератопс", "🦏", "Травоядный танк", 0.87, 0.88),
    EvolutionStage(49, "Тираннозавр", "🦖", "Король динозавров!", 0.88, 0.89),
    EvolutionStage(50, "Археоптерикс", "🦅", "Перо появилось", 0.89, 0.90),
    
    # 51-60: Птицы
    EvolutionStage(51, "Воробей", "🐦", "Мелкая птица", 0.90, 0.905),
    EvolutionStage(52, "Ворона", "🐦‍⬛", "Высокий интеллект", 0.905, 0.91),
    EvolutionStage(53, "Попугай", "🦜", "Имитация паттернов", 0.91, 0.915),
    EvolutionStage(54, "Сова", "🦉", "Ночной охотник", 0.915, 0.92),
    EvolutionStage(55, "Орёл", "🦅", "Острое зрение", 0.92, 0.925),
    EvolutionStage(56, "Пингвин", "🐧", "Нырок в глубину", 0.925, 0.93),
    EvolutionStage(57, "Страус", "🦤", "Нелетающий гигант", 0.93, 0.935),
    EvolutionStage(58, "Колибри", "🐦", "Быстрое обучение", 0.935, 0.94),
    EvolutionStage(59, "Альбатрос", "🕊️", "Дальний полёт", 0.94, 0.945),
    EvolutionStage(60, "Сокол", "🦅", "Пикирующий удар", 0.945, 0.95),
    
    # 61-70: Ранние млекопитающие
    EvolutionStage(61, "Морганукодон", "🐭", "Первое млекопитающее", 0.95, 0.952),
    EvolutionStage(62, "Опоссум", "🦨", "Сумчатое", 0.952, 0.954),
    EvolutionStage(63, "Ёж", "🦔", "Колючая защита", 0.954, 0.956),
    EvolutionStage(64, "Мышь", "🐭", "Быстрое размножение", 0.956, 0.958),
    EvolutionStage(65, "Белка", "🐿️", "Запасливость", 0.958, 0.96),
    EvolutionStage(66, "Кролик", "🐰", "Плодовитость", 0.96, 0.962),
    EvolutionStage(67, "Лиса", "🦊", "Хитрый охотник", 0.962, 0.964),
    EvolutionStage(68, "Волк", "🐺", "Стайная тактика", 0.964, 0.966),
    EvolutionStage(69, "Медведь", "🐻", "Мощь и сила", 0.966, 0.968),
    EvolutionStage(70, "Лев", "🦁", "Царь зверей", 0.968, 0.97),
    
    # 71-80: Приматы
    EvolutionStage(71, "Лемур", "🐒", "Примитивный примат", 0.97, 0.972),
    EvolutionStage(72, "Долгопят", "👀", "Большие глаза", 0.972, 0.974),
    EvolutionStage(73, "Капуцин", "🐵", "Использует инструменты!", 0.974, 0.976),
    EvolutionStage(74, "Макака", "🐒", "Социальный интеллект", 0.976, 0.978),
    EvolutionStage(75, "Павиан", "🦧", "Сложная иерархия", 0.978, 0.98),
    EvolutionStage(76, "Гиббон", "🦧", "Брахиация освоена", 0.98, 0.982),
    EvolutionStage(77, "Орангутан", "🦧", "Одиночный мыслитель", 0.982, 0.984),
    EvolutionStage(78, "Горилла", "🦍", "Сила и разум", 0.984, 0.986),
    EvolutionStage(79, "Шимпанзе", "🐵", "99% ДНК человека", 0.986, 0.988),
    EvolutionStage(80, "Бонобо", "🦧", "Эмпатия развита", 0.988, 0.99),
    
    # 81-90: Гоминиды
    EvolutionStage(81, "Сахелантроп", "🦴", "Древнейший гоминид", 0.99, 0.991),
    EvolutionStage(82, "Ардипитек", "🦴", "Прямохождение началось", 0.991, 0.992),
    EvolutionStage(83, "Австралопитек", "🦍", "Люси живёт", 0.992, 0.993),
    EvolutionStage(84, "Человек умелый", "🪨", "Первые орудия!", 0.993, 0.994),
    EvolutionStage(85, "Человек прямоходящий", "🔥", "Огонь освоен", 0.994, 0.995),
    EvolutionStage(86, "Гейдельбергский человек", "🏹", "Охотник с копьём", 0.995, 0.996),
    EvolutionStage(87, "Неандерталец", "🧊", "Ледниковый период", 0.996, 0.997),
    EvolutionStage(88, "Денисовский человек", "🏔️", "Горный житель", 0.997, 0.998),
    EvolutionStage(89, "Кроманьонец", "🎨", "Наскальная живопись", 0.998, 0.999),
    EvolutionStage(90, "Человек разумный", "👤", "Homo sapiens!", 0.999, 1.0),
    
    # 91-100: Цивилизация
    EvolutionStage(91, "Охотник-собиратель", "🏹", "Кочевая жизнь", 1.0, 1.0),
    EvolutionStage(92, "Земледелец", "🌾", "Аграрная революция", 1.0, 1.0),
    EvolutionStage(93, "Ремесленник", "🔨", "Специализация труда", 1.0, 1.0),
    EvolutionStage(94, "Купец", "💰", "Торговля знаниями", 1.0, 1.0),
    EvolutionStage(95, "Учёный", "🔬", "Научный метод", 1.0, 1.0),
    EvolutionStage(96, "Инженер", "⚙️", "Промышленность", 1.0, 1.0),
    EvolutionStage(97, "Программист", "💻", "Цифровая эра", 1.0, 1.0),
    EvolutionStage(98, "AI исследователь", "🤖", "Искусственный интеллект", 1.0, 1.0),
    EvolutionStage(99, "Мастер ML", "🧠", "Совершенное обучение!", 1.0, 1.0),
    EvolutionStage(100, "Сингулярность", "✨", "Превзошёл создателя", 1.0, 1.0),
]

class EvolutionTracker:
    """Трекер эволюции экспертов"""
    
    def __init__(self, output_dir: str = "training_viz"):
        self.output_dir = output_dir
        self.evolution_file = os.path.join(output_dir, "evolution_state.json")
        
        # Текущие уровни экспертов
        self.levels: Dict[str, int] = {
            "XGB": 0,
            "RF": 0,
            "ARF": 0,
            "NN": 0,
            "META": 0
        }
        
        # История уровней для уведомлений (чтобы не спамить)
        self.last_notified: Dict[str, int] = {
            "XGB": -1,
            "RF": -1,
            "ARF": -1,
            "NN": -1,
            "META": -1
        }
        
        self._load_state()
    
    def accuracy_to_level(self, accuracy: float) -> int:
        """Конвертирует accuracy в уровень эволюции (0-100)"""
        # Находим подходящую стадию
        for stage in EVOLUTION_STAGES:
            if stage.min_accuracy <= accuracy < stage.max_accuracy:
                return stage.level
        
        # Для accuracy >= 1.0 возвращаем максимальный уровень
        if accuracy >= 1.0:
            return 100
        
        # Для очень низких accuracy
        return 0
    
    def get_stage(self, level: int) -> EvolutionStage:
        """Получает информацию о стадии по уровню"""
        for stage in EVOLUTION_STAGES:
            if stage.level == level:
                return stage
        return EVOLUTION_STAGES[0]  # Fallback
    
    def update_expert(self, expert_name: str, accuracy: float) -> Optional[EvolutionStage]:
        """
        Обновляет уровень эксперта
        
        Returns:
            EvolutionStage если был переход на новый уровень, иначе None
        """
        new_level = self.accuracy_to_level(accuracy)
        old_level = self.levels.get(expert_name, 0)
        
        self.levels[expert_name] = new_level
        self._save_state()
        
        # Проверяем, нужно ли уведомление (переход через порог 10 уровней)
        if new_level >= old_level + 10:
            return self.get_stage(new_level)
        
        return None
    
    def should_notify(self, expert_name: str) -> bool:
        """Проверяет, нужно ли отправить уведомление"""
        current_level = self.levels.get(expert_name, 0)
        last_level = self.last_notified.get(expert_name, -1)
        
        # Уведомляем каждые 10 уровней
        if current_level >= last_level + 10:
            self.last_notified[expert_name] = current_level
            return True
        
        return False
    
    def _save_state(self):
        """Сохраняет состояние"""
        try:
            data = {
                "levels": self.levels,
                "last_notified": self.last_notified
            }
            with open(self.evolution_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            from error_logger import log_exception
            log_exception("Failed to save JSON")
    
    def _load_state(self):
        """Загружает состояние"""
        try:
            if os.path.exists(self.evolution_file):
                with open(self.evolution_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.levels = data.get("levels", self.levels)
                self.last_notified = data.get("last_notified", self.last_notified)
        except Exception:
            from error_logger import log_exception
            log_exception("Failed to load JSON")

# Глобальный экземпляр
_evolution_tracker: Optional[EvolutionTracker] = None

def get_evolution_tracker(output_dir: str = "training_viz") -> EvolutionTracker:
    """Получить глобальный трекер эволюции"""
    global _evolution_tracker
    if _evolution_tracker is None:
        _evolution_tracker = EvolutionTracker(output_dir)
    return _evolution_tracker