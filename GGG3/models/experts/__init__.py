"""
ML Experts Package

4 эксперта для предсказания вероятности роста цены:
- XGBoostExpert: Gradient boosting
- RandomForestExpert: Ensemble of decision trees
- AdaptiveRFExpert: Online learning with River
- NeuralNetworkExpert: Simplified deep learning with PyTorch (2.7K params)
"""

from .xgb_expert import XGBoostExpert
from .rf_expert import RandomForestExpert
from .arf_expert import AdaptiveRFExpert
# ✅ ИЗМЕНЕНО: Используем упрощенную версию NeuralNet (11K → 2.7K params)
from .simplified_nn_expert import SimplifiedNeuralNetworkExpert as NeuralNetworkExpert

__all__ = [
    'XGBoostExpert',
    'RandomForestExpert',
    'AdaptiveRFExpert',
    'NeuralNetworkExpert'
]
