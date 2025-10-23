import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import pearsonr

def analyze_feature_correlation(X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str] = None,
                                top_n: int = 20) -> Dict[str, float]:
    """
    Анализирует корреляцию фичей с целевой переменной.
    
    Args:
        X: матрица фичей [n_samples, n_features]
        y: целевая переменная [n_samples]
        feature_names: имена фичей
        top_n: сколько топ фичей показать
        
    Returns:
        Словарь {имя_фичи: корреляция}
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    correlations = {}
    
    print("\n" + "="*60)
    print("📊 АНАЛИЗ КОРРЕЛЯЦИИ ФИЧЕЙ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("="*60)
    
    for i, name in enumerate(feature_names):
        try:
            corr, p_value = pearsonr(X[:, i], y)
            correlations[name] = corr
        except Exception as e:
            print(f"⚠️  Не удалось вычислить корреляцию для {name}: {e}")
            correlations[name] = 0.0
    
    # Сортировка по абсолютной величине корреляции
    sorted_features = sorted(correlations.items(), 
                            key=lambda x: abs(x[1]), 
                            reverse=True)
    
    print(f"\n🔝 TOP-{min(top_n, len(sorted_features))} ФИЧЕЙ ПО КОРРЕЛЯЦИИ:\n")
    for idx, (name, corr) in enumerate(sorted_features[:top_n], 1):
        indicator = "✅" if abs(corr) > 0.05 else "⚠️"
        print(f"{idx:2d}. {indicator} {name:40s} corr={corr:+.4f}")
    
    # Статистика
    strong_corr = sum(1 for c in correlations.values() if abs(c) > 0.05)
    weak_corr = sum(1 for c in correlations.values() if abs(c) <= 0.01)
    
    print(f"\n📈 СТАТИСТИКА:")
    print(f"   Сильная корреляция (|r| > 0.05): {strong_corr}/{len(correlations)}")
    print(f"   Слабая корреляция (|r| ≤ 0.01):  {weak_corr}/{len(correlations)}")
    
    if weak_corr > len(correlations) * 0.5:
        print("\n❌ ПРОБЛЕМА: Более 50% фичей имеют очень слабую корреляцию!")
        print("   Рекомендуется пересмотреть feature engineering.")
    
    return correlations


def check_data_leakage(X: np.ndarray, feature_names: List[str] = None) -> List[str]:
    """
    Проверяет фичи на потенциальную утечку данных из будущего.
    
    Признаки data leakage:
    - Фичи с подозрительно высокими значениями
    - Фичи содержащие '_future', '_next', '_t+1' в названии
    
    Returns:
        Список подозрительных фичей
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    suspicious = []
    
    print("\n" + "="*60)
    print("🔍 ПРОВЕРКА НА DATA LEAKAGE")
    print("="*60 + "\n")
    
    # Проверка названий
    leak_keywords = ['future', 'next', 't+1', 'forward', 'ahead']
    for name in feature_names:
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in leak_keywords):
            suspicious.append(name)
            print(f"⚠️  Подозрительное название: {name}")
    
    # Проверка на идеально предсказывающие фичи (корреляция ~1.0)
    print("\n💡 Фичи с аномально высокими значениями будут показаны выше")
    
    if suspicious:
        print(f"\n❌ НАЙДЕНО {len(suspicious)} ПОДОЗРИТЕЛЬНЫХ ФИЧЕЙ!")
        print("   Проверьте что эти фичи не используют данные из будущего.")
    else:
        print("\n✅ Подозрительных фичей не обнаружено по названиям.")
    
    return suspicious