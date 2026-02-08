from itertools import product
import numpy as np

def generate_patterns(pattern_length, max_offset):
    """
    Генерация случайных паттернов длины len с максимальным смещением max_offset
    """
    if pattern_length < 2:
        raise ValueError("Длина паттерна должна быть больше 1")
    patterns = set()
    for combo in product(range(1, max_offset + 1), repeat=pattern_length - 1):
        combo_array = np.array(combo)
        if len(np.unique(combo_array)) == pattern_length - 1:
            sorted_combo = np.sort(combo_array)[::-1]
            pattern = np.append(sorted_combo, 0)
            patterns.add(tuple(pattern))
    return [np.array(p) for p in patterns]

def generate_z_vectors(data, patterns):
    """
    Генерация z векторов для данных data и паттернов patterns
    """
    data = np.array(data)
    z_vectors = []
    for pattern in patterns:
        pattern = np.array(pattern)
        indices = len(data) - pattern - 1
        z_vector = data[indices]
        z_vectors.append(z_vector)
    return np.array(z_vectors)

def generate_lorenz_series(sigma=10.0, rho=28.0, beta=8/3, n_points=1000, dt=0.01, initial_conditions=None):
    """
    Генерация временного ряда на основе уравнений Лоренца
    
    Параметры:
    sigma: параметр Прандтля (по умолчанию 10.0)
    rho: параметр Рэлея (по умолчанию 28.0)
    beta: параметр геометрии (по умолчанию 8/3)
    n_points: количество элементов временного ряда
    dt: шаг времени для интегрирования (по умолчанию 0.01)
    initial_conditions: начальные условия [x0, y0, z0] (по умолчанию [1.0, 1.0, 1.0])
    
    Возвращает:
    numpy array с временным рядом координаты x
    """
    if initial_conditions is None:
        initial_conditions = np.array([1.0, 1.0, 1.0])
    else:
        initial_conditions = np.array(initial_conditions)
    
    state = initial_conditions.copy()
    series = np.zeros(n_points)
    series[0] = state[0]
    
    for i in range(1, n_points):
        x, y, z = state
        
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        
        k1 = dt * np.array([dx_dt, dy_dt, dz_dt])
        
        k2_state = state + k1 / 2
        k2 = dt * np.array([
            sigma * (k2_state[1] - k2_state[0]),
            k2_state[0] * (rho - k2_state[2]) - k2_state[1],
            k2_state[0] * k2_state[1] - beta * k2_state[2]
        ])
        
        k3_state = state + k2 / 2
        k3 = dt * np.array([
            sigma * (k3_state[1] - k3_state[0]),
            k3_state[0] * (rho - k3_state[2]) - k3_state[1],
            k3_state[0] * k3_state[1] - beta * k3_state[2]
        ])
        
        k4_state = state + k3
        k4 = dt * np.array([
            sigma * (k4_state[1] - k4_state[0]),
            k4_state[0] * (rho - k4_state[2]) - k4_state[1],
            k4_state[0] * k4_state[1] - beta * k4_state[2]
        ])
        
        state += (k1 + 2*k2 + 2*k3 + k4) / 6
        series[i] = state[0]
    
    return series

def zscore_normalize(series, mean=None, std=None):
    if mean is None:
        mean = np.mean(series)
    if std is None:
        std = np.std(series)
    return (series - mean) / std, mean, std

def min_max_normalize(series, min=None, max=None):
    if min is None:
        min = np.min(series)
    if max is None:
        max = np.max(series)
    return (series - min) / (max - min), min, max

if __name__ == "__main__":
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    patterns = [np.array([4, 1, 0])]
    # print(generate_patterns(3, 5))
    # print(generate_z_vectors(data, patterns))
    # print(generate_lorenz_series(n_points=10))