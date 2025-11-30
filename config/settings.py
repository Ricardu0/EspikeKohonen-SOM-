"""
Configurações globais e constantes
"""

# Configurações de estilo para melhor visualização
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

RANDOM_STATE = 42

# Configurações do SOM - OTIMIZADAS
DEFAULT_SOM_CONFIG = {
    'small_dataset': {'som_x': 20, 'som_y': 20, 'sigma': 1.0, 'learning_rate': 0.5, 'iterations': 1000},
    'medium_dataset': {'som_x': 25, 'som_y': 25, 'sigma': 1.2, 'learning_rate': 0.4, 'iterations': 1500},
    'large_dataset': {'som_x': 30, 'som_y': 30, 'sigma': 1.5, 'learning_rate': 0.3, 'iterations': 2000}
}

# Configurações de otimização - OTIMIZADAS
OPTIMIZATION_PARAM_GRID = {
    'som_x': [15, 20, 25],
    'som_y': [15, 20, 25],
    'sigma': [0.8, 1.0, 1.2],
    'learning_rate': [0.3, 0.5, 0.7],
    'iterations': [500, 1000, 1500]  # ✅ Menos iterações
}