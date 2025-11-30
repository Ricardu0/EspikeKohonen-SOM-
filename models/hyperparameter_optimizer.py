"""
Módulo de otimização de hiperparâmetros
"""

import numpy as np
import logging
from typing import Dict
from models.som_trainer import MemoryEfficientSOMTrainer
from analysis.som_analyzer import KohonenAdvancedAnalyzer
from sklearn.metrics import silhouette_score
from config.settings import OPTIMIZATION_PARAM_GRID

logger = logging.getLogger(__name__)

class SOMHyperparameterOptimizer:
    """Otimizador de hiperparâmetros do SOM"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = None
        self.optimization_history = []

    def optimize_parameters(self, data, param_grid=None, max_evaluations=20):
        """Otimiza hiperparâmetros do SOM usando busca em grade"""
        logger.info("🎯 OTIMIZAÇÃO DE HIPERPARÂMETROS DO SOM")

        if param_grid is None:
            param_grid = OPTIMIZATION_PARAM_GRID

        best_score = -float('inf')
        best_params = None
        evaluations = 0

        param_combinations = self._generate_param_combinations(param_grid)

        for params in param_combinations:
            if evaluations >= max_evaluations:
                break

            try:
                logger.info(f"   🔍 Testando: {params}")

                # Treinar SOM com parâmetros atuais
                trainer = MemoryEfficientSOMTrainer(random_state=self.random_state)
                som, q_error, t_error = trainer.train_kohonen_network(data, **params)

                # ✅ CORREÇÃO: Passar data diretamente (já é array numpy)
                analyzer = KohonenAdvancedAnalyzer()
                analyzer.create_comprehensive_visualizations(som, data)
                neuron_clusters = analyzer.get_neuron_clusters()

                # Mapear pontos para clusters
                cluster_assignments = []
                for sample in data:
                    winner = som.winner(sample)
                    cluster_id = neuron_clusters[winner] if neuron_clusters is not None else 0
                    cluster_assignments.append(cluster_id)

                # Calcular score composto
                score = self._calculate_optimization_score(
                    q_error, t_error, cluster_assignments, data, som
                )

                self.optimization_history.append({
                    'params': params,
                    'q_error': q_error,
                    't_error': t_error,
                    'score': score
                })

                logger.info(f"     ✅ Score: {score:.4f} (QE: {q_error:.4f}, TE: {t_error:.4f})")

                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info("     🎉 Novo melhor score!")

            except Exception as e:
                logger.error(f"     ❌ Erro: {e}")
                continue

            evaluations += 1

        self.best_params = best_params
        logger.info(f"🏆 MELHORES PARÂMETROS: {best_params}")
        logger.info(f"🏆 MELHOR SCORE: {best_score:.4f}")

        return best_params

    def _generate_param_combinations(self, param_grid):
        """Gera combinações de parâmetros para busca em grade"""
        from itertools import product
        keys = param_grid.keys()
        values = param_grid.values()
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def _calculate_optimization_score(self, q_error, t_error, clusters, data, som):
        """Calcula score composto para otimização"""
        # Normalizar erros (menor = melhor)
        q_score = 1.0 / (1.0 + q_error)
        t_score = 1.0 / (1.0 + t_error)

        # Avaliar qualidade de clusters
        valid_mask = np.array(clusters) != 0
        valid_data = data[valid_mask]
        valid_clusters = np.array(clusters)[valid_mask]

        if len(np.unique(valid_clusters)) >= 2:
            try:
                sil_score = silhouette_score(valid_data, valid_clusters)
            except:
                sil_score = 0.0
        else:
            sil_score = 0.0

        # Score composto (pesos podem ser ajustados)
        composite_score = 0.4 * q_score + 0.3 * t_score + 0.3 * sil_score

        return composite_score