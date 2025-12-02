import numpy as np
import logging
import time
from typing import Dict, List, Optional
from models.som_trainer import MemoryEfficientSOMTrainer
from analysis.som_analyzer import KohonenAdvancedAnalyzer
from sklearn.metrics import silhouette_score
from config.settings import OPTIMIZATION_PARAM_GRID

logger = logging.getLogger(__name__)

class SOMHyperparameterOptimizer:
    """Otimizador de hiperparâmetros do SOM"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params: Optional[Dict] = None
        self.best_score: float = -float('inf')
        self.optimization_history: List[Dict] = []

    def optimize_parameters(
        self,
        data: np.ndarray,
        param_grid: Optional[Dict] = None,
        max_evaluations: int = 20,
        score_weights: Dict[str, float] = None
    ) -> Optional[Dict]:
        """Otimiza hiperparâmetros do SOM usando busca em grade"""
        logger.info("🎯 OTIMIZAÇÃO DE HIPERPARÂMETROS DO SOM")

        if not isinstance(data, np.ndarray):
            raise ValueError("Os dados devem ser um numpy.ndarray")

        if param_grid is None:
            param_grid = OPTIMIZATION_PARAM_GRID

        if score_weights is None:
            score_weights = {"q": 0.4, "t": 0.3, "sil": 0.3}

        evaluations = 0
        param_combinations = self._generate_param_combinations(param_grid)

        for params in param_combinations:
            if evaluations >= max_evaluations:
                break

            start_time = time.time()
            try:
                logger.info(f"   🔍 Testando: {params}")

                trainer = MemoryEfficientSOMTrainer(random_state=self.random_state)
                som, q_error, t_error = trainer.train_kohonen_network(data, **params)

                analyzer = KohonenAdvancedAnalyzer()
                analyzer.create_comprehensive_visualizations(som, data)
                neuron_clusters = analyzer.get_neuron_clusters()

                cluster_assignments = self._map_samples_to_clusters(data, som, neuron_clusters)

                score = self._calculate_optimization_score(
                    q_error, t_error, cluster_assignments, data, score_weights
                )

                self.optimization_history.append({
                    'params': params,
                    'q_error': q_error,
                    't_error': t_error,
                    'score': score
                })

                elapsed = time.time() - start_time
                logger.info(f"     ✅ Score: {score:.4f} (QE: {q_error:.4f}, TE: {t_error:.4f}) em {elapsed:.2f}s")

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    logger.info("     🎉 Novo melhor score!")

            except Exception as e:
                logger.error(f"     ❌ Erro: {e}")

            finally:
                evaluations += 1

        logger.info(f"🏆 MELHORES PARÂMETROS: {self.best_params}")
        logger.info(f"🏆 MELHOR SCORE: {self.best_score:.4f}")

        return self.best_params

    def _generate_param_combinations(self, param_grid: Dict):
        """Gera combinações de parâmetros para busca em grade"""
        from itertools import product
        keys = param_grid.keys()
        values = param_grid.values()
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def _map_samples_to_clusters(self, data: np.ndarray, som, neuron_clusters: Optional[Dict]) -> List[int]:
        """Mapeia cada amostra para o cluster correspondente"""
        cluster_assignments = []
        for sample in data:
            winner = som.winner(sample)
            cluster_id = neuron_clusters[winner] if neuron_clusters is not None else -1
            cluster_assignments.append(cluster_id)
        return cluster_assignments

    def _calculate_optimization_score(
        self,
        q_error: float,
        t_error: float,
        clusters: List[int],
        data: np.ndarray,
        weights: Dict[str, float]
    ) -> float:
        """Calcula score composto para otimização"""
        q_score = 1.0 / (1.0 + q_error)
        t_score = 1.0 / (1.0 + t_error)

        valid_mask = np.array(clusters) >= 0
        valid_data = data[valid_mask]
        valid_clusters = np.array(clusters)[valid_mask]

        if len(np.unique(valid_clusters)) >= 2:
            try:
                sil_score = silhouette_score(valid_data, valid_clusters)
            except Exception:
                sil_score = 0.0
        else:
            sil_score = 0.0

        composite_score = (
            weights["q"] * q_score +
            weights["t"] * t_score +
            weights["sil"] * sil_score
        )

        return composite_score
