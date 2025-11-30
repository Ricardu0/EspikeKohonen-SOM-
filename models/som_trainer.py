"""
Módulo de treinamento do SOM com gerenciamento de memória
"""

import numpy as np
from minisom import MiniSom
import logging
import gc
import psutil
import os
from typing import Tuple, Optional, Dict
from config.settings import DEFAULT_SOM_CONFIG

logger = logging.getLogger(__name__)

class MemoryEfficientSOMTrainer:
    """Treinador de SOM com gerenciamento eficiente de memória"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.som = None
        self.batch_size = 5000
        self.dtype = np.float32

    def set_batch_size(self, data_size: int, som_shape: Tuple[int, int]) -> None:
        """Define o tamanho do lote baseado na memória disponível"""
        n_neurons = som_shape[0] * som_shape[1]
        estimated_memory_per_sample = n_neurons * 4  # bytes para float32
        safe_batch_size = min(100000000 // estimated_memory_per_sample, 10000)
        self.batch_size = max(1000, min(safe_batch_size, data_size // 10))
        logger.info(f"Batch size definido para: {self.batch_size}")

    def log_memory_usage(self):
        """Log do uso de memória"""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Uso de memória: {memory_mb:.2f} MB")
        except:
            logger.info("Monitoramento de memória não disponível")

    def get_optimized_som_config(self, data_size: int, input_dim: int) -> Dict:
        """Define configurações otimizadas baseadas no tamanho dos dados"""
        if data_size > 100000:
            som_x, som_y = 25, 25
            learning_rate = 0.3
            sigma = 1.2
            iterations = 1000
        elif data_size > 50000:
            som_x, som_y = 30, 30
            learning_rate = 0.4
            sigma = 1.5
            iterations = 1500
        else:
            som_x, som_y = 35, 35
            learning_rate = 0.5
            sigma = 1.8
            iterations = 2000

        return {
            'som_x': som_x,
            'som_y': som_y,
            'learning_rate': learning_rate,
            'sigma': sigma,
            'iterations': iterations
        }

    def optimize_data_types(self, data: np.ndarray) -> np.ndarray:
        """Converte dados para tipos otimizados"""
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)
        return data

    def safe_quantization_error(self, som, data: np.ndarray) -> float:
        """Calcula erro de quantização em lotes de forma segura"""
        try:
            if len(data) * som._weights.size < 1e8:
                return som.quantization_error(data)
        except MemoryError:
            pass

        return self._batch_quantization_error(som, data)

    def _batch_quantization_error(self, som, data: np.ndarray) -> float:
        """Implementação eficiente com processamento em lotes"""
        total_distance = 0.0
        n_samples = len(data)

        for i in range(0, n_samples, self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_bmus = self._batch_winner(som, batch)
            batch_weights = som._weights[batch_bmus[:, 0], batch_bmus[:, 1]]
            distances = np.linalg.norm(batch - batch_weights, axis=1)
            total_distance += np.sum(distances)

            if i % (self.batch_size * 5) == 0:
                gc.collect()

        return total_distance / n_samples

    def _batch_winner(self, som, batch: np.ndarray) -> np.ndarray:
        """Encontra BMUs em lote de forma eficiente"""
        batch_size = len(batch)
        winners = np.empty((batch_size, 2), dtype=np.int32)

        for i in range(batch_size):
            winners[i] = som.winner(batch[i])

        return winners

    def _safe_topographic_error(self, som, data: np.ndarray) -> float:
        """Calcula erro topográfico de forma segura"""
        try:
            if len(data) < 10000:
                return som.topographic_error(data)
            else:
                sample_size = min(5000, len(data))
                indices = np.random.choice(len(data), sample_size, replace=False)
                return som.topographic_error(data[indices])
        except MemoryError:
            logger.warning("Erro de memória no cálculo topográfico, usando amostra menor")
            sample_size = min(1000, len(data))
            indices = np.random.choice(len(data), sample_size, replace=False)
            return som.topographic_error(data[indices])

    def _train_with_memory_management(self, som, data: np.ndarray, iterations: int):
        """Treinamento com monitoramento de memória"""
        try:
            for iteration in range(iterations):
                # ✅ Pular algumas iterações para acelerar (apenas durante otimização)
                if iterations > 1000 and iteration % 10 == 0 and iteration < iterations - 100:
                    continue
                    
                batch_indices = np.random.choice(
                    len(data),
                    size=min(self.batch_size, len(data)),
                    replace=False
                )
                batch = data[batch_indices]

                som.train_batch(batch, 1, verbose=False)

                if iteration % 200 == 0:  # ✅ Log menos frequente
                    gc.collect()

                if iteration % 1000 == 0:  # ✅ Log menos frequente
                    self.log_memory_usage()
                    logger.info(f"Iteração {iteration}/{iterations}")

        except KeyboardInterrupt:
            logger.info("⏹️  Treinamento interrompido pelo usuário")
            raise

    # ✅ VERIFICAR: Este método deve estar DENTRO da classe, com a indentação correta!
    def train_kohonen_network(self,
                              data: np.ndarray,
                              som_x: Optional[int] = None,
                              som_y: Optional[int] = None,
                              sigma: Optional[float] = None,
                              learning_rate: Optional[float] = None,
                              iterations: Optional[int] = None,
                              random_seed: Optional[int] = None):
        """
        Função principal de treinamento com gerenciamento de memória
        """
        if any(param is None for param in [som_x, som_y, sigma, learning_rate, iterations]):
            config = self.get_optimized_som_config(len(data), data.shape[1])
            som_x = config['som_x'] if som_x is None else som_x
            som_y = config['som_y'] if som_y is None else som_y
            sigma = config['sigma'] if sigma is None else sigma
            learning_rate = config['learning_rate'] if learning_rate is None else learning_rate
            iterations = config['iterations'] if iterations is None else iterations

        logger.info(f"Configuração do SOM: {som_x}x{som_y}, sigma={sigma}, "
                     f"learning_rate={learning_rate}, iterations={iterations}")

        self.set_batch_size(len(data), (som_x, som_y))
        data = self.optimize_data_types(data)

        try:
            som = MiniSom(som_x, som_y, data.shape[1],
                          sigma=sigma,
                          learning_rate=learning_rate,
                          neighborhood_function='gaussian',
                          random_seed=random_seed)

            som._weights = som._weights.astype(self.dtype)

            self._train_with_memory_management(som, data, iterations)

            logger.info("Calculando erro de quantização...")
            q_error = self.safe_quantization_error(som, data)

            logger.info("Calculando erro topográfico...")
            topographic_error = self._safe_topographic_error(som, data)

            self.log_memory_usage()

            self.som = som
            return som, q_error, topographic_error

        except MemoryError as e:
            logger.error(f"Erro de memória durante o treinamento: {e}")
            return self.fallback_training(data)
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            raise

    def fallback_training(self, data: np.ndarray):
        """Fallback para quando as configurações otimizadas falham"""
        logger.info("Usando configuração de fallback...")

        fallback_config = {
            'som_x': 20,
            'som_y': 20,
            'learning_rate': 0.2,
            'sigma': 0.8,
            'iterations': 500
        }

        if len(data) > 30000:
            sample_size = 30000
            indices = np.random.choice(len(data), sample_size, replace=False)
            data = data[indices]
            logger.info(f"Usando amostra de {sample_size} dados para fallback")

        return self.train_kohonen_network(data, **fallback_config)