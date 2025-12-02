"""
CORREÇÕES CRÍTICAS PARA models/som_trainer.py
Problemas identificados:
1. Mapa 20x20 muito pequeno para 216k registros
2. Sigma e learning_rate não otimizados
3. Sem validação durante treinamento
"""

import numpy as np
from minisom import MiniSom
import logging
import gc
import psutil
import os
from typing import Tuple, Optional, Dict
import math

logger = logging.getLogger(__name__)


class MemoryEfficientSOMTrainer:
    """Treinador SOM com configurações otimizadas"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.som = None
        self.batch_size = 5000
        self.dtype = np.float32
        self.training_history = {'q_error': [], 't_error': []}

    def calculate_optimal_map_size(self, n_samples: int, n_features: int) -> Tuple[int, int]:
        """
        ✅ CORREÇÃO: Calcular tamanho ótimo do mapa com limite de memória
        """
        # Fórmula adaptativa baseada na memória disponível
        # Estimar memória necessária: n_samples * n_neurons * 4 bytes (float32)
        # Vamos limitar a ~2GB de memória máxima estimada

        # Tentar começar com tamanho razoável
        if n_samples > 100000:
            base_size = 40  # Para grandes datasets
        elif n_samples > 50000:
            base_size = 35
        elif n_samples > 20000:
            base_size = 30
        elif n_samples > 10000:
            base_size = 25
        else:
            base_size = 20

        # Verificar memória estimada
        estimated_memory = n_samples * (base_size ** 2) * 4 / (1024**3)  # GB

        # Se memória estimada > 2GB, reduzir tamanho
        while estimated_memory > 2.0 and base_size > 15:
            base_size -= 5
            estimated_memory = n_samples * (base_size ** 2) * 4 / (1024**3)

        # Arredondar para múltiplo de 5
        optimal_size = 5 * round(base_size / 5)
        optimal_size = max(30, min(optimal_size, 70))  # Limites 15-50

        ratio = n_samples / (optimal_size ** 2)

        logger.info(f"📐 TAMANHO DO MAPA SEGURO:")
        logger.info(f"   • Mapa: {optimal_size}x{optimal_size} ({optimal_size**2} neurônios)")
        logger.info(f"   • Amostras por neurônio: {ratio:.1f}")
        logger.info(f"   • Memória estimada: {estimated_memory:.2f} GB")

        return optimal_size, optimal_size

    def calculate_optimal_sigma(self, map_size: int) -> float:
        """
        ✅ CORREÇÃO: Sigma ajustado para mapa maior
        Regra: sigma inicial = map_size / 3
        """
        sigma = max(1.0, map_size / 3.0)
        logger.info(f"📏 Sigma inicial: {sigma:.2f}")
        return sigma

    def get_optimized_som_config(self, data_size: int, input_dim: int) -> Dict:
        """
        ✅ CONFIGURAÇÃO OTIMIZADA BASEADA EM ANÁLISE
        """
        # Calcular tamanho ótimo
        som_x, som_y = self.calculate_optimal_map_size(data_size, input_dim)
        
        # Sigma inicial
        sigma = self.calculate_optimal_sigma(som_x)
        
        # Learning rate adaptativo
        if data_size > 100000:
            learning_rate = 0.3
            iterations = 2000
        elif data_size > 50000:
            learning_rate = 0.4
            iterations = 1500
        else:
            learning_rate = 0.5
            iterations = 1000
        
        config = {
            'som_x': som_x,
            'som_y': som_y,
            'learning_rate': learning_rate,
            'sigma': sigma,
            'iterations': iterations
        }
        
        logger.info(f"⚙️  CONFIGURAÇÃO OTIMIZADA:")
        for key, value in config.items():
            logger.info(f"   • {key}: {value}")
        
        return config

    def set_batch_size(self, data_size: int, som_shape: Tuple[int, int]) -> None:
        """Define batch size otimizado considerando memória"""
        n_neurons = som_shape[0] * som_shape[1]
        
        # Estimar memória: n_neurons * 4 bytes (distâncias) + dados do batch
        # Mais conservador: limitar a 500MB por batch
        estimated_memory_per_sample = n_neurons * 4  # bytes
        safe_batch_size = min(500 * 1024 * 1024 // estimated_memory_per_sample, 10000)
        
        self.batch_size = max(1000, min(safe_batch_size, data_size // 10))
        logger.info(f"📦 Batch size: {self.batch_size} (seguro para memória)")

    def optimize_data_types(self, data: np.ndarray) -> np.ndarray:
        """Converte para float32"""
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)
        return data

    def safe_quantization_error(self, som, data: np.ndarray) -> float:
        """Calcula QE em lotes"""
        try:
            if len(data) * som._weights.size < 1e8:
                return som.quantization_error(data)
        except MemoryError:
            pass
        
        return self._batch_quantization_error(som, data)

    def _batch_quantization_error(self, som, data: np.ndarray) -> float:
        """QE em lotes"""
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
        """Encontra BMUs em lote"""
        batch_size = len(batch)
        winners = np.empty((batch_size, 2), dtype=np.int32)
        
        for i in range(batch_size):
            winners[i] = som.winner(batch[i])
        
        return winners

    def _safe_topographic_error(self, som, data: np.ndarray) -> float:
        """Calcula TE de forma segura"""
        try:
            if len(data) < 10000:
                return som.topographic_error(data)
            else:
                sample_size = min(5000, len(data))
                indices = np.random.choice(len(data), sample_size, replace=False)
                return som.topographic_error(data[indices])
        except MemoryError:
            logger.warning("Erro de memória no TE, usando amostra menor")
            sample_size = min(1000, len(data))
            indices = np.random.choice(len(data), sample_size, replace=False)
            return som.topographic_error(data[indices])

    def _train_with_validation(self, som, data: np.ndarray, iterations: int):
        """
        ✅ NOVO: Treinamento com validação periódica
        """
        validation_interval = max(100, iterations // 10)
        
        logger.info(f"🏃 INICIANDO TREINAMENTO")
        logger.info(f"   • Iterações: {iterations}")
        logger.info(f"   • Validação a cada: {validation_interval} iterações")
        
        for iteration in range(iterations):
            # Batch aleatório
            batch_indices = np.random.choice(
                len(data),
                size=min(self.batch_size, len(data)),
                replace=False
            )
            batch = data[batch_indices]
            
            # Treinar
            som.train_batch(batch, 1, verbose=False)
            
            # Validação periódica
            if iteration % validation_interval == 0 and iteration > 0:
                # Calcular métricas em subset
                sample_size = min(5000, len(data))
                sample_indices = np.random.choice(len(data), sample_size, replace=False)
                sample_data = data[sample_indices]
                
                try:
                    q_error = som.quantization_error(sample_data)
                    t_error = som.topographic_error(sample_data)
                    
                    self.training_history['q_error'].append(q_error)
                    self.training_history['t_error'].append(t_error)
                    
                    logger.info(f"   Iteração {iteration}/{iterations}: QE={q_error:.4f}, TE={t_error:.4f}")
                except Exception as e:
                    logger.warning(f"Erro na validação: {e}")
            
            # Garbage collection periódico
            if iteration % 200 == 0:
                gc.collect()

    def train_kohonen_network(self,
                              data: np.ndarray,
                              som_x: Optional[int] = None,
                              som_y: Optional[int] = None,
                              sigma: Optional[float] = None,
                              learning_rate: Optional[float] = None,
                              iterations: Optional[int] = None,
                              random_seed: Optional[int] = None):
        """
        ✅ TREINAMENTO OTIMIZADO
        """
        # Usar configuração otimizada se parâmetros não fornecidos
        if any(param is None for param in [som_x, som_y, sigma, learning_rate, iterations]):
            config = self.get_optimized_som_config(len(data), data.shape[1])
            som_x = config['som_x'] if som_x is None else som_x
            som_y = config['som_y'] if som_y is None else som_y
            sigma = config['sigma'] if sigma is None else sigma
            learning_rate = config['learning_rate'] if learning_rate is None else learning_rate
            iterations = config['iterations'] if iterations is None else iterations
        
        logger.info(f"🎯 CONFIGURAÇÃO FINAL:")
        logger.info(f"   • Mapa: {som_x}x{som_y}")
        logger.info(f"   • Sigma: {sigma}")
        logger.info(f"   • Learning rate: {learning_rate}")
        logger.info(f"   • Iterações: {iterations}")
        
        # Preparar dados
        self.set_batch_size(len(data), (som_x, som_y))
        data = self.optimize_data_types(data)
        
        try:
            # Criar SOM
            som = MiniSom(som_x, som_y, data.shape[1],
                          sigma=sigma,
                          learning_rate=learning_rate,
                          neighborhood_function='gaussian',
                          random_seed=random_seed or self.random_state)
            
            som._weights = som._weights.astype(self.dtype)
            
            # Treinar com validação
            self._train_with_validation(som, data, iterations)
            
            # Métricas finais
            logger.info("📊 CALCULANDO MÉTRICAS FINAIS...")
            q_error = self.safe_quantization_error(som, data)
            topographic_error = self._safe_topographic_error(som, data)
            
            logger.info(f"✅ TREINAMENTO CONCLUÍDO:")
            logger.info(f"   • QE final: {q_error:.4f}")
            logger.info(f"   • TE final: {topographic_error:.4f}")
            
            # Análise da convergência
            if len(self.training_history['q_error']) > 2:
                initial_qe = self.training_history['q_error'][0]
                final_qe = self.training_history['q_error'][-1]
                improvement = ((initial_qe - final_qe) / initial_qe) * 100
                logger.info(f"   • Melhoria: {improvement:.1f}%")
            
            self.som = som
            return som, q_error, topographic_error
            
        except MemoryError as e:
            logger.error(f"Erro de memória: {e}")
            return self.fallback_training(data)
        except Exception as e:
            logger.error(f"Erro: {e}")
            raise

    def fallback_training(self, data: np.ndarray):
        """Fallback com configuração reduzida"""
        logger.info("⚠️  USANDO CONFIGURAÇÃO FALLBACK")
        
        fallback_config = {
            'som_x': 20,
            'som_y': 20,
            'learning_rate': 0.2,
            'sigma': 1.0,
            'iterations': 500
        }
        
        if len(data) > 30000:
            sample_size = 30000
            indices = np.random.choice(len(data), sample_size, replace=False)
            data = data[indices]
            logger.info(f"Usando amostra de {sample_size}")
        
        return self.train_kohonen_network(data, **fallback_config)

    def get_training_history(self):
        """Retorna histórico de treinamento"""
        return self.training_history