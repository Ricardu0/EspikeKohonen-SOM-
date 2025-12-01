"""
Módulo avançado de avaliação de qualidade de clusters
"""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging
from typing import Dict, List, Optional, Union, Tuple
import warnings

logger = logging.getLogger(__name__)

class ClusterQualityEvaluator:
    """Avaliador avançado de qualidade de clusters com métricas robustas"""
    
    # Constantes para interpretação
    SILHOUETTE_THRESHOLDS = {
        'excelente': 0.7,
        'bom': 0.5,
        'razoavel': 0.25,
        'pobre': 0.0
    }
    
    BALANCE_THRESHOLDS = {
        'excelente': 0.7,
        'bom': 0.5,
        'razoavel': 0.3,
        'pobre': 0.0
    }

    def __init__(self, random_state: int = 42):
        """
        Inicializa o avaliador de qualidade de clusters
        
        Args:
            random_state: Seed para reproducibilidade
        """
        self.random_state = random_state
        self.metrics_history = {}
        self._scaler = StandardScaler()
        
        # Configurar warnings
        warnings.filterwarnings('ignore', category=UserWarning)

    def comprehensive_cluster_quality(self, 
                                   data: np.ndarray, 
                                   clusters: np.ndarray, 
                                   som=None,
                                   feature_names: Optional[List[str]] = None,
                                   calculate_additional: bool = True) -> Dict[str, float]:
        """
        Avaliação abrangente da qualidade dos clusters
        
        Args:
            data: Array com os dados (n_samples, n_features)
            clusters: Array com labels dos clusters (n_samples,)
            som: Objeto SOM opcional para avaliação específica
            feature_names: Nomes das features para análise descritiva
            calculate_additional: Se deve calcular métricas adicionais
            
        Returns:
            Dicionário com todas as métricas calculadas
        """
        print("\n📊 AVALIAÇÃO COMPREENSIVA DE QUALIDADE DOS CLUSTERS")
        print("=" * 55)

        # Validação dos dados
        data, clusters = self._validate_inputs(data, clusters)
        
        # Filtrar clusters válidos (excluir ruído)
        valid_mask = clusters != -1  # Cluster -1 normalmente representa ruído
        valid_data = data[valid_mask]
        valid_clusters = clusters[valid_mask]

        if len(np.unique(valid_clusters)) < 2:
            logger.warning("Menos de 2 clusters válidos para avaliação")
            print("   ⚠️  Menos de 2 clusters válidos para avaliação")
            return {}

        metrics = {}

        try:
            # 1. Métricas Fundamentais de Validação Interna
            metrics.update(self._calculate_basic_metrics(valid_data, valid_clusters))
            
            # 2. Métricas de Estabilidade e Robustez
            if calculate_additional:
                metrics.update(self._calculate_stability_metrics(data, clusters))
            
            # 3. Métricas de Balanceamento e Distribuição
            metrics.update(self._calculate_distribution_metrics(valid_clusters))
            
            # 4. Métricas Específicas do SOM (se disponível)
            if som is not None:
                metrics.update(self._evaluate_som_quality(som, valid_data, valid_clusters))
            
            # 5. Análise Descritiva dos Clusters
            if feature_names is not None and calculate_additional:
                metrics.update(self._cluster_descriptive_analysis(valid_data, valid_clusters, feature_names))
            
            # 6. Interpretação e Relatório
            self._generate_quality_report(metrics)
            
            # Armazenar no histórico
            self._update_metrics_history(metrics)

        except Exception as e:
            logger.error(f"Erro no cálculo de métricas: {e}", exc_info=True)
            print(f"   ❌ Erro crítico no cálculo de métricas: {e}")

        return metrics

    def _validate_inputs(self, data: np.ndarray, clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Valida e prepara os dados de entrada"""
        data = np.array(data)
        clusters = np.array(clusters).flatten()
        
        if len(data) != len(clusters):
            raise ValueError("Dados e clusters têm tamanhos diferentes")
        
        if data.ndim != 2:
            raise ValueError("Dados devem ser uma matriz 2D")
            
        return data, clusters

    def _calculate_basic_metrics(self, data: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
        """Calcula métricas fundamentais de validação interna"""
        metrics = {}
        
        n_clusters = len(np.unique(clusters))
        n_samples = len(data)
        
        try:
            # Silhouette Score
            if n_samples > 1 and n_clusters > 1:
                metrics['silhouette_score'] = silhouette_score(data, clusters)
            else:
                metrics['silhouette_score'] = -1
                
            # Calinski-Harabasz
            if n_samples > n_clusters and n_clusters > 1:
                metrics['calinski_harabasz'] = calinski_harabasz_score(data, clusters)
            else:
                metrics['calinski_harabasz'] = -1
                
            # Davies-Bouldin
            if n_clusters > 1:
                metrics['davies_bouldin'] = davies_bouldin_score(data, clusters)
            else:
                metrics['davies_bouldin'] = float('inf')
                
        except Exception as e:
            logger.warning(f"Erro no cálculo de métricas básicas: {e}")
            # Valores padrão em caso de erro
            metrics.update({
                'silhouette_score': -1,
                'calinski_harabasz': -1,
                'davies_bouldin': float('inf')
            })
            
        return metrics

    def _calculate_stability_metrics(self, data: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de estabilidade e robustez"""
        metrics = {}
        
        try:
            # Coeficiente de variação intra-cluster
            intra_cluster_vars = []
            for cluster_id in np.unique(clusters):
                cluster_data = data[clusters == cluster_id]
                if len(cluster_data) > 1:
                    cluster_var = np.mean(np.var(cluster_data, axis=0))
                    intra_cluster_vars.append(cluster_var)
            
            if intra_cluster_vars:
                metrics['intra_cluster_variation'] = np.mean(intra_cluster_vars)
                metrics['intra_cluster_variation_cv'] = np.std(intra_cluster_vars) / np.mean(intra_cluster_vars)
            
        except Exception as e:
            logger.warning(f"Erro no cálculo de métricas de estabilidade: {e}")
            
        return metrics

    def _calculate_distribution_metrics(self, clusters: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de distribuição e balanceamento"""
        metrics = {}
        
        cluster_counts = np.bincount(clusters)
        
        # Remove clusters vazios
        cluster_counts = cluster_counts[cluster_counts > 0]
        
        if len(cluster_counts) < 2:
            return metrics
            
        # Score de balanceamento
        metrics['balance_score'] = self._calculate_balance_score(cluster_counts)
        
        # Medidas de desigualdade
        metrics['cluster_imbalance'] = np.std(cluster_counts) / np.mean(cluster_counts)
        metrics['gini_coefficient'] = self._calculate_gini_coefficient(cluster_counts)
        
        # Entropia da distribuição
        proportions = cluster_counts / np.sum(cluster_counts)
        metrics['cluster_entropy'] = -np.sum(proportions * np.log(proportions))
        
        # Tamanho relativo dos clusters
        metrics['size_ratio'] = np.min(cluster_counts) / np.max(cluster_counts)
        
        return metrics

    def _calculate_balance_score(self, cluster_counts: np.ndarray) -> float:
        """Calcula score de balanceamento (0-1, onde 1 é perfeitamente balanceado)"""
        if len(cluster_counts) < 2:
            return 0.0
            
        # Coeficiente de variação invertido (menor variação = mais balanceado)
        cv = np.std(cluster_counts) / np.mean(cluster_counts)
        return 1.0 / (1.0 + cv)

    def _calculate_gini_coefficient(self, cluster_counts: np.ndarray) -> float:
        """Calcula coeficiente de Gini para desigualdade de tamanhos"""
        sorted_counts = np.sort(cluster_counts)
        n = len(sorted_counts)
        index = np.arange(1, n + 1)
        
        return (np.sum((2 * index - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts))

    def _evaluate_som_quality(self, som, data: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
        """Avalia qualidade específica do mapeamento SOM"""
        metrics = {}
        
        try:
            # Quantization error por cluster
            q_errors = []
            topographic_errors = []
            
            for cluster_id in np.unique(clusters):
                cluster_data = data[clusters == cluster_id]
                if len(cluster_data) > 0:
                    try:
                        q_error = som.quantization_error(cluster_data)
                        q_errors.append(q_error)
                        
                        # Erro topográfico (se disponível)
                        if hasattr(som, 'topographic_error'):
                            t_error = som.topographic_error(cluster_data)
                            topographic_errors.append(t_error)
                            
                    except Exception as e:
                        logger.warning(f"Erro cálculo métricas SOM cluster {cluster_id}: {e}")
                        continue

            if q_errors:
                metrics['som_quantization_error'] = np.mean(q_errors)
                metrics['som_quantization_error_std'] = np.std(q_errors)
                
            if topographic_errors:
                metrics['som_topographic_error'] = np.mean(topographic_errors)
                
        except Exception as e:
            logger.warning(f"Erro na avaliação SOM: {e}")
            
        return metrics

    def _cluster_descriptive_analysis(self, data: np.ndarray, clusters: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, float]:
        """Análise descritiva dos clusters"""
        metrics = {}
        
        try:
            # Separação inter-cluster (distância entre centróides)
            centroids = []
            for cluster_id in np.unique(clusters):
                cluster_data = data[clusters == cluster_id]
                centroids.append(np.mean(cluster_data, axis=0))
            
            if len(centroids) > 1:
                # Distância média entre centróides
                centroid_distances = []
                for i in range(len(centroids)):
                    for j in range(i + 1, len(centroids)):
                        distance = np.linalg.norm(centroids[i] - centroids[j])
                        centroid_distances.append(distance)
                
                metrics['mean_centroid_distance'] = np.mean(centroid_distances)
                metrics['min_centroid_distance'] = np.min(centroid_distances)
                
        except Exception as e:
            logger.warning(f"Erro na análise descritiva: {e}")
            
        return metrics

    def _generate_quality_report(self, metrics: Dict[str, float]):
        """Gera relatório completo de qualidade"""
        print("\n   📈 RELATÓRIO DE QUALIDADE DOS CLUSTERS")
        print("   " + "-" * 45)
        
        # Métricas de Validação Interna
        self._print_validation_metrics(metrics)
        
        # Métricas de Distribuição
        self._print_distribution_metrics(metrics)
        
        # Métricas SOM (se disponíveis)
        self._print_som_metrics(metrics)
        
        # Score Geral (heurístico)
        overall_score = self._calculate_overall_score(metrics)
        print(f"\n   🎯 SCORE GERAL: {overall_score:.2f}/10.0")
        self._interpret_overall_score(overall_score)

    def _print_validation_metrics(self, metrics: Dict[str, float]):
        """Imprime métricas de validação interna"""
        print("   • VALIDAÇÃO INTERNA:")
        
        if 'silhouette_score' in metrics and metrics['silhouette_score'] != -1:
            sil = metrics['silhouette_score']
            interpretation = self._interpret_silhouette(sil)
            print(f"     📊 Silhouette: {sil:.4f} ({interpretation})")
            
        if 'calinski_harabasz' in metrics and metrics['calinski_harabasz'] != -1:
            ch = metrics['calinski_harabasz']
            print(f"     📈 Calinski-Harabasz: {ch:.2f} (↑ melhor)")
            
        if 'davies_bouldin' in metrics and metrics['davies_bouldin'] != float('inf'):
            db = metrics['davies_bouldin']
            print(f"     📉 Davies-Bouldin: {db:.4f} (↓ melhor)")

    def _print_distribution_metrics(self, metrics: Dict[str, float]):
        """Imprime métricas de distribuição"""
        print("   • DISTRIBUIÇÃO:")
        
        if 'balance_score' in metrics:
            balance = metrics['balance_score']
            interpretation = self._interpret_balance(balance)
            print(f"     ⚖️  Balance Score: {balance:.4f} ({interpretation})")
            
        if 'cluster_entropy' in metrics:
            entropy = metrics['cluster_entropy']
            print(f"     🔢 Entropia: {entropy:.4f} (↑ diversidade)")
            
        if 'size_ratio' in metrics:
            ratio = metrics['size_ratio']
            print(f"     📏 Razão Tamanhos: {ratio:.4f} (↑ balanceado)")

    def _print_som_metrics(self, metrics: Dict[str, float]):
        """Imprime métricas específicas do SOM"""
        som_metrics = [k for k in metrics.keys() if k.startswith('som_')]
        if som_metrics:
            print("   • QUALIDADE SOM:")
            for metric in som_metrics:
                value = metrics[metric]
                print(f"     🧮 {metric}: {value:.4f}")

    def _interpret_silhouette(self, score: float) -> str:
        """Interpreta o silhouette score"""
        if score >= self.SILHOUETTE_THRESHOLDS['excelente']:
            return "EXCELENTE"
        elif score >= self.SILHOUETTE_THRESHOLDS['bom']:
            return "BOM"
        elif score >= self.SILHOUETTE_THRESHOLDS['razoavel']:
            return "RAZOÁVEL"
        else:
            return "POBRE"

    def _interpret_balance(self, score: float) -> str:
        """Interpreta o balance score"""
        if score >= self.BALANCE_THRESHOLDS['excelente']:
            return "EXCELENTE"
        elif score >= self.BALANCE_THRESHOLDS['bom']:
            return "BOM"
        elif score >= self.BALANCE_THRESHOLDS['razoavel']:
            return "RAZOÁVEL"
        else:
            return "POBRE"

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calcula score geral heurístico (0-10)"""
        scores = []
        
        # Silhouette (0-3 pontos)
        if 'silhouette_score' in metrics and metrics['silhouette_score'] != -1:
            sil = max(0, metrics['silhouette_score'])
            scores.append(sil * 3)
            
        # Balance (0-3 pontos)
        if 'balance_score' in metrics:
            balance = metrics['balance_score']
            scores.append(balance * 3)
            
        # Davies-Bouldin (0-2 pontos) - invertido pois menor é melhor
        if 'davies_bouldin' in metrics and metrics['davies_bouldin'] != float('inf'):
            db = metrics['davies_bouldin']
            db_score = max(0, (2 - min(db, 2)) / 2) * 2
            scores.append(db_score)
            
        # Entropia (0-2 pontos)
        if 'cluster_entropy' in metrics:
            entropy = metrics['cluster_entropy']
            max_entropy = np.log(len([k for k in metrics.keys() if 'cluster' in k]))  # estimativa
            if max_entropy > 0:
                entropy_score = (entropy / max_entropy) * 2
                scores.append(entropy_score)
                
        return min(10, sum(scores)) if scores else 0

    def _interpret_overall_score(self, score: float):
        """Interpreta o score geral"""
        if score >= 8:
            interpretation = "✅ EXCELENTE - Clusters de alta qualidade"
        elif score >= 6:
            interpretation = "✅ BOM - Clusters bem definidos"
        elif score >= 4:
            interpretation = "⚠️  RAZOÁVEL - Clusters aceitáveis"
        else:
            interpretation = "❌ POBRE - Clusters precisam de ajustes"
            
        print(f"   {interpretation}")

    def _update_metrics_history(self, metrics: Dict[str, float]):
        """Atualiza o histórico de métricas"""
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Retorna resumo estatístico do histórico de métricas"""
        summary = {}
        
        for metric, values in self.metrics_history.items():
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_observations': len(values)
                }
                
        return summary

# Exemplo de uso
if __name__ == "__main__":
    # Teste básico do módulo
    evaluator = ClusterQualityEvaluator()
    
    # Dados de exemplo
    X = np.random.randn(100, 5)
    labels = np.random.randint(0, 3, 100)
    
    # Avaliação
    metrics = evaluator.comprehensive_cluster_quality(
        data=X, 
        clusters=labels,
        feature_names=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']
    )
    
    print(f"\nMétricas calculadas: {len(metrics)}")