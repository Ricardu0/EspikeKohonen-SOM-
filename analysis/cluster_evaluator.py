"""
Módulo de avaliação de qualidade de clusters
"""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)

class ClusterQualityEvaluator:
    """Avaliador avançado de qualidade de clusters"""

    def __init__(self):
        self.metrics_history = {}

    def comprehensive_cluster_quality(self, data, clusters, som=None):
        """Avaliação abrangente da qualidade dos clusters"""
        print("\n📊 AVALIAÇÃO DE QUALIDADE DOS CLUSTERS")
        print("=" * 45)

        # Filtrar clusters válidos (excluir ruído)
        valid_mask = np.array(clusters) != 0
        valid_data = data[valid_mask]
        valid_clusters = np.array(clusters)[valid_mask]

        if len(np.unique(valid_clusters)) < 2:
            print("   ⚠️  Menos de 2 clusters válidos para avaliação")
            return {}

        metrics = {}

        try:
            # 1. Métricas de Coesão e Separação
            metrics['silhouette_score'] = silhouette_score(valid_data, valid_clusters)
            metrics['calinski_harabasz'] = calinski_harabasz_score(valid_data, valid_clusters)
            metrics['davies_bouldin'] = davies_bouldin_score(valid_data, valid_clusters)

            # 2. Métricas de Balanceamento
            cluster_counts = np.bincount(valid_clusters)
            metrics['balance_score'] = self._calculate_balance_score(cluster_counts)
            metrics['cluster_imbalance'] = np.std(cluster_counts) / np.mean(cluster_counts)

            # 3. Métricas Específicas do SOM (se disponível)
            if som is not None:
                metrics['som_quality'] = self._evaluate_som_quality(som, valid_data, valid_clusters)

            # 4. Interpretação das Métricas
            self._interpret_metrics(metrics)

        except Exception as e:
            print(f"   ⚠️  Erro no cálculo de métricas: {e}")

        return metrics

    def _calculate_balance_score(self, cluster_counts):
        """Calcula score de balanceamento (0-1, onde 1 é perfeitamente balanceado)"""
        if len(cluster_counts) < 2:
            return 0.0
        # Coeficiente de variação invertido (menor variação = mais balanceado)
        cv = np.std(cluster_counts) / np.mean(cluster_counts)
        return 1.0 / (1.0 + cv)

    def _evaluate_som_quality(self, som, data, clusters):
        """Avalia qualidade específica do mapeamento SOM"""
        try:
            # Quantization error por cluster
            q_errors = []
            for cluster_id in np.unique(clusters):
                cluster_data = data[clusters == cluster_id]
                if len(cluster_data) > 0:
                    q_error = som.quantization_error(cluster_data)
                    q_errors.append(q_error)

            return np.mean(q_errors) if q_errors else float('inf')
        except:
            return float('inf')

    def _interpret_metrics(self, metrics):
        """Interpreta e explica as métricas de qualidade"""
        print("   📈 MÉTRICAS DE QUALIDADE:")

        if 'silhouette_score' in metrics:
            sil = metrics['silhouette_score']
            if sil > 0.7:
                interpretation = "EXCELENTE - Clusters bem separados"
            elif sil > 0.5:
                interpretation = "BOA - Estrutura razoável de clusters"
            elif sil > 0.25:
                interpretation = "FRACA - Clusters pouco definidos"
            else:
                interpretation = "PÉSSIMA - Sem estrutura clara"
            print(f"     • Silhouette Score: {sil:.4f} ({interpretation})")

        if 'balance_score' in metrics:
            balance = metrics['balance_score']
            if balance > 0.7:
                interpretation = "EXCELENTE - Clusters bem balanceados"
            elif balance > 0.5:
                interpretation = "BOA - Balanceamento razoável"
            else:
                interpretation = "RUIM - Clusters desbalanceados"
            print(f"     • Balance Score: {balance:.4f} ({interpretation})")

        if 'calinski_harabasz' in metrics:
            ch = metrics['calinski_harabasz']
            print(f"     • Calinski-Harabasz: {ch:.2f} (maior = melhor)")