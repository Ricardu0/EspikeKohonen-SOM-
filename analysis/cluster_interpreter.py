"""
Módulo de interpretação de clusters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from analysis.cluster_evaluator import ClusterQualityEvaluator

logger = logging.getLogger(__name__)

class SOMClusterInterpreter:
    """Interpretador de clusters baseado apenas no SOM (SEM K-MEANS)"""

    def __init__(self, preprocessor, som_trainer, som_analyzer):
        self.preprocessor = preprocessor
        self.som_trainer = som_trainer
        self.som_analyzer = som_analyzer
        self.cluster_profiles = {}
        self.quality_evaluator = ClusterQualityEvaluator()

    def analyze_som_clusters(self, X, original_df, max_clusters=15, min_cluster_size_ratio=0.01):
        """Analisa clusters baseados apenas no SOM com balanceamento"""
        logger.info("🔍 ANÁLISE DE CLUSTERS DO SOM (SEM K-MEANS)")

        if self.som_trainer.som is None:
            raise ValueError("Rede de Kohonen não treinada!")

        data = X.values.astype(np.float32)

        # Obter clusters naturais do SOM
        neuron_clusters = self.som_analyzer.get_neuron_clusters()

        if neuron_clusters is None:
            raise ValueError("Clusters naturais não foram calculados!")

        # Mapear pontos para clusters com balanceamento
        balanced_clusters = self._balance_cluster_assignment(
            self.som_trainer.som, data, neuron_clusters,
            max_clusters, min_cluster_size_ratio
        )

        # Adicionar clusters ao DataFrame original
        original_df = original_df.iloc[:len(balanced_clusters)].copy()
        original_df['CLUSTER_SOM'] = balanced_clusters

        # Avaliar qualidade dos clusters
        quality_metrics = self.quality_evaluator.comprehensive_cluster_quality(
            data, balanced_clusters, self.som_trainer.som
        )

        # Análise de distribuição
        self._analyze_cluster_distribution(original_df)

        # Análise detalhada por cluster
        self._analyze_cluster_characteristics(original_df)

        return original_df, quality_metrics

    def _balance_cluster_assignment(self, som, data, neuron_clusters, max_clusters, min_cluster_size_ratio):
        """Balanceia a atribuição de clusters para evitar desbalanceamento"""
        logger.info("   ⚖️  Balanceando atribuição de clusters...")

        # Contar pontos por neurônio inicialmente
        initial_assignments = []
        neuron_cluster_map = {}

        # Criar mapeamento neurônio -> cluster
        for i in range(neuron_clusters.shape[0]):
            for j in range(neuron_clusters.shape[1]):
                cluster_id = neuron_clusters[i, j]
                if cluster_id > 0:  # Ignorar background (cluster 0)
                    neuron_cluster_map[(i, j)] = cluster_id

        # Atribuição inicial
        for sample in data:
            winner = som.winner(sample)
            cluster_id = neuron_cluster_map.get(winner, 0)
            initial_assignments.append(cluster_id)

        # Analisar distribuição inicial
        unique_clusters, counts = np.unique(initial_assignments, return_counts=True)
        total_points = len(data)

        logger.info(f"   📊 Distribuição inicial: {len(unique_clusters)} clusters")

        # Balancear clusters muito pequenos ou muito grandes
        balanced_assignments = self._redistribute_clusters(
            initial_assignments, counts, total_points,
            max_clusters, min_cluster_size_ratio
        )

        return balanced_assignments

    def _redistribute_clusters(self, assignments, counts, total_points, max_clusters, min_cluster_size_ratio):
        """Redistribui pontos para balancear clusters"""
        min_cluster_size = int(total_points * 0.005)  # 0.5% do total

        # Identificar clusters válidos (acima do tamanho mínimo)
        valid_clusters = []
        cluster_sizes = {}

        for cluster_id in np.unique(assignments):
            if cluster_id == 0:
                continue  # Pular cluster de background

            cluster_mask = np.array(assignments) == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_sizes[cluster_id] = cluster_size

            if cluster_size >= min_cluster_size:
                valid_clusters.append(cluster_id)

        # Limitar número máximo de clusters
        if len(valid_clusters) > max_clusters:
            # Manter os maiores clusters
            valid_clusters = sorted(valid_clusters,
                                    key=lambda x: cluster_sizes[x],
                                    reverse=True)[:max_clusters]

        # Reatribuir pontos de clusters inválidos para o cluster mais próximo válido
        balanced_assignments = []
        for assignment in assignments:
            if assignment == 0 or assignment not in valid_clusters:
                # Reatribuir para cluster válido mais próximo (ou manter como 0)
                balanced_assignments.append(0)  # Ou implementar lógica de reassociação
            else:
                balanced_assignments.append(assignment)

        logger.info(f"   ✅ Clusters balanceados: {len(valid_clusters)} clusters válidos")
        return balanced_assignments

    def _analyze_cluster_distribution(self, df):
        """Analisa e visualiza a distribuição dos clusters"""
        cluster_dist = df['CLUSTER_SOM'].value_counts().sort_index()

        # Filtrar cluster 0 (ruído/background)
        filtered_dist = cluster_dist[cluster_dist.index != 0]

        logger.info(f"\n📊 DISTRIBUIÇÃO DOS CLUSTERS (excluindo ruído):")
        logger.info(f"   • Total de clusters: {len(filtered_dist)}")
        logger.info(f"   • Registros em clusters: {filtered_dist.sum():,}")
        logger.info(f"   • Registros como ruído: {cluster_dist.get(0, 0):,}")

        # Visualização melhorada
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Gráfico de barras
        colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_dist)))
        bars = ax1.bar(range(len(filtered_dist)), filtered_dist.values, color=colors)
        ax1.set_title('Distribuição de Registros por Cluster', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Número de Registros')
        ax1.set_xticks(range(len(filtered_dist)))
        ax1.set_xticklabels(filtered_dist.index)

        # Adicionar valores nas barras
        for bar, count in zip(bars, filtered_dist.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{count:,}', ha='center', va='bottom', fontweight='bold')

        # Gráfico de pizza (apenas para clusters significativos)
        significant_clusters = filtered_dist[filtered_dist > filtered_dist.sum() * 0.01]  # >1%
        if len(significant_clusters) > 1:
            ax2.pie(significant_clusters.values, labels=significant_clusters.index,
                    autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(significant_clusters))))
            ax2.set_title('Proporção dos Clusters Principais (>1%)', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Clusters muito\ndesbalanceados\npara visualização',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Distribuição de Clusters', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('som_cluster_distribution_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_cluster_characteristics(self, df):
        """Analisa características de cada cluster"""
        logger.info("\n📈 CARACTERÍSTICAS POR CLUSTER:")

        # Filtrar cluster 0 (ruído)
        valid_clusters = sorted([c for c in df['CLUSTER_SOM'].unique() if c != 0])

        for cluster_id in valid_clusters:
            cluster_data = df[df['CLUSTER_SOM'] == cluster_id]
            size = len(cluster_data)
            percentage = (size / len(df)) * 100

            logger.info(f"\n🔸 CLUSTER {cluster_id}: {size:,} registros ({percentage:.1f}%)")
            logger.info("   " + "─" * 40)

            # Análise de características principais
            self._analyze_cluster_features(cluster_data, cluster_id, df)

    def _analyze_cluster_features(self, cluster_data, cluster_id, full_df):
        """Analisa features mais importantes de cada cluster"""
        categorical_insights = {}

        for col in cluster_data.select_dtypes(include=['object']).columns:
            if cluster_data[col].nunique() < 20:
                top_value = cluster_data[col].value_counts().head(1)
                if len(top_value) > 0:
                    value, count = top_value.index[0], top_value.values[0]
                    percentage = (count / len(cluster_data)) * 100
                    if percentage > 25:  # Limite mais baixo para capturar mais padrões
                        categorical_insights[col] = (value, percentage)

        if categorical_insights:
            logger.info("   🏷️  Características principais:")
            for col, (value, percentage) in list(categorical_insights.items())[:6]:  # Mais características
                logger.info(f"     • {col}: {value} ({percentage:.1f}%)")

        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.info("   📊 Estatísticas numéricas:")
            for col in list(numeric_cols)[:4]:  # Mais colunas numéricas
                stats = cluster_data[col].describe()
                logger.info(f"     • {col}: avg={stats['mean']:.1f}, min={stats['min']:.1f}, max={stats['max']:.1f}")