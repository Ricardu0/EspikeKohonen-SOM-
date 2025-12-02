"""
Módulo avançado de interpretação de clusters para SOM (Self-Organizing Maps)
Versão melhorada com análises mais robustas e visualizações detalhadas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, Optional, Any
from scipy import stats
from analysis.cluster_evaluator import ClusterQualityEvaluator

# Configuração de estilo para visualizações
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class AdvancedSOMClusterInterpreter:
    """Interpretador avançado de clusters baseado em SOM com análises detalhadas"""
    
    def __init__(self, preprocessor, som_trainer, som_analyzer):
        self.preprocessor = preprocessor
        self.som_trainer = som_trainer
        self.som_analyzer = som_analyzer
        self.cluster_profiles = {}
        self.quality_evaluator = ClusterQualityEvaluator()
        self.feature_names = None

    def analyze_som_clusters(self, X, original_df, max_clusters=15, 
                           min_cluster_size_ratio=0.001, 
                           noise_threshold=0.10) -> Tuple[pd.DataFrame, Dict]:
        """
        Analisa clusters baseados no SOM com balanceamento avançado
        
        Args:
            X: DataFrame com features
            original_df: DataFrame original com dados completos
            max_clusters: Número máximo de clusters
            min_cluster_size_ratio: Razão mínima do tamanho do cluster
            noise_threshold: Threshold para considerar como ruído
            
        Returns:
            Tuple com DataFrame enriquecido e métricas de qualidade
        """
        logger.info("🔍 ANÁLISE AVANÇADA DE CLUSTERS DO SOM")
        
        if self.som_trainer.som is None:
            raise ValueError("Rede de Kohonen não treinada!")

        # Preparação dos dados
        data = X.values.astype(np.float32)
        self.feature_names = X.columns.tolist()

        # Obter clusters naturais do SOM
        neuron_clusters = self.som_analyzer.get_neuron_clusters()
        if neuron_clusters is None:
            raise ValueError("Clusters naturais não foram calculados!")

        # Atribuição balanceada de clusters
        balanced_clusters, cluster_metrics = self._advanced_cluster_assignment(
            data, neuron_clusters, max_clusters, 
            min_cluster_size_ratio, noise_threshold
        )

        # Preparar DataFrame final
        result_df = self._prepare_result_dataframe(original_df, balanced_clusters)
        
        # Análises completas
        quality_metrics = self._comprehensive_analysis(result_df, data, balanced_clusters)
        
        return result_df, {**quality_metrics, **cluster_metrics}

    def _advanced_cluster_assignment(self, data, neuron_clusters, max_clusters, 
                                   min_cluster_size_ratio, noise_threshold):
        """Atribuição avançada de clusters com múltiplas estratégias"""
        logger.info("   ⚖️  Atribuição avançada de clusters...")
        
        som = self.som_trainer.som
        neuron_cluster_map = self._create_neuron_cluster_map(neuron_clusters)
        
        # Atribuição inicial
        initial_assignments = self._get_initial_assignments(som, data, neuron_cluster_map)
        
        # Análise da distribuição inicial
        cluster_stats = self._analyze_initial_distribution(initial_assignments)
        
        # Estratégias de balanceamento
        balanced_assignments = self._apply_balancing_strategies(
            initial_assignments, cluster_stats, len(data), 
            max_clusters, min_cluster_size_ratio, noise_threshold
        )
        
        # Métricas do processo
        cluster_metrics = {
            'initial_clusters': len(cluster_stats['valid_clusters']),
            'final_clusters': len(np.unique(balanced_assignments)) - 1,  # Excluir ruído
            'noise_points': np.sum(np.array(balanced_assignments) == 0),
            'retained_data_ratio': np.sum(np.array(balanced_assignments) > 0) / len(data)
        }
        
        logger.info(f"   ✅ Balanceamento concluído: {cluster_metrics['final_clusters']} clusters")
        
        return balanced_assignments, cluster_metrics

    def _create_neuron_cluster_map(self, neuron_clusters):
        """Cria mapeamento neurônio -> cluster"""
        neuron_cluster_map = {}
        for i in range(neuron_clusters.shape[0]):
            for j in range(neuron_clusters.shape[1]):
                cluster_id = neuron_clusters[i, j]
                if cluster_id > 0:
                    neuron_cluster_map[(i, j)] = cluster_id
        return neuron_cluster_map

    def _get_initial_assignments(self, som, data, neuron_cluster_map):
        """Obtém atribuições iniciais dos clusters"""
        assignments = []
        for sample in data:
            winner = som.winner(sample)
            cluster_id = neuron_cluster_map.get(winner, 0)
            assignments.append(cluster_id)
        return assignments

    def _analyze_initial_distribution(self, assignments):
        """Analisa distribuição inicial dos clusters"""
        assignments_array = np.array(assignments)
        unique_clusters, counts = np.unique(assignments_array, return_counts=True)
        
        valid_clusters = []
        cluster_sizes = {}
        
        for cluster_id, count in zip(unique_clusters, counts):
            cluster_sizes[cluster_id] = count
            if cluster_id > 0:  # Excluir cluster 0 (ruído)
                valid_clusters.append(cluster_id)
        
        return {
            'unique_clusters': unique_clusters,
            'counts': counts,
            'valid_clusters': valid_clusters,
            'cluster_sizes': cluster_sizes
        }

    def _apply_balancing_strategies(self, assignments, cluster_stats, total_points,
                                    max_clusters, min_cluster_size_ratio, noise_threshold):
        """
        ✅ BALANCEAMENTO ULTRA-FLEXÍVEL
        """
        # ✅ MUDANÇA: Threshold MUITO mais baixo
        min_cluster_size = max(100, int(total_points * 0.0005))  # 0.05% ou 100 pontos

        assignments_array = np.array(assignments)

        logger.info(f"\n   📊 ANÁLISE DE BALANCEAMENTO:")
        logger.info(f"      • Total de pontos: {total_points:,}")
        logger.info(f"      • Tamanho mínimo: {min_cluster_size:,}")

        # Identificar clusters válidos
        valid_clusters = []
        small_clusters = []
        cluster_info = []

        for cluster_id in cluster_stats['valid_clusters']:
            cluster_size = cluster_stats['cluster_sizes'][cluster_id]
            percentage = (cluster_size / total_points) * 100

            cluster_info.append({
                'id': cluster_id,
                'size': cluster_size,
                'percentage': percentage
            })

            if cluster_size >= min_cluster_size:
                valid_clusters.append(cluster_id)
                logger.info(f"      ✅ Cluster {cluster_id}: {cluster_size:,} ({percentage:.2f}%) - VÁLIDO")
            else:
                small_clusters.append(cluster_id)
                logger.info(f"      ⚠️  Cluster {cluster_id}: {cluster_size:,} ({percentage:.2f}%) - PEQUENO")

        # ✅ NOVO: Se nenhum cluster válido, forçar os 5 maiores
        if len(valid_clusters) == 0:
            logger.warning("      ⚠️  NENHUM cluster válido! Forçando os maiores...")
            sorted_clusters = sorted(cluster_info, key=lambda x: x['size'], reverse=True)
            num_to_keep = min(5, len(sorted_clusters))
            valid_clusters = [c['id'] for c in sorted_clusters[:num_to_keep]]

            for c in sorted_clusters[:num_to_keep]:
                logger.info(f"      🔄 FORÇADO Cluster {c['id']}: {c['size']:,} ({c['percentage']:.2f}%)")

        # Limitar número máximo
        if len(valid_clusters) > max_clusters:
            logger.info(f"      ✂️  Limitando de {len(valid_clusters)} para {max_clusters} clusters")
            sorted_valid = sorted(cluster_info, key=lambda x: x['size'], reverse=True)
            valid_clusters = [c['id'] for c in sorted_valid[:max_clusters] if c['id'] in valid_clusters]

        # Reatribuir pontos
        balanced_assignments = []
        reallocated = 0
        noise_count = 0

        for assignment in assignments:
            if assignment == 0:
                balanced_assignments.append(0)
                noise_count += 1
            elif assignment not in valid_clusters:
                # Realocar para o cluster válido mais próximo (por tamanho)
                if len(valid_clusters) > 0:
                    # Simplificação: usar o maior cluster
                    largest = max(valid_clusters, key=lambda x: cluster_stats['cluster_sizes'][x])
                    balanced_assignments.append(largest)
                    reallocated += 1
                else:
                    balanced_assignments.append(0)
                    noise_count += 1
            else:
                balanced_assignments.append(assignment)

        logger.info(f"\n   ✅ BALANCEAMENTO CONCLUÍDO:")
        logger.info(f"      • Clusters válidos: {len(valid_clusters)}")
        logger.info(f"      • IDs: {sorted(valid_clusters)}")
        logger.info(f"      • Realocados: {reallocated:,}")
        logger.info(f"      • Ruído: {noise_count:,} ({noise_count / total_points * 100:.1f}%)")

        return balanced_assignments

    def _prepare_result_dataframe(self, original_df, clusters):
        """Prepara DataFrame final com clusters e análises"""
        result_df = original_df.iloc[:len(clusters)].copy()
        result_df['CLUSTER_SOM'] = clusters
        result_df['CLUSTER_SIZE'] = result_df['CLUSTER_SOM'].map(
            result_df['CLUSTER_SOM'].value_counts()
        )
        return result_df

    def _comprehensive_analysis(self, df, data, clusters):
        """Executa análise completa dos clusters"""
        logger.info("   📊 Iniciando análise compreensiva...")
        
        # Análise de qualidade
        quality_metrics = self.quality_evaluator.comprehensive_cluster_quality(
            data, clusters, self.som_trainer.som
        )
        
        # Análises detalhadas
        self._advanced_cluster_distribution_analysis(df)
        self._cluster_characteristics_analysis(df)
        self._create_comprehensive_visualizations(df, data, clusters)
        
        return quality_metrics

    def _advanced_cluster_distribution_analysis(self, df):
        """Análise avançada da distribuição de clusters"""
        logger.info("\n📊 DISTRIBUIÇÃO AVANÇADA DOS CLUSTERS")
        
        cluster_dist = df['CLUSTER_SOM'].value_counts().sort_index()
        valid_clusters = cluster_dist[cluster_dist.index != 0]
        
        if len(valid_clusters) == 0:
            logger.warning("   ⚠️  Nenhum cluster válido encontrado!")
            return
        
        # Estatísticas detalhadas
        total_records = len(df)
        noise_count = cluster_dist.get(0, 0)
        clustered_records = valid_clusters.sum()
        
        logger.info(f"   • Clusters válidos: {len(valid_clusters)}")
        logger.info(f"   • Registros em clusters: {clustered_records:,} ({clustered_records/total_records*100:.1f}%)")
        logger.info(f"   • Registros como ruído: {noise_count:,} ({noise_count/total_records*100:.1f}%)")
        logger.info(f"   • Tamanho médio do cluster: {valid_clusters.mean():.0f} registros")
        logger.info(f"   • Desvio padrão: {valid_clusters.std():.0f} registros")
        
        # Identificar clusters outliers
        Q1 = valid_clusters.quantile(0.25)
        Q3 = valid_clusters.quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR
        outliers = valid_clusters[valid_clusters > outlier_threshold]
        
        if len(outliers) > 0:
            logger.info(f"   • Clusters grandes (outliers): {list(outliers.index)}")

    def _cluster_characteristics_analysis(self, df):
        """Análise detalhada das características dos clusters"""
        logger.info("\n📈 ANÁLISE DETALHADA POR CLUSTER")
        
        valid_clusters = sorted([c for c in df['CLUSTER_SOM'].unique() if c != 0])
        
        if not valid_clusters:
            logger.warning("   ⚠️  Nenhum cluster válido para análise!")
            return
        
        for cluster_id in valid_clusters:
            cluster_data = df[df['CLUSTER_SOM'] == cluster_id]
            self._analyze_single_cluster(cluster_data, cluster_id, df)

    def _analyze_single_cluster(self, cluster_data, cluster_id, full_df):
        """Analisa um cluster individual"""
        size = len(cluster_data)
        percentage = (size / len(full_df)) * 100
        
        logger.info(f"\n🎯 CLUSTER {cluster_id}: {size:,} registros ({percentage:.1f}%)")
        logger.info("   " + "─" * 50)
        
        # Análise de features categóricas
        self._analyze_categorical_features(cluster_data, size)
        
        # Análise de features numéricas
        self._analyze_numeric_features(cluster_data, full_df)

    def _analyze_categorical_features(self, cluster_data, cluster_size):
        """Analisa features categóricas do cluster"""
        categorical_insights = {}
        
        for col in cluster_data.select_dtypes(include=['object', 'category']).columns:
            if cluster_data[col].nunique() < 15:  # Limite para evitar alta cardinalidade
                value_counts = cluster_data[col].value_counts()
                top_value = value_counts.head(2)  # Top 2 valores
                
                for value, count in top_value.items():
                    percentage = (count / cluster_size) * 100
                    if percentage > 20:  # Threshold mais baixo para capturar padrões
                        if col not in categorical_insights:
                            categorical_insights[col] = []
                        categorical_insights[col].append((value, percentage))
        
        if categorical_insights:
            logger.info("   🏷️  CARACTERÍSTICAS CATEGÓRICAS:")
            for col, values in list(categorical_insights.items())[:8]:
                insights_str = ", ".join([f"{val} ({pct:.1f}%)" for val, pct in values[:2]])
                logger.info(f"     • {col}: {insights_str}")

    def _analyze_numeric_features(self, cluster_data, full_df):
        """Analisa features numéricas com comparação global"""
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return

        logger.info("   📊 CARACTERÍSTICAS NUMÉRICAS:")

        for col in list(numeric_cols)[:6]:  # Limitar para não poluir
            cluster_mean = cluster_data[col].mean()
            global_mean = full_df[col].mean()

            # ✅ VALIDAÇÃO: Se coordenadas ainda corrompidas, avisar
            if col in ['LATITUDE', 'LONGITUDE'] and abs(cluster_mean) > 1000:
                logger.warning(f"      ⚠️  {col}: AINDA CORROMPIDO ({cluster_mean:.0f})")
                logger.warning(f"          Aplicar correção de escala no preprocessor!")
                continue

            difference_pct = ((cluster_mean - global_mean) / abs(global_mean)) * 100 if global_mean != 0 else 0

            significance = "↑↑" if difference_pct > 15 else "↓↓" if difference_pct < -15 else "≈"

            logger.info(f"     • {col}: {significance} avg={cluster_mean:.1f} "
                        f"(global: {global_mean:.1f}, diff: {difference_pct:+.1f}%)")

    def _create_comprehensive_visualizations(self, df, data, clusters):
        """Cria visualizações abrangentes dos clusters"""
        logger.info("   🎨 Criando visualizações...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribuição de clusters
        ax1 = plt.subplot(2, 3, 1)
        self._plot_cluster_distribution(df, ax1)
        
        # 2. Composição dos clusters (features principais)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_cluster_composition(df, ax2)
        
        # 3. Heatmap de características
        ax3 = plt.subplot(2, 3, 3)
        self._plot_feature_heatmap(df, ax3)
        
        # 4. Dimensionalidade reduzida (se disponível)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_projection(df, data, clusters, ax4)
        
        # 5. Tamanho dos clusters vs qualidade
        ax5 = plt.subplot(2, 3, 5)
        self._plot_cluster_quality(df, ax5)
        
        # 6. Matriz de correlação entre clusters
        ax6 = plt.subplot(2, 3, 6)
        self._plot_cluster_correlation(df, ax6)
        
        plt.tight_layout()
        plt.savefig('advanced_som_cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico adicional: Radar chart para perfis de cluster
        self._create_radar_chart(df)

    def _plot_cluster_distribution(self, df, ax):
        """Plot da distribuição de clusters"""
        cluster_dist = df['CLUSTER_SOM'].value_counts().sort_index()
        valid_clusters = cluster_dist[cluster_dist.index != 0]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_clusters)))
        bars = ax.bar(range(len(valid_clusters)), valid_clusters.values, color=colors)
        
        ax.set_title('Distribuição de Clusters', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Número de Registros')
        ax.set_xticks(range(len(valid_clusters)))
        ax.set_xticklabels(valid_clusters.index, rotation=45)
        
        # Adicionar valores
        for bar, count in zip(bars, valid_clusters.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{count:,}', ha='center', va='bottom', fontsize=9)

    def _plot_cluster_composition(self, df, ax):
        """Plot da composição dos clusters por features principais"""
        # Implementar análise de features mais importantes por cluster
        valid_clusters = sorted([c for c in df['CLUSTER_SOM'].unique() if c != 0])
        
        if len(valid_clusters) == 0:
            ax.text(0.5, 0.5, 'Sem clusters válidos', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Composição dos Clusters', fontsize=14)
            return
        
        # Exemplo simplificado - adaptar conforme necessidade
        composition_data = []
        for cluster_id in valid_clusters[:5]:  # Limitar a 5 clusters
            cluster_data = df[df['CLUSTER_SOM'] == cluster_id]
            # Calcular métricas de composição aqui
            
        ax.set_title('Composição dos Clusters\n(Top Features)', fontsize=14, fontweight='bold')

    def _plot_feature_heatmap(self, df, ax):
        """Heatmap de características dos clusters"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Calcular médias por cluster para heatmap
            cluster_means = df.groupby('CLUSTER_SOM')[numeric_cols[:8]].mean()  # Top 8 features
            
            if len(cluster_means) > 1:
                # Normalizar para melhor visualização
                normalized_means = (cluster_means - cluster_means.mean()) / cluster_means.std()
                sns.heatmap(normalized_means.iloc[1:], ax=ax, cmap='RdBu_r', center=0, 
                           annot=True, fmt='.2f', cbar_kws={'label': 'Z-score'})
                ax.set_title('Heatmap de Características\n(Normalizado)', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Dados insuficientes\npara heatmap', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Sem features numéricas', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Heatmap de Características', fontsize=14)

    def _plot_projection(self, df, data, clusters, ax):
        """Projeção dos clusters em 2D (se disponível)"""
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            
            # Usar PCA ou t-SNE para projeção
            if data.shape[1] > 2:
                projector = PCA(n_components=2, random_state=42)
                projection = projector.fit_transform(data)
                title = 'Projeção PCA dos Clusters'
            else:
                projection = data
                title = 'Visualização Direta dos Clusters'
            
            scatter = ax.scatter(projection[:, 0], projection[:, 1], 
                               c=clusters, cmap='tab10', alpha=0.6, s=30)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            
            # Adicionar legenda para clusters
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
            
        except ImportError:
            ax.text(0.5, 0.5, 'Scikit-learn não disponível\npara projeção', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Projeção dos Clusters', fontsize=14)

    def _plot_cluster_quality(self, df, ax):
        """Plot de qualidade vs tamanho dos clusters"""
        valid_clusters = [c for c in df['CLUSTER_SOM'].unique() if c != 0]
        
        if len(valid_clusters) < 2:
            ax.text(0.5, 0.5, 'Clusters insuficientes\npara análise', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Qualidade vs Tamanho', fontsize=14)
            return
        
        cluster_sizes = []
        cluster_qualities = []  # Métricas de qualidade podem ser adicionadas
        
        for cluster_id in valid_clusters:
            cluster_data = df[df['CLUSTER_SOM'] == cluster_id]
            cluster_sizes.append(len(cluster_data))
            # Calcular métricas de qualidade aqui
        
        ax.scatter(cluster_sizes, cluster_qualities if cluster_qualities else cluster_sizes, 
                  alpha=0.6, s=60)
        ax.set_xlabel('Tamanho do Cluster')
        ax.set_ylabel('Métrica de Qualidade' if cluster_qualities else 'Tamanho')
        ax.set_title('Relação: Tamanho vs Qualidade', fontsize=14, fontweight='bold')

    def _plot_cluster_correlation(self, df, ax):
        """Matriz de correlação entre clusters"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            ax.text(0.5, 0.5, 'Features insuficientes\npara correlação', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Correlação entre Clusters', fontsize=14)
            return
        
        # Calcular correlações médias entre clusters
        valid_clusters = [c for c in df['CLUSTER_SOM'].unique() if c != 0]
        
        if len(valid_clusters) < 2:
            ax.text(0.5, 0.5, 'Clusters insuficientes', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
         # Calcula as médias das features numéricas por cluster
        cluster_means = []
        cluster_labels = []
    
        for c in valid_clusters:
            cluster_data = df[df['CLUSTER_SOM'] == c]
            if len(cluster_data) > 1:  # Precisa ter pelo menos 2 pontos
                # Calcula a média das features numéricas para este cluster
                means = cluster_data[numeric_cols].mean().values
                cluster_means.append(means)
                cluster_labels.append(c)
    
            if len(cluster_means) < 2:
                ax.text(0.5, 0.5, 'Dados insuficientes\npara correlação', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Correlação entre Clusters', fontsize=14)
            return
    
    # Agora todas as arrays têm o mesmo comprimento (número de features)
        cluster_means_array = np.array(cluster_means)  # Shape: (n_clusters, n_features)
    
    # Calcula correlação entre os perfis médios dos clusters
        correlation_matrix = np.corrcoef(cluster_means_array)
    
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(cluster_labels)))
        ax.set_yticks(range(len(cluster_labels)))
        ax.set_xticklabels(cluster_labels)
        ax.set_yticklabels(cluster_labels)
        ax.set_title('Similaridade entre Clusters\n(Correlação dos Perfis Médias)', 
                fontsize=14, fontweight='bold')
    
        # Adiciona valores na matriz
        for i in range(len(cluster_labels)):
            for j in range(len(cluster_labels)):
                ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                   ha='center', va='center', 
                   color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black',
                   fontsize=9)
    
        plt.colorbar(im, ax=ax, label='Coeficiente de Correlação')

    def _create_radar_chart(self, df):
        """Cria gráfico radar para perfis de clusters"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Top 6 features
            
            if len(numeric_cols) < 3:
                return
                
            valid_clusters = [c for c in df['CLUSTER_SOM'].unique() if c != 0][:8]  # Top 8 clusters
            
            if len(valid_clusters) < 2:
                return
            
            # Preparar dados para radar chart
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Ângulos para cada feature
            angles = np.linspace(0, 2*np.pi, len(numeric_cols), endpoint=False).tolist()
            angles += angles[:1]  # Fechar o círculo
            
            for cluster_id in valid_clusters:
                cluster_data = df[df['CLUSTER_SOM'] == cluster_id]
                values = cluster_data[numeric_cols].mean().tolist()
                values += values[:1]  # Fechar o círculo
                
                ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(numeric_cols)
            ax.set_title('Perfil dos Clusters - Radar Chart', size=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.1, 1.1))
            
            plt.savefig('cluster_radar_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"   ⚠️  Não foi possível criar radar chart: {e}")

    def get_cluster_profiles(self) -> Dict:
        """Retorna perfis detalhados dos clusters"""
        return self.cluster_profiles

    def generate_cluster_report(self, df, output_file='cluster_analysis_report.txt'):
        """Gera relatório completo da análise"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE CLUSTERS - SOM\n")
            f.write("=" * 50 + "\n\n")
            
            # Estatísticas básicas
            cluster_dist = df['CLUSTER_SOM'].value_counts().sort_index()
            valid_clusters = cluster_dist[cluster_dist.index != 0]
            
            f.write(f"ESTATÍSTICAS GERAIS:\n")
            f.write(f"- Total de clusters válidos: {len(valid_clusters)}\n")
            f.write(f"- Total de registros: {len(df):,}\n")
            f.write(f"- Registros em clusters: {valid_clusters.sum():,}\n")
            f.write(f"- Registros como ruído: {cluster_dist.get(0, 0):,}\n\n")
            
            # Perfil de cada cluster
            f.write("PERFIS DOS CLUSTERS:\n")
            f.write("-" * 30 + "\n")
            
            for cluster_id in valid_clusters.index:
                cluster_data = df[df['CLUSTER_SOM'] == cluster_id]
                size = len(cluster_data)
                percentage = (size / len(df)) * 100
                
                f.write(f"\nCLUSTER {cluster_id} ({size:,} registros - {percentage:.1f}%):\n")
                
                # Features mais importantes
                for col in cluster_data.select_dtypes(include=['object', 'category']).columns[:3]:
                    if cluster_data[col].nunique() < 10:
                        top_value = cluster_data[col].value_counts().head(1)
                        if len(top_value) > 0:
                            value, count = top_value.index[0], top_value.values[0]
                            pct = (count / size) * 100
                            f.write(f"  • {col}: {value} ({pct:.1f}%)\n")
        
        logger.info(f"   📄 Relatório salvo em: {output_file}")

# Versão de compatibilidade para código existente
SOMClusterInterpreter = AdvancedSOMClusterInterpreter