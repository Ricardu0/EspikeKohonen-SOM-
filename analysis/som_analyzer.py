"""
Módulo avançado de análise e visualização do SOM (Self-Organizing Maps)
Versão melhorada com análises mais profundas e visualizações profissionais
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label, gaussian_filter, gaussian_gradient_magnitude
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import logging
from typing import Dict, Tuple, Optional, List, Any
import warnings

warnings.filterwarnings('ignore')

# Configuração de estilo para visualizações
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class KohonenAdvancedAnalyzer:
    """Analisador avançado para visualizações e interpretação do SOM"""
    
    def __init__(self):
        self.umatrix = None
        self.activation_map = None
        self.natural_clusters = None
        self.component_planes = None
        self.quality_metrics = {}
        self.feature_names = None

    def create_comprehensive_visualizations(self, som, X, original_features=None, 
                                         feature_names=None):
        """
        Cria visualizações abrangentes e profissionais da rede de Kohonen
        
        Args:
            som: Rede SOM treinada
            X: Dados de entrada (DataFrame ou array)
            original_features: DataFrame original com features completas
            feature_names: Nomes das features para labeling
        """
        logger.info("🎨 GERANDO VISUALIZAÇÕES AVANÇADAS DO SOM")

        if som is None:
            raise ValueError("Rede SOM não treinada!")

        # Preparação dos dados
        data = self._prepare_data(X)
        self.feature_names = feature_names or getattr(X, 'columns', None)

        # Calcular métricas de qualidade
        self._calculate_quality_metrics(som, data)

        # Criar visualizações em grid organizado
        self._create_analysis_dashboard(som, data, original_features)

        logger.info("✅ Visualizações avançadas salvas com sucesso!")

    def _prepare_data(self, X):
        """Prepara dados para análise"""
        if hasattr(X, 'values'):
            return X.values.astype(np.float32)
        return X.astype(np.float32)

    def _calculate_quality_metrics(self, som, data):
        """Calcula métricas de qualidade do SOM"""
        logger.info("   📊 Calculando métricas de qualidade...")
        
        try:
            # Quantization Error
            q_error = som.quantization_error(data)
            self.quality_metrics['quantization_error'] = q_error
            
            # Topographic Error (aproximado)
            t_error = self._calculate_topographic_error(som, data)
            self.quality_metrics['topographic_error'] = t_error
            
            # Distância média na U-Matrix
            if hasattr(som, 'distance_map'):
                umatrix = som.distance_map()
                self.quality_metrics['umatrix_mean'] = np.mean(umatrix)
                self.quality_metrics['umatrix_std'] = np.std(umatrix)
            
            logger.info(f"   ✅ Métricas: QE={q_error:.4f}, TE={t_error:.4f}")
            
        except Exception as e:
            logger.warning(f"   ⚠️  Erro no cálculo de métricas: {e}")

    def _calculate_topographic_error(self, som, data, sample_fraction=0.1):
        """Calcula erro topográfico aproximado"""
        try:
            n_samples = int(len(data) * sample_fraction)
            indices = np.random.choice(len(data), n_samples, replace=False)
            topographic_errors = 0
            
            for idx in indices:
                sample = data[idx]
                w = som.winner(sample)
                # Encontrar segundo BMU seria mais preciso, mas esta é uma aproximação
                distances = np.linalg.norm(som._weights - sample, axis=2)
                sorted_indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)
                
                # Verificar se os dois BMUs mais próximos são adjacentes
                bmu1 = (sorted_indices[0][0], sorted_indices[1][0])
                bmu2 = (sorted_indices[0][1], sorted_indices[1][1])
                
                # Verificar adjacência (simplificado)
                distance_bmus = np.sqrt((bmu1[0]-bmu2[0])**2 + (bmu1[1]-bmu2[1])**2)
                if distance_bmus > 1.5:  # Não são adjacentes
                    topographic_errors += 1
            
            return topographic_errors / n_samples
            
        except Exception:
            return 0.0  # Fallback

    def _create_analysis_dashboard(self, som, data, original_features):
        """Cria dashboard completo de análise"""
        logger.info("   📈 Criando dashboard de análise...")
        
        # Figura principal com subplots organizados
        fig = plt.figure(figsize=(25, 20))
        
        # 1. U-Matrix avançada
        ax1 = plt.subplot(3, 4, 1)
        self._create_enhanced_umatrix(som, ax1)
        
        # 2. Mapa de ativação
        ax2 = plt.subplot(3, 4, 2)
        self._create_enhanced_activation_map(som, data, ax2)
        
        # 3. Clusters naturais
        ax3 = plt.subplot(3, 4, 3)
        self.natural_clusters = self._create_natural_clusters(som, data, ax3)
        
        # 4. Component Planes (primeiras 4 features)
        ax4 = plt.subplot(3, 4, 4)
        self._create_component_planes(som, ax4, num_components=4)
        
        # 5. Análise de qualidade
        ax5 = plt.subplot(3, 4, 5)
        self._create_quality_visualization(ax5)
        
        # 6. Histograma de ativação
        ax6 = plt.subplot(3, 4, 6)
        self._create_activation_histogram(ax6)
        
        # 7. Mapa de gradiente
        ax7 = plt.subplot(3, 4, 7)
        self._create_gradient_map(som, ax7)
        
        # 8. Projeção dos dados (se possível)
        ax8 = plt.subplot(3, 4, 8)
        self._create_data_projection(data, ax8)
        
        # 9. Matriz de correlação de neurônios
        ax9 = plt.subplot(3, 4, 9)
        self._create_neuron_correlation(som, ax9)
        
        # 10. Evolução do aprendizado (se disponível)
        ax10 = plt.subplot(3, 4, 10)
        self._create_learning_analysis(ax10)
        
        # 11. Distribuição de clusters
        ax11 = plt.subplot(3, 4, 11)
        self._create_cluster_distribution(ax11)
        
        # 12. Resumo executivo
        ax12 = plt.subplot(3, 4, 12)
        self._create_executive_summary(ax12)
        
        plt.tight_layout()
        plt.savefig('kohonen_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualizações adicionais em arquivos separados
        self._create_individual_visualizations(som, data, original_features)

    def _create_enhanced_umatrix(self, som, ax):
        """Cria U-Matrix avançada com múltiplas camadas de informação"""
        try:
            if hasattr(som, 'distance_map'):
                self.umatrix = som.distance_map().T
            else:
                # Calcular U-Matrix manualmente se necessário
                self.umatrix = self._compute_umatrix(som)
            
            # Plot principal
            im = ax.imshow(self.umatrix, cmap='viridis', aspect='auto', 
                          interpolation='hanning')
            
            # Adicionar contornos
            levels = np.linspace(self.umatrix.min(), self.umatrix.max(), 15)
            contour = ax.contour(self.umatrix, levels=levels, colors='white', 
                               alpha=0.3, linewidths=0.5)
            
            ax.set_title('U-Matrix: Mapa de Distâncias\n(Estrutura Topológica)', 
                        fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Coordenada X')
            ax.set_ylabel('Coordenada Y')
            
            # Adicionar barra de cores
            plt.colorbar(im, ax=ax, label='Distância Média', shrink=0.8)
            
        except Exception as e:
            self._plot_error(ax, f"Erro na U-Matrix: {str(e)}")

    def _compute_umatrix(self, som):
        """Calcula U-Matrix manualmente se necessário"""
        weights = som._weights
        umatrix = np.zeros((weights.shape[0], weights.shape[1]))
        
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                neighbors = []
                # Coletar vizinhos
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < weights.shape[0] and 0 <= nj < weights.shape[1]:
                        neighbors.append(weights[ni, nj])
                
                if neighbors:
                    dists = [np.linalg.norm(weights[i,j] - n) for n in neighbors]
                    umatrix[i,j] = np.mean(dists)
        
        return umatrix

    def _create_enhanced_activation_map(self, som, data, ax):
        """Cria mapa de ativação avançado"""
        try:
            # Calcular mapa de ativação
            self.activation_map = np.zeros((som._weights.shape[0], som._weights.shape[1]))
            
            # Processamento em batch para eficiência
            batch_size = min(5000, len(data))
            for i in range(0, len(data), batch_size):
                end_idx = min(i + batch_size, len(data))
                batch = data[i:end_idx]
                
                for sample in batch:
                    winner = som.winner(sample)
                    self.activation_map[winner] += 1
            
            # Suavização para melhor visualização
            smoothed_activation = gaussian_filter(self.activation_map, sigma=0.8)
            
            # Plot
            im = ax.imshow(smoothed_activation.T, cmap='hot', aspect='auto',
                          interpolation='bicubic')
            
            # Adicionar contornos de densidade
            if np.max(smoothed_activation) > 0:
                levels = np.linspace(smoothed_activation.min(), 
                                   smoothed_activation.max(), 8)
                contour = ax.contour(smoothed_activation.T, levels=levels, 
                                   colors='white', alpha=0.5, linewidths=1)
            
            ax.set_title('Mapa de Densidade de Ativação\n(Pontos por Neurônio)', 
                        fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Coordenada X')
            ax.set_ylabel('Coordenada Y')
            
            plt.colorbar(im, ax=ax, label='Densidade de Pontos', shrink=0.8)
            
        except Exception as e:
            self._plot_error(ax, f"Erro no mapa de ativação: {str(e)}")

    def _create_natural_clusters(self, som, data, ax, density_threshold=0.6):
        """Identifica clusters naturais baseados em densidade"""
        try:
            # Normalizar mapa de ativação
            if np.max(self.activation_map) > 0:
                normalized_activation = self.activation_map / np.max(self.activation_map)
            else:
                normalized_activation = self.activation_map.copy()
            
            # Suavizar para melhor detecção de clusters
            smoothed_density = gaussian_filter(normalized_activation, sigma=1.0)
            
            # Identificar regiões de alta densidade
            high_density_mask = smoothed_density > density_threshold
            
            # Encontrar componentes conectados
            labeled_array, num_clusters = label(high_density_mask)
            
            logger.info(f"   📊 Identificados {num_clusters} clusters naturais")
            
            # Visualização
            background = ax.imshow(self.umatrix, cmap='gray', alpha=0.4, aspect='auto')
            
            # Criar mapa de cores para clusters
            colors = plt.cm.tab20(np.linspace(0, 1, num_clusters + 1))
            
            cluster_sizes = []
            valid_clusters = []
            
            for cluster_id in range(1, num_clusters + 1):
                cluster_mask = labeled_array == cluster_id
                y_coords, x_coords = np.where(cluster_mask)
                
                if len(x_coords) >= 2:  # Cluster deve ter pelo menos 2 neurônios
                    # Calcular estatísticas do cluster
                    centroid_x = np.mean(x_coords)
                    centroid_y = np.mean(y_coords)
                    cluster_size = len(x_coords)
                    
                    cluster_sizes.append(cluster_size)
                    valid_clusters.append(cluster_id)
                    
                    # Plotar cluster
                    scatter = ax.scatter(x_coords, y_coords, 
                                       color=colors[cluster_id],
                                       label=f'C{cluster_id}',
                                       alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
                    
                    # Adicionar label do cluster
                    ax.text(centroid_x, centroid_y, str(cluster_id),
                           fontsize=9, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle="circle,pad=0.2", facecolor='white', 
                                   alpha=0.8, edgecolor='black'))
            
            ax.set_title(f'Clusters Naturais por Densidade\n{len(valid_clusters)} Clusters Identificados',
                        fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Coordenada X')
            ax.set_ylabel('Coordenada Y')
            
            # Legenda se não for muitos clusters
            if len(valid_clusters) <= 15:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            return labeled_array
            
        except Exception as e:
            self._plot_error(ax, f"Erro nos clusters: {str(e)}")
            return None

    def _create_component_planes(self, som, ax, num_components=4):
        """Cria visualização de component planes para features importantes"""
        try:
            weights = som._weights
            
            if self.feature_names is None:
                self.feature_names = [f'Feature_{i}' for i in range(weights.shape[2])]
            
            # Selecionar features com maior variância
            feature_vars = np.var(weights.reshape(-1, weights.shape[2]), axis=0)
            top_features = np.argsort(feature_vars)[-num_components:][::-1]
            
            # Configurar subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, (feature_idx, ax_comp) in enumerate(zip(top_features, axes)):
                feature_plane = weights[:, :, feature_idx]
                
                im = ax_comp.imshow(feature_plane.T, cmap='coolwarm', aspect='auto',
                                  interpolation='nearest')
                
                feature_name = self.feature_names[feature_idx]
                ax_comp.set_title(f'{feature_name}\n(Var: {feature_vars[feature_idx]:.3f})',
                                fontsize=10, fontweight='bold')
                ax_comp.set_xlabel('X')
                ax_comp.set_ylabel('Y')
                
                plt.colorbar(im, ax=ax_comp, shrink=0.8)
            
            plt.tight_layout()
            plt.savefig('kohonen_component_planes.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Para o dashboard, mostrar apenas o primeiro component plane
            if len(top_features) > 0:
                first_feature = top_features[0]
                feature_plane = weights[:, :, first_feature]
                
                im = ax.imshow(feature_plane.T, cmap='coolwarm', aspect='auto')
                ax.set_title(f'Component Plane: {self.feature_names[first_feature]}',
                           fontsize=11, fontweight='bold', pad=10)
                ax.set_xlabel('Coordenada X')
                ax.set_ylabel('Coordenada Y')
                plt.colorbar(im, ax=ax, label='Valor do Peso', shrink=0.8)
            
        except Exception as e:
            self._plot_error(ax, f"Erro nos component planes: {str(e)}")

    def _create_quality_visualization(self, ax):
        """Visualização das métricas de qualidade"""
        try:
            metrics = self.quality_metrics
            if not metrics:
                ax.text(0.5, 0.5, 'Métricas não disponíveis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Métricas de Qualidade', fontsize=11)
                return
            
            # Preparar dados para bar plot
            metric_names = []
            metric_values = []
            colors = []
            
            for name, value in metrics.items():
                metric_names.append(name.replace('_', ' ').title())
                metric_values.append(value)
                # Cores baseadas no tipo de métrica
                if 'error' in name.lower():
                    colors.append('lightcoral')
                else:
                    colors.append('lightsteelblue')
            
            # Criar bar plot
            bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title('Métricas de Qualidade do SOM', fontsize=11, fontweight='bold', pad=10)
            ax.set_ylabel('Valor da Métrica')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
        except Exception as e:
            self._plot_error(ax, f"Erro nas métricas: {str(e)}")

    def _create_activation_histogram(self, ax):
        """Histograma da distribuição de ativações"""
        try:
            if self.activation_map is not None:
                activations = self.activation_map.flatten()
                activations = activations[activations > 0]  # Filtrar zeros
                
                if len(activations) > 0:
                    ax.hist(activations, bins=20, alpha=0.7, color='skyblue', 
                           edgecolor='black')
                    ax.set_title('Distribuição de Ativações por Neurônio', 
                               fontsize=11, fontweight='bold', pad=10)
                    ax.set_xlabel('Número de Pontos')
                    ax.set_ylabel('Frequência')
                    ax.grid(True, alpha=0.3)
                    
                    # Adicionar estatísticas
                    mean_act = np.mean(activations)
                    ax.axvline(mean_act, color='red', linestyle='--', 
                              label=f'Média: {mean_act:.1f}')
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'Sem ativações', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Mapa de ativação não disponível', 
                       ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            self._plot_error(ax, f"Erro no histograma: {str(e)}")

    def _create_gradient_map(self, som, ax):
        """Mapa de gradiente para mostrar fronteiras"""
        try:
            if self.umatrix is not None:
                gradient = gaussian_gradient_magnitude(self.umatrix, sigma=1.0)
                
                im = ax.imshow(gradient, cmap='Reds', aspect='auto')
                ax.set_title('Mapa de Gradiente\n(Fronteiras entre Clusters)',
                           fontsize=11, fontweight='bold', pad=10)
                ax.set_xlabel('Coordenada X')
                ax.set_ylabel('Coordenada Y')
                plt.colorbar(im, ax=ax, label='Magnitude do Gradiente', shrink=0.8)
            else:
                ax.text(0.5, 0.5, 'U-Matrix não disponível', 
                       ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            self._plot_error(ax, f"Erro no gradiente: {str(e)}")

    def _create_data_projection(self, data, ax):
        """Projeção dos dados em 2D usando PCA/t-SNE"""
        try:
            if len(data) > 1000:
                # Amostrar para performance
                indices = np.random.choice(len(data), 1000, replace=False)
                sample_data = data[indices]
            else:
                sample_data = data
            
            if sample_data.shape[1] > 2:
                # Usar PCA para redução dimensional
                pca = PCA(n_components=2, random_state=42)
                projection = pca.fit_transform(sample_data)
                method = 'PCA'
            else:
                projection = sample_data
                method = 'Original'
            
            scatter = ax.scatter(projection[:, 0], projection[:, 1], 
                               alpha=0.6, s=20, c='purple')
            ax.set_title(f'Projeção dos Dados ({method})', 
                        fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel(f'Componente 1')
            ax.set_ylabel(f'Componente 2')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self._plot_error(ax, f"Erro na projeção: {str(e)}")

    def _create_neuron_correlation(self, som, ax):
        """Matriz de correlação entre neurônios"""
        try:
            weights = som._weights
            flattened_weights = weights.reshape(-1, weights.shape[2])
            
            # Calcular correlação entre vetores de peso
            correlation_matrix = np.corrcoef(flattened_weights)
            
            im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', 
                          vmin=-1, vmax=1)
            ax.set_title('Correlação entre Neurônios', 
                        fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Índice do Neurônio')
            ax.set_ylabel('Índice do Neurônio')
            plt.colorbar(im, ax=ax, label='Coeficiente de Correlação', shrink=0.8)
            
        except Exception as e:
            self._plot_error(ax, f"Erro na correlação: {str(e)}")

    def _create_learning_analysis(self, ax):
        """Análise do processo de aprendizado (placeholder)"""
        ax.text(0.5, 0.5, 'Análise de Aprendizado\n(Não disponível nesta versão)',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Análise do Processo de Aprendizado', 
                    fontsize=11, fontweight='bold', pad=10)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    def _create_cluster_distribution(self, ax):
        """Distribuição de tamanhos dos clusters"""
        try:
            if self.natural_clusters is not None:
                cluster_ids, cluster_sizes = np.unique(self.natural_clusters, return_counts=True)
                
                # Filtrar cluster 0 (background) e clusters pequenos
                valid_mask = (cluster_ids > 0) & (cluster_sizes >= 2)
                valid_clusters = cluster_ids[valid_mask]
                valid_sizes = cluster_sizes[valid_mask]
                
                if len(valid_clusters) > 0:
                    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_clusters)))
                    bars = ax.bar(valid_clusters, valid_sizes, color=colors, alpha=0.7)
                    
                    ax.set_title('Distribuição de Tamanhos dos Clusters',
                               fontsize=11, fontweight='bold', pad=10)
                    ax.set_xlabel('ID do Cluster')
                    ax.set_ylabel('Número de Neurônios')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Adicionar valores
                    for bar, size in zip(bars, valid_sizes):
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                               f'{size}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'Sem clusters válidos', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Clusters não calculados', 
                       ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            self._plot_error(ax, f"Erro na distribuição: {str(e)}")

    def _create_executive_summary(self, ax):
        """Resumo executivo das análises"""
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        summary_text = "RESUMO EXECUTIVO\n\n"
        
        # Métricas de qualidade
        if self.quality_metrics:
            q_error = self.quality_metrics.get('quantization_error', 'N/A')
            t_error = self.quality_metrics.get('topographic_error', 'N/A')
            summary_text += f"Qualidade:\n"
            summary_text += f"• Erro de Quantização: {q_error:.4f}\n"
            summary_text += f"• Erro Topográfico: {t_error:.4f}\n\n"
        
        # Informações de clusters
        if self.natural_clusters is not None:
            cluster_ids = np.unique(self.natural_clusters)
            num_clusters = len(cluster_ids[cluster_ids > 0])
            summary_text += f"Clusters:\n"
            summary_text += f"• Número: {num_clusters}\n"
        
        # Informações de ativação
        if self.activation_map is not None:
            total_activations = np.sum(self.activation_map)
            active_neurons = np.sum(self.activation_map > 0)
            total_neurons = self.activation_map.size
            
            summary_text += f"\nAtivação:\n"
            summary_text += f"• Total pontos: {int(total_activations)}\n"
            summary_text += f"• Neurônios ativos: {active_neurons}/{total_neurons}\n"
            summary_text += f"• Taxa ativação: {active_neurons/total_neurons*100:.1f}%"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))

    def _create_individual_visualizations(self, som, data, original_features):
        """Cria visualizações individuais de alta qualidade"""
        # U-Matrix detalhada
        self._create_detailed_umatrix(som)
        
        # Clusters detalhados
        self._create_detailed_clusters(som, data)
        
        # Component planes completos
        self._create_all_component_planes(som)

    def _create_detailed_umatrix(self, som):
        """Cria visualização detalhada da U-Matrix"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # U-Matrix padrão
        im1 = ax1.imshow(self.umatrix, cmap='viridis', aspect='auto')
        ax1.set_title('U-Matrix - Visão Geral', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1)
        
        # U-Matrix com contornos
        im2 = ax2.imshow(self.umatrix, cmap='plasma', aspect='auto')
        levels = np.linspace(self.umatrix.min(), self.umatrix.max(), 15)
        contour = ax2.contour(self.umatrix, levels=levels, colors='white', alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_title('U-Matrix com Contornos', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2)
        
        # U-Matrix 3D (se possível)
        try:
            from mpl_toolkits.mplot3d import Axes3D
            X, Y = np.meshgrid(range(self.umatrix.shape[1]), range(self.umatrix.shape[0]))
            ax3 = fig.add_subplot(133, projection='3d')
            surf = ax3.plot_surface(X, Y, self.umatrix, cmap='coolwarm', 
                                  alpha=0.8, linewidth=0)
            ax3.set_title('U-Matrix 3D', fontsize=12, fontweight='bold')
            plt.colorbar(surf, ax=ax3, shrink=0.6)
        except ImportError:
            ax3.text(0.5, 0.5, '3D não disponível', ha='center', va='center', 
                    transform=ax3.transAxes)
            ax3.set_title('U-Matrix 3D (Não disponível)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('kohonen_detailed_umatrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_detailed_clusters(self, som, data):
        """Cria visualização detalhada dos clusters"""
        if self.natural_clusters is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Clusters com U-Matrix de fundo
        background = ax1.imshow(self.umatrix, cmap='gray', alpha=0.3, aspect='auto')
        
        cluster_ids = np.unique(self.natural_clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_ids)))
        
        for cluster_id in cluster_ids:
            if cluster_id == 0:
                continue
                
            cluster_mask = self.natural_clusters == cluster_id
            y_coords, x_coords = np.where(cluster_mask)
            
            if len(x_coords) > 0:
                ax1.scatter(x_coords, y_coords, color=colors[cluster_id], 
                          label=f'Cluster {cluster_id}', s=50, alpha=0.8)
                
                # Centroide
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                ax1.text(centroid_x, centroid_y, str(cluster_id), 
                        fontsize=10, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        ax1.set_title('Clusters Naturais - Visualização Detalhada', 
                     fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Histograma de clusters
        cluster_sizes = []
        for cluster_id in cluster_ids:
            if cluster_id > 0:
                size = np.sum(self.natural_clusters == cluster_id)
                cluster_sizes.append(size)
        
        if cluster_sizes:
            ax2.bar(range(1, len(cluster_sizes) + 1), cluster_sizes, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes))))
            ax2.set_title('Distribuição de Tamanhos dos Clusters', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Número de Neurônios')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kohonen_detailed_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_all_component_planes(self, som):
        """Cria component planes para todas as features"""
        weights = som._weights
        n_features = weights.shape[2]
        
        if self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Calcular layout do grid
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i in range(n_features):
            ax = axes[i]
            feature_plane = weights[:, :, i]
            
            im = ax.imshow(feature_plane.T, cmap='coolwarm', aspect='auto')
            ax.set_title(f'{self.feature_names[i]}', fontsize=10, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Ocultar eixos vazios
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('kohonen_all_component_planes.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error(self, ax, message):
        """Plota mensagem de erro em um subplot"""
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               transform=ax.transAxes, fontsize=9, color='red')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    def get_neuron_clusters(self):
        """Retorna os clusters dos neurônios"""
        return self.natural_clusters

    def get_quality_metrics(self):
        """Retorna as métricas de qualidade calculadas"""
        return self.quality_metrics.copy()

    def generate_analysis_report(self, output_file='som_analysis_report.txt'):
        """Gera relatório completo da análise"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DO SOM\n")
            f.write("=" * 50 + "\n\n")
            
            # Métricas de qualidade
            f.write("MÉTRICAS DE QUALIDADE:\n")
            f.write("-" * 30 + "\n")
            for name, value in self.quality_metrics.items():
                f.write(f"{name.replace('_', ' ').title()}: {value:.6f}\n")
            
            f.write("\n")
            
            # Informações de clusters
            if self.natural_clusters is not None:
                cluster_ids = np.unique(self.natural_clusters)
                num_clusters = len(cluster_ids[cluster_ids > 0])
                f.write(f"CLUSTERS NATURAIS: {num_clusters} clusters identificados\n")
                
                for cluster_id in cluster_ids:
                    if cluster_id > 0:
                        size = np.sum(self.natural_clusters == cluster_id)
                        f.write(f"  • Cluster {cluster_id}: {size} neurônios\n")
            
            f.write("\n")
            
            # Estatísticas de ativação
            if self.activation_map is not None:
                total_activations = np.sum(self.activation_map)
                active_neurons = np.sum(self.activation_map > 0)
                total_neurons = self.activation_map.size
                
                f.write("ESTATÍSTICAS DE ATIVAÇÃO:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total de ativações: {int(total_activations)}\n")
                f.write(f"Neurônios ativos: {active_neurons}/{total_neurons} "
                       f"({active_neurons/total_neurons*100:.1f}%)\n")
                f.write(f"Ativações por neurônio: {total_activations/active_neurons:.1f} (média)\n")
            
            f.write("\nRECOMENDAÇÕES:\n")
            f.write("-" * 30 + "\n")
            
            if self.quality_metrics.get('quantization_error', 1) > 0.5:
                f.write("• Considerar aumentar o número de épocas de treinamento\n")
            
            if self.quality_metrics.get('topographic_error', 1) > 0.2:
                f.write("• A topografia pode ser melhorada ajustando a taxa de aprendizado\n")
            
            if self.natural_clusters is not None and len(np.unique(self.natural_clusters)) <= 2:
                f.write("• Poucos clusters identificados - considerar ajustar parâmetros de densidade\n")
        
        logger.info(f"   📄 Relatório de análise salvo em: {output_file}")