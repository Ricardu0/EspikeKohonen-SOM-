"""
Módulo de análise e visualização do SOM
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label, gaussian_filter
import logging

logger = logging.getLogger(__name__)

class KohonenAdvancedAnalyzer:
    """Analisador avançado para visualizações e interpretação do SOM"""

    def __init__(self):
        self.umatrix = None
        self.activation_map = None
        self.natural_clusters = None

    def create_comprehensive_visualizations(self, som, X, original_features=None):
        """Cria visualizações abrangentes da rede de Kohonen"""
        logger.info("🎨 GERANDO VISUALIZAÇÕES AVANÇADAS")

        if som is None:
            raise ValueError("Rede não treinada!")

        # ✅ CORREÇÃO: Verificar se X é DataFrame ou array numpy
        if hasattr(X, 'values'):
            data = X.values.astype(np.float32)  # Se for DataFrame
        else:
            data = X.astype(np.float32)  # Se for array numpy

        # 1. U-MATRIX PRINCIPAL - FORMATO MELHORADO
        logger.info("1. Gerando U-Matrix Aprimorada...")
        self._create_enhanced_umatrix(som)

        # 2. MAPA DE ATIVAÇÃO - FORMATO MELHORADO
        logger.info("2. Gerando Mapa de Ativação Aprimorado...")
        self._create_enhanced_activation_map(som, data)

        # 3. CLUSTERS NATURAIS DO SOM (SEM K-MEANS)
        logger.info("3. Gerando Clusters Naturais...")
        self.natural_clusters = self._create_natural_clusters(som, data)

        logger.info("✅ Visualizações salvas com sucesso!")

    def _create_enhanced_umatrix(self, som):
        """Cria U-Matrix com formatação melhorada"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # U-Matrix padrão
        self.umatrix = som.distance_map().T
        im1 = ax1.imshow(self.umatrix, cmap='viridis', aspect='auto')
        ax1.set_title('U-Matrix: Mapa de Distâncias\n(Áreas escuras = clusters, claras = fronteiras)',
                      fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel('Coordenada X do Neurônio')
        ax1.set_ylabel('Coordenada Y do Neurônio')
        plt.colorbar(im1, ax=ax1, label='Distância Média')

        # U-Matrix com contornos
        im2 = ax2.imshow(self.umatrix, cmap='plasma', aspect='auto')

        # Adicionar contornos baseados na densidade
        levels = np.linspace(self.umatrix.min(), self.umatrix.max(), 10)
        contour = ax2.contour(self.umatrix, levels=levels, colors='white', alpha=0.6, linewidths=0.5)
        ax2.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

        ax2.set_title('U-Matrix com Contornos de Densidade', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Coordenada X do Neurônio')
        ax2.set_ylabel('Coordenada Y do Neurônio')
        plt.colorbar(im2, ax=ax2, label='Distância Média')

        plt.tight_layout()
        plt.savefig('kohonen_umatrix_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_enhanced_activation_map(self, som, data):
        """Cria mapa de ativação com formatação melhorada"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Calcular mapa de ativação
        self.activation_map = np.zeros((som._weights.shape[0], som._weights.shape[1]))

        batch_size = 10000
        for i in range(0, len(data), batch_size):
            end_idx = min(i + batch_size, len(data))
            batch = data[i:end_idx]
            for sample in batch:
                winner = som.winner(sample)
                self.activation_map[winner] += 1

        # Mapa de ativação padrão
        im1 = ax1.imshow(self.activation_map.T, cmap='YlOrRd', aspect='auto')
        ax1.set_title('Mapa de Ativação: Densidade de Pontos', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel('Coordenada X do Neurônio')
        ax1.set_ylabel('Coordenada Y do Neurônio')

        # Adicionar valores no mapa
        for i in range(self.activation_map.shape[0]):
            for j in range(self.activation_map.shape[1]):
                if self.activation_map[i, j] > 0:
                    ax1.text(i, j, f'{int(self.activation_map[i, j])}',
                             ha='center', va='center', fontweight='bold',
                             color='white' if self.activation_map[i, j] > np.percentile(self.activation_map, 70) else 'black',
                             fontsize=6)

        plt.colorbar(im1, ax=ax1, label='Número de Pontos')

        # Mapa de ativação com heatmap suavizado
        smoothed_activation = gaussian_filter(self.activation_map, sigma=1.0)

        im2 = ax2.imshow(smoothed_activation.T, cmap='hot', aspect='auto')
        ax2.set_title('Mapa de Ativação Suavizado', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Coordenada X do Neurônio')
        ax2.set_ylabel('Coordenada Y do Neurônio')
        plt.colorbar(im2, ax=ax2, label='Densidade Suavizada')

        plt.tight_layout()
        plt.savefig('kohonen_activation_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_natural_clusters(self, som, data, density_threshold=0.7):
        """Cria clusters naturais baseados em densidade no SOM (SEM K-MEANS)"""
        logger.info("   🎯 Identificando clusters naturais baseados em densidade...")

        # Normalizar mapa de ativação
        if np.max(self.activation_map) > 0:
            normalized_activation = self.activation_map / np.max(self.activation_map)
        else:
            normalized_activation = self.activation_map

        # Identificar regiões de alta densidade
        high_density_mask = normalized_activation > density_threshold

        # Usar connected components para encontrar clusters naturais
        labeled_array, num_clusters = label(high_density_mask)

        logger.info(f"   📊 Encontrados {num_clusters} clusters naturais")

        # Visualizar clusters naturais
        plt.figure(figsize=(14, 10))

        # Criar mapa de cores personalizado
        colors = plt.cm.Set3(np.linspace(0, 1, num_clusters + 1))

        # Plotar fundo com U-Matrix
        plt.imshow(self.umatrix, cmap='gray', alpha=0.3, aspect='auto')

        # Plotar clusters
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = labeled_array == cluster_id
            y_coords, x_coords = np.where(cluster_mask)

            if len(x_coords) > 0:  # Cluster não vazio
                # Calcular centroide
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)

                # Plotar cluster
                plt.scatter(x_coords, y_coords,
                            color=colors[cluster_id],
                            label=f'Cluster {cluster_id}',
                            alpha=0.7, s=50)

                # Adicionar número do cluster
                plt.text(centroid_x, centroid_y, str(cluster_id),
                         fontsize=12, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', alpha=0.8))

        plt.title(f'Clusters Naturais do SOM\n{num_clusters} Grupos Identificados por Densidade',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Coordenada X do Neurônio')
        plt.ylabel('Coordenada Y do Neurônio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.colorbar(label='Distância (U-Matrix)')

        plt.tight_layout()
        plt.savefig('kohonen_natural_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

        return labeled_array

    def get_neuron_clusters(self):
        """Retorna os clusters dos neurônios"""
        return self.natural_clusters