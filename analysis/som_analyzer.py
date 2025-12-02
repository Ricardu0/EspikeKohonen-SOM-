"""
Módulo avançado de análise e visualização do SOM (Self-Organizing Maps)
Versão refatorada para uso eficiente de memória em datasets grandes
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import warnings
import gc

from typing import Dict, Tuple, Optional, List, Any

from scipy import stats
from scipy.ndimage import label, gaussian_filter, gaussian_gradient_magnitude
from scipy.cluster.hierarchy import linkage, fcluster

# Evitar imports pesados até o uso
# from skimage.feature import peak_local_max
# from skimage.segmentation import watershed

warnings.filterwarnings('ignore')

# Estilo de visualização
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class KohonenAdvancedAnalyzer:
    """Analisador avançado para visualizações e interpretação do SOM (memory-safe)"""

    def __init__(self):
        self.umatrix: Optional[np.ndarray] = None
        self.activation_map: Optional[np.ndarray] = None
        self.natural_clusters: Optional[np.ndarray] = None
        self.component_planes: Optional[np.ndarray] = None
        self.quality_metrics: Dict[str, float] = {}
        self.feature_names: Optional[List[str]] = None

        # Parâmetros de segurança de memória
        self.max_samples_qe_te: int = 10000
        self.batch_size_qe: int = 1000
        self.batch_size_activation: int = 5000
        self.max_neurons_corr: int = 2500  # para matriz de correlação
        self.max_umatrix_side: int = 200   # downsample se mapa > 200x200

    def create_comprehensive_visualizations(
        self,
        som,
        X,
        original_features: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        save_path: str = 'kohonen_comprehensive_analysis.png'
    ):
        """Cria visualizações abrangentes e profissionais da rede de Kohonen com segurança de memória."""
        logger.info("🎨 GERANDO VISUALIZAÇÕES AVANÇADAS DO SOM")

        if som is None:
            raise ValueError("Rede SOM não treinada!")

        data = self._prepare_data(X)
        self.feature_names = feature_names or (list(X.columns) if hasattr(X, 'columns') else None)

        # Métricas com amostragem e batch
        q_error, t_error = self.compute_quality_metrics_safe(som, data)
        self.quality_metrics['quantization_error'] = q_error
        self.quality_metrics['topographic_error'] = t_error

        # U-Matrix segura
        self.umatrix = self._get_umatrix_safe(som)

        # Dashboard
        self._create_analysis_dashboard(som, data, original_features, save_path)

        logger.info("✅ Visualizações avançadas salvas com sucesso!")

    def _prepare_data(self, X):
        """Converte para float32 e retorna numpy array."""
        if hasattr(X, 'values'):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    # ---------------------------
    # Métricas com amostragem/batch
    # ---------------------------
    def compute_quality_metrics_safe(self, som, data) -> Tuple[float, float]:
        """Calcula QE e TE com amostragem e processamento em lotes para evitar estouro de memória."""
        # Amostragem para QE/TE
        if len(data) > self.max_samples_qe_te:
            idx = np.random.choice(len(data), self.max_samples_qe_te, replace=False)
            sample = data[idx]
        else:
            sample = data

        # QE em lotes
        q_error = self._batch_quantization_error(som, sample, batch_size=self.batch_size_qe)

        # TE com amostra menor e fallback
        try:
            te_sample_size = min(5000, len(sample))
            indices = np.random.choice(len(sample), te_sample_size, replace=False)
            t_error = som.topographic_error(sample[indices])
        except MemoryError:
            logger.warning("TE: MemoryError, usando amostra menor.")
            te_sample_size = min(1500, len(sample))
            indices = np.random.choice(len(sample), te_sample_size, replace=False)
            t_error = som.topographic_error(sample[indices])
        except Exception as e:
            logger.warning(f"TE: erro {e}, retornando 0.0.")
            t_error = 0.0

        return q_error, t_error

    def _batch_quantization_error(self, som, data, batch_size=1000) -> float:
        """Calcula QE em lotes, somando distância entre amostra e peso do BMU."""
        total_distance = 0.0
        n = len(data)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = data[start:end]
            # Processar cada amostra (MiniSom não tem winner em lote)
            for x in batch:
                w = som.winner(x)
                weight = som._weights[w[0], w[1]]
                total_distance += np.linalg.norm(x - weight)
            if (start // batch_size) % 5 == 0:
                gc.collect()
        return total_distance / n

    # ---------------------------
    # U-Matrix segura
    # ---------------------------
    def _get_umatrix_safe(self, som) -> np.ndarray:
        """Obtém U-Matrix do som ou calcula manualmente; aplica downsampling se necessário."""
        try:
            umat = som.distance_map().T
        except Exception:
            umat = self._compute_umatrix_manual(som)

        # Downsample se muito grande
        h, w = umat.shape
        max_side = self.max_umatrix_side
        if max(h, w) > max_side:
            scale_h = max_side / h
            scale_w = max_side / w
            scale = min(scale_h, scale_w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            umat = self._resize_bilinear(umat, (new_h, new_w))
            logger.info(f"U-Matrix downsample: {h}x{w} → {new_h}x{new_w}")
        return umat.astype(np.float32)

    def _compute_umatrix_manual(self, som) -> np.ndarray:
        """Calcula U-Matrix manualmente (média das distâncias para vizinhos 4-conectados)."""
        weights = som._weights
        H, W, D = weights.shape
        umat = np.zeros((H, W), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbors.append(weights[ni, nj])
                if neighbors:
                    dists = [np.linalg.norm(weights[i, j] - n) for n in neighbors]
                    umat[i, j] = np.mean(dists)
            if i % 25 == 0:
                gc.collect()
        return umat

    def _resize_bilinear(self, img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """Resize bilinear simples para 2D numpy (sem dependências extras)."""
        new_h, new_w = new_shape
        h, w = img.shape
        # Mapeamento de coordenadas
        y = (np.linspace(0, h - 1, new_h)).astype(np.float32)
        x = (np.linspace(0, w - 1, new_w)).astype(np.float32)
        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        wy = y - y0
        wx = x - x0

        out = np.zeros((new_h, new_w), dtype=np.float32)
        for i in range(new_h):
            for j in range(new_w):
                v00 = img[y0[i], x0[j]]
                v01 = img[y0[i], x1[j]]
                v10 = img[y1[i], x0[j]]
                v11 = img[y1[i], x1[j]]
                out[i, j] = (
                    (1 - wy[i]) * (1 - wx[j]) * v00 +
                    (1 - wy[i]) * wx[j] * v01 +
                    wy[i] * (1 - wx[j]) * v10 +
                    wy[i] * wx[j] * v11
                )
        return out

    # ---------------------------
    # Dashboard
    # ---------------------------
    def _create_analysis_dashboard(self, som, data, original_features, save_path):
        logger.info("   📈 Criando dashboard de análise...")

        fig = plt.figure(figsize=(22, 18))

        # 1. U-Matrix
        ax1 = plt.subplot(3, 4, 1)
        self._plot_umatrix(ax1)

        # 2. Mapa de ativação
        ax2 = plt.subplot(3, 4, 2)
        self._create_enhanced_activation_map(som, data, ax2)

        # 3. Clusters naturais (watershed) com fallback
        ax3 = plt.subplot(3, 4, 3)
        self.natural_clusters = self._create_natural_clusters_native(ax3)

        # 4. Component plane (primeira feature de maior variância)
        ax4 = plt.subplot(3, 4, 4)
        self._create_component_plane_single(som, ax4)

        # 5. Métricas
        ax5 = plt.subplot(3, 4, 5)
        self._create_quality_visualization(ax5)

        # 6. Histograma de ativação
        ax6 = plt.subplot(3, 4, 6)
        self._create_activation_histogram(ax6)

        # 7. Mapa de gradiente
        ax7 = plt.subplot(3, 4, 7)
        self._create_gradient_map(ax7)

        # 8. Projeção (amostrada)
        ax8 = plt.subplot(3, 4, 8)
        self._create_data_projection(data, ax8)

        # 9. Correlação entre neurônios (amostragem)
        ax9 = plt.subplot(3, 4, 9)
        self._create_neuron_correlation(som, ax9)

        # 10. Placeholder aprendizado
        ax10 = plt.subplot(3, 4, 10)
        self._create_learning_analysis(ax10)

        # 11. Distribuição clusters
        ax11 = plt.subplot(3, 4, 11)
        self._create_cluster_distribution(ax11)

        # 12. Resumo
        ax12 = plt.subplot(3, 4, 12)
        self._create_executive_summary(ax12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Extras detalhados (com limitações de memória)
        self._create_individual_visualizations(som, data, original_features)

    def _plot_umatrix(self, ax):
        try:
            im = ax.imshow(self.umatrix, cmap='viridis', aspect='auto', interpolation='nearest')
            levels = np.linspace(self.umatrix.min(), self.umatrix.max(), 10)
            ax.contour(self.umatrix, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
            ax.set_title('U-Matrix: Mapa de Distâncias', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Distância Média', shrink=0.8)
        except Exception as e:
            self._plot_error(ax, f"Erro na U-Matrix: {str(e)}")

    # ---------------------------
    # Clusterização nativa (watershed) com segurança
    # ---------------------------
    def _create_natural_clusters_native(self, ax, threshold_percentile=30):
        """Identifica clusters usando watershed na U-Matrix com checagens de memória."""
        try:
            # Imports sob demanda para reduzir custo de memória
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_max

            umat = self.umatrix
            threshold = np.percentile(umat, threshold_percentile)
            markers_mask = umat < threshold

            markers, num_markers = label(markers_mask)

            # Watershed
            cluster_map = watershed(umat.astype(np.float32), markers)

            # Filtrar clusters pequenos usando ativação (se disponível)
            if self.activation_map is not None:
                min_activations = max(1, int(np.sum(self.activation_map) * 0.01))
                for cluster_id in range(1, num_markers + 1):
                    mask = cluster_map == cluster_id
                    activations = int(np.sum(self.activation_map[mask]))
                    if activations < min_activations:
                        cluster_map[mask] = 0

            # Renumerar
            valid = np.unique(cluster_map[cluster_map > 0])
            final_map = np.zeros_like(cluster_map, dtype=np.int32)
            for new_id, old_id in enumerate(valid, start=1):
                final_map[cluster_map == old_id] = new_id

            self._plot_clusters_native(ax, final_map)
            return final_map

        except Exception as e:
            logger.warning(f"Watershed falhou ({e}), usando fallback por threshold.")
            labeled, _ = label(self.umatrix < np.percentile(self.umatrix, threshold_percentile))
            self._plot_clusters_native(ax, labeled)
            return labeled

    def _plot_clusters_native(self, ax, cluster_map):
        ax.imshow(self.umatrix, cmap='gray', alpha=0.3, aspect='auto')
        num_clusters = len(np.unique(cluster_map)) - 1
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, num_clusters)))
        for cluster_id in range(1, num_clusters + 1):
            y_coords, x_coords = np.where(cluster_map == cluster_id)
            if len(x_coords) > 0:
                ax.scatter(x_coords, y_coords,
                           color=colors[cluster_id - 1],
                           label=f'C{cluster_id}',
                           alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        ax.set_title(f'Clusters Naturais (Watershed)\n{num_clusters} identificados',
                     fontsize=11, fontweight='bold')
        if num_clusters > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # ---------------------------
    # Ativação (batch)
    # ---------------------------
    def _create_enhanced_activation_map(self, som, data, ax):
        """Cria mapa de ativação processando em lotes."""
        try:
            H, W, _ = som._weights.shape
            self.activation_map = np.zeros((H, W), dtype=np.int32)
            bs = min(self.batch_size_activation, len(data))
            for i in range(0, len(data), bs):
                batch = data[i:i + bs]
                for sample in batch:
                    w = som.winner(sample)
                    self.activation_map[w[0], w[1]] += 1
                if (i // bs) % 5 == 0:
                    gc.collect()

            smoothed = gaussian_filter(self.activation_map.astype(np.float32), sigma=0.8)
            im = ax.imshow(smoothed.T, cmap='hot', aspect='auto', interpolation='nearest')
            if np.max(smoothed) > 0:
                levels = np.linspace(smoothed.min(), smoothed.max(), 8)
                ax.contour(smoothed.T, levels=levels, colors='white', alpha=0.5, linewidths=0.8)
            ax.set_title('Mapa de Densidade de Ativação', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Densidade de Pontos', shrink=0.8)
        except Exception as e:
            self._plot_error(ax, f"Erro no mapa de ativação: {str(e)}")

    # ---------------------------
    # Component plane (uma feature)
    # ---------------------------
    def _create_component_plane_single(self, som, ax, num_candidates=4):
        """Exibe um único component plane da feature com maior variância (economia de memória)."""
        try:
            weights = som._weights
            H, W, D = weights.shape
            # Variância por feature
            feature_vars = np.var(weights.reshape(H * W, D), axis=0)
            top_idx = int(np.argsort(feature_vars)[-1])
            plane = weights[:, :, top_idx]
            im = ax.imshow(plane.T, cmap='coolwarm', aspect='auto', interpolation='nearest')
            name = self.feature_names[top_idx] if self.feature_names else f'Feature_{top_idx}'
            ax.set_title(f'Component Plane: {name}', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Valor do Peso', shrink=0.8)
        except Exception as e:
            self._plot_error(ax, f"Erro nos component planes: {str(e)}")

    # ---------------------------
    # Métricas (plot)
    # ---------------------------
    def _create_quality_visualization(self, ax):
        try:
            metrics = self.quality_metrics
            if not metrics:
                ax.text(0.5, 0.5, 'Métricas não disponíveis',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Métricas de Qualidade', fontsize=11)
                return

            names = [k.replace('_', ' ').title() for k in metrics.keys()]
            values = list(metrics.values())
            colors = [('lightcoral' if 'error' in k.lower() else 'lightsteelblue')
                      for k in metrics.keys()]
            bars = ax.bar(names, values, color=colors, alpha=0.8)
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2.0, v + 0.001,
                        f'{v:.4f}', ha='center', va='bottom', fontsize=9)
            ax.set_title('Métricas de Qualidade do SOM', fontsize=11, fontweight='bold', pad=10)
            ax.set_ylabel('Valor'); ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        except Exception as e:
            self._plot_error(ax, f"Erro nas métricas: {str(e)}")

    def _create_activation_histogram(self, ax):
        try:
            if self.activation_map is None:
                ax.text(0.5, 0.5, 'Mapa de ativação não disponível',
                        ha='center', va='center', transform=ax.transAxes)
                return
            activations = self.activation_map.flatten()
            activations = activations[activations > 0]
            if len(activations) == 0:
                ax.text(0.5, 0.5, 'Sem ativações',
                        ha='center', va='center', transform=ax.transAxes)
                return
            ax.hist(activations, bins=20, alpha=0.75, color='skyblue', edgecolor='black')
            ax.set_title('Distribuição de Ativações por Neurônio',
                         fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Pontos'); ax.set_ylabel('Frequência'); ax.grid(True, alpha=0.3)
            mean_act = np.mean(activations)
            ax.axvline(mean_act, color='red', linestyle='--', label=f'Média: {mean_act:.1f}')
            ax.legend()
        except Exception as e:
            self._plot_error(ax, f"Erro no histograma: {str(e)}")

    def _create_gradient_map(self, ax):
        try:
            if self.umatrix is None:
                ax.text(0.5, 0.5, 'U-Matrix não disponível',
                        ha='center', va='center', transform=ax.transAxes)
                return
            gradient = gaussian_gradient_magnitude(self.umatrix, sigma=1.0)
            im = ax.imshow(gradient, cmap='Reds', aspect='auto', interpolation='nearest')
            ax.set_title('Mapa de Gradiente (Fronteiras)', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Magnitude', shrink=0.8)
        except Exception as e:
            self._plot_error(ax, f"Erro no gradiente: {str(e)}")

    # ---------------------------
    # Projeção 2D com amostragem
    # ---------------------------
    def _create_data_projection(self, data, ax):
        try:
            if len(data) > 1500:
                idx = np.random.choice(len(data), 1500, replace=False)
                sample = data[idx]
            else:
                sample = data

            # PCA manual rápido (cov ~ 2D) se D > 2
            if sample.shape[1] > 2:
                # Centralizar
                Xc = sample - sample.mean(axis=0, keepdims=True)
                # SVD para 2 componentes
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                proj = Xc @ Vt[:2].T
                method = 'SVD (PCA)'
            else:
                proj = sample
                method = 'Original'
            ax.scatter(proj[:, 0], proj[:, 1], alpha=0.6, s=16, c='purple')
            ax.set_title(f'Projeção dos Dados ({method})', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Comp 1'); ax.set_ylabel('Comp 2'); ax.grid(True, alpha=0.3)
        except Exception as e:
            self._plot_error(ax, f"Erro na projeção: {str(e)}")

    # ---------------------------
    # Correlação entre neurônios com amostragem
    # ---------------------------
    def _create_neuron_correlation(self, som, ax):
        try:
            weights = som._weights.reshape(-1, som._weights.shape[2])
            N = weights.shape[0]
            if N > self.max_neurons_corr:
                idx = np.random.choice(N, self.max_neurons_corr, replace=False)
                weights = weights[idx]
                logger.info(f"Correlação: amostrando {self.max_neurons_corr}/{N} neurônios.")
            corr = np.corrcoef(weights)
            im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1, interpolation='nearest')
            ax.set_title('Correlação entre Neurônios (amostrada)', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Índice'); ax.set_ylabel('Índice')
            plt.colorbar(im, ax=ax, label='Correlação', shrink=0.8)
        except Exception as e:
            self._plot_error(ax, f"Erro na correlação: {str(e)}")

    # ---------------------------
    # Outros plots
    # ---------------------------
    def _create_learning_analysis(self, ax):
        ax.text(0.5, 0.5, 'Análise de Aprendizado\n(Não disponível nesta versão)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Processo de Aprendizado', fontsize=11, fontweight='bold', pad=10)
        ax.set_frame_on(False); ax.set_xticks([]); ax.set_yticks([])

    def _create_cluster_distribution(self, ax):
        try:
            if self.natural_clusters is None:
                ax.text(0.5, 0.5, 'Clusters não calculados',
                        ha='center', va='center', transform=ax.transAxes)
                return
            ids, sizes = np.unique(self.natural_clusters, return_counts=True)
            mask = (ids > 0) & (sizes >= 2)
            ids, sizes = ids[mask], sizes[mask]
            if len(ids) == 0:
                ax.text(0.5, 0.5, 'Sem clusters válidos',
                        ha='center', va='center', transform=ax.transAxes)
                return
            colors = plt.cm.Set3(np.linspace(0, 1, len(ids)))
            bars = ax.bar(ids, sizes, color=colors, alpha=0.8)
            ax.set_title('Distribuição de Tamanhos dos Clusters', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('ID'); ax.set_ylabel('Neurônios'); ax.grid(True, alpha=0.3, axis='y')
            for b, s in zip(bars, sizes):
                ax.text(b.get_x() + b.get_width() / 2.0, s + 0.1,
                        f'{s}', ha='center', va='bottom', fontsize=8)
        except Exception as e:
            self._plot_error(ax, f"Erro na distribuição: {str(e)}")

    def _create_executive_summary(self, ax):
        ax.set_frame_on(False); ax.set_xticks([]); ax.set_yticks([])
        lines = ["RESUMO EXECUTIVO", ""]
        if self.quality_metrics:
            qe = self.quality_metrics.get('quantization_error', np.nan)
            te = self.quality_metrics.get('topographic_error', np.nan)
            lines += [f"Qualidade:",
                      f"• Erro de Quantização: {qe:.4f}",
                      f"• Erro Topográfico: {te:.4f}", ""]
        if self.natural_clusters is not None:
            ids = np.unique(self.natural_clusters)
            num = int(np.sum(ids > 0))
            lines += [f"Clusters:", f"• Número: {num}"]
        if self.activation_map is not None:
            total = int(np.sum(self.activation_map))
            active = int(np.sum(self.activation_map > 0))
            total_neurons = int(self.activation_map.size)
            rate = (active / total_neurons * 100.0) if total_neurons > 0 else 0.0
            lines += ["", "Ativação:",
                      f"• Total pontos: {total}",
                      f"• Neurônios ativos: {active}/{total_neurons}",
                      f"• Taxa ativação: {rate:.1f}%"]
        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))

    # ---------------------------
    # Visualizações individuais (com limites)
    # ---------------------------
    def _create_individual_visualizations(self, som, data, original_features):
        self._create_detailed_umatrix()
        self._create_detailed_clusters()
        self._create_all_component_planes(som, max_planes=12)  # limitar para evitar imagens gigantes

    def _create_detailed_umatrix(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        im1 = ax1.imshow(self.umatrix, cmap='viridis', aspect='auto', interpolation='nearest')
        ax1.set_title('U-Matrix - Visão Geral', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(self.umatrix, cmap='plasma', aspect='auto', interpolation='nearest')
        levels = np.linspace(self.umatrix.min(), self.umatrix.max(), 10)
        contour = ax2.contour(self.umatrix, levels=levels, colors='white', alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_title('U-Matrix com Contornos', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.savefig('kohonen_detailed_umatrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_detailed_clusters(self):
        if self.natural_clusters is None:
            return
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax1.imshow(self.umatrix, cmap='gray', alpha=0.3, aspect='auto', interpolation='nearest')
        ids = np.unique(self.natural_clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(ids))))
        for cid in ids:
            if cid == 0:
                continue
            mask = self.natural_clusters == cid
            y, x = np.where(mask)
            if len(x) > 0:
                ax1.scatter(x, y, color=colors[cid % len(colors)], label=f'Cluster {cid}', s=40, alpha=0.8)
        ax1.set_title('Clusters Naturais - Visualização Detalhada', fontsize=13, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('kohonen_detailed_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_all_component_planes(self, som, max_planes=12):
        weights = som._weights
        H, W, D = weights.shape
        n = min(D, max_planes)
        feature_vars = np.var(weights.reshape(H * W, D), axis=0)
        top = np.argsort(feature_vars)[-n:][::-1]
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()
        for i in range(n):
            ax = axes[i]
            plane = weights[:, :, top[i]]
            im = ax.imshow(plane.T, cmap='coolwarm', aspect='auto', interpolation='nearest')
            name = self.feature_names[top[i]] if self.feature_names else f'Feature_{top[i]}'
            ax.set_title(name, fontsize=10, fontweight='bold')
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(n, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.savefig('kohonen_all_component_planes.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------
    # Utilitários
    # ---------------------------
    def _plot_error(self, ax, message):
        ax.text(0.5, 0.5, message, ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='red')
        ax.set_frame_on(False); ax.set_xticks([]); ax.set_yticks([])

    def get_neuron_clusters(self):
        return self.natural_clusters

    def get_quality_metrics(self):
        return self.quality_metrics.copy()

    def generate_analysis_report(self, output_file='som_analysis_report.txt'):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DO SOM\n")
            f.write("=" * 50 + "\n\n")
            f.write("MÉTRICAS DE QUALIDADE:\n")
            f.write("-" * 30 + "\n")
            for name, value in self.quality_metrics.items():
                f.write(f"{name.replace('_', ' ').title()}: {value:.6f}\n")
            f.write("\n")
            if self.natural_clusters is not None:
                cluster_ids = np.unique(self.natural_clusters)
                num_clusters = int(np.sum(cluster_ids > 0))
                f.write(f"CLUSTERS NATURAIS: {num_clusters} clusters identificados\n")
                for cid in cluster_ids:
                    if cid > 0:
                        size = int(np.sum(self.natural_clusters == cid))
                        f.write(f"  • Cluster {cid}: {size} neurônios\n")
            f.write("\n")
            if self.activation_map is not None:
                total = int(np.sum(self.activation_map))
                active = int(np.sum(self.activation_map > 0))
                total_neurons = int(self.activation_map.size)
                rate = (active / total_neurons * 100.0) if total_neurons > 0 else 0.0
                f.write("ESTATÍSTICAS DE ATIVAÇÃO:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total de ativações: {total}\n")
                f.write(f"Neurônios ativos: {active}/{total_neurons} ({rate:.1f}%)\n")
                if active > 0:
                    f.write(f"Ativações por neurônio: {total / active:.1f} (média)\n")
            f.write("\nRECOMENDAÇÕES:\n")
            f.write("-" * 30 + "\n")
            if self.quality_metrics.get('quantization_error', 1) > 0.5:
                f.write("• Considerar aumentar épocas ou mapa SOM (mais neurônios)\n")
            if self.quality_metrics.get('topographic_error', 1) > 0.2:
                f.write("• Ajustar taxa de aprendizado/sigma para melhor preservação da topologia\n")
            if self.natural_clusters is not None and len(np.unique(self.natural_clusters)) <= 2:
                f.write("• Poucos clusters: testar limiar do watershed ou aumentar resolução do mapa\n")
        logger.info(f"   📄 Relatório de análise salvo em: {output_file}")
