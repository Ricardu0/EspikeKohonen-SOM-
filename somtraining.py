# kohonen_analysis_pipeline.py
"""
Pipeline Avançado de Rede de Kohonen (SOM) com Análise Interpretável
e Gerenciamento Eficiente de Memória para Grandes Conjuntos de Dados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats
import joblib
import os
import argparse
from scipy import sparse
import warnings
import gc
import psutil
import logging
from typing import Tuple, Optional, Dict, List
from scipy.ndimage import label, gaussian_filter
from scipy.spatial.distance import cdist
import matplotlib.patches as patches

warnings.filterwarnings('ignore')

# Configurações de estilo para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
RANDOM_STATE = 42

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        logging.info(f"Batch size definido para: {self.batch_size}")

    def log_memory_usage(self):
        """Log do uso de memória"""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logging.info(f"Uso de memória: {memory_mb:.2f} MB")
        except:
            logging.info("Monitoramento de memória não disponível")

    def get_optimized_som_config(self, data_size: int, input_dim: int) -> Dict:
        """Define configurações otimizadas baseadas no tamanho dos dados"""
        if data_size > 100000:
            som_x, som_y = 25, 25  # Aumentado para melhor resolução
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
            logging.warning("Erro de memória no cálculo topográfico, usando amostra menor")
            sample_size = min(1000, len(data))
            indices = np.random.choice(len(data), sample_size, replace=False)
            return som.topographic_error(data[indices])

    def _train_with_memory_management(self, som, data: np.ndarray, iterations: int):
        """Treinamento com monitoramento de memória"""
        for iteration in range(iterations):
            batch_indices = np.random.choice(
                len(data),
                size=min(self.batch_size, len(data)),
                replace=False
            )
            batch = data[batch_indices]

            som.train_batch(batch, 1, verbose=False)

            if iteration % 100 == 0:
                gc.collect()

            if iteration % 500 == 0:
                self.log_memory_usage()
                logging.info(f"Iteração {iteration}/{iterations}")

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

        logging.info(f"Configuração do SOM: {som_x}x{som_y}, sigma={sigma}, "
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

            logging.info("Calculando erro de quantização...")
            q_error = self.safe_quantization_error(som, data)

            logging.info("Calculando erro topográfico...")
            topographic_error = self._safe_topographic_error(som, data)

            self.log_memory_usage()

            self.som = som
            return som, q_error, topographic_error

        except MemoryError as e:
            logging.error(f"Erro de memória durante o treinamento: {e}")
            return self.fallback_training(data)
        except Exception as e:
            logging.error(f"Erro inesperado: {e}")
            raise

    def fallback_training(self, data: np.ndarray):
        """Fallback para quando as configurações otimizadas falham"""
        logging.info("Usando configuração de fallback...")

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
            logging.info(f"Usando amostra de {sample_size} dados para fallback")

        return self.train_kohonen_network(data, **fallback_config)


class AdvancedDataPreprocessor:
    """Pré-processamento avançado com análise exploratória"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.feature_info = {}
        self.categorical_mappings = {}

    def load_and_analyze_data(self, csv_path='SPSafe_2022.csv', sample_frac=None):
        """Carrega e analisa dados com relatório detalhado"""
        print("📊 CARREGAMENTO E ANÁLISE EXPLORATÓRIA DE DADOS")
        print("=" * 50)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo {csv_path} não encontrado!")

        try:
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1', low_memory=False)

        original_size = len(df)
        if sample_frac and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=RANDOM_STATE)
            print(f"✅ Dataset amostrado: {len(df):,} registros ({sample_frac * 100:.1f}% do original)")

        print(f"📈 Shape do dataset: {df.shape}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        print(f"   • Numéricas: {len(numeric_cols)} colunas")
        print(f"   • Categóricas: {len(categorical_cols)} colunas")

        print(f"\n📋 Estatísticas básicas:")
        print(f"   • Registros totais: {len(df):,}")
        print(f"   • Valores missing: {df.isnull().sum().sum():,}")
        print(f"   • Memória utilizada: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

        return df

    def create_eda_visualizations(self, df):
        """Cria visualizações de análise exploratória"""
        print("\n🎨 CRIANDO VISUALIZAÇÕES EXPLORATÓRIAS...")

        plt.figure(figsize=(10, 6))
        dtype_counts = df.dtypes.value_counts()
        plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        plt.title('Distribuição de Tipos de Dados')
        plt.savefig('eda_data_types.png', dpi=300, bbox_inches='tight')
        plt.close()

        missing_data = df.isnull().sum().sort_values(ascending=False).head(20)
        plt.figure(figsize=(12, 8))
        missing_data.plot(kind='barh')
        plt.title('Top 20 Colunas com Valores Missing')
        plt.xlabel('Número de Valores Missing')
        plt.tight_layout()
        plt.savefig('eda_missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Visualizações exploratórias salvas")

    def enhanced_feature_engineering(self, df):
        """Engenharia de features com análise detalhada"""
        print("\n🔧 ENGENHARIA DE FEATURES AVANÇADA")
        print("=" * 40)

        df = df.copy()
        feature_categories = {}

        temporal_features = []
        if 'DATA_OCORRENCIA' in df.columns:
            df['DATA_OCORRENCIA'] = pd.to_datetime(df['DATA_OCORRENCIA'], errors='coerce')
            df['DIA_SEMANA'] = df['DATA_OCORRENCIA'].dt.day_name()
            df['MES'] = df['DATA_OCORRENCIA'].dt.month_name()
            df['ANO'] = df['DATA_OCORRENCIA'].dt.year
            df['FIM_SEMANA'] = df['DATA_OCORRENCIA'].dt.weekday >= 5
            temporal_features.extend(['DIA_SEMANA', 'MES', 'ANO', 'FIM_SEMANA'])

        if 'HORA_OCORRENCIA' in df.columns:
            def parse_hour_detailed(h):
                try:
                    s = str(h).strip().replace('h', ':').replace('.', ':')
                    if ':' in s:
                        return int(s.split(':')[0])
                    elif s.isdigit():
                        return int(s[:2]) if len(s) > 2 else int(s)
                except:
                    return np.nan
                return np.nan

            df['HORA'] = df['HORA_OCORRENCIA'].apply(parse_hour_detailed)

            bins = [-1, 5, 9, 12, 15, 18, 21, 24]
            labels = ['Madrugada', 'Manhã Cedo', 'Manhã', 'Tarde Cedo', 'Tarde', 'Noite', 'Noite Tardia']
            df['PERIODO_DIA'] = pd.cut(df['HORA'], bins=bins, labels=labels).astype(str)
            temporal_features.extend(['HORA', 'PERIODO_DIA'])

        geographic_features = []
        if all(col in df.columns for col in ['LATITUDE', 'LONGITUDE']):
            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
            df['TEM_COORDENADAS'] = df['LATITUDE'].notna() & df['LONGITUDE'].notna()
            geographic_features.extend(['LATITUDE', 'LONGITUDE', 'TEM_COORDENADAS'])

        demographic_features = []
        if 'IDADE_PESSOA' in df.columns:
            df['IDADE_PESSOA'] = pd.to_numeric(df['IDADE_PESSOA'], errors='coerce')
            bins = [0, 18, 30, 45, 60, 100, 200]
            labels = ['0-18', '19-30', '31-45', '46-60', '61-100', '100+']
            df['FAIXA_ETARIA'] = pd.cut(df['IDADE_PESSOA'], bins=bins, labels=labels).astype(str)
            demographic_features.extend(['IDADE_PESSOA', 'FAIXA_ETARIA'])

        categorical_features = [
            'SEXO_PESSOA', 'COR_PELE', 'TIPO_VEICULO', 'TIPO_LOCAL',
            'NATUREZA_APURADA', 'CIDADE', 'BAIRRO', 'UF'
        ]

        available_categorical = [col for col in categorical_features if col in df.columns]
        all_features = temporal_features + geographic_features + demographic_features + available_categorical
        available_features = [col for col in all_features if col in df.columns]

        print("📋 FEATURES SELECIONADAS:")
        feature_categories = {
            'Temporais': [f for f in temporal_features if f in available_features],
            'Geográficas': [f for f in geographic_features if f in available_features],
            'Demográficas': [f for f in demographic_features if f in available_features],
            'Categóricas': [f for f in available_categorical if f in available_features]
        }

        for category, features in feature_categories.items():
            if features:
                print(f"   • {category}: {len(features)} features")
                for feature in features:
                    unique_vals = df[feature].nunique()
                    print(f"     - {feature}: {unique_vals} valores únicos")

        features_df = df[available_features].copy()
        self.feature_info = feature_categories

        print(f"\n✅ Engenharia de features concluída: {features_df.shape}")
        return features_df

    def smart_encoding(self, features_df):
        """Codificação inteligente com preservação de significado"""
        print("\n🔠 CODIFICAÇÃO INTELIGENTE DE FEATURES")
        print("=" * 45)

        X = features_df.copy()

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        print(f"🔢 Features numéricas ({len(numeric_features)}):")
        for feature in numeric_features:
            stats = X[feature].describe()
            print(f"   • {feature}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")

        print(f"\n🏷️  Features categóricas ({len(categorical_features)}):")
        for feature in categorical_features:
            unique_count = X[feature].nunique()
            top_categories = X[feature].value_counts().head(3)
            print(f"   • {feature}: {unique_count} categorias")
            print(f"     Top: {', '.join([f'{k}({v})' for k, v in top_categories.items()])}")

        print("\n🔄 Processando valores missing...")
        for col in categorical_features:
            X[col] = X[col].fillna('NÃO_INFORMADO')
            if X[col].nunique() > 20:
                top_categories = X[col].value_counts().head(15).index
                X[col] = X[col].apply(lambda x: x if x in top_categories else 'OUTROS')
                print(f"   • {col}: cardinalidade reduzida para 16 categorias")

        for col in numeric_features:
            X[col] = X[col].fillna(X[col].median())

        print("\n🎯 Aplicando codificação one-hot...")
        if categorical_features:
            X_encoded = self.encoder.fit_transform(X[categorical_features])
            encoded_features = self.encoder.get_feature_names_out(categorical_features)
            print(f"   • {len(categorical_features)} features → {len(encoded_features)} colunas codificadas")
        else:
            X_encoded = sparse.csr_matrix((X.shape[0], 0))
            encoded_features = []

        if numeric_features:
            X_scaled = self.scaler.fit_transform(X[numeric_features])
            X_scaled = sparse.csr_matrix(X_scaled)
        else:
            X_scaled = sparse.csr_matrix((X.shape[0], 0))

        X_final = sparse.hstack([X_scaled, X_encoded])

        print(f"\n✅ Dataset final: {X_final.shape}")
        print(f"   • Matriz esparsa: {X_final.getnnz():,} elementos não-zero")
        print(f"   • Densidade: {X_final.getnnz() / (X_final.shape[0] * X_final.shape[1]):.4f}")

        sparse.save_npz('X_processed_sparse.npz', X_final)

        print("💾 Convertendo para formato denso para compatibilidade...")
        X_dense = X_final.toarray()
        feature_names = list(numeric_features) + list(encoded_features)
        X_df = pd.DataFrame(X_dense, columns=feature_names, index=X.index)

        return X_df

    def save_preprocessing_artifacts(self):
        """Salva artefatos do pré-processamento"""
        joblib.dump(self.scaler, 'advanced_scaler.pkl')
        joblib.dump(self.encoder, 'advanced_encoder.pkl')
        joblib.dump(self.feature_info, 'feature_info.pkl')
        print("💾 Artefatos de pré-processamento salvos")


class KohonenAdvancedAnalyzer:
    """Analisador avançado para visualizações e interpretação do SOM"""

    def __init__(self):
        self.umatrix = None
        self.activation_map = None
        self.natural_clusters = None

    def create_comprehensive_visualizations(self, som, X, original_features=None):
        """Cria visualizações abrangentes da rede de Kohonen"""
        print("\n🎨 GERANDO VISUALIZAÇÕES AVANÇADAS")
        print("=" * 40)

        if som is None:
            raise ValueError("Rede não treinada!")

        data = X.values.astype(np.float32)

        # 1. U-MATRIX PRINCIPAL - FORMATO MELHORADO
        print("1. Gerando U-Matrix Aprimorada...")
        self._create_enhanced_umatrix(som)

        # 2. MAPA DE ATIVAÇÃO - FORMATO MELHORADO
        print("2. Gerando Mapa de Ativação Aprimorado...")
        self._create_enhanced_activation_map(som, data)

        # 3. CLUSTERS NATURAIS DO SOM (SEM K-MEANS)
        print("3. Gerando Clusters Naturais...")
        self.natural_clusters = self._create_natural_clusters(som, data)

        print("✅ Visualizações salvas com sucesso!")

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
            batch = data[i:i + batch_size]
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
                             color='white' if self.activation_map[i, j] > np.percentile(self.activation_map,
                                                                                        70) else 'black',
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
        print("   🎯 Identificando clusters naturais baseados em densidade...")

        # Normalizar mapa de ativação
        normalized_activation = self.activation_map / np.max(self.activation_map)

        # Identificar regiões de alta densidade
        high_density_mask = normalized_activation > density_threshold

        # Usar connected components para encontrar clusters naturais
        labeled_array, num_clusters = label(high_density_mask)

        print(f"   📊 Encontrados {num_clusters} clusters naturais")

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
        print("\n🔍 ANÁLISE DE CLUSTERS DO SOM (SEM K-MEANS)")
        print("=" * 55)

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
        print("   ⚖️  Balanceando atribuição de clusters...")

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

        print(f"   📊 Distribuição inicial: {len(unique_clusters)} clusters")

        # Balancear clusters muito pequenos ou muito grandes
        balanced_assignments = self._redistribute_clusters(
            initial_assignments, counts, total_points,
            max_clusters, min_cluster_size_ratio
        )

        return balanced_assignments

    def _redistribute_clusters(self, assignments, counts, total_points, max_clusters, min_cluster_size_ratio):
        """Redistribui pontos para balancear clusters"""
        min_cluster_size = int(total_points * min_cluster_size_ratio)

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

        print(f"   ✅ Clusters balanceados: {len(valid_clusters)} clusters válidos")
        return balanced_assignments

    def _analyze_cluster_distribution(self, df):
        """Analisa e visualiza a distribuição dos clusters"""
        cluster_dist = df['CLUSTER_SOM'].value_counts().sort_index()

        # Filtrar cluster 0 (ruído/background)
        filtered_dist = cluster_dist[cluster_dist.index != 0]

        print(f"\n📊 DISTRIBUIÇÃO DOS CLUSTERS (excluindo ruído):")
        print(f"   • Total de clusters: {len(filtered_dist)}")
        print(f"   • Registros em clusters: {filtered_dist.sum():,}")
        print(f"   • Registros como ruído: {cluster_dist.get(0, 0):,}")

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
        print("\n📈 CARACTERÍSTICAS POR CLUSTER:")

        # Filtrar cluster 0 (ruído)
        valid_clusters = sorted([c for c in df['CLUSTER_SOM'].unique() if c != 0])

        for cluster_id in valid_clusters:
            cluster_data = df[df['CLUSTER_SOM'] == cluster_id]
            size = len(cluster_data)
            percentage = (size / len(df)) * 100

            print(f"\n🔸 CLUSTER {cluster_id}: {size:,} registros ({percentage:.1f}%)")
            print("   " + "─" * 40)

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
            print("   🏷️  Características principais:")
            for col, (value, percentage) in list(categorical_insights.items())[:6]:  # Mais características
                print(f"     • {col}: {value} ({percentage:.1f}%)")

        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("   📊 Estatísticas numéricas:")
            for col in list(numeric_cols)[:4]:  # Mais colunas numéricas
                stats = cluster_data[col].describe()
                print(f"     • {col}: avg={stats['mean']:.1f}, min={stats['min']:.1f}, max={stats['max']:.1f}")


class SOMHyperparameterOptimizer:
    """Otimizador de hiperparâmetros do SOM"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_params = None
        self.optimization_history = []

    def optimize_parameters(self, data, param_grid, max_evaluations=20):
        """Otimiza hiperparâmetros do SOM usando busca em grade"""
        print("\n🎯 OTIMIZAÇÃO DE HIPERPARÂMETROS DO SOM")
        print("=" * 45)

        best_score = -float('inf')
        best_params = None

        evaluations = 0

        # Gerar combinações de parâmetros
        param_combinations = self._generate_param_combinations(param_grid)

        for params in param_combinations:
            if evaluations >= max_evaluations:
                break

            try:
                print(f"   🔍 Testando: {params}")

                # Treinar SOM com parâmetros atuais
                trainer = MemoryEfficientSOMTrainer(random_state=self.random_state)
                som, q_error, t_error = trainer.train_kohonen_network(data, **params)

                # Avaliar qualidade
                analyzer = KohonenAdvancedAnalyzer()
                analyzer.create_comprehensive_visualizations(som, data)
                neuron_clusters = analyzer.get_neuron_clusters()

                # Mapear pontos para clusters
                cluster_assignments = []
                for sample in data:
                    winner = som.winner(sample)
                    cluster_id = neuron_clusters[winner] if neuron_clusters is not None else 0
                    cluster_assignments.append(cluster_id)

                # Calcular score composto
                score = self._calculate_optimization_score(
                    q_error, t_error, cluster_assignments, data, som
                )

                self.optimization_history.append({
                    'params': params,
                    'q_error': q_error,
                    't_error': t_error,
                    'score': score
                })

                print(f"     ✅ Score: {score:.4f} (QE: {q_error:.4f}, TE: {t_error:.4f})")

                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"     🎉 Novo melhor score!")

            except Exception as e:
                print(f"     ❌ Erro: {e}")
                continue

            evaluations += 1

        self.best_params = best_params
        print(f"\n🏆 MELHORES PARÂMETROS: {best_params}")
        print(f"🏆 MELHOR SCORE: {best_score:.4f}")

        return best_params

    def _generate_param_combinations(self, param_grid):
        """Gera combinações de parâmetros para busca em grade"""
        from itertools import product

        keys = param_grid.keys()
        values = param_grid.values()

        for combination in product(*values):
            yield dict(zip(keys, combination))

    def _calculate_optimization_score(self, q_error, t_error, clusters, data, som):
        """Calcula score composto para otimização"""
        # Normalizar erros (menor = melhor)
        q_score = 1.0 / (1.0 + q_error)
        t_score = 1.0 / (1.0 + t_error)

        # Avaliar qualidade de clusters
        valid_mask = np.array(clusters) != 0
        valid_data = data[valid_mask]
        valid_clusters = np.array(clusters)[valid_mask]

        if len(np.unique(valid_clusters)) >= 2:
            try:
                sil_score = silhouette_score(valid_data, valid_clusters)
            except:
                sil_score = 0.0
        else:
            sil_score = 0.0

        # Score composto (pesos podem ser ajustados)
        composite_score = 0.4 * q_score + 0.3 * t_score + 0.3 * sil_score

        return composite_score


def main():
    """Função principal do pipeline avançado"""
    parser = argparse.ArgumentParser(description='Pipeline Avançado de Rede de Kohonen com Análise Interpretável')
    parser.add_argument('--input', default='SPSafe_2022.csv', help='Arquivo CSV de entrada')
    parser.add_argument('--output', default='X_ready_advanced.parquet', help='Arquivo de saída')
    parser.add_argument('--sample_frac', type=float, default=0.3, help='Fração de amostragem (0.1-1.0)')
    parser.add_argument('--iterations', type=int, default=5000, help='Iterações do SOM')
    parser.add_argument('--max_clusters', type=int, default=12, help='Número máximo de clusters')
    parser.add_argument('--map_size', type=int, default=None, help='Tamanho do mapa (opcional)')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma do SOM')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='Taxa de aprendizado')
    parser.add_argument('--optimize', action='store_true', help='Otimizar hiperparâmetros automaticamente')

    args = parser.parse_args()

    print("=" * 70)
    print("🧠 PIPELINE AVANÇADO - REDE DE KOHONEN (SOM PURO)")
    print("=" * 70)

    # 1. PRÉ-PROCESSAMENTO AVANÇADO
    print("\n🎯 FASE 1: PRÉ-PROCESSAMENTO E ANÁLISE EXPLORATÓRIA")
    preprocessor = AdvancedDataPreprocessor()

    df = preprocessor.load_and_analyze_data(args.input, args.sample_frac)
    preprocessor.create_eda_visualizations(df)

    features_df = preprocessor.enhanced_feature_engineering(df)
    X_processed = preprocessor.smart_encoding(features_df)

    X_processed.to_parquet(args.output, index=False)
    preprocessor.save_preprocessing_artifacts()
    print(f"💾 Dados processados salvos: {args.output}")

    # 2. TREINAMENTO DA REDE DE KOHONEN (COM OU SEM OTIMIZAÇÃO)
    print("\n🎯 FASE 2: TREINAMENTO DA REDE DE KOHONEN")

    data_for_training = X_processed.values.astype(np.float32)

    if args.optimize:
        print("   🔧 Executando otimização de hiperparâmetros...")
        optimizer = SOMHyperparameterOptimizer(random_state=RANDOM_STATE)

        param_grid = {
            'som_x': [20, 25, 30],
            'som_y': [20, 25, 30],
            'sigma': [0.8, 1.0, 1.2, 1.5],
            'learning_rate': [0.3, 0.5, 0.7],
            'iterations': [3000, 5000, 8000]
        }

        best_params = optimizer.optimize_parameters(data_for_training, param_grid)

        # Usar melhores parâmetros
        kohonen_trainer = MemoryEfficientSOMTrainer(random_state=RANDOM_STATE)
        som, q_error, t_error = kohonen_trainer.train_kohonen_network(
            data_for_training, **best_params
        )
    else:
        # Usar parâmetros manuais
        kohonen_trainer = MemoryEfficientSOMTrainer(random_state=RANDOM_STATE)
        som, q_error, t_error = kohonen_trainer.train_kohonen_network(
            data_for_training,
            som_x=args.map_size,
            som_y=args.map_size,
            sigma=args.sigma,
            learning_rate=args.learning_rate,
            iterations=args.iterations
        )

    print(f"✅ Treinamento concluído: QE={q_error:.4f}, TE={t_error:.4f}")

    # 3. VISUALIZAÇÕES AVANÇADAS
    print("\n🎯 FASE 3: VISUALIZAÇÕES E ANÁLISES")
    analyzer = KohonenAdvancedAnalyzer()
    analyzer.create_comprehensive_visualizations(som, X_processed, features_df)

    # 4. ANÁLISE DE CLUSTERS (SOM PURO)
    print("\n🎯 FASE 4: ANÁLISE DE CLUSTERS (SOM PURO)")
    interpreter = SOMClusterInterpreter(preprocessor, kohonen_trainer, analyzer)
    df_with_clusters, quality_metrics = interpreter.analyze_som_clusters(
        X_processed, df, args.max_clusters
    )

    # Salvar resultados finais
    df_with_clusters.to_parquet('dataset_com_clusters_som.parquet', index=False)
    joblib.dump(kohonen_trainer.som, 'kohonen_model_pure_som.pkl')

    # Salvar métricas de qualidade
    if quality_metrics:
        joblib.dump(quality_metrics, 'cluster_quality_metrics.pkl')

    print("\n" + "=" * 70)
    print("🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print("\n📁 ARQUIVOS GERADOS:")
    print("   • kohonen_umatrix_enhanced.png      - U-Matrix aprimorada")
    print("   • kohonen_activation_enhanced.png   - Mapa de ativação aprimorado")
    print("   • kohonen_natural_clusters.png      - Clusters naturais do SOM")
    print("   • som_cluster_distribution_enhanced.png - Distribuição balanceada")
    print("   • eda_*.png                         - Análises exploratórias")
    print("   • dataset_com_clusters_som.parquet  - Dataset com clusters SOM")
    print("   • kohonen_model_pure_som.pkl        - Modelo SOM puro")
    print("   • cluster_quality_metrics.pkl       - Métricas de qualidade")
    print("   • advanced_*.pkl                    - Artefatos do pré-processamento")
    print(f"   • {args.output}                    - Dados processados")
    print("\n🔍 PRÓXIMOS PASSOS:")
    print("   • Analisar os mapas aprimorados para identificar padrões")
    print("   • Verificar clusters naturais identificados pelo SOM")
    print("   • Ajustar parâmetros de densidade se necessário")
    print("   • Utilizar clusters para análises específicas de segurança")


if __name__ == '__main__':
    main()