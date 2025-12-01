"""
Módulo de pré-processamento de dados - VERSÃO OTIMIZADA
Correções críticas implementadas:
1. Limpeza robusta de outliers geográficos
2. Normalização resistente a ruído (RobustScaler)
3. Redução de dimensionalidade com PCA
4. Detecção avançada de outliers (LOF + IsolationForest)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy import sparse
import joblib
import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class AdvancedDataPreprocessor:
    """Pré-processamento avançado com detecção robusta de outliers"""

    def __init__(self, pca_variance=0.90):
        # ✅ MUDANÇA 1: RobustScaler ao invés de StandardScaler
        self.scaler = RobustScaler()  # Resistente a outliers
        self.encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.pca = PCA(n_components=pca_variance, random_state=42)
        self.feature_info = {}
        self.categorical_mappings = {}
        self.outlier_stats = {}

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
            df = df.sample(frac=sample_frac, random_state=42)
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

    def clean_geographic_outliers(self, df):
        """
        ✅ CORREÇÃO CRÍTICA 1: Limpeza robusta de coordenadas geográficas
        Remove valores impossíveis que estavam destruindo o SOM
        """
        print("\n🗺️  LIMPEZA CRÍTICA DE COORDENADAS GEOGRÁFICAS")
        print("=" * 50)
        
        initial_size = len(df)
        
        # Limites válidos para São Paulo
        LAT_MIN, LAT_MAX = -25.0, -19.0
        LON_MIN, LON_MAX = -48.0, -44.0
        
        if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
            
            # Log dos outliers antes da limpeza
            lat_outliers = ((df['LATITUDE'] < LAT_MIN) | (df['LATITUDE'] > LAT_MAX)).sum()
            lon_outliers = ((df['LONGITUDE'] < LON_MIN) | (df['LONGITUDE'] > LON_MAX)).sum()
            
            print(f"⚠️  OUTLIERS DETECTADOS:")
            print(f"   • Latitudes inválidas: {lat_outliers:,} ({lat_outliers/len(df)*100:.2f}%)")
            print(f"   • Longitudes inválidas: {lon_outliers:,} ({lon_outliers/len(df)*100:.2f}%)")
            
            # Estratégia 1: Remover outliers extremos
            mask_valid = (
                (df['LATITUDE'].between(LAT_MIN, LAT_MAX)) &
                (df['LONGITUDE'].between(LON_MIN, LON_MAX))
            )
            
            df_clean = df[mask_valid].copy()
            removed = initial_size - len(df_clean)
            
            print(f"\n✅ LIMPEZA CONCLUÍDA:")
            print(f"   • Registros removidos: {removed:,} ({removed/initial_size*100:.2f}%)")
            print(f"   • Registros mantidos: {len(df_clean):,}")
            
            # Estatísticas após limpeza
            if len(df_clean) > 0:
                print(f"\n📊 COORDENADAS APÓS LIMPEZA:")
                print(f"   • Latitude: [{df_clean['LATITUDE'].min():.4f}, {df_clean['LATITUDE'].max():.4f}]")
                print(f"   • Longitude: [{df_clean['LONGITUDE'].min():.4f}, {df_clean['LONGITUDE'].max():.4f}]")
            
            self.outlier_stats['geographic_outliers_removed'] = removed
            
            return df_clean
        
        return df

    def detect_spatial_outliers(self, df):
        """
        ✅ CORREÇÃO CRÍTICA 2: Detecção de outliers espaciais usando LOF
        Implementa técnica validada pela literatura para dados criminais
        """
        print("\n🔍 DETECÇÃO AVANÇADA DE OUTLIERS ESPACIAIS (LOF)")
        print("=" * 55)
        
        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            return df
        
        # Remover NaN antes do LOF
        spatial_mask = df['LATITUDE'].notna() & df['LONGITUDE'].notna()
        df_spatial = df[spatial_mask].copy()
        
        if len(df_spatial) < 100:
            print("⚠️  Dados insuficientes para detecção LOF")
            return df
        
        # Aplicar LOF nas coordenadas geográficas
        spatial_features = df_spatial[['LATITUDE', 'LONGITUDE']].values
        
        # Ajustar n_neighbors baseado no tamanho do dataset
        n_neighbors = min(20, len(df_spatial) // 100)
        contamination = 0.05  # Espera-se 5% de outliers
        
        print(f"   • Aplicando LOF com n_neighbors={n_neighbors}")
        print(f"   • Contaminação esperada: {contamination*100:.1f}%")
        
        try:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outlier_labels = lof.fit_predict(spatial_features)
            
            # -1 = outlier, 1 = inlier
            n_outliers = (outlier_labels == -1).sum()
            
            print(f"\n✅ OUTLIERS ESPACIAIS DETECTADOS:")
            print(f"   • Outliers encontrados: {n_outliers:,} ({n_outliers/len(df_spatial)*100:.2f}%)")
            
            # Adicionar flag de outlier ao dataframe original
            df_spatial['SPATIAL_OUTLIER'] = outlier_labels == -1
            
            # Mesclar de volta ao dataframe original
            df = df.merge(
                df_spatial[['SPATIAL_OUTLIER']], 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            df['SPATIAL_OUTLIER'] = df['SPATIAL_OUTLIER'].fillna(False)
            
            # Remover outliers espaciais
            df_clean = df[~df['SPATIAL_OUTLIER']].copy()
            df_clean = df_clean.drop('SPATIAL_OUTLIER', axis=1)
            
            print(f"   • Registros mantidos: {len(df_clean):,}")
            
            self.outlier_stats['spatial_outliers_removed'] = n_outliers
            
            return df_clean
            
        except Exception as e:
            logger.warning(f"Erro no LOF: {e}. Pulando detecção espacial.")
            return df

    def detect_multivariate_outliers(self, X_numeric):
        """
        ✅ CORREÇÃO CRÍTICA 3: Detecção multivariada de outliers
        Usa Isolation Forest para detectar outliers em todas as features numéricas
        """
        print("\n🌲 DETECÇÃO MULTIVARIADA DE OUTLIERS (ISOLATION FOREST)")
        print("=" * 60)
        
        if X_numeric.shape[1] == 0:
            return np.ones(X_numeric.shape[0], dtype=bool)
        
        contamination = 0.05  # 5% esperado de outliers
        
        print(f"   • Features numéricas analisadas: {X_numeric.shape[1]}")
        print(f"   • Contaminação esperada: {contamination*100:.1f}%")
        
        try:
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            outlier_labels = iso_forest.fit_predict(X_numeric)
            
            # -1 = outlier, 1 = inlier
            inlier_mask = outlier_labels == 1
            n_outliers = (~inlier_mask).sum()
            
            print(f"\n✅ OUTLIERS MULTIVARIADOS DETECTADOS:")
            print(f"   • Outliers encontrados: {n_outliers:,} ({n_outliers/len(X_numeric)*100:.2f}%)")
            print(f"   • Registros mantidos: {inlier_mask.sum():,}")
            
            self.outlier_stats['multivariate_outliers_removed'] = n_outliers
            
            return inlier_mask
            
        except Exception as e:
            logger.warning(f"Erro no Isolation Forest: {e}")
            return np.ones(X_numeric.shape[0], dtype=bool)

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
        if len(missing_data) > 0:
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
        """
        ✅ CORREÇÃO CRÍTICA 4: Codificação com normalização robusta e PCA
        """
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

        # ✅ DETECÇÃO DE OUTLIERS MULTIVARIADOS (antes da codificação)
        if len(numeric_features) > 0:
            inlier_mask = self.detect_multivariate_outliers(X[numeric_features].values)
            X = X[inlier_mask].copy()
            print(f"\n   • Dataset após remoção de outliers: {X.shape}")

        print("\n🎯 Aplicando codificação one-hot...")
        if categorical_features:
            X_encoded = self.encoder.fit_transform(X[categorical_features])
            encoded_features = self.encoder.get_feature_names_out(categorical_features)
            print(f"   • {len(categorical_features)} features → {len(encoded_features)} colunas codificadas")
        else:
            X_encoded = sparse.csr_matrix((X.shape[0], 0))
            encoded_features = []

        # ✅ NORMALIZAÇÃO ROBUSTA (resistente a outliers)
        print("\n🎯 Aplicando normalização robusta (RobustScaler)...")
        if numeric_features:
            X_scaled = self.scaler.fit_transform(X[numeric_features])
            X_scaled = sparse.csr_matrix(X_scaled)
            print("   • RobustScaler aplicado (resistente a outliers remanescentes)")
        else:
            X_scaled = sparse.csr_matrix((X.shape[0], 0))

        X_final = sparse.hstack([X_scaled, X_encoded])

        print(f"\n✅ Dataset antes do PCA: {X_final.shape}")
        print(f"   • Matriz esparsa: {X_final.getnnz():,} elementos não-zero")
        print(f"   • Densidade: {X_final.getnnz() / (X_final.shape[0] * X_final.shape[1]):.4f}")

        # ✅ REDUÇÃO DE DIMENSIONALIDADE COM PCA
        print("\n🎯 APLICANDO PCA PARA REDUÇÃO DE RUÍDO")
        print("=" * 45)
        
        X_dense = X_final.toarray()
        
        if X_dense.shape[1] > 10:  # Só aplica PCA se tiver muitas features
            X_pca = self.pca.fit_transform(X_dense)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            
            print(f"   • Dimensões originais: {X_dense.shape[1]}")
            print(f"   • Dimensões após PCA: {X_pca.shape[1]}")
            print(f"   • Variância explicada: {explained_variance*100:.2f}%")
            print(f"   • Redução: {(1 - X_pca.shape[1]/X_dense.shape[1])*100:.1f}%")
            
            feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            X_df = pd.DataFrame(X_pca, columns=feature_names, index=X.index)
        else:
            print("   • PCA não aplicado (poucas features)")
            feature_names = list(numeric_features) + list(encoded_features)
            X_df = pd.DataFrame(X_dense, columns=feature_names, index=X.index)

        # Estatísticas finais
        print(f"\n📊 RESUMO DE LIMPEZA DE OUTLIERS:")
        for key, value in self.outlier_stats.items():
            print(f"   • {key}: {value:,}")

        return X_df

    def save_preprocessing_artifacts(self):
        """Salva artefatos do pré-processamento"""
        joblib.dump(self.scaler, 'advanced_scaler.pkl')
        joblib.dump(self.encoder, 'advanced_encoder.pkl')
        joblib.dump(self.pca, 'advanced_pca.pkl')
        joblib.dump(self.feature_info, 'feature_info.pkl')
        joblib.dump(self.outlier_stats, 'outlier_stats.pkl')
        print("💾 Artefatos de pré-processamento salvos")

    def full_pipeline(self, csv_path, sample_frac=None):
        """
        Pipeline completo com todas as correções aplicadas
        """
        # 1. Carregar dados
        df = self.load_and_analyze_data(csv_path, sample_frac)
        
        # 2. Limpeza crítica de coordenadas
        df = self.clean_geographic_outliers(df)
        
        # 3. Detecção de outliers espaciais (LOF)
        df = self.detect_spatial_outliers(df)
        
        # 4. Criar visualizações
        self.create_eda_visualizations(df)
        
        # 5. Engenharia de features
        features_df = self.enhanced_feature_engineering(df)
        
        # 6. Codificação inteligente (inclui detecção multivariada + PCA)
        X_final = self.smart_encoding(features_df)
        
        # 7. Salvar artefatos
        self.save_preprocessing_artifacts()
        
        return X_final