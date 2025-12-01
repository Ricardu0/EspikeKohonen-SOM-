"""
CORREÇÕES CRÍTICAS PARA data/preprocessor.py
Problemas identificados nos logs:
1. Coordenadas geográficas corrompidas (-245775413879999)
2. 0 clusters identificados (mapa desorganizado)
3. Alta dimensionalidade (82 features) sem PCA
4. Normalização inadequada (StandardScaler com outliers extremos)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # ✅ MUDANÇA 1
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
        # ✅ CORREÇÃO 1: RobustScaler ao invés de StandardScaler
        self.scaler = RobustScaler()  # Resistente a outliers extremos
        self.encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.pca = PCA(n_components=pca_variance, random_state=42)
        self.feature_info = {}
        self.categorical_mappings = {}
        self.outlier_stats = {}

    def clean_geographic_outliers(self, df):
        """
        ✅ CORREÇÃO CRÍTICA: Limpeza agressiva de coordenadas geográficas
        Problema: LATITUDE min=-245775413879999 (impossível!)
        Solução: Filtrar apenas coordenadas válidas para São Paulo
        """
        print("\n🗺️  LIMPEZA CRÍTICA DE COORDENADAS GEOGRÁFICAS")
        print("=" * 50)
        
        initial_size = len(df)
        
        # Limites válidos para São Paulo (expandidos para margem)
        LAT_MIN, LAT_MAX = -25.5, -19.0
        LON_MIN, LON_MAX = -49.0, -44.0
        
        if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
            # Forçar conversão numérica
            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
            
            # Log dos problemas ANTES da limpeza
            lat_outliers = ((df['LATITUDE'] < LAT_MIN) | (df['LATITUDE'] > LAT_MAX)).sum()
            lon_outliers = ((df['LONGITUDE'] < LON_MIN) | (df['LONGITUDE'] > LON_MAX)).sum()
            nan_coords = df[['LATITUDE', 'LONGITUDE']].isnull().any(axis=1).sum()
            
            print(f"⚠️  OUTLIERS DETECTADOS:")
            print(f"   • Latitudes inválidas: {lat_outliers:,} ({lat_outliers/len(df)*100:.2f}%)")
            print(f"   • Longitudes inválidas: {lon_outliers:,} ({lon_outliers/len(df)*100:.2f}%)")
            print(f"   • Coordenadas NaN: {nan_coords:,} ({nan_coords/len(df)*100:.2f}%)")
            
            # Estratégia 1: REMOVER outliers extremos
            mask_valid = (
                df['LATITUDE'].notna() &
                df['LONGITUDE'].notna() &
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
                print(f"\n📊 COORDENADAS VÁLIDAS:")
                print(f"   • Latitude: [{df_clean['LATITUDE'].min():.4f}, {df_clean['LATITUDE'].max():.4f}]")
                print(f"   • Longitude: [{df_clean['LONGITUDE'].min():.4f}, {df_clean['LONGITUDE'].max():.4f}]")
                print(f"   • Média Latitude: {df_clean['LATITUDE'].mean():.4f}")
                print(f"   • Média Longitude: {df_clean['LONGITUDE'].mean():.4f}")
            
            self.outlier_stats['geographic_outliers_removed'] = removed
            
            return df_clean
        
        return df

    def detect_spatial_outliers(self, df):
        """
        ✅ CORREÇÃO 2: Detecção de outliers espaciais usando LOF
        Literatura: Local Outlier Factor identifica pontos isolados
        """
        print("\n🔍 DETECÇÃO AVANÇADA DE OUTLIERS ESPACIAIS (LOF)")
        print("=" * 55)
        
        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            return df
        
        # Remover NaN
        spatial_mask = df['LATITUDE'].notna() & df['LONGITUDE'].notna()
        df_spatial = df[spatial_mask].copy()
        
        if len(df_spatial) < 100:
            print("⚠️  Dados insuficientes para detecção LOF")
            return df
        
        # Aplicar LOF
        spatial_features = df_spatial[['LATITUDE', 'LONGITUDE']].values
        
        # Ajustar parâmetros baseado no tamanho
        n_neighbors = min(20, len(df_spatial) // 100)
        contamination = 0.05  # 5% esperado
        
        print(f"   • LOF: n_neighbors={n_neighbors}, contamination={contamination*100:.1f}%")
        
        try:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outlier_labels = lof.fit_predict(spatial_features)
            
            n_outliers = (outlier_labels == -1).sum()
            
            print(f"\n✅ OUTLIERS ESPACIAIS:")
            print(f"   • Detectados: {n_outliers:,} ({n_outliers/len(df_spatial)*100:.2f}%)")
            
            # Remover outliers
            df_clean = df_spatial[outlier_labels == 1].copy()
            
            print(f"   • Mantidos: {len(df_clean):,}")
            
            self.outlier_stats['spatial_outliers_removed'] = n_outliers
            
            return df_clean
            
        except Exception as e:
            logger.warning(f"Erro no LOF: {e}")
            return df

    def detect_multivariate_outliers(self, X_numeric):
        """
        ✅ CORREÇÃO 3: Isolation Forest para outliers multivariados
        """
        print("\n🌲 DETECÇÃO MULTIVARIADA (ISOLATION FOREST)")
        print("=" * 50)
        
        if X_numeric.shape[1] == 0 or X_numeric.shape[0] < 10:
            return np.ones(X_numeric.shape[0], dtype=bool)
        
        contamination = 0.05
        
        print(f"   • Features: {X_numeric.shape[1]}, Contaminação: {contamination*100:.1f}%")
        
        try:
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            outlier_labels = iso_forest.fit_predict(X_numeric)
            
            inlier_mask = outlier_labels == 1
            n_outliers = (~inlier_mask).sum()
            
            print(f"\n✅ OUTLIERS MULTIVARIADOS:")
            print(f"   • Detectados: {n_outliers:,} ({n_outliers/len(X_numeric)*100:.2f}%)")
            print(f"   • Mantidos: {inlier_mask.sum():,}")
            
            self.outlier_stats['multivariate_outliers_removed'] = n_outliers
            
            return inlier_mask
            
        except Exception as e:
            logger.warning(f"Erro no Isolation Forest: {e}")
            return np.ones(X_numeric.shape[0], dtype=bool)

    def apply_pca_reduction(self, X_dense, variance_threshold=0.90):
        """
        ✅ CORREÇÃO 4: Redução de dimensionalidade com PCA
        Problema: 82 features é muito para SOM 20x20
        Solução: Reduzir mantendo 90% da variância
        """
        print(f"\n🎯 REDUÇÃO DE DIMENSIONALIDADE (PCA)")
        print("=" * 50)
        
        if X_dense.shape[1] <= 10:
            print("   • PCA não aplicado (poucas features)")
            return X_dense, list(range(X_dense.shape[1]))
        
        original_dims = X_dense.shape[1]
        
        # Aplicar PCA
        X_pca = self.pca.fit_transform(X_dense)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        n_components = X_pca.shape[1]
        reduction = (1 - n_components/original_dims) * 100
        
        print(f"   • Dimensões originais: {original_dims}")
        print(f"   • Dimensões após PCA: {n_components}")
        print(f"   • Variância explicada: {explained_variance*100:.2f}%")
        print(f"   • Redução: {reduction:.1f}%")
        
        # Top componentes por variância
        print(f"\n   📊 TOP 5 COMPONENTES:")
        for i, var in enumerate(self.pca.explained_variance_ratio_[:5]):
            print(f"     • PC{i+1}: {var*100:.2f}%")
        
        return X_pca, [f'PC{i+1}' for i in range(n_components)]

    def smart_encoding(self, features_df):
        """
        ✅ PIPELINE COMPLETO COM TODAS AS CORREÇÕES
        """
        print("\n🔠 CODIFICAÇÃO INTELIGENTE + LIMPEZA")
        print("=" * 45)

        X = features_df.copy()

        # Separar numéricas e categóricas
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        print(f"🔢 Features numéricas: {len(numeric_features)}")
        print(f"🏷️  Features categóricas: {len(categorical_features)}")

        # Processar missing values
        print("\n🔄 Tratando valores missing...")
        for col in categorical_features:
            X[col] = X[col].fillna('NÃO_INFORMADO')
            if X[col].nunique() > 20:
                top_categories = X[col].value_counts().head(15).index
                X[col] = X[col].apply(lambda x: x if x in top_categories else 'OUTROS')
                print(f"   • {col}: reduzido para 16 categorias")

        for col in numeric_features:
            X[col] = X[col].fillna(X[col].median())

        # ✅ PASSO 1: Detectar outliers ANTES da normalização
        if len(numeric_features) > 0:
            print("\n🔍 Detectando outliers multivariados...")
            inlier_mask = self.detect_multivariate_outliers(X[numeric_features].values)
            X = X[inlier_mask].copy()
            print(f"   • Dataset após limpeza: {X.shape}")

        # ✅ PASSO 2: One-Hot Encoding
        print("\n🎯 One-Hot Encoding...")
        if categorical_features:
            X_encoded = self.encoder.fit_transform(X[categorical_features])
            encoded_features = self.encoder.get_feature_names_out(categorical_features)
            print(f"   • {len(categorical_features)} → {len(encoded_features)} colunas")
        else:
            X_encoded = sparse.csr_matrix((X.shape[0], 0))
            encoded_features = []

        # ✅ PASSO 3: RobustScaler (resistente a outliers)
        print("\n📏 Normalizando com RobustScaler...")
        if numeric_features:
            X_scaled = self.scaler.fit_transform(X[numeric_features])
            X_scaled = sparse.csr_matrix(X_scaled)
            print("   • RobustScaler aplicado (usa mediana/IQR)")
        else:
            X_scaled = sparse.csr_matrix((X.shape[0], 0))

        # Combinar
        X_combined = sparse.hstack([X_scaled, X_encoded])
        X_dense = X_combined.toarray()

        print(f"\n✅ Shape antes do PCA: {X_dense.shape}")

        # ✅ PASSO 4: PCA para redução de ruído
        X_final, feature_names = self.apply_pca_reduction(X_dense, variance_threshold=0.90)

        # Criar DataFrame final
        X_df = pd.DataFrame(X_final, columns=feature_names, index=X.index)

        # Estatísticas finais
        print(f"\n📊 RESUMO DE LIMPEZA:")
        for key, value in self.outlier_stats.items():
            print(f"   • {key}: {value:,}")
        
        print(f"\n✅ DATASET FINAL: {X_df.shape}")

        return X_df

    def full_pipeline(self, csv_path, sample_frac=None):
        """
        ✅ PIPELINE COMPLETO CORRIGIDO
        """ 
        
        print("\n📊 CARREGAMENTO E ANÁLISE")
        print("=" * 50)
        
        # Carregar dados
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo {csv_path} não encontrado!")
        
        try:
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1', low_memory=False)
        
        if sample_frac and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"✅ Amostra: {len(df):,} ({sample_frac*100:.1f}%)")
        
        print(f"📈 Shape inicial: {df.shape}")
        
        # 1. Limpeza crítica de coordenadas
        df = self.clean_geographic_outliers(df)
        
        # 2. Detecção espacial (LOF)
        df = self.detect_spatial_outliers(df)
        
        # 3. Engenharia de features
        features_df = self.enhanced_feature_engineering(df)
        
        # 4. Codificação + PCA
        X_final = self.smart_encoding(features_df)
        
        # 5. Salvar artefatos
        self.save_preprocessing_artifacts()
        
        return X_final

    def enhanced_feature_engineering(self, df):
        """Engenharia de features (mantida do código original)"""
        print("\n🔧 ENGENHARIA DE FEATURES")
        print("=" * 40)
        
        df = df.copy()
        
        # Features temporais
        temporal_features = []
        if 'DATA_OCORRENCIA' in df.columns:
            df['DATA_OCORRENCIA'] = pd.to_datetime(df['DATA_OCORRENCIA'], errors='coerce')
            df['DIA_SEMANA'] = df['DATA_OCORRENCIA'].dt.day_name()
            df['MES'] = df['DATA_OCORRENCIA'].dt.month_name()
            df['ANO'] = df['DATA_OCORRENCIA'].dt.year
            df['FIM_SEMANA'] = df['DATA_OCORRENCIA'].dt.weekday >= 5
            temporal_features.extend(['DIA_SEMANA', 'MES', 'ANO', 'FIM_SEMANA'])
        
        if 'HORA_OCORRENCIA' in df.columns:
            def parse_hour(h):
                try:
                    s = str(h).strip().replace('h', ':').replace('.', ':')
                    if ':' in s:
                        return int(s.split(':')[0])
                    elif s.isdigit():
                        return int(s[:2]) if len(s) > 2 else int(s)
                except:
                    return np.nan
                return np.nan
            
            df['HORA'] = df['HORA_OCORRENCIA'].apply(parse_hour)
            
            bins = [-1, 5, 9, 12, 15, 18, 21, 24]
            labels = ['Madrugada', 'Manhã Cedo', 'Manhã', 'Tarde Cedo', 'Tarde', 'Noite', 'Noite Tardia']
            df['PERIODO_DIA'] = pd.cut(df['HORA'], bins=bins, labels=labels).astype(str)
            temporal_features.extend(['HORA', 'PERIODO_DIA'])
        
        # Features geográficas
        geographic_features = []
        if all(col in df.columns for col in ['LATITUDE', 'LONGITUDE']):
            df['TEM_COORDENADAS'] = df['LATITUDE'].notna() & df['LONGITUDE'].notna()
            geographic_features.extend(['LATITUDE', 'LONGITUDE', 'TEM_COORDENADAS'])
        
        # Features categóricas
        categorical_features = [
            'TIPO_LOCAL', 'NATUREZA_APURADA', 'BAIRRO', 'UF'
        ]
        
        available_categorical = [col for col in categorical_features if col in df.columns]
        all_features = temporal_features + geographic_features + available_categorical
        available_features = [col for col in all_features if col in df.columns]
        
        print(f"📋 FEATURES SELECIONADAS:")
        print(f"   • Temporais: {len([f for f in temporal_features if f in available_features])}")
        print(f"   • Geográficas: {len([f for f in geographic_features if f in available_features])}")
        print(f"   • Categóricas: {len([f for f in available_categorical if f in available_features])}")
        
        features_df = df[available_features].copy()
        
        return features_df

    def save_preprocessing_artifacts(self):
        """Salva artefatos"""
        joblib.dump(self.scaler, 'advanced_scaler.pkl')
        joblib.dump(self.encoder, 'advanced_encoder.pkl')
        joblib.dump(self.pca, 'advanced_pca.pkl')
        joblib.dump(self.feature_info, 'feature_info.pkl')
        joblib.dump(self.outlier_stats, 'outlier_stats.pkl')
        print("💾 Artefatos salvos")