"""
Módulo de pré-processamento de dados
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse
import joblib
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

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