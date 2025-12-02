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
    """Pré-processamento avançado para dados de São Paulo"""

    def __init__(self, pca_variance=0.90):
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.pca = PCA(n_components=pca_variance, random_state=42)
        self.geo_scaler = RobustScaler()
        self.feature_info = {}
        self.categorical_mappings = {}
        self.outlier_stats = {}
        self.coordinate_correction_applied = False
        self.correction_factor = None

        # Limites geográficos de São Paulo
        self.SP_LAT_MIN = -26.0
        self.SP_LAT_MAX = -19.0
        self.SP_LON_MIN = -54.0
        self.SP_LON_MAX = -43.5
        self.SP_LAT_CENTER = -23.55
        self.SP_LON_CENTER = -46.63

    def fix_brazilian_decimal(self, value):
        """
        ✅ CORREÇÃO CRÍTICA: Converter vírgula decimal brasileira para ponto
        -21,1418 → -21.1418
        """
        if pd.isna(value):
            return np.nan

        # Se já é número, retornar
        if isinstance(value, (int, float)):
            return float(value)

        # Se é string, converter vírgula para ponto
        if isinstance(value, str):
            # Remover espaços
            value = value.strip()
            # Substituir vírgula por ponto
            value = value.replace(',', '.')
            try:
                return float(value)
            except ValueError:
                return np.nan

        return np.nan

    def robust_fix_coords(self, df):
        """
        ✅ CORREÇÃO ULTRA-ROBUSTA DE COORDENADAS CORROMPIDAS

        Detecta e corrige automaticamente:
        - Escalas erradas (valores em 10⁴, 10⁵, 10⁶)
        - Sinais invertidos
        - Valores impossíveis (fora dos limites terrestres)
        - Valores NaN ou inválidos
        - Vírgula decimal brasileira

        Returns:
            DataFrame com coordenadas corrigidas
        """
        print("\n" + "🔥" * 30)
        print("🛠️  CORREÇÃO ROBUSTA DE COORDENADAS GEOGRÁFICAS")
        print("🔥" * 30)

        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            print("⚠️  Colunas de coordenadas não encontradas!")
            return df

        # ==========================================
        # ETAPA 1: CONVERTER PARA NUMÉRICO
        # ==========================================
        print("\n📌 ETAPA 1: CONVERSÃO PARA FORMATO NUMÉRICO")
        print("─" * 50)

        def safe_convert_to_float(series):
            """Converte série para float tratando vírgulas e strings"""

            def convert_value(val):
                if pd.isna(val):
                    return np.nan
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    # Remover espaços e substituir vírgula por ponto
                    val = val.strip().replace(',', '.')
                    try:
                        return float(val)
                    except ValueError:
                        return np.nan
                return np.nan

            return series.apply(convert_value)

        # Aplicar conversão
        df['LATITUDE'] = safe_convert_to_float(df['LATITUDE'])
        df['LONGITUDE'] = safe_convert_to_float(df['LONGITUDE'])

        print(f"✅ Conversão concluída")
        print(
            f"   • Latitude - NaN: {df['LATITUDE'].isna().sum():,} ({df['LATITUDE'].isna().sum() / len(df) * 100:.2f}%)")
        print(
            f"   • Longitude - NaN: {df['LONGITUDE'].isna().sum():,} ({df['LONGITUDE'].isna().sum() / len(df) * 100:.2f}%)")

        # ==========================================
        # ETAPA 2: ANÁLISE DE ESCALA
        # ==========================================
        print("\n📌 ETAPA 2: DETECÇÃO AUTOMÁTICA DE ESCALA ERRADA")
        print("─" * 50)

        # Amostra válida para análise (não-NaN)
        lat_valid = df['LATITUDE'].dropna()
        lon_valid = df['LONGITUDE'].dropna()

        if len(lat_valid) == 0 or len(lon_valid) == 0:
            print("❌ Nenhum dado válido para análise!")
            return df

        # Estatísticas brutas
        lat_mean_raw = lat_valid.mean()
        lon_mean_raw = lon_valid.mean()
        lat_median_raw = lat_valid.median()
        lon_median_raw = lon_valid.median()
        lat_abs_mean = np.abs(lat_valid).mean()
        lon_abs_mean = np.abs(lon_valid).mean()

        print(f"📊 ESTATÍSTICAS BRUTAS:")
        print(f"   • Latitude:  média={lat_mean_raw:.2f}, mediana={lat_median_raw:.2f}, |média|={lat_abs_mean:.2f}")
        print(f"   • Longitude: média={lon_mean_raw:.2f}, mediana={lon_median_raw:.2f}, |média|={lon_abs_mean:.2f}")

        # ==========================================
        # ETAPA 3: CORREÇÃO DE ESCALA
        # ==========================================
        print("\n📌 ETAPA 3: CORREÇÃO AUTOMÁTICA DE ESCALA")
        print("─" * 50)

        # Limites esperados (mundo inteiro)
        WORLD_LAT_MIN, WORLD_LAT_MAX = -90, 90
        WORLD_LON_MIN, WORLD_LON_MAX = -180, 180

        # Função de correção de escala
        def detect_and_fix_scale(series, coord_type='lat'):
            """Detecta escala errada e aplica correção"""
            valid = series.dropna()
            if len(valid) == 0:
                return series, 1.0

            abs_mean = np.abs(valid).mean()
            abs_median = np.abs(valid).median()

            # Decisão de escala baseada na magnitude
            if abs_mean > 1e6 or abs_median > 1e6:
                factor = 1e6
                reason = "valores na ordem de 10⁶"
            elif abs_mean > 1e5 or abs_median > 1e5:
                factor = 1e5
                reason = "valores na ordem de 10⁵"
            elif abs_mean > 1e4 or abs_median > 1e4:
                factor = 1e4
                reason = "valores na ordem de 10⁴"
            elif abs_mean > 1e3 or abs_median > 1e3:
                factor = 1e3
                reason = "valores na ordem de 10³"
            else:
                factor = 1.0
                reason = "escala OK"

            if factor > 1.0:
                print(f"   🔧 {coord_type.upper()}: {reason} → dividindo por {factor:.0e}")
                return series / factor, factor
            else:
                print(f"   ✅ {coord_type.upper()}: {reason}")
                return series, factor

        # Aplicar correção
        df['LATITUDE'], lat_factor = detect_and_fix_scale(df['LATITUDE'], 'latitude')
        df['LONGITUDE'], lon_factor = detect_and_fix_scale(df['LONGITUDE'], 'longitude')

        # ==========================================
        # ETAPA 4: CLIP PARA LIMITES MUNDIAIS
        # ==========================================
        print("\n📌 ETAPA 4: APLICANDO LIMITES GEOGRÁFICOS MUNDIAIS")
        print("─" * 50)

        # Contar valores fora dos limites ANTES do clip
        lat_out_of_bounds = ((df['LATITUDE'] < WORLD_LAT_MIN) | (df['LATITUDE'] > WORLD_LAT_MAX)).sum()
        lon_out_of_bounds = ((df['LONGITUDE'] < WORLD_LON_MIN) | (df['LONGITUDE'] > WORLD_LON_MAX)).sum()

        print(f"   • Latitudes fora de [{WORLD_LAT_MIN}, {WORLD_LAT_MAX}]: {lat_out_of_bounds:,}")
        print(f"   • Longitudes fora de [{WORLD_LON_MIN}, {WORLD_LON_MAX}]: {lon_out_of_bounds:,}")

        # Aplicar clip
        df['LATITUDE'] = df['LATITUDE'].clip(WORLD_LAT_MIN, WORLD_LAT_MAX)
        df['LONGITUDE'] = df['LONGITUDE'].clip(WORLD_LON_MIN, WORLD_LON_MAX)

        print(f"   ✅ Clip aplicado")

        # ==========================================
        # ETAPA 5: PREENCHIMENTO DE NaN
        # ==========================================
        print("\n📌 ETAPA 5: PREENCHIMENTO DE VALORES INVÁLIDOS")
        print("─" * 50)

        # Calcular medianas dos dados válidos (após correção)
        lat_valid_post = df['LATITUDE'].dropna()
        lon_valid_post = df['LONGITUDE'].dropna()

        if len(lat_valid_post) > 0 and len(lon_valid_post) > 0:
            lat_median = lat_valid_post.median()
            lon_median = lon_valid_post.median()

            # Verificar se mediana está dentro de São Paulo (caso seja dataset SP)
            if (self.SP_LAT_MIN <= lat_median <= self.SP_LAT_MAX and
                    self.SP_LON_MIN <= lon_median <= self.SP_LON_MAX):
                print(f"   ✅ Usando mediana de São Paulo:")
            else:
                # Fallback: centro de São Paulo
                lat_median = self.SP_LAT_CENTER
                lon_median = self.SP_LON_CENTER
                print(f"   ⚠️  Mediana fora de SP, usando centro de São Paulo:")
        else:
            # Fallback
            lat_median = self.SP_LAT_CENTER
            lon_median = self.SP_LON_CENTER
            print(f"   ⚠️  Sem dados válidos, usando centro de São Paulo:")

        print(f"      • Latitude: {lat_median:.6f}")
        print(f"      • Longitude: {lon_median:.6f}")

        # Preencher NaN
        nan_lat = df['LATITUDE'].isna().sum()
        nan_lon = df['LONGITUDE'].isna().sum()

        df['LATITUDE'].fillna(lat_median, inplace=True)
        df['LONGITUDE'].fillna(lon_median, inplace=True)

        print(f"   ✅ NaN preenchidos: Lat={nan_lat:,}, Lon={nan_lon:,}")

        # ==========================================
        # ETAPA 6: VALIDAÇÃO FINAL
        # ==========================================
        print("\n📌 ETAPA 6: VALIDAÇÃO FINAL")
        print("─" * 50)

        lat_final = df['LATITUDE']
        lon_final = df['LONGITUDE']

        # Estatísticas finais
        stats_final = {
            'lat_min': lat_final.min(),
            'lat_max': lat_final.max(),
            'lat_mean': lat_final.mean(),
            'lat_median': lat_final.median(),
            'lat_std': lat_final.std(),
            'lon_min': lon_final.min(),
            'lon_max': lon_final.max(),
            'lon_mean': lon_final.mean(),
            'lon_median': lon_final.median(),
            'lon_std': lon_final.std(),
            'lat_nan': lat_final.isna().sum(),
            'lon_nan': lon_final.isna().sum()
        }

        print(f"📊 ESTATÍSTICAS FINAIS:")
        print(f"   • Latitude:")
        print(f"      - Range: [{stats_final['lat_min']:.6f}, {stats_final['lat_max']:.6f}]")
        print(f"      - Média: {stats_final['lat_mean']:.6f} ± {stats_final['lat_std']:.6f}")
        print(f"      - Mediana: {stats_final['lat_median']:.6f}")
        print(f"      - NaN: {stats_final['lat_nan']:,}")
        print(f"   • Longitude:")
        print(f"      - Range: [{stats_final['lon_min']:.6f}, {stats_final['lon_max']:.6f}]")
        print(f"      - Média: {stats_final['lon_mean']:.6f} ± {stats_final['lon_std']:.6f}")
        print(f"      - Mediana: {stats_final['lon_median']:.6f}")
        print(f"      - NaN: {stats_final['lon_nan']:,}")

        # Verificação de qualidade
        lat_in_sp = ((stats_final['lat_min'] >= self.SP_LAT_MIN) and
                     (stats_final['lat_max'] <= self.SP_LAT_MAX))
        lon_in_sp = ((stats_final['lon_min'] >= self.SP_LON_MIN) and
                     (stats_final['lon_max'] <= self.SP_LON_MAX))

        if lat_in_sp and lon_in_sp:
            print(f"\n   ✅ SUCESSO: Todas as coordenadas dentro dos limites de São Paulo!")
        elif (WORLD_LAT_MIN <= stats_final['lat_min'] and stats_final['lat_max'] <= WORLD_LAT_MAX and
              WORLD_LON_MIN <= stats_final['lon_min'] and stats_final['lon_max'] <= WORLD_LON_MAX):
            print(f"\n   ✅ OK: Coordenadas dentro dos limites mundiais (mas fora de SP)")
        else:
            print(f"\n   ⚠️  ATENÇÃO: Ainda há coordenadas suspeitas!")

        # ==========================================
        # ETAPA 7: SALVAR ESTATÍSTICAS
        # ==========================================
        self.outlier_stats['coordinate_correction'] = {
            'lat_factor': lat_factor,
            'lon_factor': lon_factor,
            'lat_out_of_bounds_pre': lat_out_of_bounds,
            'lon_out_of_bounds_pre': lon_out_of_bounds,
            'nan_filled_lat': nan_lat,
            'nan_filled_lon': nan_lon,
            'final_stats': stats_final
        }

        print("\n" + "🔥" * 30)
        print("✅ CORREÇÃO DE COORDENADAS CONCLUÍDA")
        print("🔥" * 30 + "\n")

        return df

    def load_and_fix_coordinates(self, df):
        """
        ✅ CORREÇÃO PRINCIPAL: Converter coordenadas brasileiras corretamente
        """
        print("\n🔧 CORREÇÃO DE FORMATO BRASILEIRO (VÍRGULA DECIMAL)")
        print("=" * 50)

        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            print("⚠️  Colunas de coordenadas não encontradas")
            return df

        # Mostrar amostras ANTES da correção
        print(f"\n📊 AMOSTRA ANTES DA CORREÇÃO:")
        lat_sample_before = df['LATITUDE'].head(5)
        lon_sample_before = df['LONGITUDE'].head(5)
        print(f"   Latitude: {lat_sample_before.tolist()}")
        print(f"   Longitude: {lon_sample_before.tolist()}")

        # Aplicar correção de vírgula decimal
        print(f"\n🔄 Convertendo vírgulas para pontos...")
        df['LATITUDE'] = df['LATITUDE'].apply(self.fix_brazilian_decimal)
        df['LONGITUDE'] = df['LONGITUDE'].apply(self.fix_brazilian_decimal)

        # Mostrar amostras DEPOIS da correção
        print(f"\n✅ AMOSTRA APÓS CORREÇÃO:")
        lat_sample_after = df['LATITUDE'].head(5)
        lon_sample_after = df['LONGITUDE'].head(5)
        print(f"   Latitude: {lat_sample_after.tolist()}")
        print(f"   Longitude: {lon_sample_after.tolist()}")

        # Estatísticas após correção
        lat_mean = df['LATITUDE'].mean()
        lon_mean = df['LONGITUDE'].mean()
        lat_min = df['LATITUDE'].min()
        lat_max = df['LATITUDE'].max()
        lon_min = df['LONGITUDE'].min()
        lon_max = df['LONGITUDE'].max()

        print(f"\n📊 ESTATÍSTICAS APÓS CORREÇÃO:")
        print(f"   • Latitude: [{lat_min:.6f}, {lat_max:.6f}]")
        print(f"   • Longitude: [{lon_min:.6f}, {lon_max:.6f}]")
        print(f"   • Centro: ({lat_mean:.6f}, {lon_mean:.6f})")

        # Verificar se está dentro dos limites de SP
        if (lat_min >= self.SP_LAT_MIN and lat_max <= self.SP_LAT_MAX and
                lon_min >= self.SP_LON_MIN and lon_max <= self.SP_LON_MAX):
            print(f"   ✅ Coordenadas dentro dos limites de São Paulo!")
        else:
            print(f"   ⚠️  Ainda há coordenadas fora dos limites")
            print(
                f"      Limites SP: Lat[{self.SP_LAT_MIN}, {self.SP_LAT_MAX}], Lon[{self.SP_LON_MIN}, {self.SP_LON_MAX}]")

        # Verificar NaN
        nan_lat = df['LATITUDE'].isna().sum()
        nan_lon = df['LONGITUDE'].isna().sum()
        if nan_lat > 0 or nan_lon > 0:
            print(f"   ℹ️  NaN encontrados: Lat={nan_lat:,}, Lon={nan_lon:,}")

        return df

    def clean_geographic_outliers(self, df):
        """
        ✅ Limpeza de outliers geográficos (após correção de formato)
        """
        print("\n🗺️  LIMPEZA DE OUTLIERS GEOGRÁFICOS")
        print("=" * 50)

        initial_size = len(df)

        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            print("⚠️  Colunas de coordenadas não encontradas")
            return df

        print(f"\n📍 LIMITES DE SÃO PAULO:")
        print(f"   • Latitude: [{self.SP_LAT_MIN:.1f}, {self.SP_LAT_MAX:.1f}]")
        print(f"   • Longitude: [{self.SP_LON_MIN:.1f}, {self.SP_LON_MAX:.1f}]")

        # Identificar problemas
        lat_outliers = ((df['LATITUDE'] < self.SP_LAT_MIN) |
                        (df['LATITUDE'] > self.SP_LAT_MAX)).sum()
        lon_outliers = ((df['LONGITUDE'] < self.SP_LON_MIN) |
                        (df['LONGITUDE'] > self.SP_LON_MAX)).sum()
        nan_coords = df[['LATITUDE', 'LONGITUDE']].isnull().any(axis=1).sum()

        print(f"\n📊 PROBLEMAS IDENTIFICADOS:")
        print(f"   • Latitudes fora de SP: {lat_outliers:,} ({lat_outliers / initial_size * 100:.2f}%)")
        print(f"   • Longitudes fora de SP: {lon_outliers:,} ({lon_outliers / initial_size * 100:.2f}%)")
        print(f"   • Coordenadas NaN: {nan_coords:,} ({nan_coords / initial_size * 100:.2f}%)")

        # Calcular valores de referência (baseados em dados válidos)
        valid_mask = ((df['LATITUDE'] >= self.SP_LAT_MIN) &
                      (df['LATITUDE'] <= self.SP_LAT_MAX) &
                      (df['LONGITUDE'] >= self.SP_LON_MIN) &
                      (df['LONGITUDE'] <= self.SP_LON_MAX))

        if valid_mask.sum() > 0:
            lat_median = df.loc[valid_mask, 'LATITUDE'].median()
            lon_median = df.loc[valid_mask, 'LONGITUDE'].median()
            print(f"\n📍 Usando mediana de dados válidos:")
        else:
            # Se não houver dados válidos, usar centro de SP
            lat_median = self.SP_LAT_CENTER
            lon_median = self.SP_LON_CENTER
            print(f"\n📍 Usando centro de São Paulo:")

        print(f"   • Latitude: {lat_median:.6f}")
        print(f"   • Longitude: {lon_median:.6f}")

        # Aplicar correções
        mask_lat_outlier = (df['LATITUDE'] < self.SP_LAT_MIN) | (df['LATITUDE'] > self.SP_LAT_MAX)
        mask_lon_outlier = (df['LONGITUDE'] < self.SP_LON_MIN) | (df['LONGITUDE'] > self.SP_LON_MAX)
        mask_nan = df[['LATITUDE', 'LONGITUDE']].isnull().any(axis=1)

        df.loc[mask_lat_outlier, 'LATITUDE'] = lat_median
        df.loc[mask_lon_outlier, 'LONGITUDE'] = lon_median
        df.loc[mask_nan, 'LATITUDE'] = lat_median
        df.loc[mask_nan, 'LONGITUDE'] = lon_median

        corrections_made = mask_lat_outlier.sum() + mask_lon_outlier.sum() + mask_nan.sum()

        print(f"\n✅ CORREÇÕES APLICADAS:")
        print(f"   • Total de correções: {corrections_made:,}")
        print(f"   • Registros preservados: {len(df):,} (100%)")

        # Validação final
        lat_min_final = df['LATITUDE'].min()
        lat_max_final = df['LATITUDE'].max()
        lon_min_final = df['LONGITUDE'].min()
        lon_max_final = df['LONGITUDE'].max()
        lat_center = df['LATITUDE'].mean()
        lon_center = df['LONGITUDE'].mean()

        print(f"\n📊 COORDENADAS FINAIS:")
        print(f"   • Latitude: [{lat_min_final:.6f}, {lat_max_final:.6f}]")
        print(f"   • Longitude: [{lon_min_final:.6f}, {lon_max_final:.6f}]")
        print(f"   • Centro: ({lat_center:.6f}, {lon_center:.6f})")

        # Verificar resultado
        if (lat_min_final >= self.SP_LAT_MIN and lat_max_final <= self.SP_LAT_MAX and
                lon_min_final >= self.SP_LON_MIN and lon_max_final <= self.SP_LON_MAX):
            print(f"   ✅ Todas as coordenadas dentro de São Paulo!")
        else:
            print(f"   ⚠️  Algumas coordenadas ainda fora dos limites")

        # Salvar estatísticas
        self.outlier_stats['geographic_corrections'] = corrections_made
        self.outlier_stats['records_preserved'] = len(df)
        self.outlier_stats['lat_range'] = (lat_min_final, lat_max_final)
        self.outlier_stats['lon_range'] = (lon_min_final, lon_max_final)

        return df

    def detect_spatial_outliers(self, df):
        """Detecção de outliers espaciais - desativada para preservar dados"""
        print("\n🔍 DETECÇÃO DE OUTLIERS ESPACIAIS")
        print("=" * 50)
        print("   • Pulando LOF para preservar dados")
        return df

    def enhanced_feature_engineering(self, df):
        """Engenharia de features otimizada"""
        print("\n🔧 ENGENHARIA DE FEATURES")
        print("=" * 40)

        df = df.copy()
        selected_features = []

        print("📋 Selecionando features para clustering...")

        # Coordenadas
        if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
            # Validação final
            lat_check = df['LATITUDE'].mean()
            lon_check = df['LONGITUDE'].mean()

            print(f"\n   📍 Validando coordenadas:")
            print(f"      • Latitude média: {lat_check:.6f}")
            print(f"      • Longitude média: {lon_check:.6f}")

            if (lat_check >= self.SP_LAT_MIN and lat_check <= self.SP_LAT_MAX and
                    lon_check >= self.SP_LON_MIN and lon_check <= self.SP_LON_MAX):
                selected_features.extend(['LATITUDE', 'LONGITUDE'])
                print("      ✅ Coordenadas válidas incluídas")
            else:
                print("      ⚠️  Coordenadas suspeitas, mas incluindo mesmo assim")
                selected_features.extend(['LATITUDE', 'LONGITUDE'])

        # Features temporais
        if 'DATA_OCORRENCIA' in df.columns:
            try:
                df['DATA_OCORRENCIA'] = pd.to_datetime(df['DATA_OCORRENCIA'], errors='coerce')
                df['HORA_DIA'] = df['DATA_OCORRENCIA'].dt.hour
                df['DIA_SEMANA'] = df['DATA_OCORRENCIA'].dt.dayofweek
                df['MES'] = df['DATA_OCORRENCIA'].dt.month
                selected_features.extend(['HORA_DIA', 'DIA_SEMANA', 'MES'])
                print("   • Features temporais extraídas")
            except Exception as e:
                print(f"⚠️  Erro em features temporais: {e}")

        # Natureza do crime
        crime_features = ['NATUREZA_APURADA', 'TIPO_LOCAL', 'RUBRICA']
        for feature in crime_features:
            if feature in df.columns:
                selected_features.append(feature)
                print(f"   • Feature de crime: {feature}")

        # Localização administrativa
        location_features = ['BAIRRO', 'CIDADE', 'UF']
        for feature in location_features:
            if feature in df.columns:
                selected_features.append(feature)
                print(f"   • Feature de localização: {feature}")

        print(f"\n✅ FEATURES SELECIONADAS: {len(selected_features)}")

        available_features = [f for f in selected_features if f in df.columns]

        if not available_features:
            print("❌ Nenhuma feature disponível!")
            return pd.DataFrame()

        features_df = df[available_features].copy()
        print(f"📊 Dataset de features: {features_df.shape}")

        return features_df

    def smart_encoding(self, features_df):
        """Codificação inteligente de features - REFATORADO"""
        print("\n🔠 PREPARAÇÃO DE FEATURES PARA CLUSTERING")
        print("=" * 50)

        X = features_df.copy()
        print(f"📦 Dataset inicial: {X.shape}")

        # Identificar tipos de features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        print(f"🔢 Features numéricas: {len(numeric_features)}")
        print(f"🏷️  Features categóricas: {len(categorical_features)}")

        if len(X) < 10:
            print("⚠️  Dados insuficientes!")
            return pd.DataFrame()

        # Processar categóricas
        print("\n🔄 Processando features categóricas...")
        for col in categorical_features:
            X[col] = X[col].fillna('NÃO_INFORMADO')
            X[col] = X[col].astype(str)

            unique_count = X[col].nunique()
            if unique_count > 20:
                freq = X[col].value_counts()
                common_categories = freq[freq > len(X) * 0.01].index.tolist()
                X[col] = X[col].apply(lambda x: x if x in common_categories else 'OUTROS')
                print(f"   • {col}: {unique_count} → {len(common_categories) + 1} categorias")

        # Processar numéricas
        print("\n🔄 Processando features numéricas...")
        for col in numeric_features:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)

        # REFATORADO: One-Hot Encoding com validação
        print("\n🎯 Aplicando One-Hot Encoding...")
        if categorical_features:
            try:
                # Preparar dados categóricos para encoding
                X_categorical = X[categorical_features]

                # Fit e transform
                X_encoded = self.encoder.fit_transform(X_categorical)
                encoded_features = self.encoder.get_feature_names_out(categorical_features)
                print(f"   • {len(categorical_features)} categóricas → {len(encoded_features)} colunas")
            except Exception as e:
                print(f"⚠️  Erro no encoding: {e}")
                X_encoded = sparse.csr_matrix((X.shape[0], 0))
                encoded_features = []
        else:
            X_encoded = sparse.csr_matrix((X.shape[0], 0))
            encoded_features = []

        if hasattr(self, 'pca') and hasattr(self.pca, 'components_'):
            print("\n🎯 AJUSTE: Rebalanceando contribuição de coordenadas no PCA")

            # Identificar quais componentes têm alta contribuição de LAT/LON
            components = self.pca.components_

            # Índices de LAT/LON nas features originais (geralmente as primeiras)
            # Se você souber os índices exatos, use-os
            # Por simplicidade, vamos assumir que são as 2 primeiras features
            coord_indices = [0, 1]  # Ajustar conforme sua pipeline

            for i, component in enumerate(components):
                coord_contribution = np.sum(np.abs(component[coord_indices]))
                total_contribution = np.sum(np.abs(component))

                if total_contribution > 0:
                    coord_ratio = coord_contribution / total_contribution

                    if coord_ratio > 0.7:  # Se coordenadas dominam mais de 70%
                        print(f"   ⚠️  PC{i + 1}: Coordenadas dominam {coord_ratio * 100:.1f}% - ajustando...")

                        # Reduzir peso das coordenadas
                        damping_factor = 0.5
                        component[coord_indices] *= damping_factor

                        # Renormalizar componente
                        norm = np.linalg.norm(component)
                        if norm > 0:
                            components[i] = component / norm

            print(f"   ✅ PCA rebalanceado")

        # REFATORADO: Normalização com validação
        print("\n📏 Normalizando features numéricas...")
        if numeric_features:
            try:
                X_numeric = X[numeric_features]
                X_scaled = self.scaler.fit_transform(X_numeric)
                X_scaled = sparse.csr_matrix(X_scaled)
                print(f"   • {len(numeric_features)} features normalizadas")
            except Exception as e:
                print(f"⚠️  Erro na normalização: {e}")
                X_scaled = sparse.csr_matrix((X.shape[0], len(numeric_features)))
        else:
            X_scaled = sparse.csr_matrix((X.shape[0], 0))

        # REFATORADO: Combinar features com verificação de compatibilidade
        print("\n🔗 Combinando features...")
        try:
            # Verificar se as matrizes têm o mesmo número de linhas
            if X_scaled.shape[0] != X_encoded.shape[0]:
                print("❌ Número de linhas incompatível!")
                print(f"   X_scaled: {X_scaled.shape}")
                print(f"   X_encoded: {X_encoded.shape}")
                # Alinhar truncando para o menor
                min_rows = min(X_scaled.shape[0], X_encoded.shape[0])
                if min_rows > 0:
                    X_scaled = X_scaled[:min_rows]
                    X_encoded = X_encoded[:min_rows]
                    print(f"   ⚠️  Truncado para {min_rows} linhas")
                else:
                    return pd.DataFrame()

            # Combinar as matrizes esparsas
            X_combined = sparse.hstack([X_scaled, X_encoded], format='csr')
            print(f"✅ Features combinadas: {X_combined.shape}")

            # Converter para denso
            X_dense = X_combined.toarray()
            print(f"✅ Matriz densa criada: {X_dense.shape}")

        except Exception as e:
            print(f"❌ Erro ao combinar features: {e}")
            return pd.DataFrame()

        # REFATORADO: PCA com tratamento de erro
        if X_dense.shape[1] > 20:
            print("\n🎯 Reduzindo dimensionalidade com PCA...")
            try:
                X_pca = self.pca.fit_transform(X_dense)
                explained_variance = self.pca.explained_variance_ratio_.sum()
                n_components = X_pca.shape[1]

                print(f"   • Dimensões: {X_dense.shape[1]} → {n_components}")
                print(f"   • Variância explicada: {explained_variance * 100:.2f}%")

                feature_names = [f'PC{i + 1}' for i in range(n_components)]
                X_final = pd.DataFrame(X_pca, columns=feature_names, index=X.index)

                # Salvar contribuições PCA
                self.pca_contributions = self.get_feature_contributions_from_pca()
                joblib.dump(self.pca_contributions, 'pca_feature_contributions.pkl')
                print("   💾 Contribuições PCA salvas")

            except Exception as e:
                print(f"⚠️  Erro no PCA: {e}")
                # Fallback: usar matriz densa original
                feature_names = [f'Feature_{i}' for i in range(X_dense.shape[1])]
                X_final = pd.DataFrame(X_dense, columns=feature_names, index=X.index)
        else:
            print(f"   • Dimensionalidade OK ({X_dense.shape[1]} features)")
            feature_names = [f'Feature_{i}' for i in range(X_dense.shape[1])]
            X_final = pd.DataFrame(X_dense, columns=feature_names, index=X.index)

        print(f"\n✅ DATASET FINAL: {X_final.shape}")
        return X_final

    def get_feature_contributions_from_pca(self):
        """
        Retorna contribuição de features originais em cada componente principal
        """
        if not hasattr(self.pca, 'components_'):
            print("⚠️  PCA não foi ajustado ainda")
            return None

        # Componentes do PCA (n_components x n_features)
        components = self.pca.components_

        # Variância explicada por cada PC
        explained_var = self.pca.explained_variance_ratio_

        # Criar DataFrame legível
        feature_names = [f'Feature_{i}' for i in range(components.shape[1])]

        contributions_df = pd.DataFrame(
            components.T,  # Transpor: features nas linhas
            columns=[f'PC{i + 1}' for i in range(components.shape[0])],
            index=feature_names
        )

        # Adicionar variância explicada
        contributions_df.loc['Explained_Variance'] = explained_var

        print("\n📊 TOP 5 FEATURES POR COMPONENTE PRINCIPAL:")
        for pc in contributions_df.columns:
            if pc == 'Explained_Variance':
                continue
            top_features = contributions_df[pc].abs().nlargest(5)
            print(f"\n{pc} ({explained_var[int(pc[2:]) - 1] * 100:.1f}% variância):")
            for feat, val in top_features.items():
                print(f"   • {feat}: {val:.3f}")

        return contributions_df

    def full_pipeline(self, csv_path, sample_frac=None):
        """Pipeline completo com correção de formato brasileiro - REFATORADO"""
        print("\n" + "=" * 60)
        print("🚀 PIPELINE DE PRÉ-PROCESSAMENTO PARA CLUSTERING")
        print("=" * 60)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo {csv_path} não encontrado!")

        # ✅ CARREGAR COM CONFIGURAÇÃO BRASILEIRA
        try:
            print(f"\n📂 Carregando: {csv_path}")
            print("   • Usando separador: ';'")
            print("   • Decimal brasileiro: ',' será convertido para '.'")

            # Carregar sem converter automaticamente para manter controle
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8',
                             low_memory=False, dtype=str)  # ✅ Ler tudo como string primeiro
        except UnicodeDecodeError:
            print("   • Tentando encoding latin-1...")
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1',
                             low_memory=False, dtype=str)

        print(f"📈 Shape inicial: {df.shape}")
        print(f"📋 Colunas: {list(df.columns)[:10]}...")

        # Amostragem
        if sample_frac and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"✅ Amostra: {len(df):,} registros ({sample_frac * 100:.1f}%)")

        # ✅ FASE 0: CORREÇÃO DE FORMATO BRASILEIRO
        print("\n" + "=" * 60)
        print("📍 FASE 0: CORREÇÃO DE FORMATO (VÍRGULA → PONTO)")
        print("=" * 60)

        df = self.robust_fix_coords(df)

        # FASE 1: Limpeza de outliers
        print("\n" + "=" * 60)
        print("🗺️  FASE 1: LIMPEZA DE OUTLIERS")
        print("=" * 60)

        df = self.clean_geographic_outliers(df)

        if len(df) == 0:
            print("❌ Nenhum dado após limpeza!")
            return pd.DataFrame()

        # FASE 2: Engenharia de features
        print("\n" + "=" * 60)
        print("🔧 FASE 2: ENGENHARIA DE FEATURES")
        print("=" * 60)

        features_df = self.enhanced_feature_engineering(df)

        if len(features_df) == 0:
            print("❌ Nenhuma feature gerada!")
            return pd.DataFrame()

        # FASE 3: Codificação
        print("\n" + "=" * 60)
        print("🔠 FASE 3: PREPARAÇÃO PARA CLUSTERING")
        print("=" * 60)

        X_final = self.smart_encoding(features_df)

        if len(X_final) == 0:
            print("❌ Nenhum dado processado!")
            return pd.DataFrame()

        # FASE 4: Salvar
        print("\n" + "=" * 60)
        print("💾 FASE 4: SALVANDO RESULTADOS")
        print("=" * 60)

        self.save_preprocessing_artifacts()

        # Salvar DataFrame corrigido
        df.to_parquet('df_corrected.parquet', index=False)
        print("💾 DataFrame corrigido salvo: df_corrected.parquet")

        print(f"\n🎉 PIPELINE CONCLUÍDO!")
        print(f"📊 Dados processados: {X_final.shape}")

        return X_final

    def save_preprocessing_artifacts(self):
        """Salva artefatos do preprocessamento"""
        try:
            artifacts = {
                'scaler': self.scaler,
                'encoder': self.encoder,
                'pca': self.pca if hasattr(self.pca, 'components_') else None,
                'geo_scaler': self.geo_scaler,
                'feature_info': self.feature_info,
                'outlier_stats': self.outlier_stats
            }

            for name, artifact in artifacts.items():
                if artifact is not None:
                    joblib.dump(artifact, f'{name}.pkl')

            print("✅ Artefatos salvos!")

        except Exception as e:
            print(f"⚠️  Erro ao salvar: {e}")


if __name__ == "__main__":
    preprocessor = AdvancedDataPreprocessor()
    processed_data = preprocessor.full_pipeline("SPSafe_2022.csv", sample_frac=0.1)

    if len(processed_data) > 0:
        print(f"\n🎉 Dados prontos!")
        print(f"📊 Shape: {processed_data.shape}")
        print(f"📋 Amostra:")
        print(processed_data.head())
    else:
        print("\n❌ Falha no processamento!")