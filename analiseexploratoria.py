# ============================================================================
# ANÁLISE EXPLORATÓRIA DE DADOS (EDA) - CRIMINALIDADE SÃO PAULO
# Arquivo: eda_crimes_sp.py
# Descrição: Análise exploratória completa dos dados SPSafe
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# CLASSE DE ANÁLISE EXPLORATÓRIA
# ============================================================================

class CrimeEDA:
    """
    Análise Exploratória de Dados para crimes em São Paulo

    Esta classe implementa todas as análises necessárias para o critério 6:
    - Dados faltantes
    - Estatísticas descritivas  
    - Desvio padrão e distribuição
    - Correlação entre variáveis
    - Verificação de linearidade
    - Detecção de outliers
    """

    def __init__(self, data_path):
        """
        Inicializa a classe de EDA

        Args:
            data_path: Caminho para a pasta com os arquivos CSV
        """
        self.data_path = Path(data_path)
        self.df = None
        self.anos_disponiveis = [2019, 2020, 2021, 2022]

    def carregar_dados(self, anos=None):
        """
        Carrega dados de múltiplos anos

        Args:
            anos: Lista de anos a carregar. Se None, carrega todos disponíveis
        """
        if anos is None:
            anos = self.anos_disponiveis

        print("=" * 70)
        print("📂 CARREGANDO DADOS")
        print("=" * 70)

        dfs = []
        for ano in anos:
            try:
                arquivo = self.data_path / f"SPSafe_{ano}.csv"

                if not arquivo.exists():
                    print(f"⚠️  Arquivo não encontrado: {arquivo}")
                    continue

                print(f"📥 Carregando {ano}...")

                df_ano = pd.read_csv(
                    arquivo,
                    sep=None,
                    engine="python",
                    encoding_errors="ignore"
                )

                df_ano['ANO'] = ano
                dfs.append(df_ano)

                print(f"   ✅ {len(df_ano):,} registros carregados")

            except Exception as e:
                print(f"   ❌ Erro ao carregar {ano}: {e}")

        if not dfs:
            raise ValueError("❌ Nenhum arquivo foi carregado com sucesso!")

        self.df = pd.concat(dfs, ignore_index=True)

        print(f"\n✅ TOTAL: {len(self.df):,} registros de {len(dfs)} anos")
        print(f"✅ Período: {self.df['ANO'].min()} - {self.df['ANO'].max()}")
        print("=" * 70 + "\n")

        return self.df

    def executar_eda_completa(self):
        """
        Executa todas as etapas da análise exploratória
        """
        if self.df is None:
            raise ValueError("❌ Dados não carregados! Execute carregar_dados() primeiro.")

        print("\n" + "=" * 70)
        print("📊 INICIANDO ANÁLISE EXPLORATÓRIA DE DADOS")
        print("=" * 70 + "\n")

        # 1. Visão geral dos dados
        self._visao_geral()

        # 2. Análise de dados faltantes
        self._analisar_dados_faltantes()

        # 3. Estatísticas descritivas
        self._estatisticas_descritivas()

        # 4. Análise de distribuição e desvio padrão
        self._analisar_distribuicao()

        # 5. Análise de correlação
        self._analisar_correlacao()

        # 6. Verificação de linearidade
        self._verificar_linearidade()

        # 7. Detecção de outliers
        self._detectar_outliers()

        # 8. Análise temporal
        self._analisar_temporal()

        # 9. Análise espacial básica
        self._analisar_espacial()

        # 10. Análise de variáveis categóricas
        self._analisar_categoricas()

        print("\n" + "=" * 70)
        print("✅ ANÁLISE EXPLORATÓRIA CONCLUÍDA")
        print("=" * 70)
        print("\n📁 Gráficos salvos:")
        print("   • eda_dados_faltantes.png")
        print("   • eda_distribuicao.png")
        print("   • eda_correlacao.png")
        print("   • eda_linearidade.png")
        print("   • eda_outliers.png")
        print("   • eda_temporal.png")
        print("   • eda_espacial.png")
        print("   • eda_categoricas.png")
        print("=" * 70 + "\n")

    def _visao_geral(self):
        """Visão geral dos dados"""
        print("=" * 70)
        print("📋 1. VISÃO GERAL DOS DADOS")
        print("=" * 70)

        print(f"\n📊 Dimensões do Dataset:")
        print(f"   • Linhas (registros): {len(self.df):,}")
        print(f"   • Colunas (variáveis): {len(self.df.columns)}")
        print(f"   • Tamanho em memória: {self.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        print(f"\n📝 Colunas Disponíveis ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            print(f"   {i:2d}. {col:30s} ({dtype})")

        print(f"\n📅 Período de Análise:")
        print(f"   • Anos: {sorted(self.df['ANO'].unique())}")

        if 'DATA_OCORRENCIA' in self.df.columns:
            self.df['DATA_OCORRENCIA'] = pd.to_datetime(self.df['DATA_OCORRENCIA'], errors='coerce')
            print(f"   • Data inicial: {self.df['DATA_OCORRENCIA'].min()}")
            print(f"   • Data final: {self.df['DATA_OCORRENCIA'].max()}")

        print("\n" + "=" * 70 + "\n")

    def _analisar_dados_faltantes(self):
        """Análise de dados faltantes"""
        print("=" * 70)
        print("🔍 2. ANÁLISE DE DADOS FALTANTES")
        print("=" * 70)

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        missing_df = pd.DataFrame({
            'Coluna': missing.index,
            'Faltantes': missing.values,
            'Percentual (%)': missing_pct.values
        }).sort_values('Percentual (%)', ascending=False)

        print("\n📉 Dados Faltantes por Coluna:")
        print(missing_df.to_string(index=False))

        total_cells = len(self.df) * len(self.df.columns)
        total_missing = self.df.isnull().sum().sum()
        completude = ((total_cells - total_missing) / total_cells) * 100

        print(f"\n📊 Resumo Geral:")
        print(f"   • Total de células: {total_cells:,}")
        print(f"   • Células faltantes: {total_missing:,}")
        print(f"   • Completude do dataset: {completude:.2f}%")

        # Visualização
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Gráfico 1: Top 10 colunas com mais missing
        top_missing = missing_df[missing_df['Faltantes'] > 0].head(10)

        if not top_missing.empty:
            ax1.barh(top_missing['Coluna'], top_missing['Percentual (%)'], color='coral')
            ax1.set_xlabel('Percentual de Dados Faltantes (%)')
            ax1.set_title('Top 10 Colunas com Dados Faltantes', fontsize=14, weight='bold')
            ax1.grid(axis='x', alpha=0.3)

            for i, v in enumerate(top_missing['Percentual (%)']):
                ax1.text(v + 1, i, f'{v:.1f}%', va='center')
        else:
            ax1.text(0.5, 0.5, 'Nenhum dado faltante!',
                     ha='center', va='center', fontsize=16, weight='bold')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)

        # Gráfico 2: Mapa de calor de missing
        missing_matrix = self.df.isnull().astype(int)
        sample_size = min(1000, len(missing_matrix))
        missing_sample = missing_matrix.sample(sample_size, random_state=42)

        sns.heatmap(missing_sample.T, cbar=False, cmap='RdYlGn_r',
                    yticklabels=True, xticklabels=False, ax=ax2)
        ax2.set_title(f'Mapa de Calor: Dados Faltantes\n(Amostra de {sample_size:,} registros)',
                      fontsize=14, weight='bold')
        ax2.set_xlabel('Registros')
        ax2.set_ylabel('Colunas')

        plt.tight_layout()
        plt.savefig('eda_dados_faltantes.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n✅ Gráfico salvo: eda_dados_faltantes.png")
        print("=" * 70 + "\n")

    def _estatisticas_descritivas(self):
        """Estatísticas descritivas"""
        print("=" * 70)
        print("📈 3. ESTATÍSTICAS DESCRITIVAS")
        print("=" * 70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            print("\n📊 Variáveis Numéricas:")
            desc = self.df[numeric_cols].describe()
            print(desc.round(2).to_string())

            print("\n📏 Medidas Adicionais:")
            for col in numeric_cols:
                if self.df[col].notna().sum() > 0:
                    print(f"\n{col}:")
                    print(f"   • Média: {self.df[col].mean():.2f}")
                    print(f"   • Mediana: {self.df[col].median():.2f}")
                    print(f"   • Moda: {self.df[col].mode().values[0] if len(self.df[col].mode()) > 0 else 'N/A'}")
                    print(f"   • Desvio Padrão: {self.df[col].std():.2f}")
                    print(f"   • Variância: {self.df[col].var():.2f}")
                    print(f"   • Assimetria (Skewness): {self.df[col].skew():.2f}")
                    print(f"   • Curtose (Kurtosis): {self.df[col].kurtosis():.2f}")
                    print(f"   • Mínimo: {self.df[col].min():.2f}")
                    print(f"   • Máximo: {self.df[col].max():.2f}")

                    # Interpretação da assimetria
                    skew = self.df[col].skew()
                    if skew > 1:
                        print(f"   → Distribuição MUITO ASSIMÉTRICA À DIREITA")
                    elif skew > 0.5:
                        print(f"   → Distribuição ASSIMÉTRICA À DIREITA")
                    elif skew < -1:
                        print(f"   → Distribuição MUITO ASSIMÉTRICA À ESQUERDA")
                    elif skew < -0.5:
                        print(f"   → Distribuição ASSIMÉTRICA À ESQUERDA")
                    else:
                        print(f"   → Distribuição APROXIMADAMENTE SIMÉTRICA")
        else:
            print("\n⚠️  Nenhuma variável numérica encontrada")

        print("\n" + "=" * 70 + "\n")

    def _analisar_distribuicao(self):
        """Análise de distribuição e desvio padrão"""
        print("=" * 70)
        print("📊 4. ANÁLISE DE DISTRIBUIÇÃO E DESVIO PADRÃO")
        print("=" * 70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            print("\n⚠️  Nenhuma variável numérica para análise")
            print("=" * 70 + "\n")
            return

        # Análise de coeficiente de variação
        print("\n📐 Coeficiente de Variação (CV):")
        print("   (Indica dispersão relativa: quanto maior, mais disperso)")
        print()

        for col in numeric_cols:
            if self.df[col].notna().sum() > 0:
                mean = self.df[col].mean()
                std = self.df[col].std()
                cv = (std / mean * 100) if mean != 0 else 0

                print(f"   {col}:")
                print(f"      • CV = {cv:.2f}%", end="")

                if cv < 15:
                    print(" → Baixa dispersão")
                elif cv < 30:
                    print(" → Dispersão moderada")
                else:
                    print(" → Alta dispersão")

        # Visualizações
        n_plots = min(len(numeric_cols), 6)
        fig = plt.figure(figsize=(18, 12))

        for idx, col in enumerate(numeric_cols[:n_plots], 1):
            data = self.df[col].dropna()

            if len(data) == 0:
                continue

            # Subplot para cada variável
            ax1 = plt.subplot(3, n_plots, idx)
            ax1.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax1.set_title(f'Histograma: {col}', fontsize=10, weight='bold')
            ax1.set_ylabel('Frequência')
            ax1.axvline(data.mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Média: {data.mean():.1f}')
            ax1.axvline(data.median(), color='green', linestyle='--',
                        linewidth=2, label=f'Mediana: {data.median():.1f}')
            ax1.legend(fontsize=8)
            ax1.grid(axis='y', alpha=0.3)

            # Boxplot
            ax2 = plt.subplot(3, n_plots, idx + n_plots)
            ax2.boxplot(data, vert=True)
            ax2.set_title(f'Boxplot: {col}', fontsize=10, weight='bold')
            ax2.set_ylabel('Valor')
            ax2.grid(axis='y', alpha=0.3)

            # Q-Q Plot
            ax3 = plt.subplot(3, n_plots, idx + 2 * n_plots)
            stats.probplot(data, dist="norm", plot=ax3)
            ax3.set_title(f'Q-Q Plot: {col}', fontsize=10, weight='bold')
            ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('eda_distribuicao.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n✅ Gráfico salvo: eda_distribuicao.png")
        print("\n💡 INTERPRETAÇÃO:")
        print("   • Histograma: Mostra a distribuição de frequências")
        print("   • Boxplot: Identifica quartis, mediana e outliers")
        print("   • Q-Q Plot: Compara com distribuição normal")
        print("      - Pontos na linha = distribuição normal")
        print("      - Desvios = assimetria ou caudas pesadas")
        print("=" * 70 + "\n")

    def _analisar_correlacao(self):
        """Análise de correlação"""
        print("=" * 70)
        print("🔗 5. ANÁLISE DE CORRELAÇÃO")
        print("=" * 70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            print("\n⚠️  Menos de 2 variáveis numéricas para análise de correlação")
            print("=" * 70 + "\n")
            return

        # Calcular matriz de correlação
        correlation = self.df[numeric_cols].corr()

        print("\n📊 Matriz de Correlação (Pearson):")
        print(correlation.round(3).to_string())

        # Encontrar correlações fortes
        print("\n🔍 Correlações Significativas:")

        thresholds = [0.7, 0.5, 0.3]
        labels = ['MUITO FORTE', 'FORTE', 'MODERADA']

        for threshold, label in zip(thresholds, labels):
            strong_corr = []
            for i in range(len(correlation.columns)):
                for j in range(i + 1, len(correlation.columns)):
                    corr_value = correlation.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        strong_corr.append({
                            'Var1': correlation.columns[i],
                            'Var2': correlation.columns[j],
                            'Correlação': corr_value
                        })

            if strong_corr:
                print(f"\n   {label} (|r| ≥ {threshold}):")
                for corr in strong_corr:
                    sinal = "positiva" if corr['Correlação'] > 0 else "negativa"
                    print(f"      • {corr['Var1']} ↔ {corr['Var2']}: {corr['Correlação']:.3f} ({sinal})")

        # Visualização
        plt.figure(figsize=(14, 12))

        # Máscara para triângulo superior
        mask = np.triu(np.ones_like(correlation, dtype=bool))

        # Heatmap
        sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1)

        plt.title('Matriz de Correlação de Pearson', fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('eda_correlacao.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n✅ Gráfico salvo: eda_correlacao.png")
        print("\n💡 INTERPRETAÇÃO DO COEFICIENTE DE CORRELAÇÃO:")
        print("   • |r| = 0.9 a 1.0  → Correlação MUITO FORTE")
        print("   • |r| = 0.7 a 0.9  → Correlação FORTE")
        print("   • |r| = 0.5 a 0.7  → Correlação MODERADA")
        print("   • |r| = 0.3 a 0.5  → Correlação FRACA")
        print("   • |r| = 0.0 a 0.3  → Correlação MUITO FRACA")
        print("   • r > 0 → Correlação POSITIVA (aumentam juntas)")
        print("   • r < 0 → Correlação NEGATIVA (uma aumenta, outra diminui)")
        print("=" * 70 + "\n")

    def _verificar_linearidade(self):
        """Verificação de linearidade"""
        print("=" * 70)
        print("📏 6. VERIFICAÇÃO DE LINEARIDADE")
        print("=" * 70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if 'LATITUDE' not in self.df.columns or 'LONGITUDE' not in self.df.columns:
            print("\n⚠️  Colunas LATITUDE/LONGITUDE não encontradas")
            print("   Análise de linearidade requer variável dependente")
            print("=" * 70 + "\n")
            return

        # Criar variável dependente: contagem de crimes por coordenada
        print("\n📊 Criando variável alvo: contagem por localização...")

        # Arredondar coordenadas para agrupar
        self.df['LAT_ROUND'] = self.df['LATITUDE'].round(2)
        self.df['LON_ROUND'] = self.df['LONGITUDE'].round(2)

        # Contar crimes por localização
        crime_counts = self.df.groupby(['LAT_ROUND', 'LON_ROUND']).size().reset_index(name='count')

        print(f"   ✅ {len(crime_counts):,} localizações únicas")
        print(f"   • Média de crimes/localização: {crime_counts['count'].mean():.2f}")
        print(f"   • Máximo: {crime_counts['count'].max()}")

        # Scatter plots com linha de tendência
        features_to_test = ['LAT_ROUND', 'LON_ROUND']

        if len(features_to_test) == 0:
            print("\n⚠️  Nenhuma feature disponível para análise")
            print("=" * 70 + "\n")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes = axes.ravel()

        for idx, col in enumerate(features_to_test):
            ax = axes[idx]

            # Scatter plot
            ax.scatter(crime_counts[col], crime_counts['count'],
                       alpha=0.5, s=20, color='steelblue')

            # Linha de tendência
            z = np.polyfit(crime_counts[col], crime_counts['count'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(crime_counts[col].min(), crime_counts[col].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
                    label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

            # Calcular R²
            from sklearn.metrics import r2_score
            r2 = r2_score(crime_counts['count'], p(crime_counts[col]))

            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Contagem de Crimes', fontsize=12)
            ax.set_title(f'{col} vs Contagem\nR² = {r2:.4f}',
                         fontsize=14, weight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            print(f"\n   {col}:")
            print(f"      • R² = {r2:.4f}", end="")
            if r2 > 0.7:
                print(" → Relação LINEAR FORTE")
            elif r2 > 0.3:
                print(" → Relação LINEAR MODERADA")
            else:
                print(" → Relação NÃO-LINEAR ou FRACA")

        plt.tight_layout()
        plt.savefig('eda_linearidade.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n✅ Gráfico salvo: eda_linearidade.png")
        print("\n💡 CONCLUSÕES SOBRE LINEARIDADE:")
        print("   • Dados criminais geralmente apresentam relações NÃO-LINEARES")
        print("   • R² baixo indica que modelos lineares não são adequados")
        print("   • Justifica uso de modelos baseados em árvores (XGBoost, Random Forest)")
        print("   • Localização geográfica tem relação complexa com criminalidade")
        print("=" * 70 + "\n")

    def _detectar_outliers(self):
        """Detecção de outliers"""
        print("=" * 70)
        print("⚠️  7. DETECÇÃO DE OUTLIERS")
        print("=" * 70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            print("\n⚠️  Nenhuma variável numérica para análise")
            print("=" * 70 + "\n")
            return

        print("\n📊 Método IQR (Interquartile Range):")
        print("   Outliers = valores fora do intervalo [Q1 - 1.5×IQR, Q3 + 1.5×IQR]\n")

        outliers_summary = []

        for col in numeric_cols:
            data = self.df[col].dropna()

            if len(data) == 0:
                continue

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outliers_pct = (outliers / len(data)) * 100

            outliers_summary.append({
                'Coluna': col,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'Limite Inferior': lower_bound,
                'Limite Superior': upper_bound,
                'Outliers': outliers,
                'Percentual (%)': outliers_pct
            })

            if outliers > 0:
                print(f"   {col}:")
                print(f"      • Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
                print(f"      • Limites: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"      • Outliers: {outliers:,} ({outliers_pct:.2f}%)")

        outliers_df = pd.DataFrame(outliers_summary)

        # Visualização
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4))
        axes = axes.ravel() if n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            data = self.df[col].dropna()

            if len(data) == 0:
                axes[idx].text(0.5, 0.5, f'Sem dados\n{col}',
                               ha='center', va='center', fontsize=12)
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)
                continue

            bp = axes[idx].boxplot(data, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')

            # Destacar outliers
            row = outliers_df[outliers_df['Coluna'] == col].iloc[0]
            outliers_count = row['Outliers']
            outliers_pct = row['Percentual (%)']

            axes[idx].set_title(f'{col}\n{outliers_count:,} outliers ({outliers_pct:.1f}%)',
                                fontsize=11, weight='bold')
            axes[idx].set_ylabel('Valor')
            axes[idx].grid(axis='y', alpha=0.3)

        # Remover subplots vazios
        for idx in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig('eda_outliers.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n✅ Gráfico salvo: eda_outliers.png")
        print("\n💡 INTERPRETAÇÃO PARA DADOS CRIMINAIS:")
        print("   • Outliers em criminalidade são ESPERADOS e IMPORTANTES")
        print("   • Hotspots criminais naturalmente geram valores extremos")
        print("   • Representam áreas de alto risco que DEVEM ser identificadas")
        print("   • XGBoost e Random Forest são ROBUSTOS a outliers por design")
        print("   • DECISÃO: MANTER outliers (informação valiosa para predição)")
        print("=" * 70 + "\n")

    def _analisar_temporal(self):
        """Análise temporal dos dados"""
        print("=" * 70)
        print("📅 8. ANÁLISE TEMPORAL")
        print("=" * 70)

        if 'DATA_OCORRENCIA' not in self.df.columns:
            print("\n⚠️  Coluna DATA_OCORRENCIA não encontrada")
            print("=" * 70 + "\n")
            return

        # Converter para datetime
        self.df['DATA_OCORRENCIA'] = pd.to_datetime(self.df['DATA_OCORRENCIA'], errors='coerce')

        # Extrair componentes temporais
        self.df['MES'] = self.df['DATA_OCORRENCIA'].dt.month
        self.df['DIA_SEMANA'] = self.df['DATA_OCORRENCIA'].dt.dayofweek
        self.df['DIA_MES'] = self.df['DATA_OCORRENCIA'].dt.day

        print("\n📊 Distribuição Temporal:")

        # Por ano
        crimes_por_ano = self.df['ANO'].value_counts().sort_index()
        print(f"\n   Por Ano:")
        for ano, count in crimes_por_ano.items():
            print(f"      • {ano}: {count:,} crimes")

        # Por mês
        meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        crimes_por_mes = self.df['MES'].value_counts().sort_index()
        print(f"\n   Por Mês (média mensal):")
        for mes, count in crimes_por_mes.items():
            print(
                f"      • {meses_nome[mes - 1]}: {count:,} crimes ({count / len(self.df['ANO'].unique()):.0f}/ano)")

        # Por dia da semana
        dias_nome = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        crimes_por_dia = self.df['DIA_SEMANA'].value_counts().sort_index()
        print(f"\n   Por Dia da Semana:")
        for dia, count in crimes_por_dia.items():
            print(f"      • {dias_nome[dia]}: {count:,} crimes")

        # Visualizações
        fig = plt.figure(figsize=(18, 12))

        # 1. Série temporal anual
        ax1 = plt.subplot(3, 2, 1)
        crimes_por_ano.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
        ax1.set_title('Crimes por Ano', fontsize=14, weight='bold')
        ax1.set_xlabel('Ano')
        ax1.set_ylabel('Número de Crimes')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(crimes_por_ano.values):
            ax1.text(i, v + crimes_por_ano.max() * 0.02, f'{v:,}',
                     ha='center', fontsize=10, weight='bold')

        # 2. Crimes por mês
        ax2 = plt.subplot(3, 2, 2)
        crimes_por_mes.plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
        ax2.set_title('Crimes por Mês', fontsize=14, weight='bold')
        ax2.set_xlabel('Mês')
        ax2.set_ylabel('Número de Crimes')
        ax2.set_xticklabels(meses_nome, rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        # 3. Crimes por dia da semana
        ax3 = plt.subplot(3, 2, 3)
        crimes_por_dia.plot(kind='bar', ax=ax3, color='lightgreen', edgecolor='black')
        ax3.set_title('Crimes por Dia da Semana', fontsize=14, weight='bold')
        ax3.set_xlabel('Dia da Semana')
        ax3.set_ylabel('Número de Crimes')
        ax3.set_xticklabels(dias_nome, rotation=45)
        ax3.grid(axis='y', alpha=0.3)

        # 4. Heatmap: Dia da semana vs Mês
        ax4 = plt.subplot(3, 2, 4)
        pivot_table = self.df.groupby(['DIA_SEMANA', 'MES']).size().unstack(fill_value=0)
        sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Crimes'})
        ax4.set_title('Heatmap: Dia da Semana vs Mês', fontsize=14, weight='bold')
        ax4.set_xlabel('Mês')
        ax4.set_ylabel('Dia da Semana')
        ax4.set_yticklabels(dias_nome, rotation=0)
        ax4.set_xticklabels(meses_nome, rotation=45)

        # 5. Tendência temporal (se houver data)
        ax5 = plt.subplot(3, 2, 5)
        crimes_por_data = self.df.groupby(self.df['DATA_OCORRENCIA'].dt.to_period('M')).size()
        crimes_por_data.plot(ax=ax5, color='darkblue', linewidth=2)
        ax5.set_title('Tendência Temporal Mensal', fontsize=14, weight='bold')
        ax5.set_xlabel('Período')
        ax5.set_ylabel('Número de Crimes')
        ax5.grid(alpha=0.3)

        # 6. Boxplot por ano
        ax6 = plt.subplot(3, 2, 6)
        crimes_diarios = self.df.groupby(['ANO', self.df['DATA_OCORRENCIA'].dt.date]).size().reset_index(
            name='count')
        anos = sorted(crimes_diarios['ANO'].unique())
        data_boxplot = [crimes_diarios[crimes_diarios['ANO'] == ano]['count'].values for ano in anos]
        bp = ax6.boxplot(data_boxplot, labels=anos, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax6.set_title('Distribuição de Crimes Diários por Ano', fontsize=14, weight='bold')
        ax6.set_xlabel('Ano')
        ax6.set_ylabel('Crimes por Dia')
        ax6.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('eda_temporal.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n✅ Gráfico salvo: eda_temporal.png")
        print("\n💡 INSIGHTS TEMPORAIS:")

        # Identificar mês com mais crimes
        mes_max = crimes_por_mes.idxmax()
        mes_min = crimes_por_mes.idxmin()
        print(f"   • Mês com MAIS crimes: {meses_nome[mes_max - 1]} ({crimes_por_mes[mes_max]:,})")
        print(f"   • Mês com MENOS crimes: {meses_nome[mes_min - 1]} ({crimes_por_mes[mes_min]:,})")

        # Dia da semana
        dia_max = crimes_por_dia.idxmax()
        dia_min = crimes_por_dia.idxmin()
        print(f"   • Dia com MAIS crimes: {dias_nome[dia_max]} ({crimes_por_dia[dia_max]:,})")
        print(f"   • Dia com MENOS crimes: {dias_nome[dia_min]} ({crimes_por_dia[dia_min]:,})")

        # Fim de semana vs dias úteis
        fim_semana = self.df[self.df['DIA_SEMANA'].isin([5, 6])].shape[0]
        dias_uteis = self.df[~self.df['DIA_SEMANA'].isin([5, 6])].shape[0]
        print(f"   • Crimes em fins de semana: {fim_semana:,} ({fim_semana / len(self.df) * 100:.1f}%)")
        print(f"   • Crimes em dias úteis: {dias_uteis:,} ({dias_uteis / len(self.df) * 100:.1f}%)")

        print("=" * 70 + "\n")

    def _analisar_espacial(self):
        """Análise espacial básica"""
        print("=" * 70)
        print("🗺️  9. ANÁLISE ESPACIAL")
        print("=" * 70)

        if 'LATITUDE' not in self.df.columns or 'LONGITUDE' not in self.df.columns:
            print("\n⚠️  Colunas LATITUDE/LONGITUDE não encontradas")
            print("=" * 70 + "\n")
            return

        # Limpar coordenadas inválidas
        df_spatial = self.df[(self.df['LATITUDE'].between(-90, 90)) &
                             (self.df['LONGITUDE'].between(-180, 180))].copy()

        print(f"\n📍 Dados Espaciais:")
        print(f"   • Total de registros com coordenadas válidas: {len(df_spatial):,}")
        print(f"   • Registros descartados: {len(self.df) - len(df_spatial):,}")

        print(f"\n📊 Estatísticas Geográficas:")
        print(f"   Latitude:")
        print(f"      • Mínima: {df_spatial['LATITUDE'].min():.6f}")
        print(f"      • Máxima: {df_spatial['LATITUDE'].max():.6f}")
        print(f"      • Média: {df_spatial['LATITUDE'].mean():.6f}")

        print(f"   Longitude:")
        print(f"      • Mínima: {df_spatial['LONGITUDE'].min():.6f}")
        print(f"      • Máxima: {df_spatial['LONGITUDE'].max():.6f}")
        print(f"      • Média: {df_spatial['LONGITUDE'].mean():.6f}")

        # Amostragem para visualização (para não travar com muitos pontos)
        sample_size = min(50000, len(df_spatial))
        df_sample = df_spatial.sample(sample_size, random_state=42)

        print(f"\n📌 Usando amostra de {sample_size:,} pontos para visualização")

        # Visualizações
        fig = plt.figure(figsize=(18, 12))

        # 1. Scatter plot de todas as ocorrências
        ax1 = plt.subplot(2, 2, 1)
        ax1.scatter(df_sample['LONGITUDE'], df_sample['LATITUDE'],
                    alpha=0.3, s=1, c='red')
        ax1.set_title(f'Distribuição Espacial de Crimes\n(Amostra: {sample_size:,} pontos)',
                      fontsize=14, weight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(alpha=0.3)

        # 2. Heatmap 2D (densidade)
        ax2 = plt.subplot(2, 2, 2)
        h = ax2.hexbin(df_sample['LONGITUDE'], df_sample['LATITUDE'],
                       gridsize=50, cmap='YlOrRd', mincnt=1)
        ax2.set_title('Mapa de Densidade (Hexbin)', fontsize=14, weight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        plt.colorbar(h, ax=ax2, label='Número de Crimes')

        # 3. Histograma 2D
        ax3 = plt.subplot(2, 2, 3)
        h2 = ax3.hist2d(df_sample['LONGITUDE'], df_sample['LATITUDE'],
                        bins=50, cmap='hot')
        ax3.set_title('Histograma 2D de Densidade', fontsize=14, weight='bold')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        plt.colorbar(h2[3], ax=ax3, label='Frequência')

        # 4. Contorno de densidade
        ax4 = plt.subplot(2, 2, 4)

        # Criar grid de densidade
        from scipy.stats import gaussian_kde

        # Subamostrar ainda mais para KDE (computacionalmente intensivo)
        kde_sample_size = min(10000, len(df_sample))
        df_kde = df_sample.sample(kde_sample_size, random_state=42)

        x = df_kde['LONGITUDE'].values
        y = df_kde['LATITUDE'].values

        # Calcular KDE
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)

        # Criar grid
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = kde(np.vstack([Xi.flatten(), Yi.flatten()]))

        # Plot contorno
        contour = ax4.contourf(Xi, Yi, zi.reshape(Xi.shape), levels=15, cmap='YlOrRd')
        ax4.set_title('Contorno de Densidade (KDE)', fontsize=14, weight='bold')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        plt.colorbar(contour, ax=ax4, label='Densidade')

        plt.tight_layout()
        plt.savefig('eda_espacial.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n✅ Gráfico salvo: eda_espacial.png")
        print("\n💡 INSIGHTS ESPACIAIS:")
        print("   • Crimes concentram-se geograficamente (não uniformes)")
        print("   • Padrões de hotspots são claramente visíveis")
        print("   • Densidade varia significativamente por região")
        print("   • Justifica abordagem de grid espacial para modelagem")
        print("=" * 70 + "\n")

    def _analisar_categoricas(self):
        """Análise de variáveis categóricas"""
        print("=" * 70)
        print("📋 10. ANÁLISE DE VARIÁVEIS CATEGÓRICAS")
        print("=" * 70)

        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            print("\n⚠️  Nenhuma variável categórica encontrada")
            print("=" * 70 + "\n")
            return

        print(f"\n📊 Variáveis Categóricas Encontradas: {len(categorical_cols)}")

        # Analisar cada variável categórica
        for col in categorical_cols:
            if col in ['DATA_OCORRENCIA']:  # Pular datas
                continue

                n_unique = self.df[col].nunique()
                n_missing = self.df[col].isnull().sum()

                print(f"\n   {col}:")
                print(f"      • Valores únicos: {n_unique:,}")
                print(f"      • Valores faltantes: {n_missing:,} ({n_missing / len(self.df) * 100:.2f}%)")

                if n_unique <= 20:  # Mostrar contagens se tiver poucas categorias
                    top_values = self.df[col].value_counts().head(10)
                    print(f"      • Top 10 valores:")
                    for val, count in top_values.items():
                        print(f"         - {val}: {count:,} ({count / len(self.df) * 100:.2f}%)")

        # Visualização das principais categóricas
        cols_to_plot = []
        for col in categorical_cols:
            if col not in ['DATA_OCORRENCIA'] and self.df[col].nunique() <= 30:
                cols_to_plot.append(col)

        if not cols_to_plot:
            print("\n⚠️  Nenhuma variável categórica apropriada para visualização")
            print("=" * 70 + "\n")
            return

        n_plots = min(len(cols_to_plot), 6)
        fig = plt.figure(figsize=(18, n_plots * 4))

        for idx, col in enumerate(cols_to_plot[:n_plots], 1):
            ax = plt.subplot(n_plots, 1, idx)

            # Top 15 categorias
            top_categories = self.df[col].value_counts().head(15)

            bars = ax.barh(range(len(top_categories)), top_categories.values, color='steelblue')
            ax.set_yticks(range(len(top_categories)))
            ax.set_yticklabels(top_categories.index, fontsize=10)
            ax.set_xlabel('Frequência', fontsize=11)
            ax.set_title(f'Top 15: {col}', fontsize=13, weight='bold')
            ax.grid(axis='x', alpha=0.3)

            # Adicionar valores nas barras
            for i, (bar, value) in enumerate(zip(bars, top_categories.values)):
                ax.text(value + top_categories.max() * 0.01, i, f'{value:,}',
                        va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('eda_categoricas.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n✅ Gráfico salvo: eda_categoricas.png")
        print("\n💡 INTERPRETAÇÃO:")
        print("   • Variáveis categóricas fornecem contexto sobre os crimes")
        print("   • Podem ser usadas para feature engineering")
        print("   • Diversidade de categorias pode indicar vulnerabilidade")
        print("=" * 70 + "\n")

    def gerar_relatorio_sumario(self):
        """Gera relatório sumário da EDA"""
        if self.df is None:
            print("❌ Execute carregar_dados() primeiro!")
            return

        print("\n" + "=" * 70)
        print("📄 RELATÓRIO SUMÁRIO - ANÁLISE EXPLORATÓRIA")
        print("=" * 70)

        print(f"\n📊 DATASET:")
        print(f"   • Total de registros: {len(self.df):,}")
        print(f"   • Total de variáveis: {len(self.df.columns)}")
        print(f"   • Período: {self.df['ANO'].min()} - {self.df['ANO'].max()}")
        print(f"   • Tamanho em memória: {self.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        print(f"\n📈 QUALIDADE DOS DADOS:")
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        print(f"   • Completude geral: {100 - missing_pct:.2f}%")
        print(f"   • Dados faltantes: {missing_pct:.2f}%")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        print(f"\n📊 TIPOS DE VARIÁVEIS:")
        print(f"   • Numéricas: {len(numeric_cols)}")
        print(f"   • Categóricas: {len(categorical_cols)}")

        if 'LATITUDE' in self.df.columns and 'LONGITUDE' in self.df.columns:
            valid_coords = ((self.df['LATITUDE'].between(-90, 90)) &
                            (self.df['LONGITUDE'].between(-180, 180))).sum()
            print(f"\n🗺️  DADOS ESPACIAIS:")
            print(f"   • Coordenadas válidas: {valid_coords:,} ({valid_coords / len(self.df) * 100:.2f}%)")

        print(f"\n📅 DISTRIBUIÇÃO TEMPORAL:")
        for ano in sorted(self.df['ANO'].unique()):
            count = (self.df['ANO'] == ano).sum()
            print(f"   • {ano}: {count:,} registros ({count / len(self.df) * 100:.2f}%)")

        print("\n✅ ANÁLISES REALIZADAS:")
        print("   ✓ Dados faltantes")
        print("   ✓ Estatísticas descritivas")
        print("   ✓ Distribuição e desvio padrão")
        print("   ✓ Correlação entre variáveis")
        print("   ✓ Linearidade")
        print("   ✓ Detecção de outliers")
        print("   ✓ Análise temporal")
        print("   ✓ Análise espacial")
        print("   ✓ Análise de variáveis categóricas")

        print("\n📁 ARQUIVOS GERADOS:")
        print("   • eda_dados_faltantes.png")
        print("   • eda_distribuicao.png")
        print("   • eda_correlacao.png")
        print("   • eda_linearidade.png")
        print("   • eda_outliers.png")
        print("   • eda_temporal.png")
        print("   • eda_espacial.png")
        print("   • eda_categoricas.png")

        print("\n" + "=" * 70)


# ============================================================================
# FUNÇÃO PRINCIPAL - FORA DA CLASSE!
# ============================================================================

def main():
    """
    Função principal para executar a análise exploratória
    """
    print("\n" + "=" * 70)
    print("🔬 ANÁLISE EXPLORATÓRIA DE DADOS - CRIMES SÃO PAULO")
    print("   Atividade A1 - Critério 6: Análise Exploratória")
    print("=" * 70)

    # Configurar caminho dos dados
    DATA_PATH = Path("coloque o caminho do dataset aqui")

    # Verificar se o caminho existe
    if not DATA_PATH.exists():
        print(f"\n❌ ERRO: Caminho não encontrado!")
        print(f"   {DATA_PATH}")
        print(f"\n💡 SOLUÇÃO: Atualize a variável DATA_PATH no código")
        return

    try:
        # Criar instância da classe EDA
        eda = CrimeEDA(DATA_PATH)

        # Carregar dados (anos de treino: 2019-2021)
        print("\n📂 Carregando dados de TREINO (2019-2021)...")
        eda.carregar_dados(anos=[2019, 2020, 2021])

        # Executar análise exploratória completa
        eda.executar_eda_completa()

        # Gerar relatório sumário
        eda.gerar_relatorio_sumario()

        print("\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("\n💡 PRÓXIMOS PASSOS:")
        print("   1. Revisar todos os gráficos gerados")
        print("   2. Incluir no PowerPoint da apresentação")
        print("   3. Preparar interpretações para cada análise")
        print("   4. Documentar decisões de tratamento de dados")

    except Exception as e:
        print(f"\n❌ ERRO durante a execução:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EXECUÇÃO - FORA DA CLASSE!
# ============================================================================

if __name__ == "__main__":
    # Configurar warnings
    warnings.filterwarnings('ignore')

    # Configurar matplotlib para não bloquear
    plt.ion()

    # Executar análise
    main()

    print("\n" + "=" * 70)
    print("🎯 ANÁLISE EXPLORATÓRIA FINALIZADA")
    print("=" * 70)
