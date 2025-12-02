"""
Pipeline Principal - Arquivo de entrada
"""

import argparse
import logging
import joblib
import traceback
import numpy as np
import pandas as pd
from data.preprocessor import AdvancedDataPreprocessor
from models.som_trainer import MemoryEfficientSOMTrainer
from models.hyperparameter_optimizer import SOMHyperparameterOptimizer
from analysis.som_analyzer import KohonenAdvancedAnalyzer
from analysis.cluster_interpreter import SOMClusterInterpreter
from config.settings import RANDOM_STATE

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Função principal do pipeline avançado"""
    try:
        parser = argparse.ArgumentParser(description='Pipeline Avançado de Rede de Kohonen com Análise Interpretável')
        parser.add_argument('--input', default='SPSafe_2022.csv', help='Arquivo CSV de entrada')
        parser.add_argument('--output', default='X_ready_advanced.parquet', help='Arquivo de saída')
        parser.add_argument('--sample_frac', type=float, default=0.3, help='Fração de amostragem (0.1-1.0)')
        parser.add_argument('--iterations', type=int, default=1000, help='Iterações do SOM')  # ✅ Reduzido
        parser.add_argument('--max_clusters', type=int, default=12, help='Número máximo de clusters')
        parser.add_argument('--map_size', type=int, default=20, help='Tamanho do mapa (opcional)')  # ✅ Reduzido
        parser.add_argument('--sigma', type=float, default=1.0, help='Sigma do SOM')
        parser.add_argument('--learning_rate', type=float, default=0.5, help='Taxa de aprendizado')
        parser.add_argument('--optimize', action='store_true', help='Otimizar hiperparâmetros automaticamente')
        parser.add_argument('--fast_optimize', action='store_true', help='Otimização rápida (menos combinações)')  # ✅ Nova opção

        args = parser.parse_args()

        logger.info("=" * 70)
        logger.info("🧠 PIPELINE AVANÇADO - REDE DE KOHONEN (SOM PURO)")
        logger.info("=" * 70)

        # 1. PRÉ-PROCESSAMENTO AVANÇADO
        logger.info("🎯 FASE 1: PRÉ-PROCESSAMENTO E ANÁLISE EXPLORATÓRIA")
        preprocessor = AdvancedDataPreprocessor()

        # Primeiro carregamos o dataframe original para ter as colunas originais
        try:
            df = pd.read_csv(args.input, sep=';', encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(args.input, sep=';', encoding='latin-1', low_memory=False)

        # Aplicar amostragem se necessário
        if args.sample_frac and args.sample_frac < 1.0:
            df = df.sample(frac=args.sample_frac, random_state=42)
            logger.info(f"📊 Aplicada amostragem: {args.sample_frac*100}% dos dados")

        # CORREÇÃO: Agora passamos o caminho do arquivo para o preprocessor
        X_processed = preprocessor.full_pipeline(args.input, args.sample_frac)

        # IMPORTANTE: O full_pipeline remove outliers, então temos menos linhas
        # Precisamos sincronizar df com X_processed
        # Se X_processed tiver índice, usamos ele (assumindo que a ordem foi mantida)
        if hasattr(X_processed, 'index'):
            try:
                # Tentamos alinhar pelo índice
                df = df.iloc[X_processed.index].copy()
                logger.info("✅ Dados originais alinhados com dados processados pelo índice")
            except:
                # Fallback: usar o mesmo número de linhas
                df = df.head(len(X_processed)).copy()
                logger.warning("⚠️  Usando fallback para alinhamento de dados")
        else:
            # X_processed não tem índice (array numpy)
            df = df.head(len(X_processed)).copy()
            logger.warning("⚠️  X_processed não tem índice, usando fallback")

        X_processed.to_parquet(args.output, index=False)
        logger.info(f"💾 Dados processados salvos: {args.output}")
       
        # 2. TREINAMENTO DA REDE DE KOHONEN
        logger.info("🎯 FASE 2: TREINAMENTO DA REDE DE KOHONEN")

        data_for_training = X_processed.values.astype(np.float32)

        if args.optimize or args.fast_optimize:
            logger.info("   🔧 Executando otimização de hiperparâmetros...")
            optimizer = SOMHyperparameterOptimizer(random_state=RANDOM_STATE)

            # ✅ Grade de parâmetros adaptável
            if args.fast_optimize:
                param_grid = {
                    'som_x': [15, 20],  # ✅ Menos opções
                    'som_y': [15, 20],
                    'sigma': [0.8, 1.0],
                    'learning_rate': [0.3, 0.5],
                    'iterations': [500, 1000]  # ✅ Menos iterações
                }
                max_evaluations = 8  # ✅ Menos avaliações
            else:
                param_grid = {
                    'som_x': [20, 25, 30],
                    'som_y': [20, 25, 30],
                    'sigma': [0.8, 1.0, 1.2],
                    'learning_rate': [0.3, 0.5, 0.7],
                    'iterations': [1000, 2000, 3000]
                }
                max_evaluations = 15

            best_params = optimizer.optimize_parameters(
                data_for_training, param_grid, max_evaluations
            )

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

        logger.info(f"✅ Treinamento concluído: QE={q_error:.4f}, TE={t_error:.4f}")

        # 3. VISUALIZAÇÕES AVANÇADAS
        logger.info("🎯 FASE 3: VISUALIZAÇÕES E ANÁLISES")
        analyzer = KohonenAdvancedAnalyzer()
        analyzer.create_comprehensive_visualizations(som, X_processed)

        # 4. ANÁLISE DE CLUSTERS
        logger.info("🎯 FASE 4: ANÁLISE DE CLUSTERS (SOM PURO)")
        interpreter = SOMClusterInterpreter(preprocessor, kohonen_trainer, analyzer)
        df_with_clusters, quality_metrics = interpreter.analyze_som_clusters(
            X_processed, df, args.max_clusters
        )

        # Salvar resultados finais
        df_with_clusters.to_parquet('dataset_com_clusters_som.parquet', index=False)
        joblib.dump(kohonen_trainer.som, 'kohonen_model_pure_som.pkl')

        if quality_metrics:
            joblib.dump(quality_metrics, 'cluster_quality_metrics.pkl')

        logger.info("🎉 PIPELINE CONCLUÍDO COM SUCESSO!")

    except KeyboardInterrupt:
        logger.info("⏹️  Pipeline interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro no pipeline: {e}")
        logger.error(traceback.format_exc())
        raise

    if 'LATITUDE' in df_with_clusters.columns:
        lat_mean = df_with_clusters['LATITUDE'].mean()
        lon_mean = df_with_clusters['LONGITUDE'].mean()

        if abs(lat_mean) > 100 or abs(lon_mean) > 100:
            logger.error("❌ COORDENADAS AINDA CORROMPIDAS!")
            logger.error(f"   Latitude média: {lat_mean:.0f}")
            logger.error(f"   Longitude média: {lon_mean:.0f}")
            logger.error("   AÇÃO: Verificar preprocessor linha ~80")
        else:
            logger.info(f"✅ Coordenadas OK: Lat={lat_mean:.2f}, Lon={lon_mean:.2f}")

    # Verificar clusters
    n_clusters = df_with_clusters['CLUSTER_SOM'].nunique() - 1  # -1 para excluir ruído
    if n_clusters < 3:
        logger.warning(f"⚠️  Poucos clusters: {n_clusters}")
        logger.warning("   SUGESTÕES:")
        logger.warning("   1. Aumentar map_size (atual: 30 → testar 40)")
        logger.warning("   2. Reduzir sigma (atual: 1.5 → testar 1.0)")
        logger.warning("   3. Aumentar iterations (atual: 5000 → testar 8000)")
    else:
        logger.info(f"✅ Clusters identificados: {n_clusters}")

if __name__ == '__main__':
    main()

    # Instrução para rodar o script:
    # python main.py --input SPSafe_2022.csv --output X_ready_advanced.parquet --sample_frac 0.3 --iterations 1000 --max_clusters 12 --map_size 20 --sigma 1.0 --learning_rate 0.5 --optimize
    # python main.py --input SPSafe_2022.csv --sample_frac 0.1 --iterations 300 --map_size 10
    # python main.py --input SPSafe_2022.csv --sample_frac 0.3 --optimize