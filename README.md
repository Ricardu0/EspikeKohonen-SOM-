# Self-Organizing Map (SOM) - Comandos de Execução

## 📋 Comandos Básicos

```bash
# Comando completo com todos os parâmetros
python main.py --input SPSafe_2022.csv --output X_ready_advanced.parquet --sample_frac 0.3 --iterations 1000 --max_clusters 12 --map_size 20 --sigma 1.0 --learning_rate 0.5 --optimize

# Processamento rápido para testes
python main.py --input SPSafe_2022.csv --sample_frac 0.1 --iterations 300 --map_size 10

# Apenas amostragem e otimização
python main.py --input SPSafe_2022.csv --sample_frac 0.3 --optimize
