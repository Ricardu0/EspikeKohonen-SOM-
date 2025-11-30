"""
Utilitários para gerenciamento de memória
"""

import psutil
import os
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Gerenciador de memória para operações intensivas"""
    
    @staticmethod
    def get_memory_usage():
        """Retorna uso atual de memória em MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    @staticmethod
    def check_memory_available(required_mb=1000):
        """Verifica se há memória suficiente disponível"""
        try:
            available = psutil.virtual_memory().available / 1024 / 1024
            return available >= required_mb
        except:
            return True
    
    @staticmethod
    def optimize_data_types(df):
        """Otimiza tipos de dados para reduzir uso de memória"""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        return df