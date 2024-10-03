import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import joblib

# Adiciona o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from banco_dados.gerenciador_bd import GerenciadorBD

logger = logging.getLogger(__name__)

class Normalizador:
    def __init__(self):
        self.gerenciador_bd = GerenciadorBD()
        self.scalers = {}
        self.scaler = MinMaxScaler()

    def obter_min_max(self, coluna, tabela):
        if coluna not in self.scalers:
            raise ValueError(f"Scaler não encontrado para a coluna {coluna}")
        scaler = self.scalers[coluna]
        return scaler.data_min_[0], scaler.data_max_[0]

    def normalizar_dados(self, tabela):
        logger.info(f"Iniciando normalização para tabela: {tabela}")
        dados = self.gerenciador_bd.obter_dados(tabela)
        if dados.empty:
            logger.warning(f"Não há dados para normalizar na tabela {tabela}")
            return None
        
        logger.info(f"Dados obtidos para normalização: {dados.shape}")

        # Identificar colunas para normalização
        if tabela in ['acoes', 'fiis']:
            colunas_para_normalizar = ['abertura', 'maxima', 'minima', 'fechamento', 'volume']
        elif tabela == 'dados_economicos':
            colunas_para_normalizar = ['valor']
        elif tabela == 'sentimento_mercado':
            colunas_para_normalizar = ['sentimento']
        else:
            logger.error(f"Tabela não reconhecida: {tabela}")
            return None
        
        # Verificar e lidar com valores nulos
        if dados[colunas_para_normalizar].isnull().any().any():
            logger.warning(f"Encontrados valores nulos na tabela {tabela}. Realizando imputação.")
            dados[colunas_para_normalizar] = dados[colunas_para_normalizar].fillna(dados[colunas_para_normalizar].mean())

        # Normalização
        dados_normalizados = dados.copy()
        for coluna in colunas_para_normalizar:
            scaler = MinMaxScaler()
            dados_normalizados[coluna] = scaler.fit_transform(dados[[coluna]])
            self.scalers[coluna] = scaler

        logger.info(f"Dados normalizados: {dados_normalizados.shape}")
        self.salvar_scalers(tabela)
        return dados_normalizados
    
    def normalizar_sequencia(self, sequencia, colunas):
        sequencia_normalizada = np.zeros_like(sequencia)
        for i, coluna in enumerate(colunas):
            if coluna in self.scalers:
                sequencia_normalizada[:, :, i] = self.scalers[coluna].transform(sequencia[:, :, i].reshape(-1, 1)).reshape(sequencia.shape[0], -1)
        return sequencia_normalizada

    def desnormalizar_sequencia(self, sequencia_normalizada, colunas):
        sequencia = np.zeros_like(sequencia_normalizada)
        for i, coluna in enumerate(colunas):
            if coluna in self.scalers:
                sequencia[:, :, i] = self.scalers[coluna].inverse_transform(sequencia_normalizada[:, :, i].reshape(-1, 1)).reshape(sequencia_normalizada.shape[0], -1)
        return sequencia

    def salvar_dados_normalizados(self, tabela, dados_normalizados):
        if not dados_normalizados.empty:
            try:
                dados_normalizados.to_sql(f'{tabela}_normalizados', self.gerenciador_bd.engine, if_exists='replace', index=False)
                logger.info(f"Dados normalizados da tabela {tabela} salvos com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao salvar dados normalizados da tabela {tabela}: {str(e)}")
        else:
            logger.warning(f"Não há dados normalizados para salvar da tabela {tabela}.")

    def desnormalizar_valor(self, valor_normalizado, coluna, tabela):
        if coluna not in self.scalers:
            self.carregar_scalers(tabela)
        scaler = self.scalers[coluna]
        if isinstance(valor_normalizado, np.ndarray):
            valor_normalizado = np.clip(valor_normalizado, 0, 1)
        else:
            valor_normalizado = max(0, min(valor_normalizado, 1))
        return scaler.inverse_transform([[valor_normalizado]])[0][0]

    def salvar_scalers(self, tabela):
        scaler_path = os.path.join(project_root, 'data', f'{tabela}_scalers.joblib')
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.scalers, scaler_path)

    def carregar_scalers(self, tabela):
        scaler_path = os.path.join(project_root, 'data', f'{tabela}_scalers.joblib')
        if os.path.exists(scaler_path):
            try:
                self.scalers = joblib.load(scaler_path)
            except Exception as e:
                raise ValueError(f"Erro ao carregar scalers para a tabela {tabela}: {str(e)}")
        else:
            raise ValueError(f"Scalers para a tabela {tabela} não encontrados. Execute a normalização primeiro.")

if __name__ == '__main__':
    normalizador = Normalizador()
    for tabela in ['acoes', 'fiis', 'dados_economicos', 'sentimento_mercado']:
        dados_normalizados = normalizador.normalizar_dados(tabela)
        normalizador.salvar_dados_normalizados(tabela, dados_normalizados)