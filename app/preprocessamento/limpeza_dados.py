import pandas as pd
import numpy as np
import os
import sys
import logging
logger = logging.getLogger(__name__)
# Adiciona o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from banco_dados.gerenciador_bd import GerenciadorBD

class LimpadorDados:
    def __init__(self):
        self.gerenciador_bd = GerenciadorBD()

    def limpar_dados(self, tabela):
        try:
            dados = self.gerenciador_bd.obter_dados(tabela)
            if dados.empty:
                logger.warning(f"Aviso: Não há dados na tabela {tabela}")
                return pd.DataFrame()
            dados_limpos = self.remover_valores_ausentes(dados)
            dados_limpos = self.tratar_outliers(dados_limpos)
            return dados_limpos
        except Exception as e:
            logger.error(f"Erro ao limpar dados da tabela {tabela}: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def remover_valores_ausentes(self, dados):
        return dados.dropna()

    def tratar_outliers(self, dados, iqr_multiplier=1.5):
        for coluna in dados.select_dtypes(include=[np.number]).columns:
            q1 = dados[coluna].quantile(0.25)
            q3 = dados[coluna].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            dados[coluna] = dados[coluna].clip(limite_inferior, limite_superior)
        return dados

if __name__ == '__main__':
    limpador = LimpadorDados()
    for tabela in ['acoes', 'fiis', 'dados_economicos']:
        dados_limpos = limpador.limpar_dados(tabela)
        if not dados_limpos.empty:
            print(f"Dados da tabela {tabela} limpos com sucesso.")
        else:
            print(f"Não foi possível limpar os dados da tabela {tabela}.")