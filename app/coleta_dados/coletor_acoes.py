import os
import sys
from datetime import timedelta
import yfinance as yf
import pandas as pd

# Adicione o diretório raiz do projeto ao PYTHONPATH
projeto_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, projeto_dir)

from banco_dados.gerenciador_bd import GerenciadorBD

import logging

logger = logging.getLogger(__name__)

class ColetorAcoes:
    def __init__(self):
        self.gerenciador_bd = GerenciadorBD()

    def obter_lista_acoes(self):
        # Aqui você deve implementar a lógica para obter a lista de todas as ações da B3
        # Por simplicidade, vamos usar uma lista estática por enquanto
        return ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']

    def coletar_dados_acoes(self, periodo='1y'):
        acoes = self.obter_lista_acoes()
        logger.info(f"Coletando dados para {len(acoes)} ações")
        for acao in acoes:
            try:
                logger.debug(f"Coletando dados para {acao}")
                dados = yf.Ticker(acao).history(period=periodo)
                if dados.empty:
                    logger.warning(f"Nenhum dado retornado para {acao}")
                    continue
                logger.debug(f"Dados coletados para {acao}: {len(dados)} registros")
                self.salvar_dados_acao(acao, dados)
            except Exception as e:
                logger.error(f"Erro ao coletar dados para {acao}: {str(e)}")
        logger.info("Coleta de dados de ações concluída")

    def salvar_dados_acao(self, acao, dados):
        dados_filtrados = []
        for index, row in dados.iterrows():
            dados_filtrados.append({
                'acao': acao,
                'data': index.date(),
                'abertura': row['Open'],
                'maxima': row['High'],
                'minima': row['Low'],
                'fechamento': row['Close'],
                'volume': row['Volume']
            })
        self.gerenciador_bd.upsert_dados('acoes', dados_filtrados, ['acao', 'data'])

    def atualizar_dados_acoes(self):
        acoes = self.obter_lista_acoes()
        end_date = pd.Timestamp.now().floor('D')
        for acao in acoes:
            ultima_data = self.gerenciador_bd.obter_ultima_data('acoes', acao)
            if ultima_data:
                start_date = pd.Timestamp(ultima_data) + pd.Timedelta(days=1)
                if start_date >= end_date:
                    logger.info(f"Dados já atualizados para {acao}. Pulando.")
                    continue
                logger.info(f"Coletando dados incrementais para {acao} de {start_date} até {end_date}")
                dados = yf.Ticker(acao).history(start=start_date, end=end_date)
            else:
                logger.info(f"Coletando todos os dados disponíveis para {acao} até {end_date}")
                dados = yf.Ticker(acao).history(period="max", end=end_date)
            self.salvar_dados_acao(acao, dados)

    def obter_dados_historicos(self, acao, periodo='30d'):
        dados = yf.Ticker(acao).history(period=periodo)
        return dados[['Open', 'High', 'Low', 'Close', 'Volume']].rename(
            columns={'Open': 'abertura', 'High': 'maxima', 'Low': 'minima', 'Close': 'fechamento'}
        )

if __name__ == '__main__':
    coletor = ColetorAcoes()
    coletor.coletar_dados_acoes()