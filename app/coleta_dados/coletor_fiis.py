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

class ColetorFIIs:
    def __init__(self):
        self.gerenciador_bd = GerenciadorBD()

    def obter_lista_fiis(self):
        # Implemente uma lógica mais robusta para obter a lista de FIIs
        # Por exemplo, você pode buscar essa lista de uma fonte confiável ou de um arquivo de configuração
        return ['HGLG11.SA', 'KNRI11.SA', 'MXRF11.SA', 'BCFF11.SA', 'XPLG11.SA']

    def coletar_dados_fiis(self, periodo='1y'):
        fiis = self.obter_lista_fiis()
        logger.info(f"Coletando dados para {len(fiis)} FIIs")
        for fii in fiis:
            try:
                logger.debug(f"Coletando dados para {fii}")
                dados = yf.Ticker(fii).history(period=periodo)
                if dados.empty:
                    logger.warning(f"Nenhum dado retornado para {fii}")
                    continue
                logger.debug(f"Dados coletados para {fii}: {len(dados)} registros")
                self.salvar_dados_fii(fii, dados)
            except Exception as e:
                logger.error(f"Erro ao coletar dados para {fii}: {str(e)}")
        logger.info("Coleta de dados de FIIs concluída")

    def salvar_dados_fii(self, fii, dados):
        dados_filtrados = []
        for index, row in dados.iterrows():
            dados_filtrados.append({
                'fii': fii,
                'data': index.date(),
                'abertura': row['Open'],
                'maxima': row['High'],
                'minima': row['Low'],
                'fechamento': row['Close'],
                'volume': row['Volume'],
                'dividendos': row['Dividends']
            })
        self.gerenciador_bd.upsert_dados('fiis', dados_filtrados, ['fii', 'data'])

    def atualizar_dados_fiis(self):
        fiis = self.obter_lista_fiis()
        end_date = pd.Timestamp.now().floor('D')
        for fii in fiis:
            ultima_data = self.gerenciador_bd.obter_ultima_data('fiis', fii)
            if ultima_data:
                start_date = pd.Timestamp(ultima_data) + pd.Timedelta(days=1)
                if start_date >= end_date:
                    logger.info(f"Dados já atualizados para {fii}. Pulando.")
                    continue
                logger.info(f"Coletando dados incrementais para {fii} de {start_date} até {end_date}")
                dados = yf.Ticker(fii).history(start=start_date, end=end_date)
            else:
                logger.info(f"Coletando todos os dados disponíveis para {fii} até {end_date}")
                dados = yf.Ticker(fii).history(period="max", end=end_date)
            self.salvar_dados_fii(fii, dados)

    def obter_dados_historicos(self, fii, periodo='30d'):
        dados = yf.Ticker(fii).history(period=periodo)
        return dados[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']].rename(
            columns={'Open': 'abertura', 'High': 'maxima', 'Low': 'minima', 'Close': 'fechamento'}
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    coletor = ColetorFIIs()
    coletor.coletar_dados_fiis()