from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import requests
from functools import lru_cache
from retrying import retry
import logging
import json

logger = logging.getLogger(__name__)

# Adicione o diretório raiz do projeto ao PYTHONPATH
projeto_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, projeto_dir)

from banco_dados.gerenciador_bd import GerenciadorBD

class ColetorDadosEconomicos:
    def __init__(self):
        self.gerenciador_bd = GerenciadorBD()
        self.base_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json"
        logger.info("ColetorDadosEconomicos inicializado")

    def coletar_dados_economicos(self):
        series = {
            'pib': 4380,
            'inflacao': 433,
            'selic': 432,
            'cambio': 1
        }

        for nome, codigo in series.items():
            dados = self.obter_dados_serie(codigo)
            self.salvar_dados_economicos(nome, dados)

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def obter_dados_serie(self, codigo, data_inicial=None):
        now = datetime.now().date()
        if data_inicial:
            if isinstance(data_inicial, datetime):
                data_inicial = data_inicial.date()
            if data_inicial > now:
                logger.warning(f"Data inicial {data_inicial} é no futuro. Usando a data atual.")
                data_inicial = now
        else:
            data_inicial = now - timedelta(days=30)  # Coleta dados dos últimos 30 dias por padrão

        url = f"{self.base_url.format(codigo)}&dataInicial={data_inicial.strftime('%d/%m/%Y')}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            dados = response.json()
            if not dados:
                logger.warning(f"Nenhum dado retornado para o código {codigo}")
                return pd.DataFrame()
            df = pd.DataFrame(dados)
            df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
            return df
        except requests.RequestException as e:
            logger.error(f"Erro ao obter dados para código {codigo}: {str(e)}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON para código {codigo}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro inesperado ao obter dados para código {codigo}: {str(e)}")
            return pd.DataFrame()

    def salvar_dados_economicos(self, nome, dados):
        if not dados.empty:
            logger.debug(f"Dados coletados para {nome}:\n{dados.head()}")
            dados['indicador'] = nome
            dados.rename(columns={'data': 'data', 'valor': 'valor'}, inplace=True)
            self.gerenciador_bd.upsert_dados('dados_economicos', dados.to_dict('records'), ['indicador', 'data'])
            logger.info(f"Dados econômicos para {nome} atualizados com sucesso. Total de registros: {len(dados)}")
        else:
            logger.warning(f"Nenhum dado coletado para {nome}")
   
    def atualizar_dados_economicos(self):
        series = {
            'pib': 4380,
            'inflacao': 433,
            'selic': 432,
            'cambio': 1
        }

        for nome, codigo in series.items():
            ultima_data = self.gerenciador_bd.obter_ultima_data('dados_economicos', nome)
            if ultima_data:
                data_inicial = ultima_data + timedelta(days=1)
            else:
                data_inicial = datetime.now().date() - timedelta(days=30)
            
            # Ajuste específico para o câmbio
            if nome == 'cambio':
                data_inicial = datetime.now().date() - timedelta(days=7)  # Coleta dados da última semana
            
            dados = self.obter_dados_serie(codigo, data_inicial)
            
            if not dados.empty:
                self.salvar_dados_economicos(nome, dados)
            else:
                logger.warning(f"Nenhum novo dado disponível para {nome} desde {data_inicial}")

if __name__ == '__main__':
    coletor = ColetorDadosEconomicos()
    coletor.atualizar_dados_economicos()