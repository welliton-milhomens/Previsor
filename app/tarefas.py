from celery_config import app
from banco_dados.gerenciador_bd import GerenciadorBD
import logging
logger = logging.getLogger(__name__)

@app.task
def atualizar_sentimentos_background(tickers):
    from app.coleta_dados.coletor_sentimento import ColetorSentimento
    try:
        coletor = ColetorSentimento()
        coletor.atualizar_sentimentos(tickers)
    except Exception as e:
        # Log the error and handle it
        logger.error(f"Erro ao atualizar sentimentos em background: {str(e)}")

@app.task
def limpar_cache_antigo():
    try:
        gerenciador_bd = GerenciadorBD()
        gerenciador_bd.limpar_cache_antigo()
    except Exception as e:
        # Log the error and handle it
        logger.error(f"Erro ao limpar cache antigo: {str(e)}")