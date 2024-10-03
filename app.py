import os
import sys
from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime, timedelta
import numpy as np
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from tweepy.errors import TweepyException, TooManyRequests
from transformers import BertTokenizer, BertForSequenceClassification, logging as hf_logging

# Adicionar o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.coleta_dados.coletor_sentimento import ColetorSentimento
from app.coleta_dados.coletor_acoes import ColetorAcoes
from app.coleta_dados.coletor_fiis import ColetorFIIs
from app.coleta_dados.coletor_dados_economicos import ColetorDadosEconomicos
from app.modelo.treinador import TreinadorLSTM
from app.preprocessamento.normalizacao import Normalizador
from banco_dados.gerenciador_bd import GerenciadorBD

# Carregar variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Inicializar componentes
gerenciador_bd = GerenciadorBD()
normalizador = Normalizador()
treinador = TreinadorLSTM(output_dim=1)
coletor_sentimento = ColetorSentimento()
coletor_acoes = ColetorAcoes()
coletor_fiis = ColetorFIIs()
coletor_dados_economicos = ColetorDadosEconomicos()

def configurar_huggingface_autenticacao():
    nome_modelo = 'neuralmind/bert-base-portuguese-cased'
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        raise ValueError("Token de autenticação do Hugging Face não definido.")
    
    hf_logging.set_verbosity_warning()
    
    try:
        tokenizer = BertTokenizer.from_pretrained(nome_modelo, token=token, clean_up_tokenization_spaces=False)
        model = BertForSequenceClassification.from_pretrained(nome_modelo, token=token)
        logger.info("Acesso ao modelo Hugging Face verificado com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao acessar o modelo Hugging Face: {str(e)}")
        raise

configurar_huggingface_autenticacao()

@app.route('/')
def index():
    acoes = coletor_acoes.obter_lista_acoes()
    fiis = coletor_fiis.obter_lista_fiis()
    return render_template('index.html', acoes=acoes, fiis=fiis)

def inicializar_sistema():
    logger.info("Iniciando processo de atualização e treinamento do sistema")
    try:
        gerenciador_bd.criar_tabelas()
        atualizar_dados_sentimento()
        coletor_acoes.atualizar_dados_acoes()
        coletor_fiis.atualizar_dados_fiis()
        coletor_dados_economicos.atualizar_dados_economicos()
        
        tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'HGLG11.SA', 'KNRI11.SA', 'MXRF11.SA', 'BCFF11.SA', 'XPLG11.SA']
        coletor_sentimento.atualizar_sentimentos(tickers)
        
        for tabela in ['acoes', 'fiis', 'dados_economicos', 'sentimento_mercado']:
            dados_normalizados = normalizador.normalizar_dados(tabela)
            if dados_normalizados is not None and not dados_normalizados.empty:
                normalizador.salvar_dados_normalizados(tabela, dados_normalizados)
        
        coletor_sentimento.treinar_modelo_bert()

        for tabela in ['acoes', 'fiis', 'dados_economicos']:
            treinador.treinar(tabela)
        
        logger.info("Processo de atualização e treinamento concluído")
    except Exception as e:
        logger.error(f"Erro durante a inicialização do sistema: {str(e)}", exc_info=True)

def atualizar_dados_sentimento():
    ultima_atualizacao = gerenciador_bd.obter_ultima_data('sentimento_mercado')
    if ultima_atualizacao is None or ultima_atualizacao < datetime.now().date():
        logger.info("Atualizando dados de sentimento")
        tickers = coletor_acoes.obter_lista_acoes() + coletor_fiis.obter_lista_fiis()
        try:
            coletor_sentimento.atualizar_sentimentos(tickers)
        except Exception as e:
            logger.error(f"Erro ao atualizar sentimentos: {str(e)}", exc_info=True)

scheduler = BackgroundScheduler()
scheduler.add_job(func=inicializar_sistema, trigger="cron", hour=0, minute=0)
scheduler.start()

inicializar_sistema()

@app.route('/treinar', methods=['POST'])
def treinar():
    tabela = request.form['tabela']
    try:
        if not gerenciador_bd.verificar_dados_suficientes(tabela):
            return jsonify({'erro': f'Dados insuficientes para {tabela}'}), 400

        dados_normalizados = normalizador.normalizar_dados(tabela)
        if dados_normalizados is not None and not dados_normalizados.empty:
            normalizador.salvar_dados_normalizados(tabela, dados_normalizados)
        else:
            return jsonify({'erro': f'Falha ao normalizar dados para {tabela}'}), 400

        resultado = treinador.treinar(tabela)
        if resultado is None:
            return jsonify({'mensagem': f'Não foi possível treinar o modelo para {tabela} devido a dados insuficientes ou erros.'}), 400
        return jsonify({'mensagem': f'Modelo treinado e salvo com êxito para {tabela}'})
    except Exception as e:
        logger.error(f'Erro ao treinar modelo para {tabela}: {str(e)}', exc_info=True)
        return jsonify({'erro': f'Erro ao treinar modelo: {str(e)}'}), 500

@app.route('/prever', methods=['GET'])
def prever():
    tabela = request.args.get('tabela')
    tickers = request.args.getlist('tickers')
    dias = int(request.args.get('dias', 7))
    
    logger.debug(f"Iniciando previsão para tabela: {tabela}, tickers: {tickers}, dias: {dias}")
    
    normalizador.carregar_scalers(tabela)
    
    try:
        resultados = {}
        
        for ticker in tickers:
            X, _ = treinador.preparar_dados(tabela, ticker)
            if X is None or len(X) < treinador.janela_tempo:
                logger.warning(f"Dados históricos insuficientes para {ticker}")
                continue

            input_dim = X.shape[2]
            
            # Recriar o modelo com o número correto de features
            treinador.criar_modelo(input_dim)

            try:
                previsoes = treinador.prever(X, dias)
            except ValueError as e:
                logger.error(f"Erro ao fazer previsão para {ticker}: {str(e)}")
                continue

            ultimos_valores_reais = [normalizador.desnormalizar_valor(valor[3], 'fechamento', tabela) for valor in X[-7:, :, 3]]
            previsoes_desnormalizadas = [normalizador.desnormalizar_valor(valor, 'fechamento', tabela) for valor in previsoes]
            
            datas_historicas = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7, 0, -1)]
            datas_futuras = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(dias)]
            
            sentimento_medio = float(X[-1, -1, -1]) if X.shape[2] > 5 else 0.0
            
            resultados[ticker] = {
                'ultimos_valores_reais': ultimos_valores_reais,
                'datas_historicas': datas_historicas,
                'previsao': previsoes_desnormalizadas,
                'datas_futuras': datas_futuras,
                'sentimento_medio': sentimento_medio,
                'sentimento_incorporado': X.shape[2] > 5
            }
        
        logger.debug(f"Resultados: {resultados}")
        return jsonify(resultados)
    except Exception as e:
        logger.error(f'Erro ao fazer previsão: {str(e)}', exc_info=True)
        return jsonify({'erro': f'Erro ao fazer previsão: {str(e)}'}), 500

@app.route('/avaliar', methods=['POST'])
def avaliar():
    tabela = request.form['tabela']
    ticker = request.form['ticker']
    
    X, y = treinador.preparar_dados(tabela, ticker)
    if X is None or y is None:
        return jsonify({'erro': 'Dados insuficientes para avaliação'}), 400
    
    X_test, y_test = X[-100:], y[-100:]
    metricas = treinador.avaliar_modelo(X_test, y_test)
    
    return jsonify(metricas)

@app.route('/dados_historicos', methods=['GET'])
def dados_historicos():
    tabela = request.args.get('tabela')
    ticker = request.args.get('ticker')
    
    if not tabela or not ticker:
        return jsonify({'erro': 'Parâmetros inválidos'}), 400
    
    dados = obter_dados_historicos(tabela, ticker)
    if dados is None:
        return jsonify({'erro': 'Dados não encontrados'}), 404
    
    return jsonify({'dados': dados.tolist()})

@app.route('/comparar', methods=['GET'])
def comparar():
    tabela = request.args.get('tabela')
    tickers = request.args.getlist('tickers')
    
    if not tabela or not tickers:
        return jsonify({'erro': 'Parâmetros inválidos'}), 400
    
    resultados = {}
    for ticker in tickers:
        dados = obter_dados_historicos(tabela, ticker)
        if dados is not None:
            resultados[ticker] = dados.tolist()
    
    return jsonify(resultados)

@app.route('/atualizar_sentimentos', methods=['POST'])
def atualizar_sentimentos():
    tickers = request.json.get('tickers', [])
    if not tickers:
        return jsonify({'erro': 'Nenhum ticker fornecido'}), 400
    
    try:
        coletor_sentimento.atualizar_sentimentos(tickers)
        return jsonify({'mensagem': 'Sentimentos atualizados com sucesso'})
    except Exception as e:
        logger.error(f'Erro ao atualizar sentimentos: {str(e)}', exc_info=True)
        return jsonify({'erro': f'Erro ao atualizar sentimentos: {str(e)}'}), 500

def obter_dados_historicos(tabela, ticker):
    logger.debug(f"Obtendo dados históricos para tabela: {tabela}, ticker: {ticker}")
    try:
        if tabela == 'acoes':
            query = "SELECT abertura, maxima, minima, fechamento, volume FROM acoes_normalizados WHERE acao = :ticker ORDER BY data DESC LIMIT 30"
        elif tabela == 'fiis':
            query = "SELECT abertura, maxima, minima, fechamento, volume, dividendos FROM fiis_normalizados WHERE fii = :ticker ORDER BY data DESC LIMIT 30"
        else:
            raise ValueError(f"Tabela não reconhecida: {tabela}")

        dados = gerenciador_bd.executar_query(query, {'ticker': ticker})
        
        if dados.empty or len(dados) < 30:
            logger.warning(f"Dados insuficientes para {ticker}")
            return None
        
        dados = dados.iloc[::-1].astype('float32')
        
        logger.debug(f"Dados históricos (normalizados):\n{dados.head()}")
        
        return dados.values
    except Exception as e:
        logger.error(f"Erro ao obter dados históricos: {str(e)}", exc_info=True)
        return None

if __name__ == '__main__':
    print("Servidor iniciando...")
    print("Acesse http://127.0.0.1:5000 no seu navegador")
    app.run(debug=True, use_reloader=False)
    gerenciador_bd.criar_tabelas()
    coletor_sentimento.testar_autenticacao()