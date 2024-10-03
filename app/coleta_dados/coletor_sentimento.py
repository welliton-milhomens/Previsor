import os
import random
import pandas as pd
from datetime import datetime, timedelta
import logging

import torch
from banco_dados.gerenciador_bd import GerenciadorBD
from app.modelo.treinar_bert import BERTTrainer
from app.preprocessamento.analisador_sentimento import AnalisadorSentimento, mapear_sentimento
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
import time
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class ColetorSentimento:
    def __init__(self):
        self.gerenciador_bd = GerenciadorBD()
        self.analisador = AnalisadorSentimento()
        self.bert_trainer = BERTTrainer()
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                                  client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                                  user_agent=os.getenv('REDDIT_USER_AGENT'))
        self.vader = SentimentIntensityAnalyzer()

    def coletar_texto_de_url(self, url, ticker):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            titulo = soup.find('h1', class_='page-title')
            conteudo = soup.find('div', class_='article-content')
            
            if not titulo:
                titulo = soup.find('h1') or soup.find('title')
            if not conteudo:
                conteudo = soup.find('div', class_='content') or soup.find('article')
            
            titulo = titulo.text.strip() if titulo else ''
            conteudo = conteudo.text.strip() if conteudo else ''
            
            if not conteudo:
                conteudo = soup.body.text.strip() if soup.body else ''
            
            texto_completo = f"{titulo}\n\n{conteudo}"
            return texto_completo if texto_completo else None
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Erro 404: Página não encontrada para {ticker} em {url}. Continuando com outras fontes.")
            else:
                logger.error(f"Erro HTTP ao acessar {url} para {ticker}: {str(e)}")
        except requests.RequestException as e:
            logger.error(f"Erro ao fazer requisição para {url} para {ticker}: {str(e)}")
        except Exception as e:
            logger.error(f"Erro inesperado ao coletar texto para {ticker}: {str(e)}")
        return None

    def coletar_noticias_alpha_vantage(self, ticker):
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.alpha_vantage_api_key}"
            response = requests.get(url)
            data = response.json()
            noticias = []
            if 'feed' in data:
                for item in data['feed']:
                    noticias.append({
                        'data': item['time_published'],
                        'titulo': item['title'],
                        'texto': item['summary'],
                        'sentimento': item['overall_sentiment_score']
                    })
            logger.info(f"Coletadas {len(noticias)} notícias da Alpha Vantage para {ticker}")
            return pd.DataFrame(noticias)
        except Exception as e:
            logger.error(f"Erro ao coletar notícias da Alpha Vantage para {ticker}: {str(e)}")
            return pd.DataFrame()

    def coletar_posts_reddit(self, subreddit_name, query, limit=100):
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            for post in subreddit.search(query, limit=limit):
                posts.append({
                    'data': datetime.fromtimestamp(post.created_utc),
                    'titulo': post.title,
                    'texto': post.selftext,
                    'score': post.score
                })
            logger.info(f"Coletados {len(posts)} posts do Reddit para {query}")
            return pd.DataFrame(posts)
        except Exception as e:
            logger.error(f"Erro ao coletar posts do Reddit para {query}: {str(e)}")
            return pd.DataFrame()

    def simular_tweets_stocktwits(self, symbol, num_tweets=30):
        tweets_simulados = []
        sentimentos = ['Bullish', 'Bearish', 'Neutral']
        for _ in range(num_tweets):
            tweets_simulados.append({
                'data': datetime.now() - timedelta(minutes=random.randint(1, 1440)),
                'texto': f"Simulação de tweet sobre {symbol}. #stocks #investing",
                'sentimento': random.choice(sentimentos)
            })
        logger.info(f"Simulados {len(tweets_simulados)} tweets para {symbol}")
        return pd.DataFrame(tweets_simulados)

    def coletar_textos_de_fontes(self, ticker):
        textos = pd.DataFrame()
        
        # Alpha Vantage
        textos = pd.concat([textos, self.coletar_noticias_alpha_vantage(ticker)], ignore_index=True)
        
        # Reddit
        subreddit_name = 'investimentos'  # ou 'farialimabets'
        query = f"Petrobras OR {ticker}"
        textos = pd.concat([textos, self.coletar_posts_reddit(subreddit_name, query)], ignore_index=True)
        
        # Dados simulados no lugar do StockTwits
        textos = pd.concat([textos, self.simular_tweets_stocktwits(ticker)], ignore_index=True)
        
        # Infomoney (mantendo a coleta original)
        url = f'https://www.infomoney.com.br/cotacoes/b3/acao/{ticker.lower()}/'
        texto_infomoney = self.coletar_texto_de_url(url, ticker)
        if texto_infomoney:
            textos = pd.concat([textos, pd.DataFrame([{
                'data': datetime.now(),
                'texto': texto_infomoney,
                'ticker': ticker
            }])], ignore_index=True)
        
        return textos

    def analisar_sentimentos_em_lote(self, textos):
        sentimentos = []
        for _, linha in textos.iterrows():
            # Verifique se a chave 'ticker' existe antes de tentar acessá-la
            ticker = linha.get('ticker', 'N/A')  # Use 'N/A' ou outro valor padrão se não existir
            sentimento = self.analisador.analisar(linha['texto'])
            sentimentos.append({
                'data': linha['data'].date() if isinstance(linha['data'], datetime) else linha['data'],
                'ticker': ticker,
                'texto': linha['texto'],
                'sentimento': mapear_sentimento(sentimento) if isinstance(sentimento, dict) else sentimento
            })
        return sentimentos

    def salvar_sentimentos(self, sentimentos):
        if not sentimentos:
            logger.warning("Nenhum sentimento para salvar")
            return

        # Converte os valores de sentimento para um tipo compatível com double precision
        for sentimento in sentimentos:
            if isinstance(sentimento['sentimento'], list):
                sentimento['sentimento'] = sum(sentimento['sentimento']) / len(sentimento['sentimento'])

        try:
            self.gerenciador_bd.upsert_dados('sentimento_mercado', sentimentos, ['data', 'ticker'])
            logger.info(f"Sentimentos salvos com sucesso: {len(sentimentos)} registros")
        except Exception as e:
            logger.error(f"Erro ao salvar sentimentos: {str(e)}", exc_info=True)

    def atualizar_sentimentos(self, tickers):
        for ticker in tickers:
            try:
                textos = self.coletar_textos_de_fontes(ticker)
                if textos.empty:
                    logger.warning(f"Nenhum texto coletado para {ticker}. Usando dados simulados.")
                    textos = self.simular_tweets_stocktwits(ticker)
                sentimentos = self.analisar_sentimentos_em_lote(textos)
                self.salvar_sentimentos(sentimentos)
            except Exception as e:
                logger.error(f"Erro ao atualizar sentimentos para {ticker}: {str(e)}")

    def coletar_tweets_stocktwits(self, symbol, limit=30):
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        headers = {'Authorization': f'OAuth {os.getenv("STOCKTWITS_API_KEY")}'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        tweets = []
        if 'messages' in data:
            for message in data['messages'][:limit]:
                tweets.append({
                    'data': message['created_at'],
                    'texto': message['body'],
                    'sentimento': message['entities']['sentiment']['basic'] if 'sentiment' in message['entities'] else 'Neutral'
                })
        return tweets

    def analisar_sentimento_vader(self, texto):
        return self.vader.polarity_scores(texto)['compound']
    
    def analisar(self, texto):
        inputs = self.tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = (probabilities[0][1] - probabilities[0][0]).item()  # Range: -1 to 1
        return sentiment_score

    def combinar_fontes(self, ticker):
        noticias = self.coletar_noticias_alpha_vantage(ticker)
        posts_reddit = self.coletar_posts_reddit('wallstreetbets', ticker)
        tweets = self.coletar_tweets_stocktwits(ticker)

        # Analisar sentimento para Reddit e StockTwits
        posts_reddit['sentimento'] = posts_reddit['texto'].apply(self.analisar_sentimento_vader)
        tweets['sentimento'] = tweets['sentimento'].map({'Bullish': 1, 'Bearish': -1, 'Neutral': 0})

        # Combinar todas as fontes
        todas_fontes = pd.concat([noticias, posts_reddit, tweets], ignore_index=True)
        todas_fontes['data'] = pd.to_datetime(todas_fontes['data'])
        todas_fontes = todas_fontes.sort_values('data', ascending=False)

        return todas_fontes

    def calcular_sentimento_geral(self, dados):
        return dados['sentimento'].mean()


    def preparar_dados_para_bert(self):
        logger.info("Preparando dados para treinamento do modelo BERT")
        query = "SELECT texto, sentimento FROM sentimento_mercado WHERE texto IS NOT NULL ORDER BY data DESC"
        dados = self.gerenciador_bd.executar_query(query)
        
        if dados.empty:
            logger.warning("Nenhum dado disponível para treinamento do BERT")
            return [], [], [], []

        logger.info(f"Dados obtidos do banco: {len(dados)} registros")

        dados['sentimento'] = pd.to_numeric(dados['sentimento'], errors='coerce')
        dados = dados.dropna()

        if dados.empty:
            logger.warning("Todos os dados foram removidos após limpeza")
            return [], [], [], []

        train_size = int(0.8 * len(dados))
        train_texts = dados['texto'][:train_size].tolist()
        train_labels = dados['sentimento'][:train_size].tolist()
        val_texts = dados['texto'][train_size:].tolist()
        val_labels = dados['sentimento'][train_size:].tolist()
        
        logger.info(f"Dados preparados: {len(train_texts)} para treino, {len(val_texts)} para validação")
        return train_texts, train_labels, val_texts, val_labels

    def treinar_modelo_bert(self):
        logger.info("Iniciando preparação de dados para treinamento do modelo BERT")
        train_texts, train_labels, val_texts, val_labels = self.preparar_dados_para_bert()

        logger.info(f"Dados preparados: {len(train_texts)} para treino, {len(val_texts)} para validação")

        # Ajuste o critério de treinamento, se necessário
        if len(train_texts) < 50 or len(val_texts) < 10:  # Exemplo de ajuste para menos dados
            logger.warning("Dados insuficientes para treinar o modelo BERT. Necessário pelo menos 50 amostras de treino e 10 de validação.")
            return

        try:
            logger.info("Iniciando treinamento do modelo BERT")
            
            # Convertendo labels para inteiros
            train_labels = [int(label) for label in train_labels]
            val_labels = [int(label) for label in val_labels]
            
            if train_texts:
                logger.info(f"Exemplo de dados de treinamento: Texto: {train_texts[0][:100]}..., Label: {train_labels[0]}")
            
            self.bert_trainer.train(train_texts, train_labels, val_texts, val_labels, epochs=5, batch_size=16)

            logger.info("Treinamento do modelo BERT concluído com sucesso")

            logger.info("Salvando modelo BERT treinado")
            modelo_json = self.bert_trainer.model.config.to_json_string()
            pesos = self.bert_trainer.model.state_dict()
            self.gerenciador_bd.salvar_modelo("modelo_bert_sentimento", modelo_json, pesos)
            logger.info("Modelo de sentimento salvo com sucesso no banco de dados")
        except Exception as e:
            logger.error(f"Erro durante o treinamento do modelo BERT: {str(e)}", exc_info=True)

    def prever_sentimentos(self, textos):
        return self.bert_trainer.predict(textos)

if __name__ == '__main__':
    coletor = ColetorSentimento()
    tickers_teste = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'HGLG11.SA', 'KNRI11.SA', 'MXRF11.SA', 'BCFF11.SA', 'XPLG11.SA']
    
    coletor.atualizar_sentimentos(tickers_teste)
    coletor.treinar_modelo_bert()

    query = "SELECT * FROM sentimento_mercado ORDER BY data DESC, ticker LIMIT 10"
    resultado = coletor.gerenciador_bd.executar_query(query)
    print("\nÚltimos dados inseridos na tabela sentimento_mercado:")
    print(resultado)

    textos_teste = ["A empresa teve um ótimo desempenho este trimestre", "As ações caíram significativamente hoje"]
    previsoes = coletor.prever_sentimentos(textos_teste)
    print("\nPrevisões de sentimento para textos de teste:")
    for texto, previsao in zip(textos_teste, previsoes):
        print(f"Texto: {texto}")
        print(f"Previsão: {previsao}")
        print()