import json
import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import tensorflow as tf
from tensorflow import keras

# Configuração de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adiciona o diretório raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from banco_dados.gerenciador_bd import GerenciadorBD
from app.coleta_dados.coletor_sentimento import ColetorSentimento
from app.preprocessamento.analisador_sentimento import AnalisadorSentimento
from app.preprocessamento.normalizacao import Normalizador
from app.modelo.kan import LSTMModel, LSTMWithAttention
from app.utils.metricas import calcular_metricas

class TreinadorLSTM:
    def __init__(self, output_dim, janela_tempo=30):
        self.output_dim = output_dim
        self.janela_tempo = janela_tempo
        self.model = None
        self.gerenciador_bd = GerenciadorBD()
        self.coletor_sentimento = ColetorSentimento()
        self.analisador_sentimento = AnalisadorSentimento()
        self.normalizador = Normalizador()
        self.tabela_atual = None
        self.input_dim = None 

    def criar_modelo(self, input_dim):
        logger.debug(f"Criando modelo LSTM com input_dim: {input_dim}")
        self.model = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(self.janela_tempo, input_dim)),
            LSTMWithAttention(input_dim, self.output_dim)
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        logger.debug("Modelo LSTM criado com sucesso")

    def preparar_dados(self, tabela, ticker=None, janela_tempo=30, horizonte_previsao=1):
        self.tabela_atual = tabela
        try:
            if tabela == 'dados_economicos':
                return self.preparar_dados_economicos(janela_tempo, horizonte_previsao)
            
            query = f"""
                SELECT DISTINCT ON (n.data) n.data, n.abertura, n.maxima, n.minima, n.fechamento, n.volume, COALESCE(s.sentimento, 0) as sentimento
                FROM {tabela}_normalizados n
                LEFT JOIN sentimento_mercado s ON n.data = s.data AND n.{'acao' if tabela == 'acoes' else 'fii'} = s.ticker
                WHERE n.{'acao' if tabela == 'acoes' else 'fii'} = :ticker
                ORDER BY n.data DESC
                LIMIT 1000
            """
            dados = self.gerenciador_bd.executar_query(query, {'ticker': ticker})
            
            if dados.empty:
                logger.warning(f"Nenhum dado encontrado para {ticker}")
                return None, None

            dados = dados.sort_values('data')
            colunas_para_treinamento = ['abertura', 'maxima', 'minima', 'fechamento', 'volume', 'sentimento']
            
            dados_economicos = self.gerenciador_bd.obter_dados_economicos(dados['data'].min(), dados['data'].max())
            if not dados_economicos.empty:
                dados = pd.merge(dados, dados_economicos, on='data', how='left')
                colunas_para_treinamento.extend(dados_economicos.columns.drop('data').tolist())

            dados = dados.fillna(method='ffill').fillna(method='bfill')
            
            X, y = [], []
            for i in range(len(dados) - janela_tempo - horizonte_previsao + 1):
                X.append(dados[colunas_para_treinamento].iloc[i:i+janela_tempo].values)
                y.append(dados['fechamento'].iloc[i+janela_tempo:i+janela_tempo+horizonte_previsao].values)
            
            return np.array(X), np.array(y)
        
        except Exception as e:
            logger.error(f"Erro ao preparar dados para {ticker}: {str(e)}", exc_info=True)
            return None, None
        
    def determinar_input_dim(self, tabela):
        X, _ = self.preparar_dados(tabela)
        if X is not None:
            return X.shape[2]
        else:
            logger.warning(f"Não foi possível determinar input_dim para {tabela}. Usando valor padrão.")
            return 10  # Ajuste este valor para o número correto de features

    def preparar_dados_economicos(self, janela_tempo=30, horizonte_previsao=1):
        query = "SELECT * FROM dados_economicos_normalizados ORDER BY data"
        dados = self.gerenciador_bd.executar_query(query)
        
        if dados.empty:
            logger.warning("Nenhum dado econômico encontrado")
            return None, None

        dados = dados.pivot(index='data', columns='indicador', values='valor').sort_index()
        dados = dados.interpolate(method='linear', limit_direction='both').fillna(0)

        X, y = [], []
        for i in range(len(dados) - janela_tempo - horizonte_previsao + 1):
            X.append(dados.iloc[i:i+janela_tempo].values)
            y.append(dados.iloc[i+janela_tempo:i+janela_tempo+horizonte_previsao].values)
        
        return np.array(X), np.array(y)

    def treinar(self, tabela, epocas=100, batch_size=32):
        logger.info(f"Iniciando treinamento para tabela: {tabela}")
        try:
            if tabela == 'sentimento_mercado':
                return self.treinar_modelo_bert(tabela)

            X, y = self.preparar_dados(tabela)
            if X is None or y is None or len(X) < self.janela_tempo:
                logger.warning(f"Dados insuficientes para treinar o modelo para {tabela}")
                return

            input_dim = X.shape[2]
            self.criar_modelo(input_dim)

            tscv = TimeSeriesSplit(n_splits=5)
            historicos = []

            for train_index, val_index in tscv.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                history = self.model.fit(
                    X_train, y_train,
                    epochs=epocas,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    callbacks=[
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
                        keras.callbacks.ModelCheckpoint(f'./checkpoints/modelo_{tabela}.keras', save_best_weights_only=True)
                    ]
                )
                historicos.append(history)

            self.salvar_modelo(f"modelo_{tabela}")
            logger.info(f"Treinamento concluído com sucesso para {tabela}")
            return historicos
        except Exception as e:
            logger.error(f"Erro durante o treinamento para {tabela}: {str(e)}", exc_info=True)
            raise

    def treinar_modelo_bert(self, tabela):
        dados = self.gerenciador_bd.obter_dados(tabela)
        if dados.empty or len(dados) < 100:
            logger.warning(f"Dados insuficientes na tabela {tabela} para treinar o modelo BERT.")
            return

        textos = dados['texto'].tolist()
        labels = dados['sentimento'].tolist()

        train_size = int(0.8 * len(textos))
        train_texts, val_texts = textos[:train_size], textos[train_size:]
        train_labels, val_labels = labels[:train_size], labels[train_size:]

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            torch.tensor(train_labels)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_encodings['input_ids']),
            torch.tensor(val_encodings['attention_mask']),
            torch.tensor(val_labels)
        )

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        model.save_pretrained('./modelo_bert_sentimento')
        tokenizer.save_pretrained('./modelo_bert_sentimento')

        self.gerenciador_bd.salvar_modelo("modelo_bert_sentimento", model.config.to_json_string(), model.state_dict())
        logger.info("Modelo BERT de sentimento treinado e salvo com sucesso.")

    def prever(self, X, dias):
        if self.model is None:
            raise ValueError("Modelo não treinado. Por favor, treine o modelo primeiro.")

        previsoes = []
        input_seq = X[-1:]

        for _ in range(dias):
            previsao = self.model.predict(input_seq)
            previsoes.append(previsao[0, 0])
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1, 3] = previsao[0, 0]

        return self.validar_previsoes(previsoes, [X[-1, -1, 3]])

    def avaliar_modelo(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred)
        for nome, valor in metricas.items():
            logger.info(f"{nome}: {valor:.4f}")
        return metricas

    def carregar_modelo(self, tabela):
        nome_modelo = f"modelo_{tabela}"
        try:
            modelo_json, pesos = self.gerenciador_bd.carregar_modelo(nome_modelo)
            if modelo_json and pesos:
                config = json.loads(modelo_json)
                # Remova a camada KAN se estiver presente
                if 'config' in config and 'layers' in config['config']:
                    config['config']['layers'] = [layer for layer in config['config']['layers'] if layer['class_name'] != 'KAN']
                
                # Corrigir o problema com 'batch_shape' e 'dtype'
                for layer in config['config']['layers']:
                    if 'batch_shape' in layer['config']:
                        layer['config']['input_shape'] = layer['config']['batch_shape'][1:]
                        del layer['config']['batch_shape']
                    if 'dtype' in layer['config']:
                        layer['config']['dtype'] = None  # Use o dtype padrão

                self.model = keras.models.model_from_json(
                    json.dumps(config),
                    custom_objects={'LSTMModel': LSTMModel, 'LSTMWithAttention': LSTMWithAttention}
                )
                self.model.set_weights(pesos)
                logger.info(f"Modelo para {tabela} carregado com sucesso")
            else:
                logger.warning(f"Nenhum modelo encontrado para {tabela}")
                input_dim = self.determinar_input_dim(tabela)
                self.criar_modelo(input_dim)
        except Exception as e:
            logger.error(f"Erro ao carregar modelo para {tabela}: {str(e)}", exc_info=True)
            input_dim = self.determinar_input_dim(tabela)
            self.criar_modelo(input_dim)

    def salvar_modelo(self, nome_modelo):
        try:
            modelo_json = self.model.to_json()
            config = json.loads(modelo_json)
            for layer in config['config']['layers']:
                if layer['class_name'] == 'InputLayer':
                    layer['config']['input_shape'] = layer['config'].get('batch_input_shape', [])[1:]
                    if 'batch_input_shape' in layer['config']:
                        del layer['config']['batch_input_shape']

            modelo_json = json.dumps(config)
            pesos = self.model.get_weights()
            self.gerenciador_bd.salvar_modelo(nome_modelo, modelo_json, pesos)
            logger.info(f"Modelo {nome_modelo} salvo com sucesso no banco de dados")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo {nome_modelo}: {str(e)}", exc_info=True)

    def validar_previsoes(self, previsoes, ultimos_valores_reais):
        variacao_maxima_diaria = max(ultimos_valores_reais) * 0.05
        ultimo_valor_real = ultimos_valores_reais[-1]
        previsoes_validadas = []
        for i, previsao in enumerate(previsoes):
            variacao = previsao - ultimo_valor_real
            if abs(variacao) > variacao_maxima_diaria:
                logger.warning(f"Previsão {i+1} ({previsao:.2f}) fora da faixa esperada.")
                previsao = ultimo_valor_real + (variacao_maxima_diaria if variacao > 0 else -variacao_maxima_diaria)
            previsoes_validadas.append(previsao)
            ultimo_valor_real = previsao
        return previsoes_validadas

if __name__ == '__main__':
    treinador = TreinadorLSTM(output_dim=1)
    for tabela in ['acoes', 'fiis', 'dados_economicos']:
        logger.info(f'Treinando modelo para {tabela}')
        treinador.treinar(tabela)
        logger.info(f'Avaliando modelo para {tabela}')
        X, y = treinador.preparar_dados(tabela)
        if X is not None and y is not None:
            treinador.avaliar_modelo(X[-100:], y[-100:])