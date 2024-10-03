from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import re

class AnalisadorSentimento:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.model = AutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased")

    def analisar(self, texto):
        inputs = self.tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities[0].tolist()

    def limpar_texto(self, texto):
        texto = re.sub(r'http\S+', '', texto)  # remove URLs
        texto = re.sub(r'@\w+', '', texto)     # remove menções
        texto = re.sub(r'#\w+', '', texto)     # remove hashtags
        return texto.strip()

    def analisar_sentimento(self, texto):
        texto_limpo = self.limpar_texto(texto)
        analise = TextBlob(texto_limpo)
        return analise.sentiment.polarity

    def analisar_tweets(self, df):
        df['sentimento'] = df['texto'].apply(self.analisar_sentimento)
        return df
    
# Função para mapear as probabilidades para um valor de sentimento entre -1 e 1
def mapear_sentimento(probabilidades):
    return (probabilidades[1] - probabilidades[0]) * 2 - 1