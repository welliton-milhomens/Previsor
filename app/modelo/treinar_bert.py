import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from transformers import logging as transformers_logging

logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_error()

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class BERTTrainer:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_texts, train_labels, val_texts, val_labels, epochs=5, batch_size=16):
        logger.info("Iniciando treinamento do modelo BERT")
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, max_length=128)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, max_length=128)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        early_stopping = EarlyStopping(patience=3, verbose=True)

        for epoch in range(epochs):
            logger.info(f"Época {epoch+1}/{epochs}")
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Perda média de treinamento: {avg_train_loss:.4f}")

            val_loss, val_accuracy = self.evaluate(val_loader)
            logger.info(f"Perda de validação: {val_loss:.4f}, Acurácia: {val_accuracy:.4f}")

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("Treinamento concluído com sucesso")

    def evaluate(self, data_loader):
        self.model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_val_loss = total_val_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        return avg_val_loss, accuracy

    def predict(self, texts):
        self.model.eval()
        dataset = SentimentDataset(texts, [0] * len(texts), self.tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=len(texts))
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.softmax(outputs.logits, dim=-1)

        return predictions.cpu().numpy()

def preparar_e_treinar_bert(textos, sentimentos, test_size=0.2, random_state=42):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        textos, sentimentos, test_size=test_size, random_state=random_state, stratify=sentimentos
    )

    bert_trainer = BERTTrainer()
    bert_trainer.train(train_texts, train_labels, val_texts, val_labels)
    return bert_trainer

if __name__ == "__main__":
    # Exemplo de uso
    textos = [
        "Bom resultado financeiro",
        "Queda nas ações",
        "Expectativa positiva",
        "Prejuízo no trimestre",
        "Aumento nas vendas"
    ]
    sentimentos = [1, 0, 1, 0, 1]  # 1 para positivo, 0 para negativo

    bert_trainer = preparar_e_treinar_bert(textos, sentimentos)

    # Exemplo de previsão
    novos_textos = ["A empresa cresceu significativamente", "Houve uma grande perda no mercado"]
    previsoes = bert_trainer.predict(novos_textos)
    
    for texto, previsao in zip(novos_textos, previsoes):
        sentimento = "positivo" if previsao[1] > previsao[0] else "negativo"
        confianca = max(previsao)
        print(f"Texto: '{texto}'")
        print(f"Sentimento previsto: {sentimento}")
        print(f"Confiança: {confianca:.4f}\n")