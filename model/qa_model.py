# models/qa_model.py
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader
from data_preparation import QADataset, load_data

class QAModel:
    def __init__(self, model_name='DeepPavlov/rubert-base-cased-sentiment', max_len=512, batch_size=8, epochs=3, lr=3e-5):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

    def prepare_dataloader(self, data_path):
        df = load_data(data_path)
        dataset = QADataset(df, self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader

    def train(self, train_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

    def save(self, save_directory='qa_model'):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def load(self, model_directory='qa_model'):
        self.tokenizer = BertTokenizer.from_pretrained(model_directory)
        self.model = BertForQuestionAnswering.from_pretrained(model_directory)
