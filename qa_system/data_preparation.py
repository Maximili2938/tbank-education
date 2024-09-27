# data_preparation.py
import pandas as pd
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row['question']
        context = row['context']
        answer = row['answer']
        start = row['answer_start']
        end = start + len(answer)

        encoding = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_offsets_mapping=True
        )

        offset_mapping = encoding.pop("offset_mapping")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]

        start_position = None
        end_position = None

        for i, (start_offset, end_offset) in enumerate(offset_mapping):
            if start_offset <= start < end_offset:
                start_position = i
            if start_offset <= end <= end_offset:
                end_position = i
                break

        if start_position is None:
            start_position = 0
        if end_position is None:
            end_position = 0

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df
