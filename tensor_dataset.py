import math
from itertools import compress
import numpy as np
import torch
from torch.utils.data import (
    TensorDataset, DataLoader, RandomSampler, SequentialSampler
)
# two data pipeline, you can use either of them
# if you want to use DataGenerator, you have to create a DataLoader object as well
# -------------------------------------tensorDataset------------------------------------------------------


def get_tensor_dataset(
    examples, label_mapper, tokenizer,
    max_seq_length=512, batch_size=64, sampler=RandomSampler
):
    input_ids = []
    input_mask_array = []
    label_mask_array = []
    label_id_array = []

    for ((text, label), label_mask) in examples:
        encoded_sent = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True
        )
        input_ids.append(encoded_sent)
        label_id_array.append(label_mapper[label])
        label_mask_array.append(label_mask)

    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        input_mask_array.append(att_mask)
    
    input_ids = torch.tensor(input_ids)
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)
    label_mask_array = torch.tensor(label_mask_array)

    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

    if sampler:
        sampler = sampler
    else:
        sampler = SequentialSampler

    return DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size)


# ---------------------------------Torch.utils.data.Dataset-----------------------------------------------


class DataGenerator(torch.utils.data.Dataset):
    def __init__(
        self, data, tokenizer, label2class,
        preprocessor=lambda x: x,
        max_len=512
    ):
        self.dataset = data
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_seq_length = max_len
        self.label2class = label2class

        print(f"Total {len(data)} Data found!!!")

    def _shorten_data(self, dataset_split=.1):
        print('Shortening The Dataset to:', str(100*dataset_split) + '%')

        selected_idx = np.arange(len(self.dataset))
        np.random.shuffle(selected_idx)

        selected_idx = selected_idx[:int(dataset_split*len(selected_idx))]
        self.dataset = list(compress(self.dataset, selected_idx))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        (text, label), label_mask = self.dataset[index]

        input_ids, attention_mask = self.preprocessor(
            text, self.tokenizer,
            add_special_tokens=True,
            max_seq_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )

        return torch.tensor(input_ids),\
               torch.tensor(attention_mask),\
               torch.tensor(self.label2class[label], dtype=torch.long),\
               torch.tensor(label_mask)


def text_processor(
    example, tokenizer,
    add_special_tokens=True,
    max_seq_length=512,
    padding="max_length",
    truncation=True
):
    encoded_sent = tokenizer.encode(
        example,
        add_special_tokens=add_special_tokens,
        max_length=max_seq_length,
        padding=padding,
        truncation=truncation
    )

    attention_mask = [int(token_id > 0) for token_id in encoded_sent]    

    return encoded_sent, attention_mask
