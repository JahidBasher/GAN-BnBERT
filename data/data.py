import torch
from itertools import compress
import math
import numpy as np


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

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label_id': torch.tensor(self.label2class[label], dtype=torch.long),
            'label_mask': torch.tensor(label_mask)
        }


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


def process_all_data(input_examples, label_masks, balance_label_examples=False):
    examples = []

    num_labeled_examples = sum(label_masks)
    label_mask_rate = num_labeled_examples/len(input_examples)

    if balance_label_examples:
        print(
            f"Total Data: {len(input_examples)},"
            + f" Labeled Data: {num_labeled_examples},"
            + f" Masking Ratio: {label_mask_rate}"
        )

    if label_mask_rate == 1 or not balance_label_examples:
        examples = [(example, label_mask) for (example, label_mask) in zip(input_examples, label_masks)]
    else:
        for index, example in enumerate(input_examples):
            if label_masks[index]:
                balance = int(1/label_mask_rate)
                balance = max(1, int(math.log(balance, 2)))
                examples.extend(balance * [(example, label_masks[index])])

            else:
                examples.append((example, label_masks[index]))

    return examples

