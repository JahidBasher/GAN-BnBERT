import math
import torch
from torch.utils.data import (
    TensorDataset, DataLoader, RandomSampler, SequentialSampler
)


def get_qc_examples_from_file(input_file):
    examples = []

    with open(input_file, 'r') as f:
        contents = f.read()
        file_as_list = contents.splitlines()
        for line in file_as_list[1:]:
            split = line.split(" ")
            question = ' '.join(split[1:])

            text_a = question
            inn_split = split[0].split(":")
            label = inn_split[0] + "_" + inn_split[1]
            examples.append((text_a, label))
        f.close()

    return examples


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