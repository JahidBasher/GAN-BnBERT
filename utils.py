import math
import numpy as np


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


def balance_train_data(train_examples, unlabeled_examples):

    train_label_masks = np.ones(len(train_examples), dtype=bool)

    if unlabeled_examples:
        train_examples = train_examples + unlabeled_examples
        tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
        train_label_masks = np.concatenate([train_label_masks, tmp_masks])
    
    return train_examples, train_label_masks




