import math
import numpy as np

def get_qc_examples_from_file(files):
    import json
    examples = {'train': [], 'test': [], 'val': []}
    for f in files:
        with open(f, 'r') as f:
            data = json.load(f)
            for data_point in data:
                text = data_point['bn_text']
                intent = data_point['intent']
                split = data_point['split']

                if text == 'TRANSLATION_FAILED':
                    continue
                examples[split].append((text, intent))

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
