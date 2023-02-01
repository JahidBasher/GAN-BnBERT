import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from config import Config
from trainer import Trainer
from modeling import Discriminator, Generator
from tensor_dataset import get_tensor_dataset
import utils


def seed_every_thing(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)


def get_device(device='cpu'):
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device


def get_model_components(cfg: Config):
    encoder = AutoModel.from_pretrained(cfg.encoder_model_name)
    discriminator = Discriminator(
        input_size=cfg.input_size,
        hidden_sizes=[cfg.hidden_size]*cfg.num_hidden_layers_d,
        num_labels=len(cfg.label_list)+1,
        dropout_rate=cfg.out_dropout_rate
    )
    generator = Generator(
        noise_size=cfg.generator_noise_size,
        output_size=cfg.hidden_size,
        hidden_sizes=[cfg.hidden_size]*cfg.num_hidden_layers_g,
        dropout_rate=cfg.out_dropout_rate
    )
    return encoder, generator, discriminator


def main(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model_name)

    labeled_examples = utils.get_qc_examples_from_file(cfg.labeled_file)
    unlabeled_examples = utils.get_qc_examples_from_file(cfg.unlabeled_file)
    test_examples = utils.get_qc_examples_from_file(cfg.test_filename)

    train_examples, train_label_masks = utils.balance_train_data(
        train_examples=labeled_examples,
        unlabeled_examples=unlabeled_examples
    )

    train_data = utils.process_all_data(
        train_examples,
        train_label_masks,
        balance_label_examples=cfg.apply_balance
    )
    test_data = utils.process_all_data(
        test_examples,
        np.ones(len(test_examples), dtype=bool),
        balance_label_examples=False
    )
    train_dataloader = get_tensor_dataset(
        train_data,
        label_mapper=cfg.label2class,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        batch_size=cfg.train_batch_size
    )
    test_dataloader = get_tensor_dataset(
        test_data,
        label_mapper=cfg.label2class,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        batch_size=cfg.train_batch_size
    )

    encoder, generator, discriminator = get_model_components(cfg)

    trainer = Trainer(
        config=cfg,
        encoder=encoder,
        discriminator=discriminator,
        generator=generator,
        train_loader=train_dataloader,
        val_loader=test_dataloader,
        device="cuda"
    )
    trainer.to_device()
    trainer.configure_optimizer()
    trainer.train(epochs=cfg.num_train_epochs)


if __name__ == "__main__":
    main(Config())









