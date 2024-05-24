## Overview
This repository contains the implementation of a GAN-BERT model for text classification tasks. The model leverages a Transformer encoder combined with a Generative Adversarial Network (GAN) framework to enhance performance on labeled and unlabeled data. The core components include an encoder (BanglaBERT), a discriminator, and a generator, designed to work together to classify text data more effectively. The corresponding paper is available at [Researchgate: Bengali Intent Classification with Generative Adversarial BERT (https://www.researchgate.net/publication/378530127_Bengali_Intent_Classification_with_Generative_Adversarial_BERT).

## Abstract
Intent classification is a fundamental task in natural language understanding, aiming to categorize user queries or sentences into predefined classes to understand user intent. The most challenging aspect of this particular task lies in effectively incorporating all possible classes of intent into a dataset while ensuring adequate linguistic variation. Plenty of research has been conducted in the related domains in rich-resource languages like English. In this study, we introduce BNIntent30, a comprehensive Bengali intent classification dataset containing 30 intent classes. The dataset is excerpted and translated from the CLINIC150 dataset containing a diverse range of user intents categorized over 150 classes. Furthermore, we propose a novel approach for Bengali intent classification using Generative Adversarial BERT to evaluate the proposed dataset, which we call GAN-BnBERT. Our approach leverages the power of BERT-based contextual embeddings to capture salient linguistic features and contextual information from the text data, while the generative adversarial network (GAN) component complements the model's ability to learn diverse representations of existing intent classes through generative modeling. Our experimental results demonstrate that the GAN-BnBERT model achieves superior performance on the newly introduced BNIntent30 dataset, surpassing the existing Bi-LSTM and the stand-alone BERT-based classification model.

## Technical Novelty
The model's novelty lies in the integration of GANs with BERT architecture to improve text classification. Key aspects include:

- Transformer Encoder: Utilizes a pre-trained BERT model (csebuetnlp/banglabert) for initial text embeddings.
- GAN Framework: Incorporates a generator to produce synthetic samples and a discriminator to distinguish between real and generated samples.
- Joint Training: Simultaneously trains the encoder, generator, and discriminator, allowing the model to leverage both labeled and unlabeled data effectively.
- Data-Efficient Learning: Integration of GAN allows the model to learn to classify even with small number of training data

## Use Cases
The model can be applied to various text classification tasks such as:

- Intent detection in conversational AI
- Sentiment analysis
- Topic classification
- Any other NLP task requiring robust text classification

## Installation
To install the necessary dependencies, clone the repository and install the required packages:

```bash
https://github.com/JahidBasher/GAN-BnBERT.git
cd GAN-BnBERT
pip install -r requirements.txt
```
## Training
To train the model, follow these steps:
- Prepare your dataset in the required format.
- Update the configuration in config.py as needed.
- Run the training script:

```bash
python main.py
```
This will train the model for the specified number of epochs and save the checkpoints in the defined artifact path.

##  Inference
To perform inference with the trained model, you need to set the mode to 'inference' and use the val_step function from the Trainer class:

```python
from config import Config
from trainer import Trainer

cfg = Config()
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
results = trainer.val_epoch(test_dataloader, mode='inference')
```
## Colab:
- How to replicate: [colab example](https://colab.research.google.com/drive/17km0Zmu7_m6Kv7UYt7pqKLjP9w4JDXHs#scrollTo=Nhe7bVIduxmT)

## BibTex
```latex
@inproceedings{inproceedings,
author = {Hasan, Mehedi and Ibna Basher, Mohammad Jahid and Shawon, Md},
year = {2023},
month = {12},
pages = {1-6},
title = {Bengali Intent Classification with Generative Adversarial BERT},
doi = {10.1109/ICCIT60459.2023.10440989}
}
```
