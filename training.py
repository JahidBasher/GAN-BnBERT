import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        config,
        encoder,
        discriminator,
        generator,
        device,
        artifact_path='artifact/checkpoint'
    ):
        self.config = config
        self.encoder = encoder
        self.discriminator = discriminator
        self.generator = generator
        self.device = device
        self.negative_log_likelihood_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.artifact_path = artifact_path

        os.makedirs(artifact_path, exist_ok=True)

    def to_device(self, device="cuda"):
        if device == "cuda" and torch.cuda.is_available():
            device = torch.device(device) 
        else:
            print(f"Selected Device: CPU")
            torch.device('cpu')
        self.encoder.to(device)
        self.discriminator.to(device)
        self.generator.to(device)

    def configure_optimizer(self):
        discriminator_params = (
            [i for i in self.encoder.parameters()]
            + [p for p in self.discriminator.parameters()]
        )
        generator_params = [p for p in self.generator.parameters()]

        self.discriminator_optimizer = torch.optim.AdamW(
            discriminator_params, lr=self.config.learning_rate_discriminator
        )
        self.generator_optimizer = torch.optim.AdamW(
            generator_params, lr=self.config.learning_rate_generator
        )

    def set_train_dataloader(self, dataset, sampler=None):
        self.train_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler(dataset) if sampler else None,
            batch_size=self.config.train_batch_size
        )

    def set_val_dataloader(self, dataset, sampler=None):
        self.val_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler(dataset) if sampler else None,
            batch_size=self.config.val_batch_size
        )

    def _generate_noise(self, batch_size):
        return torch.zeros(
            batch_size,
            self.config.generator_noise_size,
            device=self.device
        ).uniform_(0, 1)

    def train(self, epochs=10):
        for epoch in tqdm(range(epochs)):
            print(f"Training Epoch: {epoch}/{epochs}")
            train_loss = self.train_epoch(self.train_loader)
            print("Train Metric: ", train_loss)
            val_loss = self.val_epoch(self.val_loader)
            print("Val Metric: ", val_loss)

            self.save_model(epoch, train_loss, val_loss)

    def train_epoch(self, train_dataloader):
        generator_loss, discriminator_loss = 0, 0

        for step, batch in tqdm(enumerate(train_dataloader)):
            _all_losses = self.train_step(batch)
            generator_loss += _all_losses['generator_loss']
            discriminator_loss += _all_losses['discriminator_loss']

        return {
            'generator_loss': generator_loss/len(train_dataloader),
            'discriminator_loss': discriminator_loss/len(train_dataloader)
        }
    
    def train_step(self, batch):
        self.encoder.train(), self.generator.train(), self.discriminator.train()

        # b_input_ids = batch['input_ids'].to(self.device)
        # b_input_mask = batch['attention_mask'].to(self.device)
        # b_labels = batch['label_id'].to(self.device)
        # b_label_mask = batch['label_mask'].to(self.device)

        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = batch[2].to(self.device)
        b_label_mask = batch[3].to(self.device)

        real_batch_size = b_input_ids.shape[0]

        model_outputs = self.encoder(b_input_ids, attention_mask=b_input_mask)
        hidden_states = model_outputs[-1]
        
        generator_representation = self.generator(self._generate_noise(real_batch_size))

        disciminator_input = torch.cat([hidden_states, generator_representation], dim=0)
        features, logits, probs = self.discriminator(disciminator_input)

        dis_real_features, dis_fake_features = torch.split(features, real_batch_size)
        dis_real_logits, dis_fake_logits = torch.split(logits, real_batch_size)
        dis_real_probs, dis_fake_probs = torch.split(probs, real_batch_size)

        generator_loss = (
            -1 * torch.mean(torch.log(1 - dis_fake_probs[:,-1] + self.config.epsilon))
            + torch.mean(torch.pow(torch.mean(dis_real_features, dim=0) - torch.mean(dis_fake_features, dim=0), 2))
        )

        log_probs = F.log_softmax(dis_real_logits[:, 0:-1], dim=-1)
        # The discriminator provides an output for labeled and unlabeled real data
        # so the loss evaluated for unlabeled data is ignored (masked)
        label2one_hot = torch.nn.functional.one_hot(b_labels, len(self.config.label_list))
        single_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
        selected_single_example_loss = torch.masked_select(
            single_example_loss, b_label_mask.to(self.device)
        )

        labeled_example_count = selected_single_example_loss.type(torch.float32).numel()

        dl_supervised = (
            0 if labeled_example_count == 0
            else torch.div(torch.sum(selected_single_example_loss.to(self.device)), labeled_example_count)
        )

        dl_unsupervised1u = -1 * torch.mean(torch.log(1 - dis_real_probs[:, -1] + self.config.epsilon))
        dl_unsupervised2u = -1 * torch.mean(torch.log(dis_fake_probs[:, -1] + self.config.epsilon))
        discriminator_loss = dl_supervised + dl_unsupervised1u + dl_unsupervised2u

        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        # retain_graph=True is required since the underlying graph will be deleted after backward
        generator_loss.backward(retain_graph=True)
        discriminator_loss.backward()

        self.generator_optimizer.step()
        self.discriminator_optimizer.step()

        return {
            'generator_loss': generator_loss.detach().cpu().numpy(),
            'discriminator_loss': discriminator_loss.detach().cpu().numpy()
        }                   
                
    def val_epoch(self, val_dataloader):

        validation_loss, predictions, ground_truths = 0, [], []

        for step, batch in tqdm(enumerate(val_dataloader)):
            _validation_loss, _predictions, _ground_truths = self.val_step(
                batch, device=self.device
            )
            validation_loss += _validation_loss
            predictions += _predictions
            ground_truths += _ground_truths
            # print(_validation_loss, _predictions, _ground_truths)

        predictions = torch.stack(predictions).numpy()
        ground_truths = torch.stack(ground_truths).numpy()

        validation_loss = (validation_loss / len(val_dataloader)).item()
        validation_accouracy = self._compute_metrics(
            predictions=predictions,
            ground_truths=ground_truths
        )

        return {
            "validation_accuracy": validation_accouracy,
            "validation_loss": validation_loss
        }
        
    def val_step(self, batch, inference_mode=False, device=None):

        self.encoder.eval()
        self.discriminator.eval()
        self.generator.eval()

        # b_input_ids = batch['input_ids'].to(self.device)
        # b_input_mask = batch['attention_mask'].to(self.device)
        # b_labels = None if inference_mode else batch['label_id'].to(device)

        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = None if inference_mode else batch[2].to(device)
 
        with torch.no_grad():
            model_outputs = self.encoder(b_input_ids, attention_mask=b_input_mask)
            hidden_states = model_outputs[-1]
            _, logits, probs = self.discriminator(hidden_states)
            # log_probs = F.log_softmax(probs[:,1:], dim=-1)
            filtered_logits = logits[:, 0:-1]
            # Accumulate the test loss.
            validation_loss = 0 if inference_mode else self.negative_log_likelihood_loss(filtered_logits, b_labels)
             
        # Accumulate the predictions and the input labels
        _, preds = torch.max(filtered_logits, 1)
        predictions = preds.detach().cpu()
        ground_truths = None if inference_mode else b_labels.detach().cpu()

        return validation_loss, predictions, ground_truths
    
    def _compute_metrics(self, predictions, ground_truths):
        accuracy = np.sum(predictions == ground_truths) / len(predictions)
        return accuracy

    def save_model(self, epoch, train_loss, val_loss):
        val_loss_log = '_'.join([f"{k}_{v:.5f}" for k, v in val_loss.items()])
        train_loss_log = '_'.join([f"{k}_{v:.5f}" for k, v in train_loss.items()])
        saving_path = f"{self.artifact_path}/epoch_{epoch}_{val_loss_log}_{train_loss_log}.pt"

        torch.save(
            {
                "epoch": epoch,
                "encoder": self.encoder.state_dict(),
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "generator_optim": self.generator_optimizer.state_dict(),
                "discriminator_optim": self.discriminator_optimizer.state_dict(),
                "train_loss": train_loss,
                "val_los": val_loss
            },
            saving_path
        )

    def tensorboard_logging(self):
        pass

    def logging(self):
        pass

