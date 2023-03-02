import torch
import pytorch_lightning as pl

from transformers import AutoModel
from torch.nn import functional as F
from sentence_transformers.models import Pooling
from torchmetrics.classification import BinaryF1Score

# TO DO
class AgreementAdversary(torch.nn.Module):
    def __init__(self,
        input_dim: int,
        embedding_dim: int,
        semantic_filters: int
    ):
        """
        Args:
            input_dim (int): Dimensionality of the input representation.
            embedding_dim (int): Dimensionality of the semantic (output)
            representation.
            semantic_filters (int, optional): Number of semantic features to
            extract.
        """
        super().__init__()

        # Semantic Embedder
        # calc the dimensionality of the conv kernel
        stride = 1
        kernel_dim = input_dim - stride * (embedding_dim - 1)

        assert kernel_dim > 0, "`embedding_dim` too big."

        self.infuser = torch.nn.Conv1d(
            in_channels=1,
            out_channels=semantic_filters,
            kernel_size=kernel_dim,
            stride=stride
        )

        # A 1-D conv layer to format the output. Function similar to 1x1 2d conv in
        # InceptionNet.
        self.presenter = torch.nn.Conv1d(
            in_channels=semantic_filters,
            out_channels=1,
            kernel_size=1,
            stride=1
        )

    def forward(self, plain_embedding):
        # Reshape to get add the channel dim
        plain_embedding = plain_embedding.unsqueeze(1)
        # Get the semantic embedding
        sem_embedding = self.infuser(plain_embedding)
        sem_embedding = self.presenter(sem_embedding)

        # Remove the channel dim
        return sem_embedding.squeeze(1)


class AgreementModel_kldiv(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        embedding_dim: int = 128,
        semantic_filters: int = 4,
        soft_label_imp: float = 0.8,
        backbone_lr: float = 2e-5,
        task_head_lr: float = 2e-4,
        sem_infuser_lr: float = 2e-3,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            embedding_dim (int): Dimensionality of the semantic representation.
            Defaults to 128.
            semantic_filters (int, optional): Number of semantic features to
            extract. Defaults to 4.
            soft_label_imp (float): How much importance (range 0-1) to assign
            to the soft_label loss, 0 would mean no dissagreement loss and 1 would
            mean the classification loss is ignored. Defaults to 0.8.
            backbone_lr (float, optional): The LLM's learning rate.
            Defaults to 2e-5.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
            sem_infuser_lr (float, optional): Sem infuser's learning rate.
            Defaults to 2e-3.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.backbone_lr = backbone_lr
        self.task_head_lr = task_head_lr
        self.weight_decay = weight_decay
        self.soft_label_imp = soft_label_imp

        # Base model        
        self.backbone = AutoModel.from_pretrained(mpath)
        # Pooling layer to get the sentence embedding
        self.pooling = Pooling(
            self.backbone.config.hidden_size,
            pooling_mode='mean'
        )
        # Freeze the backbone model
        if not backbone_lr:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Soft label predictor
        self.soft_label_head = torch.nn.Linear(
            self.backbone.config.hidden_size,
            2
        )

        # Hard label predictor
        self.hard_label_head = torch.nn.Linear(
            self.backbone.config.hidden_size,
            1
        )

        self.hard_label_loss = torch.nn.BCEWithLogitsLoss()
        self.soft_label_loss = torch.nn.KLDivLoss(
            reduction="batchmean"
        )

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, **tokens):
        # Push all inputs to the device in use
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Get the topic interaction vector ([CLS] vector)
        token_embeddings = self.backbone(**tokens)[0]

        # Vanilla embedding from the backbone
        sentence_embedding = self.pooling({
            'token_embeddings': token_embeddings,
            'attention_mask': tokens['attention_mask']
        })['sentence_embedding']

        # L2 normalize the embeddings
        sentence_embedding = F.normalize(
            sentence_embedding,
            dim=1
        )

        # Get hard Labels
        hard_labels = self.hard_label_head(sentence_embedding)

        # Get Soft Labels
        soft_labels = self.soft_label_head(sentence_embedding)

        return hard_labels, soft_labels

    def common_step(self, batch, batch_idx):
        _, tokens, (targets, soft_targets) = batch
        hard_labels, soft_labels = self(**tokens)

        __hard =  self.hard_label_loss(
            hard_labels, targets
        )

        __soft = self.soft_label_loss(
            soft_labels.log_softmax(dim=1),
            soft_targets
        )

        # Instead of taking the weighted average we compute the
        # more important loss as a multiple of the less important loss.
        __hard_multiplier = max(1, (1-self.soft_label_imp)/self.soft_label_imp)
        __soft_multiplier = max(1, self.soft_label_imp/(1-self.soft_label_imp))
        return {
            'loss': __soft_multiplier * __soft + \
                __hard_multiplier * __hard,
            'soft_loss': __soft,
            'hard_loss': __hard
        }

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("val_" + k, v.item(), prog_bar=True)

        return loss_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        hard_metric = BinaryF1Score().to(self.device)
        _, tokens, (targets, soft_targets) = batch
        hard_labels, soft_labels = self(**tokens)

        # Do not consider the samples with 50-50 split as the class was
        # randomly chosen
        keep_indices = torch.nonzero(soft_targets[:, 0] != soft_targets[:, 1])
        hard_labels = hard_labels[keep_indices]
        targets = targets[keep_indices]

        self.log_dict({
            'soft_score': F.cross_entropy(soft_labels, soft_targets),
            'hard_score': hard_metric(
                # soft_labels.argmax(dim=1).reshape(targets.shape),
                hard_labels.sigmoid().round(decimals=0),
                targets
            )
        })      

    def predict_step(self, batch, batch_idx):
        ids, tokens = batch

        with torch.no_grad():
            hard_labels, soft_labels = self(**tokens)

        return (
            ids,
            hard_labels.sigmoid().round(decimals=0),
            soft_labels.softmax(dim=1)
        )

    def configure_optimizers(self):
        param_dicts = [
              {"params": self.soft_label_head.parameters()},
              {"params": self.hard_label_head.parameters()},
              {
                  "params": self.backbone.parameters(),
                  "lr": self.backbone_lr,
              },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.task_head_lr,
            weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=4),
                "monitor": "val_soft_loss"
            },
        }


class AgreementModel(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        embedding_dim: int = 128,
        semantic_filters: int = 4,
        soft_label_imp: float = 0.8,
        backbone_lr: float = 2e-5,
        task_head_lr: float = 2e-4,
        sem_infuser_lr: float = 2e-3,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            embedding_dim (int): Dimensionality of the semantic representation.
            Defaults to 128.
            semantic_filters (int, optional): Number of semantic features to
            extract. Defaults to 4.
            soft_label_imp (float): How much importance (range 0-1) to assign
            to the soft_label loss, 0 would mean no dissagreement loss and 1 would
            mean the classification loss is ignored. Defaults to 0.8.
            backbone_lr (float, optional): The LLM's learning rate.
            Defaults to 2e-5.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
            sem_infuser_lr (float, optional): Sem infuser's learning rate.
            Defaults to 2e-3.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.backbone_lr = backbone_lr
        self.task_head_lr = task_head_lr
        self.weight_decay = weight_decay

        # Base model        
        self.backbone = AutoModel.from_pretrained(mpath)
        # Pooling layer to get the sentence embedding
        self.pooling = Pooling(
            self.backbone.config.hidden_size,
            pooling_mode='mean'
        )
        # Freeze the backbone model
        if not backbone_lr:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Soft label predictor
        self.task_head = torch.nn.Linear(
            self.backbone.config.hidden_size,
            2
        )

        self.soft_label_loss = torch.nn.KLDivLoss(
            reduction="batchmean"
        )

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, **tokens):
        # Push all inputs to the device in use
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Get the topic interaction vector ([CLS] vector)
        token_embeddings = self.backbone(**tokens)[0]

        # Vanilla embedding from the backbone
        sentence_embedding = self.pooling({
            'token_embeddings': token_embeddings,
            'attention_mask': tokens['attention_mask']
        })['sentence_embedding']

        # L2 normalize the embeddings
        sentence_embedding = F.normalize(
            sentence_embedding,
            dim=1
        )

        return self.task_head(sentence_embedding)

    def common_step(self, batch, batch_idx):
        _, tokens, (_, soft_targets) = batch
        soft_labels = self(**tokens)

        __soft = self.soft_label_loss(
            soft_labels.log_softmax(dim=1),
            soft_targets
        )

        return {
            'loss': __soft,
            'soft_loss': __soft
        }

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("val_" + k, v.item(), prog_bar=True)

        return loss_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        hard_metric = BinaryF1Score().to(self.device)
        _, tokens, (targets, soft_targets) = batch
        soft_labels = self(**tokens)
        hard_labels = soft_labels.argmax(dim=1).reshape_as(targets)

        # Do not consider the samples with 50-50 split as the class was
        # randomly chosen
        keep_indices = torch.nonzero(soft_targets[:, 0] != soft_targets[:, 1])
        hard_labels = hard_labels[keep_indices]
        targets = targets[keep_indices]

        self.log_dict({
            'soft_score': F.cross_entropy(soft_labels, soft_targets),
            'hard_score': hard_metric(
                hard_labels,
                targets
            )
        })      

    def predict_step(self, batch, batch_idx):
        ids, tokens = batch

        with torch.no_grad():
            soft_labels = self(**tokens)
            soft_labels = soft_labels.softmax(dim=1)

        return (
            ids,
            soft_labels.argmax(dim=1).reshape(soft_labels.shape[0], 1),
            soft_labels
        )

    def configure_optimizers(self):
        param_dicts = [
              {"params": self.task_head.parameters()},
              {
                  "params": self.backbone.parameters(),
                  "lr": self.backbone_lr,
              },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.task_head_lr,
            weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=4),
                "monitor": "val_soft_loss"
            },
        }


class AgreementModel_mse(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        embedding_dim: int = 128,
        semantic_filters: int = 4,
        soft_label_imp: float = 0.8,
        backbone_lr: float = 2e-5,
        task_head_lr: float = 2e-4,
        sem_infuser_lr: float = 2e-3,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            embedding_dim (int): Dimensionality of the semantic representation.
            Defaults to 128.
            semantic_filters (int, optional): Number of semantic features to
            extract. Defaults to 4.
            soft_label_imp (float): How much importance (range 0-1) to assign
            to the soft_label loss, 0 would mean no dissagreement loss and 1 would
            mean the classification loss is ignored. Defaults to 0.8.
            backbone_lr (float, optional): The LLM's learning rate.
            Defaults to 2e-5.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
            sem_infuser_lr (float, optional): Sem infuser's learning rate.
            Defaults to 2e-3.
            weight_decay (float, optional): Common weight decay co-efficient.
            Defaults to 1e-4.
        """
        super().__init__()
        self.backbone_lr = backbone_lr
        self.task_head_lr = task_head_lr
        self.weight_decay = weight_decay

        # Base model        
        self.backbone = AutoModel.from_pretrained(mpath)
        # Pooling layer to get the sentence embedding
        self.pooling = Pooling(
            self.backbone.config.hidden_size,
            pooling_mode='mean'
        )
        # Freeze the backbone model
        if not backbone_lr:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Hard label predictor
        self.task_head = torch.nn.Linear(
            self.backbone.config.hidden_size,
            1
        )

        self.hard_label_loss = torch.nn.BCEWithLogitsLoss()
        self.soft_label_loss = torch.nn.MSELoss()

        # Save the init arguments
        self.save_hyperparameters()

    def forward(self, return_logits=False, **tokens):
        # Push all inputs to the device in use
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Get the topic interaction vector ([CLS] vector)
        token_embeddings = self.backbone(**tokens)[0]

        # Vanilla embedding from the backbone
        sentence_embedding = self.pooling({
            'token_embeddings': token_embeddings,
            'attention_mask': tokens['attention_mask']
        })['sentence_embedding']

        # L2 normalize the embeddings
        sentence_embedding = F.normalize(
            sentence_embedding,
            dim=1
        )

        # Get label logits
        label_logits = self.task_head(sentence_embedding)

        if return_logits:
            # Get label logits
            return label_logits

        else:
            probs = F.sigmoid(label_logits)
            return torch.cat((1-probs, probs), dim=1)

    def common_step(self, batch, batch_idx):
        _, tokens, (targets, soft_targets) = batch
        label_logits = self(return_logits=True, **tokens)

        # No back prop just keeping track
        __bce_loss =  self.hard_label_loss(label_logits, targets)

        # Target logits (we consider the '1' soft_label probs)
        target_logits = torch.logit(
            soft_targets[:, 1].reshape_as(label_logits),
            eps=1e-6
        )

        __soft = torch.sqrt(self.soft_label_loss(
            label_logits,
            target_logits
        ))

        return {
            'loss': __soft,
            'soft_loss': __soft,
            'hard_loss': __bce_loss
        }

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=True)

        return loss_dict

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        for k,v in loss_dict.items():
            self.log("val_" + k, v.item(), prog_bar=True)

        return loss_dict

    def test_step(self, batch, batch_idx, *args, **kwargs):
        hard_metric = BinaryF1Score().to(self.device)
        _, tokens, (targets, soft_targets) = batch
        soft_labels = self(**tokens)
        
        # Compute hard labels
        hard_labels = soft_labels.argmax(dim=1).reshape_as(targets)

        # Do not consider the samples with 50-50 split as the class was
        # randomly chosen
        keep_indices = torch.nonzero(soft_targets[:, 0] != soft_targets[:, 1])
        hard_labels = hard_labels[keep_indices]
        targets = targets[keep_indices]

        self.log_dict({
            'soft_score': F.cross_entropy(soft_labels, soft_targets),
            'hard_score': hard_metric(
                hard_labels,
                targets
            )
        })      

    def predict_step(self, batch, batch_idx):
        ids, tokens = batch

        with torch.no_grad():
            soft_labels = self(**tokens)

        return (
            ids,
            soft_labels.argmax(dim=1).reshape(
                (soft_labels.shape[0], 1)
            ),
            soft_labels
        )

    def configure_optimizers(self):
        param_dicts = [
              {"params": self.task_head.parameters()},
              {
                  "params": self.backbone.parameters(),
                  "lr": self.backbone_lr,
              },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.task_head_lr,
            weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=4),
                "monitor": "val_soft_loss"
            },
        }


class ActiavteAdversarialTraining(pl.Callback):
    def __init__(
        self
    ):
        pass
