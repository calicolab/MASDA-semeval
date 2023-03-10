import torch
import pytorch_lightning as pl

from transformers import AutoModel
from torch.nn import functional as F
from sentence_transformers.models import Pooling
from torchmetrics.classification import BinaryF1Score

class InteractionModel_dep(torch.nn.Module):
    """Model With  interannotaor dependence modelled."""
    def __init__(
        self,
        text_dim: int,
        ann_dim: int = 128,
        unique_ann_cnt: int = 8,
        modalities: int = 4,
        decoder_depth: int = 1,
        mapper_depth: int = 2,
        dropout: float = 0.2,
        pad_token_id: int = 0
    ):
        """
        Args:
            ann_dim (int, optional): _description_. Defaults to 128.
            unique_ann_cnt (int, optional): _description_. Defaults to 8.
            modalities (int, optional): Number of attention heads to use while
            computing the Text-Annotator interaction.
            As a rule of thumb set as equal to half the number of catgories
            each annotator is annotating for, refer `other_annotations`the 
            in Conv Abuse dataset. Defaults to 4.
            decoder_depth (int, optional): How many Decoder layers to use.
            Defaults to 1.
            mapper_depth (int, optional): How many Linear layers to use while
            generating the low dimensional representation of the text embeddings.
            Defaults to 2.
            pad_token_id (int, optional): Token Id assigned to the padding token.
            Defaults to 0.
        """
        super().__init__()

        # Low dimension linear transformation for the high dim conv embeddings
        self.linmap_block = self._get_linmap_block(
            text_dim,
            ann_dim,
            mapper_depth,
            dropout
        )

        # Annotator embedding
        self.ann_embedder = torch.nn.Embedding(
            unique_ann_cnt+1, #Plus one to account for the padding
            ann_dim,
            padding_idx=pad_token_id
        )

        # Text-Annotator (How does each annotator `attend` to
        # the Text)
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=ann_dim,
            nhead=modalities,
            batch_first=True
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_depth
        )      

        # Pooling to get the group annotator embedding
        self.pooling = Pooling(ann_dim, pooling_mode='mean')

        # Useful  properties
        self.ann_dim = ann_dim
        self.num_heads = modalities
        self.pad_token_id = pad_token_id

    def forward(self, text_embeddings, ann_tokens):
        # Compute low dim representation of the text embeddings
        for mod in self.linmap_block:
            text_embeddings = mod(text_embeddings)
        
        ann_embedding = self.ann_embedder(ann_tokens)

        # Add a singleton dimension at axis=1 to make the text_embeddings
        # compatible in shape to the ann_embeddings
        text_embeddings = text_embeddings.unsqueeze(dim=1)

        # Get the annotator attention masks
        key_padding_mask = (ann_tokens!=self.pad_token_id).type(torch.float32)

        # Inverting them as torch ignores the `True` value positions
        key_padding_mask_inv = (key_padding_mask==0).type(torch.bool)

        ann_embedding = self.decoder(
            ann_embedding,
            text_embeddings,
            # `tgt_mask` not required, refer discussion here:
            # https://stackoverflow.com/a/62633542/10944913
            tgt_key_padding_mask=key_padding_mask_inv
        ) # Shape N x num_ann x ann_dim

        # Get a annotator group embedding
        pooled_ann_embedding = self.pooling({
            'token_embeddings': ann_embedding,
            # Expects a huggingface style mask, torch is opp to that
            'attention_mask': (key_padding_mask_inv==0).type(torch.long)
        })['sentence_embedding']

        # L2 normalize the embeddings
        pooled_ann_embedding = F.normalize(
            pooled_ann_embedding,
            dim=1
        )

        return (
            pooled_ann_embedding,
            text_embeddings.squeeze(dim=1),
            ann_embedding # Un-pooled annotator embedding
        )

    @classmethod
    def _get_linmap_block(cls, inp_dim, opt_dim, num_layers, dropout=0.1):
        """Constructs an evenly spaced (intermediate dims are evenly spaced)
        down-sampling network.
        """
        if not inp_dim > opt_dim:
            raise ValueError(f"Can't down-sample {inp_dim}-dimensional "
                f"text vectors to {opt_dim}-dimensions.")

        step = (inp_dim - opt_dim) / num_layers

        # Intermediate dims
        int_dims = [int(inp_dim - step * i) for i in range(1, num_layers+1)]
        linmap_block = torch.nn.ModuleList()
        for idim in int_dims:
            linmap_block.extend([
                torch.nn.Linear(inp_dim, idim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            ])
            inp_dim = idim
        
        return linmap_block

    @classmethod
    def _get_attn_mask(cls, masks, num_heads=2):
        attn_masks = []
        for mask in masks:
            uns_mask = mask.unsqueeze(0)
            # Unsqueeze the result again so that the masks of individual
            # sequences remain seperated
            attn_masks.append((uns_mask.T @ uns_mask).unsqueeze(0))
        attn_masks = torch.vstack(attn_masks)

        # Check this stackoverflow question for why we do this:
        # https://stackoverflow.com/q/68205894/10944913
        return attn_masks.repeat_interleave(num_heads, dim=0)


class InteractionModel(torch.nn.Module):
    def __init__(
        self,
        text_dim: int,
        ann_dim: int = 128,
        unique_ann_cnt: int = 8,
        modalities: int = 4,
        mapper_depth: int = 2,
        dropout: float = 0.2,
        pad_token_id: int = 0,
        sentence_mode: bool = True
    ):
        """
        Args:
            ann_dim (int, optional): _description_. Defaults to 128.
            unique_ann_cnt (int, optional): _description_. Defaults to 8.
            modalities (int, optional): Number of attention heads to use while
            computing the Text-Annotator interaction.
            As a rule of thumb set as equal to half the number of catgories
            each annotator is annotating for, refer `other_annotations`the 
            in Conv Abuse dataset. Defaults to 4.
            mapper_depth (int, optional): How many Linear layers to use while
            generating the low dimensional representation of the text embeddings.
            Defaults to 2.
            pad_token_id (int, optional): Token Id assigned to the padding token.
            Defaults to 0.
            sentence_mode (bool, optional): Indicates whether the conversation embedding
            is a sentence embedding (N x hidden_dims) or token embedding
            (N x seq_len x hidden_dims). Defaults to True.
        """
        super().__init__()

        self.sentence_mode = sentence_mode

        # Low dimension linear transformation for the high dim conv embeddings
        self.linmap_block = self._get_linmap_block(
            text_dim,
            ann_dim,
            mapper_depth,
            dropout
        )

        # Annotator embedding
        self.ann_embedder = torch.nn.Embedding(
            unique_ann_cnt+1, #Plus one to account for the padding
            ann_dim,
            padding_idx=pad_token_id
        )

        # Text-Annotator (How does each annotator `attend` to
        # the Text)
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=ann_dim,
            num_heads=modalities,
            batch_first=True
        )

        # Layer Norm
        self.norm = torch.nn.LayerNorm(normalized_shape=ann_dim)

        # Pooling to get the group annotator embedding
        self.pooling = Pooling(ann_dim, pooling_mode='mean')

        # Useful  properties
        self.ann_dim = ann_dim
        self.num_heads = modalities
        self.pad_token_id = pad_token_id

    def forward(self, text_embeddings, ann_tokens):
        # Compute low dim representation of the text embeddings
        for mod in self.linmap_block:
            text_embeddings = mod(text_embeddings)
        
        ann_embedding = self.ann_embedder(ann_tokens)

        # Add a singleton dimension at axis=1 to make the text_embeddings
        # compatible in shape to the ann_embeddings
        if self.sentence_mode:
            text_embeddings = text_embeddings.unsqueeze(dim=1)

        # Get the annotator attention masks (as required by HF transformers)
        attention_mask = (ann_tokens!=self.pad_token_id).type(torch.long)

        # We dont need a query mask as the embeddinng layer makes
        # the padding vectors all 0s, leading to a cos similarity
        # score(QK) = 0 for them.
        ann_embedding_with_attn = self.cross_attention(
            ann_embedding,
            text_embeddings,
            text_embeddings,
            need_weights=False
        )[0] # Shape N x num_ann x ann_dim

        # Layer Norming
        ann_embedding = self.norm(ann_embedding+ann_embedding_with_attn)

        # Get a annotator group embedding
        pooled_ann_embedding = self.pooling({
            'token_embeddings': ann_embedding,
            # Expects a huggingface style mask, torch is opp to that
            'attention_mask': attention_mask
        })['sentence_embedding']

        # L2 normalize the embeddings
        pooled_ann_embedding = F.normalize(
            pooled_ann_embedding,
            dim=1
        )

        return (
            pooled_ann_embedding,
            text_embeddings.squeeze(dim=1),
            ann_embedding # Un-pooled annotator embedding
        )

    @classmethod
    def _get_linmap_block(cls, inp_dim, opt_dim, num_layers, dropout=0.1):
        """Constructs an evenly spaced (intermediate dims are evenly spaced)
        down-sampling network.
        """
        if not inp_dim > opt_dim:
            raise ValueError(f"Can't down-sample {inp_dim}-dimensional "
                f"text vectors to {opt_dim}-dimensions.")

        step = (inp_dim - opt_dim) / num_layers

        # Intermediate dims
        int_dims = [int(inp_dim - step * i) for i in range(1, num_layers+1)]
        linmap_block = torch.nn.ModuleList()
        for idim in int_dims:
            linmap_block.extend([
                torch.nn.Linear(inp_dim, idim),
                torch.nn.GELU(),
                torch.nn.LayerNorm(idim),
                torch.nn.Dropout(dropout),
            ])
            inp_dim = idim
        
        return linmap_block

    @classmethod
    def _get_attn_mask(cls, masks, num_heads=2):
        attn_masks = []
        for mask in masks:
            uns_mask = mask.unsqueeze(0)
            # Unsqueeze the result again so that the masks of individual
            # sequences remain seperated
            attn_masks.append((uns_mask.T @ uns_mask).unsqueeze(0))
        attn_masks = torch.vstack(attn_masks)

        # Check this stackoverflow question for why we do this:
        # https://stackoverflow.com/q/68205894/10944913
        return attn_masks.repeat_interleave(num_heads, dim=0)



class AgreementModel_vanilla(pl.LightningModule):
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
        _, (text_tokens, _), (_, soft_targets) = batch
        soft_labels = self(**text_tokens)

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
        _, (text_tokens, _), (_, soft_targets) = batch
        soft_labels = self(**text_tokens)
        hard_labels = soft_labels.argmax(dim=1).reshape(soft_labels.shape[0], 1)
        hard_targets = soft_targets.argmax(dim=1).reshape(soft_labels.shape[0], 1)

        # Do not consider the samples with 50-50 split as the class was
        # randomly chosen
        keep_indices = torch.nonzero(soft_targets[:, 0] != soft_targets[:, 1])
        hard_labels = hard_labels[keep_indices]
        hard_targets = hard_targets[keep_indices]

        self.log_dict({
            'soft_score': F.cross_entropy(soft_labels, soft_targets),
            'hard_score': hard_metric(
                hard_labels,
                hard_targets
            )
        })      

    def predict_step(self, batch, batch_idx):
        ids, (text_tokens, _) = batch

        with torch.no_grad():
            soft_labels = self(**text_tokens)
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


class AgreementModel_pooled(pl.LightningModule):
    def __init__(
        self,
        mpath: str,
        interaction_model: InteractionModel,
        backbone_lr: float = 2e-5,
        task_head_lr: float = 2e-4,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            interaction_model (InteractionModel): The model to account for
            annotator differences.
            backbone_lr (float, optional): The LLM's learning rate.
            Defaults to 2e-5.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
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

        # Annotator interaction Model
        self.interaction_model = interaction_model

        # Soft label predictor
        self.task_head = torch.nn.Linear(
            self.interaction_model.ann_dim,
            2
        )

        self.soft_label_loss = torch.nn.KLDivLoss(
            reduction="batchmean"
        )

        # Save the init arguments
        self.save_hyperparameters(ignore=['interaction_model'])

    def forward(self, text_tokens, ann_tokens):
        # Push all inputs to the device in use
        ann_tokens.to(self.device)
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        # Get the topic interaction vector ([CLS] vector)
        token_embeddings = self.backbone(**text_tokens)[0]

        # Vanilla embedding from the backbone
        sent_embedding = self.pooling({
            'token_embeddings': token_embeddings,
            'attention_mask': text_tokens['attention_mask']
        })['sentence_embedding']

        # L2 normalize the embeddings
        sent_embedding = F.normalize(
            sent_embedding,
            dim=1
        )

        pooled_ann_embedding, _, _ = self.interaction_model(
            sent_embedding, ann_tokens
        )

        return self.task_head(pooled_ann_embedding)

    def common_step(self, batch, batch_idx):
        _, (text_tokens, ann_tokens), (_, soft_targets) = batch
        soft_labels = self(text_tokens, ann_tokens)

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
        _, (text_tokens, ann_tokens), (_, soft_targets) = batch
        soft_labels = self(text_tokens, ann_tokens)
        hard_labels = soft_labels.argmax(dim=1).reshape(soft_labels.shape[0], 1)
        hard_targets = soft_targets.argmax(dim=1).reshape(soft_labels.shape[0], 1)

        # Do not consider the samples with 50-50 split as the class was
        # randomly chosen
        keep_indices = torch.nonzero(soft_targets[:, 0] != soft_targets[:, 1])
        hard_labels = hard_labels[keep_indices]
        hard_targets = hard_targets[keep_indices]

        self.log_dict({
            'soft_score': F.cross_entropy(soft_labels, soft_targets),
            'hard_score': hard_metric(
                hard_labels,
                hard_targets
            )
        })      

    def predict_step(self, batch, batch_idx):
        ids, (text_tokens, ann_tokens) = batch

        with torch.no_grad():
            soft_labels = self(text_tokens, ann_tokens)
            soft_labels = soft_labels.softmax(dim=1)

        return (
            ids,
            soft_labels.argmax(dim=1).reshape(soft_labels.shape[0], 1),
            soft_labels
        )

    def configure_optimizers(self):
        param_dicts = [
              {"params": self.task_head.parameters()},
              {"params": self.interaction_model.parameters()},
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
        interaction_model: InteractionModel,
        backbone_lr: float = 2e-5,
        task_head_lr: float = 2e-4,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            mpath (str): Path to the `transformer` artifact files.
            interaction_model (InteractionModel): The model to account for
            annotator differences.
            backbone_lr (float, optional): The LLM's learning rate.
            Defaults to 2e-5.
            task_head_lr (float, optional): Task head/s' learning rate.
            Defaults to 2e-4.
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

        # Annotator interaction Model
        self.interaction_model = interaction_model

        # Soft label predictor
        self.task_head = torch.nn.Linear(
            self.interaction_model.ann_dim,
            2
        )

        self.soft_label_loss = torch.nn.KLDivLoss(
            reduction="batchmean"
        )

        # Save the init arguments
        self.save_hyperparameters(ignore=['interaction_model'])

    def forward(self, text_tokens, ann_tokens):
        # Push all inputs to the device in use
        ann_tokens.to(self.device)
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

        token_embeddings = self.backbone(**text_tokens)[0]

        pooled_ann_embedding, _, _ = self.interaction_model(
            token_embeddings, ann_tokens
        )

        return self.task_head(pooled_ann_embedding)

    def common_step(self, batch, batch_idx):
        _, (text_tokens, ann_tokens), (_, soft_targets) = batch
        soft_labels = self(text_tokens, ann_tokens)

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
        _, (text_tokens, ann_tokens), (_, soft_targets) = batch
        soft_labels = self(text_tokens, ann_tokens)
        hard_labels = soft_labels.argmax(dim=1).reshape(soft_labels.shape[0], 1)
        hard_targets = soft_targets.argmax(dim=1).reshape(soft_labels.shape[0], 1)

        # Do not consider the samples with 50-50 split as the class was
        # randomly chosen
        keep_indices = torch.nonzero(soft_targets[:, 0] != soft_targets[:, 1])
        hard_labels = hard_labels[keep_indices]
        hard_targets = hard_targets[keep_indices]

        self.log_dict({
            'soft_score': F.cross_entropy(soft_labels, soft_targets),
            'hard_score': hard_metric(
                hard_labels,
                hard_targets
            )
        })      

    def predict_step(self, batch, batch_idx):
        ids, (text_tokens, ann_tokens) = batch

        with torch.no_grad():
            soft_labels = self(text_tokens, ann_tokens)
            soft_labels = soft_labels.softmax(dim=1)

        return (
            ids,
            soft_labels.argmax(dim=1).reshape(soft_labels.shape[0], 1),
            soft_labels
        )

    def configure_optimizers(self):
        param_dicts = [
              {"params": self.task_head.parameters()},
              {"params": self.interaction_model.parameters()},
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

