""" Custom classes for AIS trajectory modelling with BERT. Adapted from Huggingface transformers library. """

import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from transformers import BertPreTrainedModel, BertModel

from src.utils.datasets import AISDataset

class AISEmbeddings(nn.Module):
    """ Embedding layer for AIS trajectory data."""
    def __init__(self, config):
        super().__init__()
        # Project 4-dim feature vector to hidden_dim
        self.feature_projector = nn.Linear(4, config.hidden_size)
        
        # Standard BERT position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features has shape (batch_size, seq_len, 4)
        
        inputs_embeds = self.feature_projector(input_features)
        
        # Add position embeddings
        seq_length = inputs_embeds.size(1)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class AISBERT(BertPreTrainedModel):
    """ A BERT-style model Masked Trajectory Modeling (MTM) for AIS data."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Embedding layer
        self.embeddings = AISEmbeddings(config)
        
        # Standard BERT Encoder
        self.encoder = BertModel(config, add_pooling_layer=False).encoder
        
        # Create the [CLS] token
        self.pooler = BertModel(config).pooler
        
        # MTM Prediction Head
        self.prediction_head = nn.Linear(config.hidden_size, 4)
        
        self.loss_fct = nn.MSELoss()
        
        self.init_weights()

    def forward(self, input_features, attention_mask = None, labels = None) -> MaskedLMOutput:
        
        # Get embeddings
        embedding_output = self.embeddings(input_features)
        
        # Pass through BERT encoder
        if attention_mask is None:
            attention_mask = torch.ones(input_features.shape[:2], device=input_features.device)
            
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, 
            input_features.shape[:2]
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=True,
        )
        
        sequence_output = encoder_outputs.last_hidden_state
        predictions = self.prediction_head(sequence_output)
        
        # Calculate Loss
        loss = None
        if labels is not None:
            # Only calculate loss on the masked tokens (label != -100)
            active_loss = (labels != -100).all(dim=-1)
            
            active_loss_expanded = active_loss.unsqueeze(-1).expand_as(predictions)
            active_predictions = predictions[active_loss_expanded].view(-1, 4)
            active_labels = labels[active_loss_expanded].view(-1, 4)
            
            loss = self.loss_fct(active_predictions, active_labels)

        return MaskedLMOutput(
            loss=loss,
            logits=predictions,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
class AISDatasetBERT(AISDataset):
    """ Wrapper for training with BERT """
    def __init__(self, data_dir: str, max_seq_len: int):
        super().__init__(data_dir)
        self.max_seq_len = max_seq_len
        
        # Special feature vectors for CLS and SEP tokens
        cls_vector = [-1.0, -1.0, -1.0, -1.0]
        sep_vector = [-2.0, -2.0, -2.0, -2.0]
        self.cls_tensor = torch.tensor(cls_vector, dtype=torch.float32)
        self.sep_tensor = torch.tensor(sep_vector, dtype=torch.float32)

    def __getitem__(self, idx) -> torch.Tensor:
        """ Returns a trajectory tensor with [CLS] and [SEP] tokens added."""
        traj, _,_,_ = super().__getitem__(idx)
        
        # Truncate if longer than max_seq_len
        if len(traj) > self.max_seq_len - 2:
            traj = traj[:self.max_seq_len - 2]
        
        # Add [CLS] at the beginning and [SEP] at the end
        full_traj = torch.cat(
            [self.cls_tensor.unsqueeze(0), traj, self.sep_tensor.unsqueeze(0)], 
            dim=0
        )
        
        return full_traj

class DataCollator:
    """For handling padding and masking for the MTM task.
    Similar to transformers DataCollatorForLanguageModeling"""
    def __init__(self, mask_prob=0.15):
        self.mask_prob = mask_prob
        self.pad_vector_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        # [PAD] token is [0.0, 0.0, 0.0, 0.0]

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        
        # Pad all trajectories to the same length
        max_len = max(traj.shape[0] for traj in batch)
        padded_features = torch.full(
            (len(batch), max_len, 4), 
            fill_value=0.0,
            dtype=torch.float32
        )
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, traj in enumerate(batch):
            seq_len = traj.shape[0]
            padded_features[i, :seq_len] = traj
            attention_mask[i, :seq_len] = 1 # 1 for real tokens

        labels = torch.full_like(padded_features, fill_value=-100.0) # -100 is ignored by loss
        
        # Create random mask (mask_prob% probability)
        rand = torch.rand(padded_features.shape[:2])
        mask_arr = (rand < self.mask_prob)
        
        mask_arr = mask_arr & (attention_mask == 1)
        mask_arr[:, 0] = False # Don't mask [CLS]
        
        for i in range(len(batch)):
            last_token_idx = attention_mask[i].sum() - 1
            mask_arr[i, last_token_idx] = False # Don't mask [SEP]

        # Apply mask
        for i in range(len(batch)):
            for j in range(max_len):
                if mask_arr[i, j]:
                    labels[i, j] = padded_features[i, j].clone()
                    padded_features[i, j] = 0.0
        
        return {
            "input_features": padded_features,
            "attention_mask": attention_mask,
            "labels": labels
        }