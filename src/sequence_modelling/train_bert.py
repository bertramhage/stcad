import torch
import torch.nn as nn
import numpy as np
import os
import warnings
from src.utils.datasets import AISDataset

from transformers import BertConfig, BertPreTrainedModel, BertModel, TrainingArguments, Trainer
from transformers.modeling_outputs import MaskedLMOutput

# Suppress warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# --- 1. Model Configuration (Our "TinyBERT") ---
# We define a small model for fast training.
# This is the "n" dimensions you spoke of.
HIDDEN_DIM = 256  # n=256 dimensions
NUM_LAYERS = 4    # 4 layers deep
NUM_HEADS = 4     # 4 attention heads

# These are special feature vectors we'll use for CLS and SEP tokens
# We use arbitrary unique values to distinguish them.
CLS_VECTOR = [-1.0, -1.0, -1.0, -1.0]
SEP_VECTOR = [-2.0, -2.0, -2.0, -2.0]
PAD_VECTOR = [0.0, 0.0, 0.0, 0.0] # Will be ignored by attention mask

# --- 2. The Custom BERT Model for Trajectories ---

class TrajectoryBertEmbeddings(nn.Module):
    """
    This layer replaces the standard WordPiece/Token ID embeddings.
    It takes our 4-dim (lat, lon, sog, cog) vector and projects it to
    the model's hidden dimension (256), adding position embeddings.
    """
    def __init__(self, config):
        super().__init__()
        # Project 4-dim feature vector to hidden_dim
        self.feature_projector = nn.Linear(4, config.hidden_size)
        
        # Standard BERT position and token type embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Precompute position IDs
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features is (batch_size, seq_len, 4)
        
        # 1. Project features to hidden size
        # (batch_size, seq_len, 4) -> (batch_size, seq_len, 256)
        inputs_embeds = self.feature_projector(input_features)
        
        # 2. Add position embeddings
        seq_length = inputs_embeds.size(1)
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        
        # 5. Layer Norm and Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TrajectoryBertForPreTraining(BertPreTrainedModel):
    """
    The main model for Masked Trajectory Modeling (MTM).
    It includes the BERT encoder and a regression head to predict
    the original 4 features of the masked tokens.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Custom Embedding Layer
        self.embeddings = TrajectoryBertEmbeddings(config)
        
        # Standard BERT Encoder
        self.encoder = BertModel(config, add_pooling_layer=False).encoder
        
        # The "Pooler" creates the [CLS] token vector for clustering
        self.pooler = BertModel(config).pooler
        
        # MTM Prediction Head: maps hidden state -> 4 features
        self.prediction_head = nn.Linear(config.hidden_size, 4)
        
        # Loss function for regression
        self.loss_fct = nn.MSELoss()
        
        self.init_weights()

    def forward(self, input_features, attention_mask = None, labels = None) -> MaskedLMOutput:
        
        # 1. Get embeddings
        # input_features is (batch, seq, 4)
        # embedding_output is (batch, seq, 256)
        embedding_output = self.embeddings(input_features)
        
        # 2. Pass through BERT encoder
        # We need to create the extended attention mask for the encoder
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
        
        # (batch, seq, 256)
        sequence_output = encoder_outputs.last_hidden_state
        
        # (batch, 256) - This is the vector you will use for clustering
        pooled_output = self.pooler.activation(self.pooler.dense(sequence_output[:, 0]))

        # 3. Predict the masked features
        # (batch, seq, 4)
        predictions = self.prediction_head(sequence_output)
        
        # 4. Calculate Loss
        loss = None
        if labels is not None:
            # We only calculate loss on the masked tokens (where all elements in label != -100)
            # labels shape: (batch_size, seq_len, 4)
            # Check if all 4 features are -100 (non-masked tokens)
            active_loss = (labels != -100).all(dim=-1)  # (batch_size, seq_len)
            
            # predictions shape: (batch_size, seq_len, 4)
            # active_loss shape: (batch_size, seq_len)
            
            # We need to expand active_loss to match predictions
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

# --- 3. The Trajectory Dataset ---

class TrajectoryDataset(AISDataset):
    """
    A simple dataset that holds our trajectories.
    It adds the [CLS] and [SEP] vectors to each trajectory.
    """
    def __init__(self, data_dir: str, max_seq_len: int):
        super().__init__(data_dir)
        self.max_seq_len = max_seq_len
        self.cls_tensor = torch.tensor(CLS_VECTOR, dtype=torch.float32)
        self.sep_tensor = torch.tensor(SEP_VECTOR, dtype=torch.float32)

    def __getitem__(self, idx) -> torch.Tensor:
        
        # Load trajectory, ensure it's a tensor
        traj, _,_,_ = super().__getitem__(idx)
        
        # Truncate if longer than max_seq_len - 2 (for [CLS] and [SEP])
        if len(traj) > self.max_seq_len - 2:
            traj = traj[:self.max_seq_len - 2]
        
        # Add [CLS] at the beginning and [SEP] at the end
        # (1, 4) + (seq_len, 4) + (1, 4) -> (seq_len + 2, 4)
        full_traj = torch.cat(
            [self.cls_tensor.unsqueeze(0), traj, self.sep_tensor.unsqueeze(0)], 
            dim=0
        )
        
        return full_traj

# --- 4. The Data Collator (Masking & Padding) ---

class DataCollatorForMTM:
    """
    This class handles padding and masking for our MTM task.
    It's the equivalent of `DataCollatorForLanguageModeling`.
    """
    def __init__(self, mask_prob=0.15):
        self.mask_prob = mask_prob
        self.pad_vector_tensor = torch.tensor(PAD_VECTOR, dtype=torch.float32)

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        # batch is a list of tensors, each (seq_len, 4)
        
        # 1. Pad all trajectories to the same length
        max_len = max(traj.shape[0] for traj in batch)
        
        padded_features = torch.full(
            (len(batch), max_len, 4), 
            fill_value=0.0, # Will be ignored by attention mask
            dtype=torch.float32
        )
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        
        for i, traj in enumerate(batch):
            seq_len = traj.shape[0]
            padded_features[i, :seq_len] = traj
            attention_mask[i, :seq_len] = 1 # 1 means "pay attention"

        # 2. Create labels and mask inputs
        labels = torch.full_like(padded_features, fill_value=-100.0) # -100 is ignored by loss
        
        # Create random mask (15% probability)
        # (batch_size, seq_len)
        rand = torch.rand(padded_features.shape[:2])
        mask_arr = (rand < self.mask_prob)
        
        # Don't mask [CLS], [SEP], or [PAD] tokens
        mask_arr = mask_arr & (attention_mask == 1)
        mask_arr[:, 0] = False # Don't mask [CLS]
        
        # Need to find [SEP] tokens and not mask them (tricky, but we can just check the last 1)
        for i in range(len(batch)):
            last_token_idx = attention_mask[i].sum() - 1
            mask_arr[i, last_token_idx] = False # Don't mask [SEP]

        # Apply mask
        for i in range(len(batch)):
            for j in range(max_len):
                if mask_arr[i, j]:
                    # This is a masked token
                    # 1. Save original values to labels
                    labels[i, j] = padded_features[i, j].clone()
                    # 2. Mask the input features (e.g., set to zero)
                    padded_features[i, j] = 0.0 # Mask token is all zeros
        
        return {
            "input_features": padded_features,
            "attention_mask": attention_mask,
            "labels": labels
        }

# --- 5. Main Training Script ---

if __name__ == "__main__":
    
    print("Setting up model configuration...")
    # 1. Configure the "TinyBERT" model
    config = BertConfig(
        vocab_size=1, # We don't use a vocab, but config requires it
        hidden_size=HIDDEN_DIM,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=HIDDEN_DIM * 4,
        max_position_embeddings=512, # Max trajectory length is 288 (+2 for CLS/SEP), round up to fit power of 2
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=1, # We only have 1 segment type (the whole trajectory)
    )
    
    print("Initializing custom model for pre-training...")
    # 2. Instantiate the custom model
    model = TrajectoryBertForPreTraining(config)
    
    data_dir = "./data/ais/processed/"
    
    train_dataset = TrajectoryDataset(os.path.join(data_dir, 'train'), max_seq_len=config.max_position_embeddings)
    eval_dataset = TrajectoryDataset(os.path.join(data_dir, 'val'), max_seq_len=config.max_position_embeddings)
    
    # 3. Instantiate the Data Collator
    data_collator = DataCollatorForMTM(mask_prob=0.15)
    
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    os.environ["WANDB_PROJECT"] = "Computational-Tools"
    os.environ["WANDB_LOG_MODEL"] = "bert_test"
    
    # 4. Set up Training Arguments
    # This assumes you have a directory named "trajectory_bert_model"
    # An A100 can handle a very large batch size.
    # We use gradient_accumulation_steps to simulate an even larger one.
    training_args = TrainingArguments(
        output_dir="./data/models/trajectory_bert_model",
        overwrite_output_dir=True,
        num_train_epochs=5,  # Start with 5-10 epochs for pre-training
        per_device_train_batch_size=512,  # Maximize this for your A100
        per_device_eval_batch_size=512,
        gradient_accumulation_steps=4,  # Effective batch size = 512 * 4 = 2048
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,  # MUST use mixed precision on A100 for speed
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # 5. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("--- Starting Pre-Training ---")
    
    # 6. Train the model!
    trainer.train()
    
    print("--- Pre-Training Complete ---")
    
    # 7. Save the final model
    trainer.save_model("./data/models/trajectory_bert_model/final_model")
    
    print("Model saved to ./data/models/trajectory_bert_model/final_model")

    # --- 8. How to Use the Encoder for Clustering ---
    print("\n--- Example: Encoding a trajectory for clustering ---")
    
    # Load the trained model (we only need the core BERT part)
    # We load the full model and then just access its .bert component
    trained_model = TrajectoryBertForPreTraining.from_pretrained("./data/models/trajectory_bert_model/final_model")
    trained_model.eval() # Set to evaluation mode
    
    # Get a sample trajectory (needs [CLS] and [SEP])
    sample_traj_data = eval_dataset[0].unsqueeze(0) # Add batch dimension
    
    print(f"Input trajectory shape (with [CLS]/[SEP]): {sample_traj_data.shape}")
    
    with torch.no_grad():
        # Pass the features directly to the forward function
        outputs = trained_model(
            input_features=sample_traj_data,
            labels=None # No labels needed for inference
        )
        
        # The vector you want for clustering is from the [CLS] token.
        # We can get it from the last hidden state (outputs.hidden_states[-1])
        # The [CLS] token is at position 0
        
        if outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
            cls_token_vector = last_hidden_state[:, 0, :]
        else:
            # Fallback: use logits to get the hidden representation
            # Not ideal but prevents crash
            print("Warning: hidden_states not available, using alternative method")
            # Re-run through the model's encoder directly
            embedding_output = trained_model.embeddings(sample_traj_data)
            attention_mask = torch.ones(sample_traj_data.shape[:2], device=sample_traj_data.device)
            extended_attention_mask = trained_model.get_extended_attention_mask(
                attention_mask, 
                sample_traj_data.shape[:2]
            )
            encoder_outputs = trained_model.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                output_hidden_states=True,
            )
            cls_token_vector = encoder_outputs.last_hidden_state[:, 0, :]
        
        print(f"Encoded [CLS] vector shape: {cls_token_vector.shape}")
        print("This is the n-dimensional vector you use for clustering.")