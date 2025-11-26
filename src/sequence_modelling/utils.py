import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import Trainer
from torchinfo import summary
from src.preprocessing.train_test_split import train_test_split_tracks
from src.sequence_modelling.configs import get_training_args, get_bert_config
from src.sequence_modelling.models import AISBERT, DataCollator
from src.utils.datasets import AISDataset

def get_embeddings(data: AISDataset, model: AISBERT, l2_normalize: bool = False):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode
    
    data_collator = DataCollator(is_inference=True)
    dataloader = DataLoader(data, batch_size=32, shuffle=False, collate_fn=data_collator)

    all_cls_vectors = []
    all_mmsi = [] # To keep track of which vector belongs to which MMSI
    with torch.no_grad(): # No gradient computation
        for batch in dataloader: # Batch is size 32
            input_features = batch["input_features"].to(device) # Shape (batch_size, seq_len, feature_dim)
            attention_mask = batch["attention_mask"].to(device)
            mmsis = batch["mmsi"] # List of MMSIs in the batch
            
            # This is the output from the model
            outputs = model(
                input_features=input_features,
                attention_mask=attention_mask,
                labels=None
            )
            
            # To get the [CLS] token representation (classification of the whole sequence)
            # we extract the last hidden state and take the first token's embedding
            # Shape (batch_size, hidden_dim) = (batch_size, 256)
            last_hidden_state = outputs.hidden_states[-1]
            cls_token_vector = last_hidden_state[:, 0, :]
            
            all_cls_vectors.append(cls_token_vector)
            all_mmsi.extend(mmsis)

    all_cls_vectors = torch.cat(all_cls_vectors, dim=0).cpu().numpy() # Unpack batches, shape (num_samples, hidden_dim)
    if l2_normalize:
        all_cls_vectors = l2_normalize_embeddings(all_cls_vectors)
    return all_cls_vectors, all_mmsi

def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalizes the embeddings along each row (sample)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings