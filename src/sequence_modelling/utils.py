import torch
import numpy as np
from torch.utils.data import DataLoader
from src.sequence_modelling.models import AISBERT, DataCollator
from src.utils.datasets import AISDataset
from tqdm import tqdm

def get_embeddings(data: AISDataset, model: AISBERT, l2_normalize: bool = False, batch_size: int = 32, verbose: int = 0) -> tuple[np.ndarray, list]:
    
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

    num_samples = len(data)
    hidden_dim = model.config.hidden_size
    all_cls_vectors = np.zeros((num_samples, hidden_dim), dtype=np.float32) # Preallocate array
    
    all_mmsi = [] # To keep track of which vector belongs to which MMSI
    all_start_times = []
    start_idx = 0
    if verbose > 0:
        dataloader = tqdm(dataloader, total=len(dataloader), desc="Computing Embeddings")
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mmsis = batch["mmsi"]
            start_times = batch["time_start"]
            
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
            
            batch_size = cls_token_vector.size(0)
            end_idx = start_idx + batch_size
            all_cls_vectors[start_idx:end_idx] = cls_token_vector.cpu().numpy()
            
            all_mmsi.extend(mmsis)
            all_start_times.extend(start_times)
            start_idx = end_idx

    if l2_normalize:
        all_cls_vectors = l2_normalize_embeddings(all_cls_vectors)
    return all_cls_vectors, all_mmsi, all_start_times

def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalizes the embeddings along each row (sample)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings