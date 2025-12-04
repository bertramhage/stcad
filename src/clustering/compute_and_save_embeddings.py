""" Batch script to compute and save embeddings as a .npz file. """

from argparse import ArgumentParser
import os
from src.sequence_modelling.utils import get_embeddings
from src.sequence_modelling.models import AISBERT
from src.utils.datasets import AISDataset
import torch
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the AIS dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the computed embeddings.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for embedding computation.")
    parser.add_argument("--l2_normalize", action="store_true", default=True, help="Whether to L2 normalize the embeddings.")
    args = parser.parse_args()
    
    print("Loading model...")
    model = AISBERT.from_pretrained(args.model_path)
    print(f"Loaded model from {args.model_path}\n")
    
    print("Loading dataset...")
    ds = AISDataset(data_dir=args.dataset_path)
    print(f"Loaded dataset from {args.dataset_path} with {len(ds)} samples.\n")
    
    print("Computing embeddings...")
    embeddings, mmsis, start_times = get_embeddings(ds, model, l2_normalize=args.l2_normalize, batch_size=args.batch_size, verbose=1)
    print("Computed embeddings.\n")
    
    print(f"Saving embeddings to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez_compressed(args.output_path, embeddings=embeddings, mmsis=mmsis, start_times=start_times)
    print("Done.")