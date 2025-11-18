import torch
import torch.nn as nn
import numpy as np
import os
import warnings
from argparse import ArgumentParser
from transformers import Trainer

from src.sequence_modelling.configs import get_training_args, get_bert_config
from src.sequence_modelling.models import AISBERT, AISDatasetBERT, DataCollator
from src.utils.logging import CustomLogger, CustomLoggingCallback

warnings.filterwarnings("ignore", message=".*pin_memory.*")

if __name__ == "__main__":
    parser = ArgumentParser("Train Trajectory BERT Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training and validation data")
    parser.add_argument("--output_dir", type=str, default="./data/models/trajectory_bert_model", help="Directory to save the trained model")
    parser.add_argument("--run_name", type=str, default="tiny-bert-pretrain", help="Run name for logging")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    args = parser.parse_args()
    
    logger = CustomLogger(project_name='Computational-Tools', group='map_reduce_preprocessing', run_name=args.run_name)
    logger.log_config(args.__dict__)
    
    # Set up model, datasets, and trainer
    logger.info("Setting up configurations...")
    config = get_bert_config(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    model = AISBERT(config)
    
    train_dataset = AISDatasetBERT(os.path.join(args.data_dir, 'train'), max_seq_len=config.max_position_embeddings)
    eval_dataset = AISDatasetBERT(os.path.join(args.data_dir, 'val'), max_seq_len=config.max_position_embeddings)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    data_collator = DataCollator(mask_prob=0.15)
    
    logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    training_args = get_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
    )
    
    logger.log_config({f"training/{key}": value for key, value in training_args.to_dict().items()})
    logger.log_config({f"model/{key}": value for key, value in config.to_dict().items()})
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[CustomLoggingCallback(logger)] # Use CustomLogger
    )
    
    # Start training
    logger.info("Starting MTM training...")
    trainer.train()
    
    logger.info("MTM training complete.")