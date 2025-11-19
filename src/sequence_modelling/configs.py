# Contains default configurations

from transformers import BertConfig, TrainingArguments

def get_training_args(**kargs):
    training_args = {
        "output_dir": "./models/saved_models/trajectory_bert_model",
        "overwrite_output_dir": True,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 512,
        "per_device_eval_batch_size": 512,
        "gradient_accumulation_steps": 4,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 10,
        "fp16": True,
        "dataloader_num_workers": 4,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none"
    }
    training_args.update(kargs)
    return TrainingArguments(**training_args)

def get_bert_config(**kargs):
    hidden_dim = 256
    config = {
        "vocab_size": 1, # Dummy value, not used
        "hidden_size": hidden_dim,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": hidden_dim * 4,
        "max_position_embeddings": 512, # Max trajectory length is 288 (+2 for CLS/SEP), round up to fit power of 2
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "type_vocab_size": 1,
    }
    config.update(kargs)
    return BertConfig(**config)