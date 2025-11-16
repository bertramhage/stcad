# This script implements a custom Logger class logging to WandB and the terminal.

import wandb
import os
from tqdm import tqdm
import logging
import sys
import traceback

def running_on_hpc():
    hpc_env_vars = [
        "SLURM_JOB_ID",     # SLURM scheduler
        "LSB_JOBID",        # LSF scheduler
        "PBS_JOBID",        # PBS/Torque
        "SGE_JOB_ID",       # Sun Grid Engine
    ]
    return any(var in os.environ for var in hpc_env_vars)

class Customtqdm(tqdm):
    """
    A wrapper for tqdm
    """
    if running_on_hpc():
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, file=TqdmToNull())
    else:
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

class CustomLogger:
    """
    A custom logger that logs to both WandB and the terminal and supports running on HPC systems.
    
    Attributes:
        run: The WandB run object.
        using_wandb: Boolean indicating if WandB logging is active.
        logger: The terminal logger instance.
        
    Methods:
        info(message): Log an info message.
        warning(message): Log a warning message.
        error(message, exception=None): Log an error message with traceback.
        log_metrics(metrics, step=None): Log a dictionary of metrics.
        log_config(config): Log configuration parameters.
        finish(exit_code=0): Finish the WandB run.
        tqdm(iterable, **tqdm_kwargs): Wrap an iterable with tqdm for progress tracking
        artifact(artifact, name, type): Log an artifact to WandB.
        add_tags(tags): Add tags to the WandB run.
    """
    def __init__(self, project_name, group=None, run_name=None, use_wandb=True):
        
        try:
            assert use_wandb # Will raise if False
            self.run = wandb.init(project=project_name, group=group, name=run_name)
            self.using_wandb = True
        except:
            print("WandB init failed, proceeding without WandB logging.")
            self.using_wandb = False
            self.run = None
        
        # Set up terminal logger
        self.logger = logging.getLogger("custom_logger")
        self.logger.setLevel(logging.DEBUG)
        if self.logger.hasHandlers(): # Clear existing handlers
            self.logger.handlers.clear()
        self.logger.propagate = False
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Force unbuffered output
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(line_buffering=True)
        
    def info(self, message: str):
        """Log an info message to the terminal and WandB."""
        self.logger.info(message)
        sys.stdout.flush()  # Force flush
        
    def warning(self, message: str):
        """Log a warning message to the terminal and WandB."""
        self.logger.warning(message)
        sys.stdout.flush()  # Force flush
        
    def error(self, message: str, exception: Exception = None):
        """Log an error message with traceback to the terminal and WandB."""
        if exception:
            tb_str = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            self.logger.error(f"{message}\n{tb_str}")
            self.run.alert(title="Error", text=f"{message}\n\n{tb_str}") if self.using_wandb else None
            sys.stdout.flush()  # Force flush
            raise exception
        else:
            tb_str = traceback.format_exc()
            self.logger.error(f"{message}\n{tb_str}")
            self.run.alert(title="Error", text=f"{message}\n\n{tb_str}") if self.using_wandb else None
            sys.stdout.flush()  # Force flush
            raise Exception(message)
        
    def log_metrics(self, metrics: dict, step: int = None):
        """Log a dictionary of metrics to WandB."""
        if not self.using_wandb:
            for key, value in metrics.items():
                self.logger.info(f"Metric - {key}: {value}")
            return # Silently return
        
        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)
        
    def log_config(self, config: dict):
        """Log configuration parameters to WandB."""
        if not self.using_wandb:
            for key, value in config.items():
                self.logger.info(f"Config - {key}: {value}")
            return # Silently return
        
        self.run.config.update(config)
        
    def finish(self, exit_code: int = 0):
        """Finish the WandB run."""
        if not self.using_wandb:
            return # Silently return
        
        self.run.finish(exit_code=exit_code)
        
    def tqdm(self, iterable, **tqdm_kwargs):
        """Wrap an iterable with tqdm for progress tracking."""
        return Customtqdm(iterable, **tqdm_kwargs)
    
    def artifact(self, artifact, name: str, type: str):
        """Log an artifact to WandB."""
        if not self.using_wandb:
            return # Silently return
        
        wandb_artifact = wandb.Artifact(name, type=type)
        wandb_artifact.add_file(artifact)
        self.run.log_artifact(wandb_artifact)
        
    def add_tags(self, tags: list):
        """Add tags to the WandB run."""
        if not self.using_wandb:
            return # Silently return
        
        self.run.tags = self.run.tags + tuple(tags)
        
    def log_summary(self, summary: dict):
        """Log a summary dictionary to WandB."""
        if not self.using_wandb:
            for key, value in summary.items():
                self.logger.info(f"Summary - {key}: {value}")
            
            return # Silently return
        
        for key, value in summary.items():
            self.run.summary[key] = value
        
class TqdmToNull:
    """A file-like object that silently discards any writes."""
    def write(self, x):
        pass # Ignore the output
    def flush(self):
        pass