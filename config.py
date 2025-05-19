# config.py
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65 
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    # Model type: 'penalized' or 'standard'
    model_type: str = 'penalized' # Default to your custom model

    # Training specific
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    batch_size: int = 64 # B
    max_iters: int = 5000
    lr_decay_iters: int = 5000 # Should be ~= max_iters
    warmup_iters: int = 100
    min_lr: float = 3e-5 # learning_rate / 10 usually
    eval_interval: int = 250
    eval_iters: int = 20
    log_interval: int = 10
    device: str = 'cuda'
    compile_model: bool = False
    # Tensorboard logging
    out_dir: str = 'out' # Directory for checkpoints and logs
    run_name: str = 'gpt_experiment' # Name for this run, used in log dir