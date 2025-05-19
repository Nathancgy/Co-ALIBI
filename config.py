# config.py
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256 # Context length
    vocab_size: int = 65  # Character-level: Andrej Karpathy's char-rnn vocab size for tinyshakespeare
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384 # n_head * head_dim = 6 * 64 = 384
    dropout: float = 0.1
    bias: bool = False

    # Model type: 'penalized' or 'standard'
    model_type: str = 'penalized'

    # Training specific
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    batch_size: int = 64 # B
    max_iters: int = 5000
    lr_decay_iters: int = 5000
    warmup_iters: int = 100
    min_lr: float = 3e-5
    eval_interval: int = 250
    eval_iters: int = 20
    log_interval: int = 10 # Corrected from log_interv  al
    device: str = 'cuda'
    compile_model: bool = False
    # Tensorboard logging
    out_dir: str = 'out'
    run_name: str = 'gpt_experiment'