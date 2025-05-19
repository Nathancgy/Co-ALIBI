# train.py
import os
import time
import math
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
import argparse # For command-line arguments

from config import GPTConfig
from model import GPT # Assuming model.py contains the GPT class

# --- Data Loading and Preparation (same as before) ---
class CharDataset:
    def __init__(self, data_path, block_size):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.block_size = block_size
        
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        print(f"Data loaded: {len(self.data)} characters, {self.vocab_size} unique.")

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        # Return number of possible starting positions for a full block + 1 sequence
        return len(self.data) - (self.block_size + 1)


    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_batch(split, train_data, val_data, batch_size, block_size, device):
    data = train_data if split == 'train' else val_data
    # Ensure we don't pick indices too close to the end of the data
    max_idx = len(data) - block_size -1 
    if max_idx <=0: # Handle very small datasets for testing
        ix = torch.zeros(batch_size, dtype=torch.long)
    else:
        ix = torch.randint(max_idx, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Training Loop ---
def train(args): # Pass args from command line
    # Initialize config
    config = GPTConfig(model_type=args.model_type) # Use model_type from args
    
    # Setup TensorBoard
    run_specific_log_dir = os.path.join(config.out_dir, f"{config.run_name}_{config.model_type}_{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=run_specific_log_dir)
    print(f"TensorBoard logs will be saved to: {run_specific_log_dir}")

    device = config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
        config.device = 'cpu'
    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Create dataset
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'shakespeare_char')
    input_file_path = os.path.join(data_dir, 'input.txt')
    
    if not os.path.exists(input_file_path):
        print(f"Error: Dataset file not found at {input_file_path}")
        # ... (error message as before) ...
        return

    dataset = CharDataset(input_file_path, config.block_size)
    config.vocab_size = dataset.get_vocab_size()

    n = len(dataset.data)
    train_data = dataset.data[:int(n*0.9)]
    val_data = dataset.data[int(n*0.9):]

    # Initialize model
    model = GPT(config)
    model.to(device)
    print(f"Model ({config.model_type}) initialized on {device}.")

    if config.compile_model and hasattr(torch, 'compile'):
        print("Compiling the model...")
        try:
            model = torch.compile(model)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}. Running in eager mode.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, 
                                  weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))

    def get_lr(it):
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        if it > config.lr_decay_iters:
            return config.min_lr
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    
    best_val_loss = float('inf')
    
    for iter_num in range(config.max_iters):
        t0 = time.time()

        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % config.eval_interval == 0 or iter_num == config.max_iters - 1:
            model.eval()
            losses = torch.zeros(config.eval_iters)
            for k_eval in range(config.eval_iters):
                X, Y = get_batch('val', train_data, val_data, config.batch_size, config.block_size, device)
                with torch.no_grad():
                    logits, loss = model(X, Y)
                losses[k_eval] = loss.item()
            val_loss = losses.mean()
            writer.add_scalar('Loss/val', val_loss, iter_num)
            print(f"Iter {iter_num}: Val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Checkpoint saving
                checkpoint_path = os.path.join(run_specific_log_dir, f'ckpt_{config.model_type}.pt')
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            model.train()

        xb, yb = get_batch('train', train_data, val_data, config.batch_size, config.block_size, device)

        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        dt = time.time() - t0
        tokens_per_sec = (config.batch_size * config.block_size) / dt
        
        if iter_num % config.log_interval == 0:
            writer.add_scalar('Loss/train', loss.item(), iter_num)
            writer.add_scalar('LearningRate', lr, iter_num)
            writer.add_scalar('Speed/tokens_per_sec', tokens_per_sec, iter_num)
            writer.add_scalar('Speed/iter_time_ms', dt * 1000, iter_num)
            print(f"Iter {iter_num}: Loss {loss.item():.4f}, Time {dt*1000:.2f}ms, LR {lr:.6f}, Tokens/sec {tokens_per_sec:.2f}")

    writer.close()
    print(f"Training finished. Best val loss: {best_val_loss:.4f}")
    print(f"TensorBoard logs saved to: {run_specific_log_dir}")

    # (Generation part same as before)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a NanoGPT-like model with custom or standard attention.')
    parser.add_argument('--model_type', type=str, default='penalized', choices=['penalized', 'standard'],
                        help='Type of attention model to use: penalized or standard.')
    args = parser.parse_args()

    data_dir_main = os.path.join(os.path.dirname(__file__), 'data', 'shakespeare_char')
    if not os.path.exists(data_dir_main):
        os.makedirs(data_dir_main)
        print(f"Created directory: {data_dir_main}")
        # ... (instructions to download data)

    train(args)