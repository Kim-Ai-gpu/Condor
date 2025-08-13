import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import math
import time
from tqdm import tqdm

# --- 1. Neural KY-Attention Model ---

class ConnectionNetwork(nn.Module):
    """Learnable connection function network."""
    def __init__(self, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, positions):
        return self.net(positions)

class NeuralKYAttention(nn.Module):
    """Neural KY-Attention Layer - Improved with parallel processing."""
    def __init__(self, d_model, window_size=7, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.connection_nets = nn.ModuleList([
            ConnectionNetwork(hidden_dim=32, output_dim=1)
            for _ in range(num_heads)
        ])

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout_layer = nn.Dropout(dropout)
        self.padding = self.window_size // 2

    def forward(self, x, mask=None, return_weights=False):
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        K_padded = F.pad(K, (0, 0, self.padding, self.padding), 'constant', 0)
        V_padded = F.pad(V, (0, 0, self.padding, self.padding), 'constant', 0)

        K_windows = K_padded.unfold(2, self.window_size, 1).permute(0, 1, 2, 4, 3)
        V_windows = V_padded.unfold(2, self.window_size, 1).permute(0, 1, 2, 4, 3)

        attn_scores = torch.einsum('bhld,bhlwd->bhlw', Q, K_windows) / math.sqrt(self.head_dim)
        attn_scores = F.softmax(attn_scores, dim=-1)

        positions = torch.linspace(0, 1, self.window_size, device=x.device).view(1, 1, 1, self.window_size, 1)
        connection_weights_list = [
            net(positions.expand(B, -1, L, -1, -1)).squeeze(-1)
            for net in self.connection_nets
        ]
        connection_weights = torch.cat(connection_weights_list, dim=1)
        connection_weights = F.softmax(connection_weights, dim=-1)

        final_weights = attn_scores * connection_weights
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-9)

        if mask is not None:
            mask_padded = F.pad(mask.float(), (self.padding, self.padding), 'constant', 1)
            mask_windows = mask_padded.unfold(1, self.window_size, 1).unsqueeze(1)
            final_weights = final_weights.masked_fill(mask_windows == 0, 0)
            final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-9)

        output = torch.einsum('bhlw,bhlwd->bhld', final_weights, V_windows)
        output = output.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        output = self.out_proj(output)
        output = self.dropout_layer(output)

        if return_weights:
            final_weights_dense = torch.zeros(B, self.num_heads, L, L, device=x.device)
            for i in range(L):
                start = max(0, i - self.padding)
                end = min(L, i + self.padding + 1)
                win_start = max(0, self.padding - i)
                win_end = self.window_size - max(0, i + self.padding + 1 - L)
                final_weights_dense[:, :, i, start:end] = final_weights[:, :, i, win_start:win_end]
            return output, final_weights_dense

        return output

class BaseTransformer(nn.Module):
    """Base class for all Transformer models."""
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 ff_dim=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, return_weights=False):
        raise NotImplementedError

class Condor(BaseTransformer):
    """Condor for WikiText."""
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 ff_dim=2048, max_len=512, dropout=0.1, window_size=7):
        super().__init__(vocab_size, d_model, num_layers, num_heads, ff_dim, max_len, dropout)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': NeuralKYAttention(d_model, window_size, num_heads, dropout),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, ff_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_dim, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            self.layers.append(layer)

    def forward(self, input_ids, attention_mask=None, return_weights=False):
        B, L = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:L]
        x = self.dropout(x)

        all_attention_weights = []
        for layer in self.layers:
            if return_weights:
                attn_out, attn_weights = layer['attention'](x, attention_mask, return_weights=True)
                all_attention_weights.append(attn_weights)
            else:
                attn_out = layer['attention'](x, attention_mask)

            x = layer['norm1'](x + attn_out)
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if return_weights:
            return logits, all_attention_weights
        return logits

# --- 2. Standard Transformer Model ---

class StandardTransformer(BaseTransformer):
    """Standard Transformer for WikiText."""
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 ff_dim=2048, max_len=512, dropout=0.1):
        super().__init__(vocab_size, d_model, num_layers, num_heads, ff_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        src_mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(input_ids.device)

        x = self.token_embedding(input_ids) + self.pos_embedding[:L]
        x = self.dropout(x)
        
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        output = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.final_norm(output)
        logits = self.lm_head(output)
        return logits

# --- 3. Data Processing ---

class WikiTextDataset(Dataset):
    """Wrapper for WikiText dataset."""
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [
            tokenizer.encode(text, max_length=self.max_length, truncation=True)
            for text in texts if len(text.strip()) > 50
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        attention_mask = torch.ones_like(input_ids)

        pad_len = self.max_length - 1 - len(input_ids)
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, pad_len), value=-100)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

        return input_ids, attention_mask, labels

def load_wikitext_data(tokenizer, max_samples=2000):
    """Load WikiText-2 data."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    train_texts = [text for text in dataset['train']['text'] if len(text.strip()) > 10][:max_samples]
    valid_texts = [text for text in dataset['validation']['text'] if len(text.strip()) > 10][:max_samples//5]
    print(f"Loaded {len(train_texts)} training samples and {len(valid_texts)} validation samples.")
    
    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length=256)
    valid_dataset = WikiTextDataset(valid_texts, tokenizer, max_length=256)
    return train_dataset, valid_dataset

# --- 4. Training, Evaluation, and Benchmarking ---

def train_epoch(model, loader, optimizer, criterion, scheduler, device, model_type):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Training (Loss: ?)")
    for input_ids, attention_mask, labels in pbar:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        if model_type == 'nky':
            logits = model(input_ids, attention_mask=attention_mask.bool())
        else: # standard
            logits = model(input_ids, attention_mask=attention_mask)

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Training (Loss: {loss.item():.4f})")
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, model_type):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating (Loss: ?)")
        for input_ids, attention_mask, labels in pbar:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            if model_type == 'nky':
                 logits = model(input_ids, attention_mask=attention_mask.bool())
            else: # standard
                 logits = model(input_ids, attention_mask=attention_mask)
                 
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            pbar.set_description(f"Evaluating (Loss: {loss.item():.4f})")
    return total_loss / len(loader)

def run_experiment(model, model_name, train_loader, valid_loader, num_epochs, device, model_type):
    """Run a single model experiment."""
    print(f"\n--- Starting Experiment: {model_name} ---")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    history = {
        'train_loss': [], 'valid_loss': [], 'perplexity': [],
        'train_mem_mb': [], 'valid_mem_mb': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Reset memory stats for training
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device, model_type)
        train_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0
        
        # Reset memory stats for validation
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            
        valid_loss = evaluate(model, valid_loader, criterion, device, model_type)
        valid_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0
        
        perplexity = math.exp(valid_loss)
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['perplexity'].append(perplexity)
        history['train_mem_mb'].append(train_mem)
        history['valid_mem_mb'].append(valid_mem)
        
        print(f"Epoch {epoch+1} - Summary:")
        print(f"  - Train Loss: {train_loss:.4f} | Peak Train Memory: {train_mem:.2f} MB")
        print(f"  - Valid Loss: {valid_loss:.4f} | Peak Valid Memory: {valid_mem:.2f} MB")
        print(f"  - Perplexity: {perplexity:.2f}")
        
    return history

def measure_inference_performance(model, loader, device, model_type, num_batches=50):
    """Measure inference speed and memory usage."""
    model.eval().to(device)
    
    total_time = 0
    total_samples = 0
    
    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        # Warm-up
        for i, (input_ids, attention_mask, _) in enumerate(loader):
            if i >= 5: break
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            if model_type == 'nky': model(input_ids, attention_mask=attention_mask.bool())
            else: model(input_ids, attention_mask=attention_mask)
        
        # Measurement
        pbar = tqdm(loader, desc="Measuring Inference Performance", total=num_batches)
        for i, (input_ids, attention_mask, _) in enumerate(pbar):
            if i >= num_batches: break
            
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()
            
            if model_type == 'nky': model(input_ids, attention_mask=attention_mask.bool())
            else: model(input_ids, attention_mask=attention_mask)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_samples += input_ids.size(0)

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0
    samples_per_sec = total_samples / total_time
    
    return samples_per_sec, peak_mem


# --- 5. Main Function ---

def main():
    """Main comparison experiment function."""
    print("Neural KY-Attention vs Standard Transformer Comparison")
    print("=" * 60)
    
    # --- Configuration ---
    config = {
        'd_model': 256,
        'num_layers': 4,
        'num_heads': 8,
        'ff_dim': 1024,
        'max_len': 256,
        'dropout': 0.1,
        'window_size': 15, # Used by NKY Attention only
        'batch_size': 16,
        'num_epochs': 3,
        'max_samples': 5000
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Tokenizer and Data Loading ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, valid_dataset = load_wikitext_data(tokenizer, max_samples=config['max_samples'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'])
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    
    # --- Model Creation ---
    vocab_size = len(tokenizer)
    
    nky_model = Condor(
        vocab_size=vocab_size, d_model=config['d_model'], num_layers=config['num_layers'],
        num_heads=config['num_heads'], ff_dim=config['ff_dim'], max_len=config['max_len'],
        dropout=config['dropout'], window_size=config['window_size']
    )
    
    std_model = StandardTransformer(
        vocab_size=vocab_size, d_model=config['d_model'], num_layers=config['num_layers'],
        num_heads=config['num_heads'], ff_dim=config['ff_dim'], max_len=config['max_len'],
        dropout=config['dropout']
    )

    print("\nModel Parameter Counts:")
    print(f"  - Neural KY Transformer: {sum(p.numel() for p in nky_model.parameters()):,}")
    print(f"  - Standard Transformer:  {sum(p.numel() for p in std_model.parameters()):,}")
    
    # --- Run Experiments ---
    overall_start_time = time.time()
    
    nky_history = run_experiment(nky_model, "Neural KY Transformer", train_loader, valid_loader, config['num_epochs'], device, 'nky')
    std_history = run_experiment(std_model, "Standard Transformer", train_loader, valid_loader, config['num_epochs'], device, 'std')
    
    # --- Measure Inference Performance ---
    print("\n--- Measuring Inference Performance ---")
    nky_speed, nky_inf_mem = measure_inference_performance(nky_model, valid_loader, device, 'nky')
    std_speed, std_inf_mem = measure_inference_performance(std_model, valid_loader, device, 'std')

    total_time = time.time() - overall_start_time
    print(f"\nTotal experiment time: {total_time/60:.2f} minutes")

    # --- Final Results Summary ---
    print("\n--- Final Results ---")
    print(f"{'Metric':<25} | {'Neural KY Transformer':<25} | {'Standard Transformer':<25}")
    print("-" * 80)
    print(f"{'Final Valid Loss':<25} | {nky_history['valid_loss'][-1]:<25.4f} | {std_history['valid_loss'][-1]:<25.4f}")
    print(f"{'Final Perplexity':<25} | {nky_history['perplexity'][-1]:<25.2f} | {std_history['perplexity'][-1]:<25.2f}")
    print(f"{'Peak Training Memory (MB)':<25} | {max(nky_history['train_mem_mb']):<25.2f} | {max(std_history['train_mem_mb']):<25.2f}")
    print(f"{'Inference Speed (s/sec)':<25} | {nky_speed:<25.2f} | {std_speed:<25.2f}")
    print(f"{'Inference Memory (MB)':<25} | {nky_inf_mem:<25.2f} | {std_inf_mem:<25.2f}")


    # --- Results Visualization ---
    epochs = range(1, config['num_epochs'] + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Loss and Perplexity Curves
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle('Model Performance Comparison', fontsize=16)

    ax1.plot(epochs, nky_history['valid_loss'], 'o-', label='Neural KY (Valid Loss)')
    ax1.plot(epochs, std_history['valid_loss'], 's-', label='Standard (Valid Loss)')
    ax1.set_title('Validation Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, nky_history['perplexity'], 'o-', label='Neural KY (Perplexity)')
    ax2.plot(epochs, std_history['perplexity'], 's-', label='Standard (Perplexity)')
    ax2.set_title('Validation Perplexity per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('model_performance_curves.png', dpi=150)
    plt.show()

    # Performance Metrics Bar Chart
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Inference Performance & Memory Usage', fontsize=16)
    
    models = ['Neural KY', 'Standard']
    speeds = [nky_speed, std_speed]
    memories = [nky_inf_mem, std_inf_mem]
    
    ax3.bar(models, speeds, color=['#1f77b4', '#ff7f0e'])
    ax3.set_title('Inference Speed')
    ax3.set_ylabel('Samples / Second')
    for i, v in enumerate(speeds):
        ax3.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')

    ax4.bar(models, memories, color=['#1f77b4', '#ff7f0e'])
    ax4.set_title('Peak Inference Memory')
    ax4.set_ylabel('Peak Memory (MB)')
    for i, v in enumerate(memories):
        ax4.text(i, v + 2, f"{v:.2f}", ha='center', va='bottom')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('inference_performance_comparison.png', dpi=150)
    plt.show()


    # --- Learned Pattern Analysis (Neural KY) ---
    print("\nAnalyzing learned connection function patterns of Neural KY model...")
    sample_text = "The Transformer architecture, introduced in the paper 'Attention Is All You Need', has revolutionized the field of Natural Language Processing."
    try:
        analyze_learned_patterns(nky_model, sample_text, tokenizer, device, config['window_size'])
    except Exception as e:
        print(f"An error occurred during pattern analysis: {e}")


def analyze_learned_patterns(model, sample_input, tokenizer, device, window_size):
    """Analyze learned connection function patterns."""
    model.eval().to(device)
    
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(sample_input)[:50]], device=device)
        _, attention_weights = model(input_ids, return_weights=True)
        
        first_layer_weights = attention_weights[0] 
        num_heads = model.layers[0]['attention'].num_heads
        
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(16, 8), sharey=True)
        fig.suptitle('Learned Connection Functions (First Layer)', fontsize=16)
        
        positions = torch.linspace(0, 1, 100, device=device).unsqueeze(-1)
        
        for head in range(num_heads):
            row, col = head // axes.shape[1], head % axes.shape[1]
            ax = axes[row, col]
            
            connection_net = model.layers[0]['attention'].connection_nets[head]
            learned_function = connection_net(positions.unsqueeze(0)).squeeze().cpu().numpy()
            
            ax.plot(positions.squeeze().cpu().numpy(), learned_function, 'b-', linewidth=2)
            ax.set_title(f'Head {head+1}')
            ax.set_xlabel('Normalized Position')
            ax.set_ylabel('Connection Weight')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('learned_connection_functions.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    main()