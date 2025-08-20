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
from einops import rearrange, repeat

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

def apply_rotary_emb(q, k, freqs_cis):
    q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
    k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)
    
    q_complex = torch.view_as_complex(q_reshaped)
    k_complex = torch.view_as_complex(k_reshaped)
    
    freqs_cis = freqs_cis.unsqueeze(1)
    
    q_rotated = q_complex * freqs_cis
    k_rotated = k_complex * freqs_cis
    
    q_out = torch.view_as_real(q_rotated).flatten(3)
    k_out = torch.view_as_real(k_rotated).flatten(3)
    
    return q_out.type_as(q), k_out.type_as(k)

def precompute_freqs_cis(d_model, max_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
    t = torch.arange(max_len, device=inv_freq.device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.unsqueeze(0)

class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class NeuralKYAttention(nn.Module):
    def __init__(self, d_model, window_size=7, num_heads=8, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.connection_nets = nn.ModuleList([
            SwiGLU(in_dim=1,hidden_dim=128, out_dim=1)
            for _ in range(num_heads)
        ])

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, L, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, 'b l (h d) -> b h l d', h=self.num_heads)
        K = rearrange(K, 'b l (h d) -> b h l d', h=self.num_heads)
        V = rearrange(V, 'b l (h d) -> b h l d', h=self.num_heads)
        
        Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis[:, :L])

        causal_padding = self.window_size - 1
        K_padded = F.pad(K, (0, 0, causal_padding, 0), 'constant', 0)
        V_padded = F.pad(V, (0, 0, causal_padding, 0), 'constant', 0)

        K_windows = K_padded.unfold(2, self.window_size, 1)
        V_windows = V_padded.unfold(2, self.window_size, 1)

        K_windows = rearrange(K_windows, 'b h l d w -> b h l w d')
        V_windows = rearrange(V_windows, 'b h l d w -> b h l w d')
        
        Q_expanded = rearrange(Q, 'b h l d -> b h l 1 d')

        attn_scores = torch.einsum('bhlqd,bhlwd->bhlw', Q_expanded, K_windows) / math.sqrt(self.head_dim)
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
        
        output = torch.einsum('bhlw,bhlwd->bhld', final_weights, V_windows)
        output = rearrange(output, 'b h l d -> b l (h d)')
        return self.out_proj(output)

class BaseLlama(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 ff_dim_multiplier=4, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.register_buffer('freqs_cis', precompute_freqs_cis(d_model // num_heads, max_len * 2))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        raise NotImplementedError

class Condor(BaseLlama):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 ff_dim_multiplier=4, max_len=512, window_size=15):
        super().__init__(vocab_size, d_model, num_layers, num_heads, ff_dim_multiplier, max_len)
        
        ff_dim = int(2 * (d_model * ff_dim_multiplier) / 3)
        ff_dim = 256 * ((ff_dim + 256 - 1) // 256)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': NeuralKYAttention(d_model, window_size, num_heads),
                'norm1': RMSNorm(d_model),
                'ffn': SwiGLU(d_model, ff_dim, d_model),
                'norm2': RMSNorm(d_model)
            })
            self.layers.append(layer)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)
        
        freqs_cis = self.freqs_cis[:, :L]

        for layer in self.layers:
            h = x + layer['attention'](layer['norm1'](x), freqs_cis, mask=attention_mask)
            x = h + layer['ffn'](layer['norm2'](h))

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

class StandardLlama(BaseLlama):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8,
                 ff_dim_multiplier=4, max_len=512):
        super().__init__(vocab_size, d_model, num_layers, num_heads, ff_dim_multiplier, max_len)
        
        ff_dim = int(2 * (d_model * ff_dim_multiplier) / 3)
        ff_dim = 256 * ((ff_dim + 256 - 1) // 256)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=False),
                'norm1': RMSNorm(d_model),
                'ffn': SwiGLU(d_model, ff_dim, d_model),
                'norm2': RMSNorm(d_model)
            })
            self.layers.append(layer)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)

        causal_mask = torch.triu(torch.ones(L, L, device=input_ids.device, dtype=torch.bool), diagonal=1)

        for layer in self.layers:
            normed_x = layer['norm1'](x)
            
            if hasattr(F, 'scaled_dot_product_attention'):
                 q_proj_weight = layer['attention'].in_proj_weight[:self.d_model, :]
                 k_proj_weight = layer['attention'].in_proj_weight[self.d_model:2*self.d_model, :]
                 v_proj_weight = layer['attention'].in_proj_weight[2*self.d_model:, :]
                 
                 Q = F.linear(normed_x, q_proj_weight)
                 K = F.linear(normed_x, k_proj_weight)
                 V = F.linear(normed_x, v_proj_weight)

                 Q = rearrange(Q, 'b l (h d) -> b h l d', h=layer['attention'].num_heads)
                 K = rearrange(K, 'b l (h d) -> b h l d', h=layer['attention'].num_heads)
                 V = rearrange(V, 'b l (h d) -> b h l d', h=layer['attention'].num_heads)

                 Q, K = apply_rotary_emb(Q, K, freqs_cis=self.freqs_cis[:, :L])
                 
                 attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True)
                 attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
                 attn_output = layer['attention'].out_proj(attn_output)
            else:
                attn_output, _ = layer['attention'](normed_x, normed_x, normed_x, attn_mask=causal_mask, need_weights=False)

            h = x + attn_output
            x = h + layer['ffn'](layer['norm2'](h))

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [
            tokenizer.encode(text, max_length=self.max_length, truncation=True)
            for text in texts if len(text.strip()) > 0
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

def load_wikitext_data(tokenizer):
    print("Loading Wikitext-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-v1', trust_remote_code=True)
    
    train_texts = [text for text in dataset['train']['text'] if len(text.strip()) > 0]
    valid_texts = [text for text in dataset['validation']['text'] if len(text.strip()) > 0]
    print(f"Loaded {len(train_texts)} training samples and {len(valid_texts)} validation samples.")
    
    train_dataset = CustomTextDataset(train_texts, tokenizer, max_length=256)
    valid_dataset = CustomTextDataset(valid_texts, tokenizer, max_length=256)
    return train_dataset, valid_dataset

def train_epoch(model, loader, optimizer, criterion, scheduler, device, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Training (Loss: ?)")
    for input_ids, attention_mask, labels in pbar:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(dtype=torch.float16, device_type=device.type):
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Training (Loss: {loss.item():.4f})")
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating (Loss: ?)")
        for input_ids, attention_mask, labels in pbar:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            pbar.set_description(f"Evaluating (Loss: {loss.item():.4f})")
    return total_loss / len(loader)

def run_experiment(model, model_name, train_loader, valid_loader, num_epochs, device):
    print(f"\n--- Starting Experiment: {model_name} ---")
    
    try:
        model = torch.compile(model)
        print(f"Successfully compiled {model_name} model.")
    except Exception as e:
        print(f"Could not compile {model_name} model: {e}")
        
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    history = {
        'train_loss': [], 'valid_loss': [], 'perplexity': [],
        'train_mem_mb': [], 'valid_mem_mb': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        torch.cuda.reset_peak_memory_stats(device)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device, scaler)
        train_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        torch.cuda.reset_peak_memory_stats(device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        valid_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
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

def measure_inference_performance(model, loader, device, num_batches=50):
    model.eval().to(device)
    
    total_time = 0
    total_samples = 0
    
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for i, (input_ids, attention_mask, _) in enumerate(loader):
            if i >= 5: break
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                model(input_ids, attention_mask=attention_mask)
        
        pbar = tqdm(loader, desc="Measuring Inference Performance", total=num_batches)
        for i, (input_ids, attention_mask, _) in enumerate(pbar):
            if i >= num_batches: break
            
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                model(input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_samples += input_ids.size(0)

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    samples_per_sec = total_samples / total_time
    
    return samples_per_sec, peak_mem

def main():
    print("Neural KY-Attention vs Standard Llama Comparison on Wikitext-2")
    print("=" * 60)
    
    config = {
        'd_model': 512,        
        'num_layers': 8,       
        'num_heads': 16,
        'ff_dim_multiplier': 4,
        'max_len': 256,
        'window_size': 15,
        'batch_size': 16,      
        'num_epochs': 3        
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        print("Warning: float16 is not supported on this GPU. The script may run slower or fail.")
        
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, valid_dataset = load_wikitext_data(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'])
    
    vocab_size = len(tokenizer)
    
    condor_model = Condor(
        vocab_size=vocab_size, d_model=config['d_model'], num_layers=config['num_layers'],
        num_heads=config['num_heads'], ff_dim_multiplier=config['ff_dim_multiplier'], 
        max_len=config['max_len'], window_size=config['window_size']
    )
    
    llama_model = StandardLlama(
        vocab_size=vocab_size, d_model=config['d_model'], num_layers=config['num_layers'],
        num_heads=config['num_heads'], ff_dim_multiplier=config['ff_dim_multiplier'],
        max_len=config['max_len']
    )

    print("\nModel Parameter Counts:")
    print(f"  - Condor (NKY): {sum(p.numel() for p in condor_model.parameters()):,}")
    print(f"  - Standard Llama: {sum(p.numel() for p in llama_model.parameters()):,}")
    
    overall_start_time = time.time()
    
    nky_history = run_experiment(condor_model, "Condor (Neural KY)", train_loader, valid_loader, config['num_epochs'], device)
    std_history = run_experiment(llama_model, "Standard Llama", train_loader, valid_loader, config['num_epochs'], device)
    
    print("\n--- Measuring Inference Performance ---")
    compiled_condor = torch.compile(condor_model).to(device)
    compiled_llama = torch.compile(llama_model).to(device)
    nky_speed, nky_inf_mem = measure_inference_performance(compiled_condor, valid_loader, device)
    std_speed, std_inf_mem = measure_inference_performance(compiled_llama, valid_loader, device)

    total_time = time.time() - overall_start_time
    print(f"\nTotal experiment time: {total_time/60:.2f} minutes")

    print("\n--- Final Results ---")
    print(f"{'Metric':<25} | {'Condor (NKY)':<25} | {'Standard Llama':<25}")
    print("-" * 80)
    print(f"{'Final Valid Loss':<25} | {nky_history['valid_loss'][-1]:<25.4f} | {std_history['valid_loss'][-1]:<25.4f}")
    print(f"{'Final Perplexity':<25} | {nky_history['perplexity'][-1]:<25.2f} | {std_history['perplexity'][-1]:<25.2f}")
    print(f"{'Peak Training Memory (MB)':<25} | {max(nky_history['train_mem_mb']):<25.2f} | {max(std_history['train_mem_mb']):<25.2f}")
    print(f"{'Inference Speed (s/sec)':<25} | {nky_speed:<25.2f} | {std_speed:<25.2f}")
    print(f"{'Inference Memory (MB)':<25} | {nky_inf_mem:<25.2f} | {std_inf_mem:<25.2f}")

    epochs = range(1, config['num_epochs'] + 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle('Model Performance Comparison on Wikitext-2', fontsize=16)

    ax1.plot(epochs, nky_history['valid_loss'], 'o-', label='Condor (NKY) Valid Loss')
    ax1.plot(epochs, std_history['valid_loss'], 's-', label='Standard Llama (Valid Loss)')
    ax1.set_title('Validation Loss per Epoch'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()

    ax2.plot(epochs, nky_history['perplexity'], 'o-', label='Condor (NKY) Perplexity')
    ax2.plot(epochs, std_history['perplexity'], 's-', label='Standard Llama (Perplexity)')
    ax2.set_title('Validation Perplexity per Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Perplexity'); ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

if __name__ == "__main__":
    main()
