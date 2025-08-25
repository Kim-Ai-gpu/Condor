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
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

def apply_rotary_emb(q, k, freqs_cis):
    q_reshaped, k_reshaped = q.float().reshape(*q.shape[:-1], -1, 2), k.float().reshape(*k.shape[:-1], -1, 2)
    q_complex, k_complex = torch.view_as_complex(q_reshaped), torch.view_as_complex(k_reshaped)
    freqs_cis = freqs_cis.unsqueeze(1)
    q_rotated, k_rotated = q_complex * freqs_cis, k_complex * freqs_cis
    q_out, k_out = torch.view_as_real(q_rotated).flatten(3), torch.view_as_real(k_rotated).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)

def precompute_freqs_cis(d_model, max_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
    t = torch.arange(max_len, device=inv_freq.device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs).unsqueeze(0)

class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1, self.w2, self.w3 = (nn.Linear(in_dim, hidden_dim, bias=False),
                                     nn.Linear(hidden_dim, out_dim, bias=False),
                                     nn.Linear(in_dim, hidden_dim, bias=False))
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class StandardAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, **kwargs):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, freqs_cis, mask=None):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q, K, V = (rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads) for t in (Q, K, V))
        Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis[:, :x.size(1)])
        attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        return self.out_proj(rearrange(attn_output, 'b h l d -> b l (h d)'))

class LocalAttention(nn.Module):
    def __init__(self, d_model, window_size=15, num_heads=8, **kwargs):
        super().__init__()
        self.d_model, self.window_size, self.num_heads, self.head_dim = d_model, window_size, num_heads, d_model // num_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))
    def forward(self, x, freqs_cis, mask=None):
        B, L, _ = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q, K, V = (rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads) for t in (Q, K, V))
        Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis[:, :L])

        pad = self.window_size - 1
        K_padded, V_padded = F.pad(K, (0, 0, pad, 0)), F.pad(V, (0, 0, pad, 0))
        K_windows, V_windows = K_padded.unfold(2, self.window_size, 1), V_padded.unfold(2, self.window_size, 1)
        K_windows, V_windows = (rearrange(t, 'b h l d w -> b h l w d') for t in (K_windows, V_windows))

        attn_scores = torch.einsum('bhld,bhlwd->bhlw', Q, K_windows) / math.sqrt(self.head_dim)
        final_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum('bhlw,bhlwd->bhld', final_weights, V_windows)
        return self.out_proj(rearrange(output, 'b h l d -> b l (h d)'))

class NeuralKYAttention(nn.Module):
    def __init__(self, d_model, window_size=15, num_heads=8, connection_type='swiglu', **kwargs):
        super().__init__()
        self.d_model, self.window_size, self.num_heads, self.head_dim = d_model, window_size, num_heads, d_model // num_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))
        
        connection_map = {
            'swiglu': lambda: SwiGLU(1, 128, 1),
        }
        self.connection_nets = nn.ModuleList([connection_map[connection_type]() for _ in range(num_heads)])
        self.last_weights = None
    def forward(self, x, freqs_cis, mask=None, save_weights=False):
        B, L, _ = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q, K, V = (rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads) for t in (Q, K, V))
        Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis[:, :L])
        
        pad = self.window_size - 1
        K_padded, V_padded = F.pad(K, (0, 0, pad, 0)), F.pad(V, (0, 0, pad, 0))
        K_windows, V_windows = K_padded.unfold(2, self.window_size, 1), V_padded.unfold(2, self.window_size, 1)
        K_windows, V_windows = (rearrange(t, 'b h l d w -> b h l w d') for t in (K_windows, V_windows))

        attn_scores = torch.einsum('bhld,bhlwd->bhlw', Q, K_windows) / math.sqrt(self.head_dim)
        attn_scores_softmax = F.softmax(attn_scores, dim=-1)

        pos = torch.linspace(0, 1, self.window_size, device=x.device).view(1, 1, 1, self.window_size, 1)
        conn_weights_list = [net(pos.expand(B, -1, L, -1, -1)) for net in self.connection_nets]
        conn_weights = torch.cat(conn_weights_list, dim=1).squeeze(-1)
        conn_weights = F.softmax(conn_weights, dim=-1)

        final_weights = attn_scores_softmax * conn_weights
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        if save_weights: self.last_weights = conn_weights.detach()
        
        output = torch.einsum('bhlw,bhlwd->bhld', final_weights, V_windows)
        return self.out_proj(rearrange(output, 'b h l d -> b l (h d)'))

class LinformerAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, proj_dim=128, **kwargs):
        super().__init__()
        self.d_model, self.num_heads, self.proj_dim, self.max_len = d_model, num_heads, proj_dim, kwargs['max_len']
        self.head_dim = d_model // num_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))
        
        self.k_proj_mat = nn.Parameter(torch.empty(self.proj_dim, self.max_len))
        self.v_proj_mat = nn.Parameter(torch.empty(self.proj_dim, self.max_len))
        nn.init.xavier_uniform_(self.k_proj_mat)
        nn.init.xavier_uniform_(self.v_proj_mat)
        
    def forward(self, x, freqs_cis, mask=None):
        B, L, _ = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q, K, V = (rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads) for t in (Q, K, V))
        Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis[:, :L])
        
        K = rearrange(K, 'b h l d -> b h d l')
        V = rearrange(V, 'b h l d -> b h d l')
        
        pad_len = self.max_len - L
        if pad_len > 0:
            K = F.pad(K, (0, pad_len))
            V = F.pad(V, (0, pad_len))
            
        proj_causal_mask = torch.tril(torch.ones(self.proj_dim, self.max_len, device=x.device))
        
        masked_k_weight = self.k_proj_mat * proj_causal_mask
        masked_v_weight = self.v_proj_mat * proj_causal_mask
        
        K_proj = torch.matmul(K, masked_k_weight.t())
        V_proj = torch.matmul(V, masked_v_weight.t())

        K_proj = rearrange(K_proj, 'b h d p -> b h p d', p=self.proj_dim)
        V_proj = rearrange(V_proj, 'b h d p -> b h p d', p=self.proj_dim)
        
        attn_scores = torch.einsum('bhld,bhmd->bhlm', Q, K_proj) / math.sqrt(self.head_dim)
        
        mask_cols = attn_scores.size(-1) 
        causal_mask = torch.triu(torch.ones(L, mask_cols, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask, -torch.inf)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum('bhlm,bhmd->bhld', attn_weights, V_proj)
        return self.out_proj(rearrange(output, 'b h l d -> b l (h d)'))

class LongformerAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, window_size=15, **kwargs):
        super().__init__()
        self.d_model, self.num_heads, self.window_size = d_model, num_heads, window_size
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))

    def forward(self, x, freqs_cis, mask=None):
        B, L, _ = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q, K, V = (rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads) for t in (Q, K, V))
        Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis[:, :L])

        causal_mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
        
        window_mask = torch.zeros(L, L, device=x.device, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2 + 1)
            window_mask[i, start:end] = 1
            
        global_mask = torch.zeros(L, L, device=x.device, dtype=torch.bool)
        global_mask[:, 0] = 1
        global_mask[0, :] = 1
        
        attn_mask = (causal_mask & window_mask) | global_mask
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        return self.out_proj(rearrange(attn_output, 'b h l d -> b l (h d)'))

class BigBirdAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, window_size=15, num_random=2, **kwargs):
        super().__init__()
        self.d_model, self.num_heads, self.window_size, self.num_random = d_model, num_heads, window_size, num_random
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))

    def forward(self, x, freqs_cis, mask=None):
        B, L, _ = x.shape
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        Q, K, V = (rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads) for t in (Q, K, V))
        Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis[:, :L])

        causal_mask = torch.tril(torch.ones(L, L, device=x.device)).bool()
        
        window_mask = torch.zeros(L, L, device=x.device, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2 + 1)
            window_mask[i, start:end] = 1
            
        global_mask = torch.zeros(L, L, device=x.device, dtype=torch.bool)
        global_mask[:, 0] = 1
        global_mask[0, :] = 1
        
        random_mask = torch.zeros(L, L, device=x.device, dtype=torch.bool)
        for i in range(L):
            num_options = i + 1
            if num_options > self.num_random:
                rand_indices = torch.randperm(num_options)[:self.num_random]
                random_mask[i, rand_indices] = 1
            else:
                random_mask[i, :num_options] = 1

        attn_mask = (causal_mask & (window_mask | random_mask)) | global_mask
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        return self.out_proj(rearrange(attn_output, 'b h l d -> b l (h d)'))

class BaseLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.final_norm = RMSNorm(config['d_model'])
        self.register_buffer('freqs_cis', precompute_freqs_cis(config['d_model'] // config['num_heads'], config['max_len'] * 2))
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, attention_mask=None):
        raise NotImplementedError

class LlamaModel(BaseLlama):
    def __init__(self, config, attention_module_class, attention_args={}):
        super().__init__(config)
        ff_dim = int(2 * (config['d_model'] * config['ff_dim_multiplier']) / 3)
        ff_dim = 256 * ((ff_dim + 256 - 1) // 256)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': attention_module_class(d_model=config['d_model'], num_heads=config['num_heads'], **attention_args),
                'norm1': RMSNorm(config['d_model']),
                'ffn': SwiGLU(config['d_model'], ff_dim, config['d_model']),
                'norm2': RMSNorm(config['d_model'])
            }) for _ in range(config['num_layers'])
        ])
        
        if config['task_type'] == 'classification':
            self.task_head = nn.Linear(config['d_model'], config['num_classes'])
        else:
            self.task_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def forward(self, input_ids, attention_mask=None, save_weights=False):
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)
        freqs_cis = self.freqs_cis[:, :L]
        
        for layer in self.layers:
            attn_args = {'save_weights': save_weights} if isinstance(layer['attention'], NeuralKYAttention) else {}
            h = x + layer['attention'](layer['norm1'](x), freqs_cis, mask=attention_mask, **attn_args)
            x = h + layer['ffn'](layer['norm2'](h))
        
        x = self.final_norm(x)

        if self.config['task_type'] == 'classification':
            return self.task_head(x[:, 0, :])
        else:
            return self.task_head(x)

class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.examples = [tokenizer.encode(t, max_length=max_length, truncation=True) for t in texts if len(t.strip()) > 0]
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        pad_len = self.max_length - 1 - len(input_ids)
        if pad_len > 0:
            input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            labels = F.pad(labels, (0, pad_len), value=-100)
        return {'input_ids': input_ids, 'labels': labels}

class CustomClassificationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.examples = dataset
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        item = self.examples[idx]
        text, label = item['text'], item['label']
        encoded = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': encoded['input_ids'].squeeze(0), 'labels': torch.tensor(label, dtype=torch.long)}

def get_data(task_name, tokenizer, config):
    print(f"Loading and preparing data for task: {task_name}")
    if task_name in ['wikitext-2', 'wikitext-103']:
        dataset = load_dataset('wikitext', f'{task_name}-v1')
        train_dset = CustomTextDataset(dataset['train']['text'], tokenizer, config['max_len'])
        valid_dset = CustomTextDataset(dataset['validation']['text'], tokenizer, config['max_len'])
    elif task_name == 'ag_news':
        dataset = load_dataset('ag_news')
        train_dset = CustomClassificationDataset(dataset['train'], tokenizer, config['max_len'])
        valid_dset = CustomClassificationDataset(dataset['test'], tokenizer, config['max_len'])
        
    elif task_name == 'gsm8k':
        dataset = load_dataset('gsm8k', 'main',trust_remote_code=True)
        train_texts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in dataset['train']]
        valid_texts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in dataset['test']]
        train_dset = CustomTextDataset(train_texts, tokenizer, config['max_len'])
        valid_dset = CustomTextDataset(valid_texts, tokenizer, config['max_len'])
        
    elif task_name == 'code_search_net':
        dataset = load_dataset("espejelomar/code_search_net_python_10000_examples", "default")
        
        def combine_code_docstring(examples):
            return [f"# {doc}\n{code}" for doc, code in zip(examples['func_documentation_string'], examples['func_code_string'])]
        
        all_texts = combine_code_docstring(dataset['train'])
        from sklearn.model_selection import train_test_split
        train_texts, valid_texts = train_test_split(
            all_texts, 
            test_size=0.2, 
            random_state=42
        )
        
        train_dset = CustomTextDataset(train_texts, tokenizer, config['max_len'])
        valid_dset = CustomTextDataset(valid_texts, tokenizer, config['max_len'])

    elif task_name == 'codeparrot':
        dataset = load_dataset('codeparrot/codeparrot-clean', split='train').train_test_split(test_size=0.01, seed=42)
        train_dset = CustomTextDataset(dataset['train']['content'], tokenizer, config['max_len'])
        valid_dset = CustomTextDataset(dataset['test']['content'], tokenizer, config['max_len'])

    elif task_name == 'tiny_stories':
        dataset = load_dataset('roneneldan/TinyStories')
        train_dset = CustomTextDataset(dataset['train']['text'], tokenizer, config['max_len'])
        valid_dset = CustomTextDataset(dataset['validation']['text'], tokenizer, config['max_len'])

    elif task_name == 'arxiv_papers':
        dataset = load_dataset('rbiswasfc/arxiv-papers', split='train')
        
        all_texts = [f"Title: {ex['title']}\nAbstract: {ex['abstract']}" for ex in tqdm(dataset, desc="Processing arxiv-papers")]

        from sklearn.model_selection import train_test_split
        train_texts, valid_texts = train_test_split(all_texts, test_size=0.1, random_state=42)

        train_dset = CustomTextDataset(train_texts, tokenizer, config['max_len'])
        valid_dset = CustomTextDataset(valid_texts, tokenizer, config['max_len'])
    
    else:
        raise ValueError(f"Task {task_name} not supported.")
    
    train_loader = DataLoader(train_dset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dset, batch_size=config['batch_size'])
    return train_loader, valid_loader

def train_epoch(model, loader, optimizer, criterion, scheduler, device, scaler, task_type):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.float16, device_type=device.type):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        total_loss += loss.item()
        pbar.set_description(f"Training (Loss: {loss.item():.4f})")
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, task_type):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for batch in pbar:
            input_ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            with torch.amp.autocast(dtype=torch.float16, device_type=device.type):
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            if task_type == 'classification':
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
    avg_loss = total_loss / len(loader)
    metrics = {'loss': avg_loss}
    if task_type == 'language_modeling':
        metrics['perplexity'] = math.exp(avg_loss)
    if task_type == 'classification':
        metrics['accuracy'] = total_correct / total_samples
    return metrics

def run_experiment(model, model_name, train_loader, valid_loader, config, device):
    print(f"\n--- Starting Experiment: {model_name} ---")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 5e-4), weight_decay=0.01)
    
    if config['task_type'] == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * config['num_epochs'])
    scaler = torch.amp.GradScaler('cuda')
    history = {'train_loss': [], 'valid_loss': [], 'perplexity': [], 'accuracy': []}

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device, scaler, config['task_type'])
        valid_metrics = evaluate(model, valid_loader, criterion, device, config['task_type'])
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_metrics['loss'])
        summary_str = f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Valid Loss: {valid_metrics['loss']:.4f}"
        
        if 'perplexity' in valid_metrics:
            history['perplexity'].append(valid_metrics['perplexity'])
            summary_str += f" | Perplexity: {valid_metrics['perplexity']:.2f}"
        if 'accuracy' in valid_metrics:
            history['accuracy'].append(valid_metrics['accuracy'])
            summary_str += f" | Accuracy: {valid_metrics['accuracy']:.4f}"
        print(summary_str)
            
    return history

def plot_results(results, config):
    num_epochs = config['num_epochs']
    epochs = range(1, num_epochs + 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    metrics_to_plot = ['valid_loss']
    if config['task_type'] == 'language_modeling': metrics_to_plot.append('perplexity')
    if config['task_type'] == 'classification': metrics_to_plot.append('accuracy')

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(8 * len(metrics_to_plot), 6))
    if len(metrics_to_plot) == 1: axes = [axes]
    fig.suptitle(f"Model Comparison on {config.get('task_name', 'default task')}", fontsize=16)

    markers = ['o', 's', '^', 'd', 'v', '*', 'p', 'X']
    for i, (name, history) in enumerate(results.items()):
        marker = markers[i % len(markers)]
        for ax, metric in zip(axes, metrics_to_plot):
            ax.plot(epochs, history[metric], marker=marker, linestyle='-', label=f'{name}')
            ax.set_title(f'{metric.replace("_", " ").title()} per Epoch')
            ax.set_xlabel('Epoch'); ax.set_ylabel(metric.title()); ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def main():
    print("=" * 60); print("Neural Attention Model Experiment Framework"); print("=" * 60)
    
    base_config = {
        'd_model': 256, 'num_layers': 4, 'num_heads': 4, 'ff_dim_multiplier': 4,
        'max_len': 256, 'window_size': 32, 'batch_size': 16, 'num_epochs': 3,
        'task_type': 'language_modeling', 'task_name': 'wikitext-2'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.cls_token is None: tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    base_config['vocab_size'] = len(tokenizer)

    print("\nSelect Experiment Phase:")
    print("1: Component Isolation"); print("2: Fair Comparison"); print("3: Scale Study")
    print("4: Ablation Studies"); print("5: Multi-Task Evaluation"); print("6: Dataset Variety Experiments")
    choice = input("Enter phase number: ")

    models_to_run = {}
    exp_config = base_config.copy()
    
    if choice == '3':
        print("\n--- Phase 3: Scale Study ---")
        print("Running experiments for Tiny, Small, and Medium scales automatically.")
        
        scale_map = {
            'Tiny': {'d_model': 128, 'num_layers': 2, 'num_heads': 2},
            'Small': {'d_model': 256, 'num_layers': 4, 'num_heads': 4},
            'Medium': {'d_model': 512, 'num_layers': 8, 'num_heads': 8}
        }
        
        results = {}
        train_loader, valid_loader = get_data(base_config['task_name'], tokenizer, base_config)

        for scale_name, scale_cfg in scale_map.items():
            print(f"\n--- Running Scale: {scale_name} ---")
            current_config = base_config.copy()
            current_config.update(scale_cfg)
            
            models_for_scale = {
                f'KY_Attention_{scale_name}': (NeuralKYAttention, {'window_size': current_config['window_size']}),
                f'Baseline_{scale_name}': (StandardAttention, {})
            }

            for name, (attn_class, attn_args) in models_for_scale.items():
                print(f"\nInstantiating model: {name}")
                model = LlamaModel(current_config, attn_class, attn_args)
                print(f"  - Params: {sum(p.numel() for p in model.parameters()):,}")
                history = run_experiment(model, name, train_loader, valid_loader, current_config, device)
                results[name] = history
        return

    elif choice == '1':
        print("\n--- Phase 1: Component Isolation ---")
        models_to_run['1A_Baseline'] = (StandardAttention, {})
        models_to_run['1B_LocalWindow'] = (LocalAttention, {'window_size': exp_config['window_size']})
        models_to_run['1C_KY_Attention'] = (NeuralKYAttention, {'window_size': exp_config['window_size']})
    
    elif choice == '2':
        print("\n--- Phase 2: Fair Comparison ---")
        models_to_run['Local'] = (LocalAttention, {'window_size': exp_config['window_size']})
        models_to_run['KY-Attention'] = (NeuralKYAttention, {'window_size': exp_config['window_size']})
        models_to_run['Linformer'] = (LinformerAttention, {'proj_dim': 128, 'max_len': exp_config['max_len']})
        models_to_run['Longformer'] = (LongformerAttention, {'window_size': exp_config['window_size']})
        models_to_run['BigBird'] = (BigBirdAttention, {'window_size': exp_config['window_size'], 'num_random': 2})
        print("\nNote: Longformer and BigBird are simplified implementations for comparison.")
        print("They capture the core ideas but are not as optimized as official library versions.")

    elif choice == '4':
        print("\n--- Phase 4: Ablation Studies ---")
        ablation_choice = input("Select study (1:Window Size): ")
        if ablation_choice == '1':
            for w in [8, 16, 32]:
                models_to_run[f'KY_W{w}'] = (NeuralKYAttention, {'window_size': w})
        else: print("Invalid choice."); return

    elif choice == '5':
        print("\n--- Phase 5: Multi-Task Evaluation ---")
        task_choice = input("Select task (1:WikiText-2 [LM], 2:AG_News [Classification]): ")
        if task_choice == '1':
            exp_config.update({'task_type': 'language_modeling', 'task_name': 'wikitext-2'})
        elif task_choice == '2':
            dataset = load_dataset('ag_news')
            num_classes = dataset['train'].features['label'].num_classes
            exp_config.update({'task_type': 'classification', 'task_name': 'ag_news', 'num_classes': num_classes})
        else: print("Invalid choice."); return
        models_to_run['KY_Attention'] = (NeuralKYAttention, {'window_size': exp_config['window_size']})
        models_to_run['Baseline'] = (StandardAttention, {})
        
    elif choice == '6':
        print("\n--- Phase 6: Dataset Variety Experiments ---")
        print("Select a dataset to experiment on:")
        print("1: wikitext-2")
        print("2: wikitext-103")
        print("3: gsm8k (Mathematical Reasoning)")
        print("4: code_search_net (Code/Docstring)")
        print("5: codeparrot (Python Code)")
        print("6: tiny_stories (Simple Narrative)")
        print("7: arxiv-papers (Scientific Articles)")
        dataset_choice = input("Enter dataset number: ")

        if dataset_choice == '1':
            exp_config.update({'task_name': 'wikitext-2'})
        elif dataset_choice == '2':
            exp_config.update({'task_name': 'wikitext-103'})

        elif dataset_choice == '3':
            exp_config.update({'task_name': 'gsm8k'})
        elif dataset_choice == '4':
            exp_config.update({'task_name': 'code_search_net'})
        elif dataset_choice == '5':
            exp_config.update({'task_name': 'codeparrot'})
        elif dataset_choice == '6':
            exp_config.update({'task_name': 'tiny_stories'})
        elif dataset_choice == '7':
            exp_config.update({'task_name': 'arxiv_papers'})
        else:
            print("Invalid dataset choice. Exiting.")
            return
            
        models_to_run['KY_Attention'] = (NeuralKYAttention, {'window_size': exp_config['window_size']})
        models_to_run['Baseline'] = (StandardAttention, {})

    else: print("Invalid choice. Exiting."); return
    
    train_loader, valid_loader = get_data(exp_config['task_name'], tokenizer, exp_config)

    results = {}
    for name, (attn_class, attn_args) in models_to_run.items():
        print(f"\nInstantiating model: {name}")
        model = LlamaModel(exp_config, attn_class, attn_args)
        print(f"  - Params: {sum(p.numel() for p in model.parameters()):,}")
        history = run_experiment(model, name, train_loader, valid_loader, exp_config, device)
        results[name] = history
    
    if results:
        print("\n--- Final Results Summary ---")
        plot_results(results, exp_config)

if __name__ == "__main__":
    main()
