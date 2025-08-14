# Condor🦅: Fast-Converging Neural Attention via Learnable Connection Functions

PyTorch implementation of "Condor: Neural Connection Networks for Enhanced Attention" 

**Key Result**: Achieves 50x better perplexity (13.77 vs 717.88) compared to standard Transformer on WikiText-2 with only 3 epochs of training.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Kim-Ai-gpu/Condor
cd Condor

# Install dependencies
pip install torch torchvision transformers datasets

# Run experiment
python condor.py
```

## 📊 Results

| Model | Perplexity | Epochs | Convergence |
|-------|------------|--------|-------------|
| Standard Transformer | 717.88 | 3 | ❌ Not converged |
| **Neural KY-Attention** | **13.77** | **3** | ✅ **Fast convergence** |

## 🔧 Architecture

Neural KY-Attention extends traditional self-attention with learnable connection functions:

- **Learnable Connection Networks**: Each attention head learns unique connection patterns
- **Linear Complexity**: O(L×W×H) instead of O(L²×H) 
- **Window-based**: Efficient local attention with global context

## 🎯 Key Features

- ⚡ **Extremely fast convergence** - achieves good performance in just 3 epochs
- 🧠 **Novel mathematical foundation** - based on Kim-Youngseong (KY) Transform theory
- 📈 **Better sample efficiency** - learns more from limited data
- 🔧 **Easy to integrate** - drop-in replacement for standard attention

## 📝 Citation

```bibtex
@article{kim2025condor,
  title={Condor: Condor: Neural Connection Networks for Enhanced Attention},
  author={Kim-Youngseong},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## 📄 Paper

- **arXiv**: [Link coming soon]
- **Code**: You're looking at it! 🎉

## 🛠 Requirements

- Python 3.8+
- PyTorch 1.9+
- transformers
- datasets

## 🤝 Contributing

Issues and pull requests are welcome! This is research code, so expect some rough edges.

## 📧 Contact

Kim-Youngseong - dafaafafaf33@gmail.com

## 📜 License

MIT License - feel free to use for research and commercial applications!
