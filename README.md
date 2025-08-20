# 🦅 Condor: Fast-Converging Neural Attention via Learnable Connection Functions

PyTorch implementation of "Condor: Neural Connection Networks for Enhanced Attention" 

**Key Result**: Achieves 18.4% better perplexity (51.00 vs 62.51) compared to standard Transformer on wikitext-2 with only 3 epochs of training.

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

| Model | Perplexity | Epochs | Speed |
|-------|------------|--------|-------------|
| Standard Transformer | 62.51 | 3 | ❌ Slow |
| **Neural KY-Attention** | **51.00** | **3** | ✅ **Fast** |

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
- datasets==1.18.4 (for wikitext-2 dataset support)

## 🤝 Contributing

Issues and pull requests are welcome! This is research code, so expect some rough edges.

## 📧 Contact

Kim-Youngseong - dafaafafaf33@gmail.com
Our discord - https://discord.gg/tfuYKTGTk5
Our kakaotalk - https://open.kakao.com/o/gi5BuKGh

## 📜 License

MIT License - feel free to use for research and commercial applications!
