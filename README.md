# LLMs-03 - DataLoader

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TorchText](https://img.shields.io/badge/TorchText-0.17.2-orange)](https://pytorch.org/text)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org)

A comprehensive comparison of text preprocessing approaches using PyTorch DataLoader for NLP applications. This project demonstrates two distinct methodologies for tokenization and vocabulary mapping in multilingual text processing pipelines.

## 🚀 Features

- 📊 **Dual Preprocessing Approaches**: Item-level vs Batch-level text processing comparison
- 🌍 **Multilingual Support**: English (basic tokenizer) and French (spaCy) text processing  
- ⚡ **Performance Benchmarking**: Built-in timing analysis for method comparison
- 🔧 **Custom DataLoader Implementation**: Flexible collate functions for variable-length sequences
- 📝 **Educational Focus**: Clear code documentation and methodology explanations
- 🎯 **PyTorch Integration**: Native torch tensors and padding sequences

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

**Core Libraries:**
- `torch==2.2.2` - Deep learning framework
- `torchtext==0.17.2` - Text processing utilities
- `transformers==4.42.1` - Hugging Face transformers
- `pandas` - Data manipulation  
- `numpy==1.26.0` - Numerical computing
- `spacy` - Optional multilingual NLP processing (used in examples)

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Performance Analysis](#performance-analysis)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Contact](#contact)

## 🔧 Installation

### Prerequisites
- Python 3.7+
- Google Colab (recommended) or Jupyter Notebook environment

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Net-AI-Git/LLMs-03---DataLoader.git
cd LLMs-03---DataLoader
```

2. **Open in Google Colab:**
- Upload the notebook to Google Drive
- Open with Google Colab
- Run the installation cell to install all dependencies

3. **Local Installation (alternative):**
```bash
pip install torch==2.2.2 torchtext==0.17.2
pip install transformers==4.42.1 sentencepiece
pip install spacy pandas numpy==1.26.0 scikit-learn
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm  
python -m spacy download fr_core_news_sm
```

## 🎯 Usage

### Basic Example

```python
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Initialize tokenizer and vocabulary
tokenizer = get_tokenizer("basic_english")
sentences = ["Your text data here..."]
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Method 1: Item-level preprocessing
dataset = CustomDataset(sentences, tokenizer, vocab)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Method 2: Batch-level preprocessing  
dataset = CustomDataset(sentences)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=batch_collate_fn)
```

### Running the Complete Analysis

1. Open `DataLoader.ipynb` in Google Colab
2. Execute all cells sequentially
3. Compare timing results between methods
4. Analyze the tokenized output for both approaches

### Multilingual Processing (Example)

```python
# Basic English tokenizer (main approach)
tokenizer = get_tokenizer("basic_english")

# spaCy French tokenizer (demonstration)
tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
corpus = ["Ceci est une phrase.", "C'est un autre exemple."]
vocab = build_vocab_from_iterator(map(tokenizer, corpus))
```

## 📁 Project Structure

```
LLMs-03---DataLoader/
│
├── DataLoader.ipynb          # Main notebook with implementations
└── README.md                # Project documentation
```

## 🔬 Methodology

### Method 1: Item-Level Preprocessing
- Tokenization and vocabulary mapping performed in `Dataset.__getitem__()`
- Individual tensor creation per sample
- Padding applied at batch level via `collate_fn`

### Method 2: Batch-Level Preprocessing  
- Raw strings returned from `Dataset.__getitem__()`
- Tokenization and mapping performed in `collate_fn`
- Batch-wise tensor creation and padding

### Key Differences
- **Memory Usage**: Method 1 pre-processes individual items
- **Flexibility**: Method 2 allows dynamic batch processing
- **Performance**: Timing analysis reveals efficiency trade-offs

## 📊 Performance Analysis

The notebook includes built-in performance benchmarking:

```python
start = time.perf_counter()
# Processing logic here
end = time.perf_counter()
print("Elapsed time:", end - start, "seconds")
```

**Typical Results:**
- Method 1: ~0.009 seconds
- Method 2: ~0.004 seconds

*Note: Results may vary based on hardware and data size*

## 🚀 Future Work

- [ ] Integration with modern transformer tokenizers (BERT, GPT)
- [ ] Implementation of attention-based sequence processing
- [ ] Support for larger vocabulary sizes and datasets
- [ ] Memory optimization for large-scale text processing
- [ ] Distributed processing capabilities
- [ ] Advanced padding strategies and sequence bucketing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Contact

**Netanel Itzhak**
- 💼 LinkedIn: [linkedin.com/in/netanelitzhak](https://www.linkedin.com/in/netanelitzhak)
- 📧 Email: ntitz19@gmail.com
- 🐙 GitHub: [@Net-AI-Git](https://github.com/Net-AI-Git)

## 🙏 Acknowledgments

- PyTorch team for excellent documentation and examples
- spaCy community for multilingual NLP support
- Hugging Face for transformer model accessibility

---

⭐ **Star this repository if you found it helpful!**
