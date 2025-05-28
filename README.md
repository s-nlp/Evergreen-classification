<h1 align="center">🌲EverGreen QA 🍂 </h1>
<h3 align="center">Will It Still Be True Tomorrow? Multilingual Evergreen Question Classification to Improve Trustworthy QA</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2505.21115">📄arXiv</a> •
  <a href="https://github.com/s-nlp/Evergreen-classification">🌐GitHub</a> •
  <a href="https://huggingface.co/collections/s-nlp/evergreen-683465909575cb89d6b904fe">🤗HuggingFace</a>
</p>


![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains the implementation of a multilingual text classification system that categorizes questions based on their temporal mutability.

## Abstract

Large Language Models (LLMs) often hallucinate in question answering (QA) tasks. A key yet underexplored factor contributing to this is the temporality of questions -- whether they are evergreen (answers remain stable over time) or mutable (answers change). In this work, we introduce EverGreenQA, the first multilingual QA dataset with evergreen labels, supporting both evaluation and training.
Using EverGreenQA, we benchmark 12 modern LLMs to assess whether they encode question temporality explicitly (via verbalized judgments) or implicitly (via uncertainty signals). We also train EG-E5, a lightweight multilingual classifier that achieves SoTA performance on this task. Finally, we demonstrate the practical utility of evergreen classification across three applications: improving self-knowledge estimation, filtering QA datasets, and explaining GPT-4o’s retrieval behavior.

## Overview

This project implements a classifier that determines whether questions have answers that are:
- **Evergreen**: Answers that almost never change (e.g., "What was Cain's brother's name?")
- **Mutable**: Answers that typically change over several years or less (e.g., "What breed of dog is considered the smallest in the world?")

The system supports classification across 7 languages: English, Russian, French, German, Hebrew, Arabic, and Chinese.

## Features

- 🌍 **Multilingual Support**: Train and evaluate on 7 different languages
- 🤖 **Multiple Model Support**: Compatible with various transformer models (mDeBERTa, E5, mBERT)
- 📊 **Per-Language Metrics**: Detailed F1 scores for each language
- 🚀 **Efficient Generation**: Uses vLLM for fast inference with multiple LLMs

## Installation

```bash
# Clone the repository
git clone https://github.com/s-nlp/EverGreen-classification.git
cd EverGreen-classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
evergreen-classification/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── train.py            # Training script
│   ├── generate.py         # Generation/evaluation script
│   └── utils/
│       ├── data_utils.py
├── datasets/
│   └── aliases/           
├── models/                 # Saved models directory
├── results/               # Training results
└── docs/
    └── paper.pdf          # Research paper
```

## Data Format

The expected CSV format for training data:
- `is_evergreen`: Binary classification label (0 or 1)
- Language columns: `Russian`, `English`, `French`, `German`, `Hebrew`, `Arabic`, `Chinese`

## Usage

### Training

```bash
# Basic training
python src/train.py \
    --model_name "intfloat/multilingual-e5-large-instruct" \
    --output_dir "./results/multilingual-e5-large" \
    --num_epochs 8 \
    --batch_size 16

# Training with custom configuration
python src/train.py --config config/config.yaml
```

### Generation and Evaluation

```bash
# Generate translations using OpenAI
python src/generate.py \
    --mode translate \
    --input_file "data/datasets/train.csv" \
    --api_key "your-openai-api-key"

# Generate classifications using vLLM
python src/generate.py \
    --mode classify \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --input_file "data/datasets/train.csv" \
    --output_file "data/datasets/output.csv"
```

## Results

### Model Performance

| Model | Overall F1 | English | Russian | French | German | Hebrew | Arabic | Chinese |
|-------|-----------|---------|---------|--------|--------|--------|--------|---------|
| multilingual-e5-large-instruct | 0.89 | 0.92 | 0.88 | 0.90 | 0.89 | 0.87 | 0.86 | 0.91 |
| mdeberta-v3-base | 0.87 | 0.90 | 0.86 | 0.88 | 0.87 | 0.85 | 0.84 | 0.89 |
| bert-base-multilingual-cased | 0.85 | 0.88 | 0.84 | 0.86 | 0.85 | 0.83 | 0.82 | 0.87 |

### Training Configuration

- Learning Rate: 4.676e-05
- Batch Size: 16
- Epochs: 8
- Warmup Steps: 500
- Weight Decay: 0.01

## Configuration

Create a `config/config.yaml` file:

```yaml
model:
  name: "intfloat/multilingual-e5-large-instruct"
  max_length: 64
  
training:
  epochs: 8
  batch_size: 16
  learning_rate: 4.676e-05
  warmup_steps: 500
  weight_decay: 0.01
  
data:
  train_path: "/datasets/train.csv"
  test_path: "/datasets/test.csv"
  languages: ["Russian", "English", "French", "German", "Hebrew", "Arabic", "Chinese"]
```

## API Keys

For OpenAI translation features:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pletenev2025truetomorrowmultilingualevergreen,
      title={Will It Still Be True Tomorrow? Multilingual Evergreen Question Classification to Improve Trustworthy QA}, 
      author={Sergey Pletenev and Maria Marina and Nikolay Ivanov and Daria Galimzianova and Nikita Krayko and Mikhail Salnikov and Vasily Konovalov and Alexander Panchenko and Viktor Moskvoretskii},
      year={2025},
      eprint={2505.21115},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.21115}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

