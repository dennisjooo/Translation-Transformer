# Translation with Transformer

This project implements a translation system using a Transformer model for English to French translation. Currently,
the model can generate French-like sentences, though the output quality is still experimental. This implementation
serves primarily as a learning exercise for understanding Transformer architecture and debugging practices.

## Features

- Full Transformer architecture implementation for sequence-to-sequence translation
- PyTorch Lightning for structured and efficient training
- SentencePiece tokenization for subword-level vocabulary
- Multiple decoding strategies:
  - Greedy search
  - Random sampling with temperature
  - Beam search
- Training optimizations:
  - L2 regularization
  - Learning rate scheduling with warmup
  - Gradient accumulation
  - Early stopping

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies:

- PyTorch
- PyTorch Lightning
- TorchMetrics
- SentencePiece
- Pandas
- NumPy

## Project Structure

```bash
├── main.py                 # Inference script for translation
├── train.py                # Training script with PyTorch Lightning
├── config.py               # Configuration and hyperparameters
├── download_data.py        # Dataset download from statmt.org
├── train_tokenizer.py      # SentencePiece tokenizer training
├── run_training.sh         # Training pipeline automation
├── src/
│   ├── model.py            # Transformer architecture implementation
│   ├── dataset.py          # Data processing and loading
│   ├── lightning_module.py # PyTorch Lightning training module
│   ├── sampler.py          # Decoding strategies (greedy, beam, random)
│   └── test_sampler.py     # Sampler unit tests
├── tokenizer/              # Pre-trained tokenizer files
│   ├── en_tokenizer.model  # English SentencePiece model
│   ├── en_tokenizer.vocab  # English vocabulary
│   ├── fr_tokenizer.model  # French SentencePiece model
│   └── fr_tokenizer.vocab  # French vocabulary
└── requirements.txt        # Project dependencies
```

## Usage

### Training

To train the model, run:

```bash
python train.py --data_path /path/to/your/data.csv --max_epochs 40
```

Use `python train.py --help` to see all available training options.

### Inference

To translate sentences using a trained model:

```bash
python main.py --model_path /path/to/your/model.pth --input_text "Hello, how are you?"
```

Use `python main.py --help` to see all available inference options.
This will start an interactive session where you can input sentences for translation.

## Current Limitations & Development Notes

- **Training Data**: Limited to 2²³ samples, affecting overall model performance
- **Sampling Issues**: Current implementation of greedy search and other sampling strategies needs optimization
- **Dataset Processing**: Current approach of chopping sequences before adding `<BOS>` and `<EOS>` tokens may be suboptimal
- **Hyperparameter Tuning**: Current parameters are experimental and need systematic optimization

### Development Decisions

The project prioritizes learning and debugging over production-ready performance. Key areas for improvement include:

- Optimizing the token addition sequence (`<BOS>`, `<EOS>`) in data preprocessing
- Implementing more robust hyperparameter search
- Improving sampling strategies, especially for greedy decoding
