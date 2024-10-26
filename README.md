# Translation with Transformer

This project implements a translation system using a Transformer model. It's designed for translating between two
languages, with a focus on English to French translation. As of now it produces some French sounding sentences but not very good.

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

## Limitations

- The amount of data used for training is limited ($2^{23}$), so the model may not perform well at all.
But I guess what matters to
  me is it's more a debugging exercise and to get a better understanding of the concepts.
- Sampling strategy is off especially if it uses some greedy like policy.
- I'm debating on how to make the dataset. Should I add `<BOS>` and `<EOS>` tokens first and then chop it to max_length or should I add them later.
  Right now, I chop it first and then add `<BOS>` and `<EOS>` which might cause some inefficiency.
- Hyperparameter search is whatever I think of at the moment, not robust at all.
