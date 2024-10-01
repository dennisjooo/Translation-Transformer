# Neural Machine Translation with Transformer

This project implements a neural machine translation system using a Transformer model. It's designed for translating between two languages, with a focus on English to French translation.

## Features

- Transformer-based architecture for sequence-to-sequence translation
- PyTorch Lightning for efficient training and evaluation
- SentencePiece tokenization for handling large vocabularies
- Customizable sampling strategies (greedy, random, beam search)
- L2 regularization and learning rate scheduling
- Support for different inference methods (greedy, random sampling, beam search)

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- torch
- lightning
- torchmetrics
- sentencepiece
- pandas
- numpy

## Project Structure

- `main.py`: Entry point for running inference with a trained model
- `train.py`: Script for training the Transformer model
- `model.py`: Implementation of the Transformer architecture
- `dataset.py`: Data loading and preprocessing utilities
- `lightning_module.py`: PyTorch Lightning module for training
- `sampler.py`: Sampling strategies for inference
- `config.py`: Configuration settings for the project

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

- The amount of data used for training is limited, so the model may not perform well at all.
But I guess what matters to
  me is it's more a debugging exercise and to get a better understanding of the concepts.
- Sampling strategy is off especially if it uses some greedy like policy.
- Training SPM on higher amount of vocab_size is slow, so I cap it to 9999.
- I'm debating on how to make the dataset. Should I add `<BOS>` and `<EOS>` tokens first and then chop it to max_length or should I add them later.
  Right now, I chop it first and then add `<BOS>` and `<EOS>` which might cause some inefficiency.
- Hyperparameter search is whatever I think of at the moment, not robust at all.
