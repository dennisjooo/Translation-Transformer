import argparse
import torch
import sentencepiece as spm
from typing import Tuple
from src.model import Transformer
from config import config
from src.sampler import Sampler

def load_model(model_path: str, device: torch.device) -> Transformer:
    """
    Load the Transformer model from a file.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model onto.

    Returns:
        Transformer: The loaded Transformer model.
    """
    model = Transformer(config["n_embed"], config["n_head"], config["n_hidden"], config["n_layers"],
                        config["max_vocab"], config["max_len"], config["padding_value"], config["dropout_p"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_tokenizers(src_tokenizer_path: str, tgt_tokenizer_path: str) -> Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
    """
    Load source and target tokenizers.

    Args:
        src_tokenizer_path (str): Path to the source tokenizer model file.
        tgt_tokenizer_path (str): Path to the target tokenizer model file.

    Returns:
        Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]: Source and target tokenizers.
    """
    src_tokenizer = spm.SentencePieceProcessor(model_file=src_tokenizer_path)
    tgt_tokenizer = spm.SentencePieceProcessor(model_file=tgt_tokenizer_path)
    return src_tokenizer, tgt_tokenizer

def translate(sampler: Sampler, src_tokenizer: spm.SentencePieceProcessor, 
              tgt_tokenizer: spm.SentencePieceProcessor, sentence: str, device: torch.device, 
              sampling_strategy: str) -> str:
    """
    Translate a sentence using the given model and tokenizers.

    Args:
        sampler (Sampler): The sampler for generating translations.
        src_tokenizer (spm.SentencePieceProcessor): Source language tokenizer.
        tgt_tokenizer (spm.SentencePieceProcessor): Target language tokenizer.
        sentence (str): The sentence to translate.
        device (torch.device): The device to perform computations on.
        sampling_strategy (str): The sampling strategy to use for translation.

    Returns:
        str: The translated sentence.
    """
    # Tokenize the input sentence
    src_tokens = src_tokenizer.encode(sentence)
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)

    # Generate translation
    tgt_tensor = sampler(src_tensor, sampling_strategy=sampling_strategy)

    # Decode the translation
    translation = tgt_tokenizer.decode(tgt_tensor.squeeze().tolist())
    return translation

def main():
    """
    Main function to run the translation program.
    """
    parser = argparse.ArgumentParser(description="Translate sentences using the trained Transformer model.")
    parser.add_argument("--model_path", type=str, default="transformer.pth", help="Path to the trained model")
    parser.add_argument("--src_tokenizer", type=str, default=config["src_tokenizer_path"], help="Path to the source tokenizer")
    parser.add_argument("--tgt_tokenizer", type=str, default=config["tgt_tokenizer_path"], help="Path to the target tokenizer")
    parser.add_argument("--max_len", type=int, default=config["max_len"], help="Maximum length of the generated sequence")
    parser.add_argument("--sampling_strategy", type=str, default="beam", choices=["greedy", "random", "beam"], help="Sampling strategy for translation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    args = parser.parse_args()

    # Load model and tokenizers
    device = torch.device(args.device)
    model = load_model(args.model_path, device)
    src_tokenizer, tgt_tokenizer = load_tokenizers(args.src_tokenizer, args.tgt_tokenizer)

    # Create sampler
    sampler = Sampler(model, args.max_len, device)

    print(f"Model loaded. Using {args.device} for inference.")
    print(f"Sampling strategy: {args.sampling_strategy}")
    print("Enter a sentence to translate (or 'quit' to exit):")

    while True:
        sentence = input("> ")
        if sentence.lower() == 'quit':
            break

        translation = translate(model, sampler, src_tokenizer, tgt_tokenizer, sentence, device, args.max_len, args.sampling_strategy)
        print(f"Translation: {translation}")

if __name__ == "__main__":
    main()