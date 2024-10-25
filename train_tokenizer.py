import os
from typing import List
import pandas as pd
import sentencepiece as spm
from pandas import Series

from config import config

def train_tokenizers(text_list: Series, tokenizer_path: str) -> None:
    """
    Train a SentencePiece tokenizer on the given text data.
    
    Args:
        text_list (pandas.Series): Series containing text data to train the tokenizer on
        tokenizer_path (str): Path where the trained tokenizer model should be saved
        
    Returns:
        None
    """
    # Cleanup the tokenizer_path
    tokenizer_path = tokenizer_path.split('.')[0]
    
    # Temporary file to write the string to
    with open('tmp.txt', 'w') as f:
        # Write the data to the file in chunks of 5 entries per line
        for i in range(0, len(text_list), 5):
            chunk = text_list.iloc[i:i+5]
            chunk = chunk.astype(str)
            f.write('\t'.join(chunk) + '\n')
        
    # Train the tokenizer
    spm.SentencePieceTrainer.train(input='tmp.txt', model_prefix=tokenizer_path, 
                                    vocab_size=config['max_vocab'], model_type='bpe',
                                    bos_id=1, eos_id=2, pad_id=0, unk_id=3,
                                    bos_piece='<s>', eos_piece='</s>', 
                                    pad_piece='<pad>', unk_piece='<unk>',
                                    normalization_rule_name="nmt_nfkc",
                                    remove_extra_whitespaces=True,
                                    shuffle_input_sentence=True,
                                    character_coverage=0.9999,
                                    split_digits=True,
                                    split_by_unicode_script=True,
                                    split_by_whitespace=True,
                                    split_by_number=True,
                                    add_dummy_prefix=True)
    
    # Remove the temporary file
    os.remove('tmp.txt')

def main() -> None:
    """
    Main function to train tokenizers for source and target languages.
    Checks if tokenizers already exist and trains new ones if needed.
    
    Returns:
        None
    """
    # Define tokenizer paths
    src_tokenizer_path = config['src_tokenizer_path']
    tgt_tokenizer_path = config['tgt_tokenizer_path']
    
    # Create a tokenizer directory if it doesn't exist
    os.makedirs(os.path.dirname(src_tokenizer_path), exist_ok=True)
    
    # Check if tokenizers need to be trained
    need_training = False
    if not os.path.exists(src_tokenizer_path) or not os.path.exists(tgt_tokenizer_path):
        need_training = True
        print("Tokenizer files not found. Training new tokenizers...")
    
    if need_training:
        # Load and prepare the dataset
        df = pd.read_csv(config['data_path'], nrows=config['max_data'])
        df.rename(columns={'en': 'src', 'fr': 'tgt'}, inplace=True)
        
        # Train tokenizers
        print("Training source tokenizer...")
        train_tokenizers(df['src'], src_tokenizer_path.split('.')[0])
        print("Training target tokenizer...")
        train_tokenizers(df['tgt'], tgt_tokenizer_path.split('.')[0])
        print("Tokenizer training completed!")
    else:
        print("Tokenizers already exist, skipping training.")

if __name__ == "__main__":
    main()
