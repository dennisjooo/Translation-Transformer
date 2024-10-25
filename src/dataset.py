import lightning as L
import os
import pandas as pd
import re
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Union

class TranslateDataset(Dataset):
    """
    A dataset for machine translation tasks.

    This dataset handles tokenization, padding, and special token insertion for
    both source and target sequences.
    """

    def __init__(self, data: pd.DataFrame, src_tokenizer: spm.SentencePieceProcessor, 
                 tgt_tokenizer: spm.SentencePieceProcessor, start_token_id: int, 
                 end_token_id: int, padding_value: int = 0, max_len: int = 512):
        """
        Initialize the TranslateDataset.

        Args:
            data (pd.DataFrame): The dataset containing source and target sentences.
            src_tokenizer (spm.SentencePieceProcessor): Tokenizer for source language.
            tgt_tokenizer (spm.SentencePieceProcessor): Tokenizer for target language.
            start_token_id (int): ID of the start token.
            end_token_id (int): ID of the end token.
            padding_value (int, optional): Value used for padding. Defaults to 0.
            max_len (int, optional): Maximum length of sequences. Defaults to 512.
        """
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.start_token = start_token_id
        self.end_token = end_token_id
        self.padding_value = padding_value
        self.max_len = max_len

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Source sequence, target input sequence, 
            and target output sequence.
        """
        X_src = self.src_tokenizer.encode(self.data.iloc[idx]['src'])[:self.max_len]
        tgt = self.tgt_tokenizer.encode(self.data.iloc[idx]['tgt'])[:self.max_len - 1] 
        
        y_tgt = tgt[1:].copy() + [self.end_token]
        X_tgt = [self.start_token] + tgt[:-1].copy()
        
        return (torch.tensor(X_src).long(), 
                torch.tensor(X_tgt).long(), 
                torch.tensor(y_tgt).long())
    
    def collate_fn(self, 
                   batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function to create batches.

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): List of individual samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Batched and padded tensors.
        """
        X_src, X_tgt, y_tgt = zip(*batch)
        X_src = torch.nn.utils.rnn.pad_sequence(X_src, batch_first=True, padding_value=self.padding_value)
        X_tgt = torch.nn.utils.rnn.pad_sequence(X_tgt, batch_first=True, padding_value=self.padding_value)
        y_tgt = torch.nn.utils.rnn.pad_sequence(y_tgt, batch_first=True, padding_value=self.padding_value)
        
        return X_src, X_tgt, y_tgt

class DataModule(L.LightningDataModule):
    """
    A LightningDataModule for handling translation datasets.

    This module handles data loading, preprocessing, and creation of DataLoaders
    for both training and validation sets.
    """

    def __init__(self, data_path: str, max_vocab: int, batch_size: int, max_len: int = 512, 
                 num_workers: int = 4, split_size: float = 0.8,
                 src_tokenizer_path: Union[str, None] = None, 
                 tgt_tokenizer_path: Union[str, None] = None):
        """
        Initialize the DataModule.

        Args:
            data_path (str): Path to the data file.
            max_vocab (int): Maximum vocabulary size.
            batch_size (int): Batch size for DataLoaders.
            max_len (int, optional): Maximum sequence length. Defaults to 512.
            num_workers (int, optional): Number of workers for DataLoaders. Defaults to 4.
            split_size (float, optional): Train/val split ratio. Defaults to 0.8.
            src_tokenizer_path (Union[str, None], optional): Path to source tokenizer. Defaults to None.
            tgt_tokenizer_path (Union[str, None], optional): Path to target tokenizer. Defaults to None.
        """
        super().__init__()
        
        self.data_path = data_path
        self.max_vocab = max_vocab
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.src_tokenizer_path = src_tokenizer_path
        self.tgt_tokenizer_path = tgt_tokenizer_path
        self.split_size = split_size
        
        self.compile_cleanup_regex()
        
    def set_tokenizers(self, src_tokenizer_path: str, tgt_tokenizer_path: str) -> None:
        """
        Set or train tokenizers.

        Args:
            src_tokenizer_path (str): Path to source tokenizer.
            tgt_tokenizer_path (str): Path to target tokenizer.
        """
        if not os.path.exists(src_tokenizer_path):
            self.train_tokenizers(self.data['src'], src_tokenizer_path)
        if not os.path.exists(tgt_tokenizer_path):
            self.train_tokenizers(self.data['tgt'], tgt_tokenizer_path)
        
        self.src_tokenizer = spm.SentencePieceProcessor(model_file=src_tokenizer_path)
        self.tgt_tokenizer = spm.SentencePieceProcessor(model_file=tgt_tokenizer_path)
    
    def compile_cleanup_regex(self) -> None:
        """Compile regex patterns for text cleanup."""
        self.remove_multiple_spaces_dots = re.compile(r'[\.\s]{2,}')
        self.remove_punctuation = re.compile(r'[^\w\s\d\']')
        
    def text_cleanup(self, text: str) -> str:
        """
        Clean up the input text.

        Args:
            text (str): Input text to clean.

        Returns:
            str: Cleaned text.
        """
        text = text.lower().strip()
        text = self.remove_punctuation.sub(' ', text)
        text = self.remove_multiple_spaces_dots.sub(' ', text)
        return text
    
    def prepare_data(self, max_data: int = int(3.5e6)) -> None:
        """
        Prepare the dataset.

        Args:
            max_data (int, optional): Maximum number of data points to use. Defaults to int(3.5e6).
        """
        self.data = pd.read_csv(self.data_path, nrows=max_data)
        self.data = self.data.dropna()
        self.data.columns = ['src', 'tgt']
        self.data['src'] = self.data['src'].apply(self.text_cleanup)
        self.data['tgt'] = self.data['tgt'].apply(self.text_cleanup)
        self.set_tokenizers(self.src_tokenizer_path, self.tgt_tokenizer_path)
        
    def setup(self, stage: str) -> None:
        """
        Set up the datasets for the given stage.

        Args:
            stage (str): Current stage ('fit', 'validate', 'test', or 'predict').
        """
        self.train_data = self.data.sample(frac=self.split_size)
        self.val_data = self.data.drop(self.train_data.index)
        
        self.train_dataset = TranslateDataset(self.train_data, self.src_tokenizer, self.tgt_tokenizer, 
                                              1, 2, max_len=self.max_len)
        self.val_dataset = TranslateDataset(self.val_data, self.src_tokenizer, self.tgt_tokenizer, 
                                            1, 2, max_len=self.max_len)
        
    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for training data."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self.train_dataset.collate_fn)
        
    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for validation data."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.val_dataset.collate_fn)
        
    def train_tokenizers(self, text_list: List[str], tokenizer_path: str) -> None:
        """
        Train a SentencePiece tokenizer.

        Args:
            text_list (List[str]): List of texts to train on.
            tokenizer_path (str): Path to save the trained tokenizer.
        """
        tokenizer_path = tokenizer_path.split('.')[0]
        
        # Temporary file to write the string to
        with open('tmp.txt', 'w') as f:
            # Write the data to the file in chunks of 5 entries per line
            for i in range(0, len(text_list), 5):
                chunk = text_list[i:i+5]
                f.write('\t'.join(chunk.astype(str)) + '\n')
            
        # Train the tokenizer
        spm.SentencePieceTrainer.train(input='tmp.txt', model_prefix=tokenizer_path, 
                                       vocab_size=self.max_vocab, model_type='bpe',
                                       bos_id=1, eos_id=2, pad_id=0, unk_id=3,
                                       bos_piece='<s>', eos_piece='</s>', 
                                       pad_piece='<pad>', unk_piece='<unk>',
                                       normalization_rule_name="nmt_nfkc",
                                       byte_fallback=True,
                                       remove_extra_whitespaces=True,
                                       shuffle_input_sentence=True,
                                       character_coverage=0.9999,
                                       split_digits=True,
                                       split_by_unicode_script=True,
                                       split_by_whitespace=True,
                                       split_by_number=True,
                                       add_dummy_prefix=True)
        
        os.remove('tmp.txt')