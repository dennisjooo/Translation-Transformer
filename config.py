config = {
    # Data and preprocessing
    "data_path": "/kaggle/input/language-translation-englishfrench/eng_-french.csv",
    "max_vocab": 9999,
    "max_len": 1024,
    "src_tokenizer_path": "tokenizer/en_tokenizer.model",
    "tgt_tokenizer_path": "tokenizer/fr_tokenizer.model",
    "max_data": 2**17,
    "split_size": 0.8,
    
    # Model architecture
    "n_embed": 1024,
    "n_head": 8,
    "n_hidden": 1024 * 4,
    "n_layers": 4,
    "dropout_p": 0.15,
    
    # Training parameters
    "batch_size": 16,
    "num_workers": 2,
    "lr": 2e-4,
    "padding_value": 0,
    "max_epochs": 50,
    "grad_accum_steps": 4,
    "lambda_val": 5e-4
}
