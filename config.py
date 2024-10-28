import os

config = {
    # Data and preprocessing
    "max_vocab": 16384,
    "max_len": 1024,
    "src_tokenizer_path": "tokenizer/en_tokenizer.model",
    "tgt_tokenizer_path": "tokenizer/fr_tokenizer.model",
    "max_data": 2007723,
    "split_size": 0.9,
    'data_folder': os.path.join(os.path.dirname(__file__), 'data'),
    'data_path': os.path.join(os.path.dirname(__file__), 'data', 'en_fr_dataset.csv'),
    
    # Model architecture
    "n_embed": 512,
    "n_head": 16,
    "n_hidden": 512 * 4,
    "n_layers": 6,
    "dropout_p": 0.15,
    
    # Training parameters
    "batch_size": 16,
    "num_workers": 16,
    "lr": 1e-4,
    "padding_value": 0,
    "max_epochs": 5,
    "grad_accum_steps": 4,
    "lambda_val": 1e-4,
    "precision": "16-mixed",
    "warmup_steps": 5000
}
