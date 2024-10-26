import lightning as L
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

class Sampler:
    def __init__(self, model: nn.Module, max_len: int, device: torch.device = 'cpu'):
        """
        Initialize the Sampler class.

        Args:
            model (nn.Module): The model used for inference.
            max_len (int): The maximum length of the generated sequence.
            device (torch.device): The device to run the model on. Defaults to 'cpu'.
        """
        self.model = model
        self.max_len = max_len
        self.device = device
        self.end_token = 2  # Define your end token here

    def sample_random(self, src: torch.Tensor, tgt: torch.Tensor = None, top_n: int = None, 
                      temperature: float = 1.0) -> torch.Tensor:
        """
        Generate a random sample using a given model.

        Args:
            src (torch.Tensor): The source tensor.
            tgt (torch.Tensor, optional): The target tensor. Defaults to None.
            top_n (int, optional): The number of top tokens to consider. Defaults to None.
            temperature (float, optional): The temperature for sampling. Defaults to 1.0. 
            Higher values make the sampling more random.
            
        Returns:
            torch.Tensor: The generated sample tensor.
        """
        # Make sure that the temperature is greater than 0
        temperature = max(temperature, 1e-3)
        
        # Assign tgt to device
        tgt = torch.tensor([1]).long().unsqueeze(0).to(self.device) if tgt is None else tgt.squeeze(0)

        for _ in range(self.max_len):
            logits = self.model(src, tgt) / temperature

            if top_n is not None:
                logits = logits.masked_fill(logits < torch.topk(logits, top_n)[0][..., -1, None], -1e9)

            pred = torch.multinomial(F.softmax(logits[:, -1], dim=-1), 1).item()
            tgt = torch.cat([tgt, torch.tensor([pred]).long().unsqueeze(0).to(self.device)], dim=-1)

            if pred == self.end_token:
                break

        return tgt

    def sample_greedy(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Generate a sample using a greedy sampling strategy.

        Args:
            src (torch.Tensor): The source tensor.
            tgt (torch.Tensor, optional): The target tensor. Defaults to None.
            
        Returns:
            torch.Tensor: The generated sample tensor.
        """
        tgt = torch.tensor([1]).long().unsqueeze(0).to(self.device) if tgt is None else tgt.squeeze(0)

        for _ in range(self.max_len):
            logits = self.model(src, tgt)
            pred = torch.argmax(logits[:, -1], dim=-1).item()
            tgt = torch.cat([tgt, torch.tensor([pred]).long().unsqueeze(0).to(self.device)], dim=-1)

            if pred == self.end_token:
                break

        return tgt

    def sample_beam_search(self, src: torch.Tensor, tgt: torch.Tensor = None, 
                           beam_size: int = 5) -> torch.Tensor:
        """
        Perform beam search to generate a sequence of tokens.

        Args:
            src (torch.Tensor): The source tensor.
            tgt (torch.Tensor, optional): The target tensor. Defaults to None.
            beam_size (int, optional): The size of the beam. Defaults to 5.
            
        Returns:
            torch.Tensor: The generated sequence of tokens.
        """
        tgt = torch.tensor([1]).long().unsqueeze(0).to(self.device) if tgt is None else tgt.squeeze(0)
        beams = [(tgt, 0)]

        for _ in range(self.max_len):
            new_beams = []

            for beam, score in beams:
                logits = self.model(src, beam).squeeze(0)
                topk = torch.topk(logits[-1], beam_size)

                for i in range(beam_size):
                    token = topk.indices[i]
                    token_score = topk.values[i]
                    new_beam = torch.cat([beam, torch.tensor([token]).long().unsqueeze(0).to(self.device)], dim=-1)
                    new_beams.append((new_beam, score + token_score.item()))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            if beams[0][0][0][-1].item() == self.end_token:
                break

        return beams[0][0]
    
    def __call__(self, src: torch.Tensor, tgt: torch.Tensor = None, 
                 sampling_strategy: str = 'greedy', top_n: int = None, 
                 temperature: float = 1.0, beam_size: int = 5) -> torch.Tensor:
        """
        Generate a sample using the specified sampling strategy.

        Args:
            src (torch.Tensor): The source tensor.
            tgt (torch.Tensor, optional): The target tensor. Defaults to None.
            sampling_strategy (str, optional): The sampling strategy to use. Defaults to 'greedy'.
            top_n (int, optional): The number of top tokens to consider. Only used when sampling_strategy is 'random'. 
            Defaults to None.
            temperature (float, optional): The temperature for sampling. Only used when sampling_strategy is 'random'. 
            Defaults to 1.0.
            beam_size (int, optional): The size of the beam. Only used when sampling_strategy is 'beam'. Defaults to 5.
            
        Returns:
            torch.Tensor: The generated sample tensor.
        """
        # Assign tgt to device
        tgt = None if tgt is None else tgt.long().unsqueeze(0).to(self.device)
        
        # Perform sampling based on the sampling strategy
        if sampling_strategy == 'greedy':
            return self.sample_greedy(src, tgt)
        elif sampling_strategy == 'random':
            return self.sample_random(src, tgt, top_n, temperature)
        elif sampling_strategy == 'beam':
            return self.sample_beam_search(src, tgt, beam_size)
        else:
            raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")

class SamplerCallback(Callback):
    def __init__(self, model: nn.Module, src_tokenizer: spm.SentencePieceProcessor, 
                 tgt_tokenizer: spm.SentencePieceProcessor, max_len: int = 30, every_n_steps: int = 100):
        """
        Initialize the SamplerCallback class.
        
        Args:
            model (nn.Module): The model used for inference.
            src_tokenizer (spm.SentencePieceProcessor): The source tokenizer.
            tgt_tokenizer (spm.SentencePieceProcessor): The target tokenizer.
            max_len (int): The maximum length of the generated sequence. Defaults to 30.
            every_n_steps (int): The number of steps to wait before generating a sample. Defaults to 100.
        """
        # Initialize the callback
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.every_n_steps = every_n_steps
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampler = Sampler(self.model, self.max_len, self.device)
        
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, 
                           outputs: torch.Tensor, batch: torch.Tensor, batch_idx: int):
        """
        Callback function called at the end of each training batch.
        
        Args:
            trainer (L.Trainer): The Lightning Trainer object.
            pl_module (L.LightningModule): The LightningModule being trained.
            outputs (torch.Tensor): The output tensor from the forward pass.
            batch (torch.Tensor): The input batch tensor.
            batch_idx (int): The index of the current batch.
        """
        # Check if the global step is divisible by every_n_steps
        if trainer.global_step % self.every_n_steps == 0:
            self.model.eval()
            self.sample_translation(trainer)
            self.model.train()
            
    def sample_translation(self, trainer: L.Trainer, sampling_strategy: str = 'beam'):
        """
        Generate a translation sample using the specified sampling strategy.
        Args:
            trainer (L.Trainer): The trainer object.
            sampling_strategy (str, optional): The sampling strategy to use. Defaults to 'beam'.
        Returns:
            None: If either X_src, X_context, or y_tgt is empty.
            str: The translated text.
        """
        # Get a sample from the validation set
        sample = trainer.datamodule.val_dataset[
            torch.randint(0, len(trainer.datamodule.val_dataset), (1,)).item()
        ]
        
        # Get the source and target tensors
        X_src, X_context, y_tgt = sample
        
        # If either X_src, X_context, y_tgt is empty return
        if X_src.size(0) == 0 or X_context.size(0) == 0 or y_tgt.size(0) == 0:
            return
        
        # Assign X_src to device
        X_src = X_src.long().unsqueeze(0).to(self.device)
        
        # Create some partial context
        _X_context = torch.cat([
            torch.tensor([[1]]).to(self.device), X_context.clone()[:3].to(self.device).unsqueeze(0)
        ], dim=1)
        
        # Generate a sample using beam search
        X_tgt = self.sampler(X_src, _X_context, sampling_strategy, beam_size=8)
    
        # Decode the target
        source = self.src_tokenizer.decode_ids(X_src.squeeze().tolist())
        context = self.tgt_tokenizer.decode_ids(X_context.squeeze().tolist())
        target = self.tgt_tokenizer.decode_ids(y_tgt.squeeze().tolist())
        translation = self.tgt_tokenizer.decode_ids(X_tgt.squeeze().tolist())
        
        # Log the translation
        print(f"\n=====Step: {trainer.global_step}======")
        print(f"\n ->Source: {source}")
        print(f"\n ->Context: {context}")
        print(f"\n ->Target: {target}")
        print(f"\n ->**Translation: {translation}")
