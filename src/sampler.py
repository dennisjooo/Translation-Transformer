import lightning as L
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from typing import List, Tuple, Set

class Sampler:
    """
    A class for generating sequences using various sampling strategies.
    
    Implements greedy sampling, random sampling, and beam search with penalties for length,
    repetition and diversity.
    
    Args:
        model (nn.Module): The model used for sequence generation
        max_len (int): Maximum length of generated sequences
        device (torch.device): Device to run inference on. Defaults to 'cpu'
    """

    def __init__(self, model: nn.Module, max_len: int, device: torch.device = 'cpu'):
        """Initialize the sampler with model and parameters."""
        self.model = model
        self.max_len = max_len
        self.device = device
        self.end_token = 2
        self.alpha = 0.6  # Length penalty parameter
        self.n_gram_size = 3  # Size for n-gram repetition detection
        self.repetition_penalty_value = 0.7  # Penalty value for repetitions

    def _prepare_initial_state(self, src: torch.Tensor, tgt: torch.Tensor = None) -> Tuple[torch.Tensor, int]:
        """
        Prepare initial state for sampling.
        
        Args:
            src (torch.Tensor): Source sequence tensor
            tgt (torch.Tensor, optional): Target sequence tensor. Defaults to None.
            
        Returns:
            Tuple[torch.Tensor, int]: Initial target tensor and vocabulary size
            
        Raises:
            ValueError: If input tensor is empty
        """
        if src.numel() == 0:
            raise ValueError("Empty input tensor")
        
        tgt = torch.tensor([[1]]).long().to(self.device) if tgt is None else tgt
        
        # Get vocabulary size
        with torch.no_grad():
            first_logits = self.model(src, tgt)
            vocab_size = first_logits.size(-1)
            
        return tgt, vocab_size

    def _get_length_penalty(self, step: int) -> float:
        """
        Calculate length penalty based on current step.
        
        Args:
            step (int): Current generation step
            
        Returns:
            float: Calculated length penalty
        """
        return ((5 + step + 1) ** self.alpha) / (6 ** self.alpha)

    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits (torch.Tensor): Input logits
            temperature (float): Temperature value for scaling
            
        Returns:
            torch.Tensor: Temperature-scaled logits
        """
        return logits / max(temperature, 1e-5)

    def _get_n_grams(self, tokens: List[int]) -> Set[Tuple[int, ...]]:
        """
        Extract n-grams from token sequence.
        
        Args:
            tokens (List[int]): List of token IDs
            
        Returns:
            Set[Tuple[int, ...]]: Set of n-gram tuples
        """
        n_grams = set()
        for i in range(len(tokens) - self.n_gram_size + 1):
            if i >= 0:
                n_gram = tuple(tokens[i:i + self.n_gram_size])
                n_grams.add(n_gram)
        return n_grams

    def _apply_diversity_penalty(self, logits: torch.Tensor, beams: List[Tuple[torch.Tensor, float]], 
                               beam_idx: int, diversity_penalty: float) -> torch.Tensor:
        """
        Apply diversity penalty based on tokens in other beams.
        
        Args:
            logits (torch.Tensor): Input logits
            beams (List[Tuple[torch.Tensor, float]]): List of beam candidates and scores
            beam_idx (int): Index of current beam
            diversity_penalty (float): Penalty value for repeated tokens
            
        Returns:
            torch.Tensor: Logits with diversity penalty applied
        """
        if diversity_penalty > 0:
            for other_beam_idx, (other_beam, _) in enumerate(beams):
                if other_beam_idx != beam_idx:
                    other_tokens = other_beam.squeeze().tolist()
                    for token in other_tokens:
                        logits[token] -= diversity_penalty
        return logits

    def sample_random(self, src: torch.Tensor, tgt: torch.Tensor = None, top_n: int = None, 
                     temperature: float = 1.0) -> torch.Tensor:
        """
        Generate using random sampling.
        
        Args:
            src (torch.Tensor): Source sequence tensor
            tgt (torch.Tensor, optional): Target sequence tensor. Defaults to None.
            top_n (int, optional): Number of top tokens to sample from. Defaults to None.
            temperature (float, optional): Temperature for sampling. Defaults to 1.0.
            
        Returns:
            torch.Tensor: Generated sequence
        """
        tgt, _ = self._prepare_initial_state(src, tgt)
        batch_size = src.size(0)
        
        for _ in range(self.max_len):
            with torch.no_grad():
                logits = self._apply_temperature(self.model(src, tgt), temperature)

            if top_n is not None:
                logits = logits.masked_fill(logits < torch.topk(logits, top_n)[0][..., -1, None], -1e9)

            pred = torch.multinomial(F.softmax(logits[:, -1], dim=-1), 1)
            tgt = torch.cat([tgt, pred], dim=1)

            if (pred == self.end_token).all():
                break

        return tgt.squeeze() if batch_size == 1 else tgt

    def sample_greedy(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Generate using greedy sampling.
        
        Args:
            src (torch.Tensor): Source sequence tensor
            tgt (torch.Tensor, optional): Target sequence tensor. Defaults to None.
            
        Returns:
            torch.Tensor: Generated sequence
        """
        tgt, _ = self._prepare_initial_state(src, tgt)
        batch_size = src.size(0)
        
        for _ in range(self.max_len):
            with torch.no_grad():
                logits = self.model(src, tgt)
                pred = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                tgt = torch.cat([tgt, pred], dim=1)

            if (pred == self.end_token).all():
                break

        return tgt.squeeze() if batch_size == 1 else tgt

    def _process_beam(self, src: torch.Tensor, beam: torch.Tensor, score: float,
                     beam_idx: int, beams: List[Tuple[torch.Tensor, float]], 
                     beam_n_grams: List[Set[Tuple[int, ...]]], step: int,
                     candidates_per_beam: int, vocab_size: int,
                     temperature: float, diversity_penalty: float) -> List[Tuple[torch.Tensor, float]]:
        """
        Process a single beam and return new candidates.
        
        Args:
            src (torch.Tensor): Source sequence tensor
            beam (torch.Tensor): Current beam sequence
            score (float): Current beam score
            beam_idx (int): Index of current beam
            beams (List[Tuple[torch.Tensor, float]]): List of all beam candidates
            beam_n_grams (List[Set[Tuple[int, ...]]]): N-grams for each beam
            step (int): Current generation step
            candidates_per_beam (int): Number of candidates to generate per beam
            vocab_size (int): Size of vocabulary
            temperature (float): Temperature for sampling
            diversity_penalty (float): Penalty for repeated tokens
            
        Returns:
            List[Tuple[torch.Tensor, float]]: List of new candidates and their scores
        """
        with torch.no_grad():
            logits = self.model(src, beam)
            logits = self._apply_temperature(logits, temperature)
            
        length_penalty = self._get_length_penalty(step)
        last_token_logits = logits[0, -1]
        last_token_logits = self._apply_diversity_penalty(last_token_logits, beams, beam_idx, diversity_penalty)
        
        topk = torch.topk(last_token_logits, min(candidates_per_beam, vocab_size - 1))
        new_candidates = []
        
        for token, token_score in zip(topk.indices, topk.values):
            token = token.item()
            new_beam = torch.cat([beam, torch.tensor([[token]]).long().to(self.device)], dim=1)
            
            # Apply repetition penalty
            tokens = new_beam.squeeze().tolist()
            n_grams = self._get_n_grams(tokens)
            repetition_penalty = (self.repetition_penalty_value 
                                if any(n_gram in beam_n_grams[beam_idx] for n_gram in n_grams)
                                else 1.0)
            
            new_score = (score + token_score.item()) / length_penalty * repetition_penalty
            new_candidates.append((new_beam, new_score))
            
        return new_candidates

    def sample_beam_search(self, src: torch.Tensor, tgt: torch.Tensor = None, 
                         beam_size: int = 5, temperature: float = 1.0,
                         diversity_penalty: float = 0.0) -> torch.Tensor:
        """
        Generate using beam search with various penalties.
        
        Args:
            src (torch.Tensor): Source sequence tensor
            tgt (torch.Tensor, optional): Target sequence tensor. Defaults to None.
            beam_size (int, optional): Size of beam. Defaults to 5.
            temperature (float, optional): Temperature for sampling. Defaults to 1.0.
            diversity_penalty (float, optional): Penalty for repeated tokens. Defaults to 0.0.
            
        Returns:
            torch.Tensor: Generated sequence
        """
        tgt, vocab_size = self._prepare_initial_state(src, tgt)
        batch_size = src.size(0)
        
        # Initialize beam states
        batch_beams = [[(tgt, 0)] for _ in range(batch_size)]
        batch_beam_n_grams = [[set() for _ in range(beam_size)] for _ in range(batch_size)]
        candidates_per_beam = min(beam_size * 2, vocab_size - 1)
        
        for step in range(self.max_len - 1):
            batch_new_beams = []
            
            for batch_idx in range(batch_size):
                new_beams = []
                beams = batch_beams[batch_idx]
                beam_n_grams = batch_beam_n_grams[batch_idx]
                
                for beam_idx, (beam, score) in enumerate(beams):
                    if beam[0, -1].item() == self.end_token and step > 0:
                        new_beams.append((beam, score))
                        continue
                        
                    candidates = self._process_beam(
                        src[batch_idx:batch_idx+1], beam, score, beam_idx,
                        beams, beam_n_grams, step, candidates_per_beam,
                        vocab_size, temperature, diversity_penalty
                    )
                    new_beams.extend(candidates)
                
                # Sort and keep top beams
                batch_new_beams.append(sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size])
            
            # Update beams and n-grams
            for batch_idx in range(batch_size):
                batch_beams[batch_idx] = batch_new_beams[batch_idx]
                for i, (beam, _) in enumerate(batch_beams[batch_idx]):
                    tokens = beam.squeeze().tolist()
                    batch_beam_n_grams[batch_idx][i] = self._get_n_grams(tokens)
            
            # Early stopping check
            if all(all(beam[0, -1].item() == self.end_token for beam, _ in beams) 
                   for beams in batch_beams):
                break
        
        # Handle output dimensions
        if batch_size == 1:
            # For single sequence, return 1D tensor
            return batch_beams[0][0][0].squeeze()
        else:
            # For batch, ensure all sequences have same length by padding
            sequences = [beams[0][0] for beams in batch_beams]
            max_len = max(seq.size(1) for seq in sequences)
            padded_sequences = []
            
            for seq in sequences:
                if seq.size(1) < max_len:
                    padding = torch.zeros(1, max_len - seq.size(1), device=seq.device, dtype=seq.dtype)
                    seq = torch.cat([seq, padding], dim=1)
                padded_sequences.append(seq)
            
            return torch.cat(padded_sequences, dim=0)

    def __call__(self, src: torch.Tensor, tgt: torch.Tensor = None, 
                 sampling_strategy: str = 'greedy', top_n: int = None, 
                 temperature: float = 1.0, beam_size: int = 5,
                 diversity_penalty: float = 0.0) -> torch.Tensor:
        """
        Main interface for sampling.
        
        Args:
            src (torch.Tensor): Source sequence tensor
            tgt (torch.Tensor, optional): Target sequence tensor. Defaults to None.
            sampling_strategy (str, optional): Strategy for sampling - 'greedy', 'random', or 'beam'. Defaults to 'greedy'.
            top_n (int, optional): Number of top tokens to sample from. Defaults to None.
            temperature (float, optional): Temperature for sampling. Defaults to 1.0.
            beam_size (int, optional): Size of beam for beam search. Defaults to 5.
            diversity_penalty (float, optional): Penalty for repeated tokens. Defaults to 0.0.
            
        Returns:
            torch.Tensor: Generated sequence
            
        Raises:
            ValueError: If sampling strategy is invalid
        """
        tgt = None if tgt is None else tgt.long().unsqueeze(0).to(self.device)
        
        strategies = {
            'greedy': lambda: self.sample_greedy(src, tgt),
            'random': lambda: self.sample_random(src, tgt, top_n, temperature),
            'beam': lambda: self.sample_beam_search(src, tgt, beam_size, temperature, diversity_penalty)
        }
        
        if sampling_strategy not in strategies:
            raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
            
        return strategies[sampling_strategy]()

class SamplerCallback(Callback):
    """
    Callback for generating translation samples during training.
    
    Periodically generates and logs translation samples using the model being trained.
    """

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
            
    def sample_translation(self, trainer: L.Trainer, sampling_strategy: str = 'beam', 
                          temperature: float = 1.0, diversity_penalty: float = 0.1):
        """
        Generate a translation sample using the specified sampling strategy.
        
        Args:
            trainer (L.Trainer): The trainer object.
            sampling_strategy (str, optional): The sampling strategy to use. Defaults to 'beam'.
            temperature (float, optional): Temperature for sampling. Higher values increase diversity.
            diversity_penalty (float, optional): Penalty for beam diversity. Higher values encourage diverse beams.
            
        Returns:
            None: If either X_src, X_context, or y_tgt is empty.
            str: The translated text.
        """
        # Get a sample from the validation set
        sample = trainer.datamodule.val_dataset[
            torch.randint(0, len(trainer.datamodule.val_dataset), (1,)).item()
        ]
        
        X_src, X_context, y_tgt = sample
        
        if X_src.size(0) == 0 or X_context.size(0) == 0 or y_tgt.size(0) == 0:
            return
        
        X_src = X_src.long().unsqueeze(0).to(self.device)
        
        _X_context = torch.cat([
            torch.tensor([[1]]).to(self.device), X_context.clone()[:3].to(self.device).unsqueeze(0)
        ], dim=1)
        
        # Generate samples with different parameters
        X_tgt = self.sampler(
            X_src, _X_context, 
            sampling_strategy=sampling_strategy,
            beam_size=8,
            temperature=temperature,
            diversity_penalty=diversity_penalty
        )

        # Decode and log translations
        source = self.src_tokenizer.decode_ids(X_src.squeeze().tolist())
        context = self.tgt_tokenizer.decode_ids(X_context.squeeze().tolist())
        target = self.tgt_tokenizer.decode_ids(y_tgt.squeeze().tolist())
        translation = self.tgt_tokenizer.decode_ids(X_tgt.squeeze().tolist())
        
        print(f"\n=====Step: {trainer.global_step}======")
        print(f"\n ->Source: {source}")
        print(f"\n ->Context: {context}")
        print(f"\n ->Target: {target}")
        print(f"\n ->**Translation (T={temperature:.1f}, D={diversity_penalty:.1f}): {translation}")
