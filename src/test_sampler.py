import torch
from model import Transformer
from sampler import Sampler

def test_sampler():
    # Define model parameters
    n_embed = 512
    n_head = 8
    n_hidden = 2048
    n_layers = 6
    vocab_size = 10000
    max_len = 100
    pad_idx = 0

    # Initialize the Transformer model
    model = Transformer(n_embed, n_head, n_hidden, n_layers, vocab_size, max_len, pad_idx)

    # Create a Sampler instance
    sampler = Sampler(model, max_len=max_len)  # Assuming 2 is the end token

    # Create a dummy input tensor
    src = torch.randint(1, vocab_size, (1, 10))  # Batch size 1, sequence length 10

    print("Testing Greedy Sampling:")
    greedy_output = sampler.sample_greedy(src)
    print(f"Greedy Output: {greedy_output}")

    print("\nTesting Beam Search:")
    beam_output = sampler.sample_beam_search(src, beam_size=3)
    print(f"Beam Search Output: {beam_output}")

if __name__ == "__main__":
    test_sampler()
