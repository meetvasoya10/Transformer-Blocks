import pytest
import torch
import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_blocks import TokenEmbedding

# Fixture with explicit device index
@pytest.fixture
def device():
    # Use "cuda:0" explicitly if CUDA is available to match PyTorch's default behavior
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def token_emb(device):
    return TokenEmbedding(vocab_size=100, d_model=256, padding_idx=0, device=device)

def test_init_valid(token_emb):
    assert token_emb.vocabulary_size == 100
    assert token_emb.embedding_dim == 256
    assert token_emb.embedding.padding_idx == 0
    # Compare device type and index explicitly
    assert token_emb.embedding.weight.device.type == token_emb.device.type
    if torch.cuda.is_available():
        assert token_emb.embedding.weight.device.index == token_emb.device.index

def test_init_invalid_vocab_size():
    with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
        TokenEmbedding(vocab_size=0, d_model=256)
    with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
        TokenEmbedding(vocab_size=-1, d_model=256)

def test_init_invalid_d_model():
    with pytest.raises(ValueError, match="d_model must be a positive integer"):
        TokenEmbedding(vocab_size=100, d_model=0)
    with pytest.raises(ValueError, match="d_model must be a positive integer"):
        TokenEmbedding(vocab_size=100, d_model=-1)

def test_forward_valid(token_emb, device):
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long, device=device)
    output = token_emb(x)
    assert output.shape == (2, 3, 256)
    raw_embedding = token_emb.embedding(x)
    scaled = raw_embedding * math.sqrt(256)
    assert torch.allclose(output, scaled, rtol=1e-5)
    assert torch.allclose(output[0, 0], token_emb.embedding.weight[1] * math.sqrt(256), rtol=1e-5)

def test_forward_padding(token_emb, device):
    x = torch.tensor([[0, 1, 2], [3, 0, 4]], dtype=torch.long, device=device)
    output = token_emb(x)
    pad_embedding = token_emb.embedding.weight[0] * math.sqrt(256)
    assert torch.allclose(output[0, 0], pad_embedding, rtol=1e-5)
    assert torch.allclose(output[1, 1], pad_embedding, rtol=1e-5)

def test_forward_invalid_type(token_emb):
    with pytest.raises(ValueError, match="Input must be a PyTorch tensor"):
        token_emb([1, 2, 3])

def test_forward_invalid_dtype(token_emb, device):
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float, device=device)
    with pytest.raises(ValueError, match="Input tensor must be of integer type"):
        token_emb(x)

def test_forward_out_of_range(token_emb, device):
    x = torch.tensor([[1, 100]], dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="Token indices must be in range"):
        token_emb(x)
    x = torch.tensor([[1, -1]], dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="Token indices must be in range"):
        token_emb(x)

def test_forward_large_batch(token_emb, device):
    x = torch.randint(0, 100, (16, 32), dtype=torch.long, device=device)
    output = token_emb(x)
    assert output.shape == (16, 32, 256)
    # Compare device type and index explicitly
    assert output.device.type == device.type
    if torch.cuda.is_available():
        assert output.device.index == device.index

if __name__ == "__main__":
    result = pytest.main(["-v", __file__])
    if result == 0:
        print("\nWorking Perfectly")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print("GPU Name: ", torch.cuda.get_device_name())
    else:
        print("\nSome tests failed. Please check the output above for details.")