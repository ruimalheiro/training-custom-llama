import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from tokenizer import init_tokenizer
from model import Transformer, ModelConfig


@pytest.fixture(scope='session')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope='session')
def tokenizer():
    return init_tokenizer('HuggingFaceTB/SmolLM2-360M', True)

@pytest.fixture(scope='session')
def dummy_prompt_tokens(tokenizer):
    return tokenizer.encode('Language models are')

@pytest.fixture(scope='session')
def model(device, tokenizer):
    model_config = ModelConfig(
        dim=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        multiple_of=128,
        ffn_dim_multiplier=1.0,
        norm_eps=1e-05,
        rope_theta=500000.0,
        max_batch_size=2,
        max_seq_len=32,
        tokenizer = tokenizer,
        vocab_size = tokenizer.vocab_size,
        pad_token_id = tokenizer.pad_id,
        stop_tokens = tokenizer.stop_tokens
    )

    model = Transformer(model_config)
    model.to(device)
    model.eval()
    return model
