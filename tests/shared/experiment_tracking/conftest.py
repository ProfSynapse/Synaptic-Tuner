import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

@pytest.fixture
def tracking_fixtures_dir() -> Path:
    return Path(__file__).parent.parent.parent / "fixtures" / "tracking"

@pytest.fixture
def sample_dataset_path(tracking_fixtures_dir, tmp_path) -> Path:
    """Provides a small JSONL dataset for testing."""
    path = tmp_path / "sample_training_data.jsonl"
    examples = [
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
        {"conversations": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I am good, thanks."}]},
        {"messages": [{"role": "user", "content": "Calculate 2+2"}, {"role": "assistant", "content": "It is 4."}]},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return path

@pytest.fixture
def sample_losses_a_path(tmp_path) -> Path:
    path = tmp_path / "sample_losses_a.jsonl"
    losses = [
        {"index": 0, "loss": 1.2, "num_completion_tokens": 10, "num_total_tokens": 20, "jsonl_hash": "aaaa1111"},
        {"index": 1, "loss": 1.5, "num_completion_tokens": 15, "num_total_tokens": 30, "jsonl_hash": "bbbb2222"},
        {"index": 2, "loss": 0.8, "num_completion_tokens": 12, "num_total_tokens": 25, "jsonl_hash": "cccc3333"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for ex in losses:
            f.write(json.dumps(ex) + "\n")
    return path

@pytest.fixture
def sample_losses_b_path(tmp_path) -> Path:
    path = tmp_path / "sample_losses_b.jsonl"
    losses = [
        {"index": 0, "loss": 1.1, "num_completion_tokens": 10, "num_total_tokens": 20, "jsonl_hash": "aaaa1111"},
        {"index": 1, "loss": 1.6, "num_completion_tokens": 15, "num_total_tokens": 30, "jsonl_hash": "bbbb2222"},
        {"index": 2, "loss": 0.9, "num_completion_tokens": 12, "num_total_tokens": 25, "jsonl_hash": "cccc3333"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for ex in losses:
            f.write(json.dumps(ex) + "\n")
    return path

@pytest.fixture
def fake_base_losses_path(tmp_path) -> Path:
    path = tmp_path / "base_losses.jsonl"
    losses = [
        {"index": 0, "loss": 2.0, "num_completion_tokens": 10, "num_total_tokens": 20, "jsonl_hash": "aaaa1111"},
        {"index": 1, "loss": 2.5, "num_completion_tokens": 15, "num_total_tokens": 30, "jsonl_hash": "bbbb2222"},
        {"index": 2, "loss": 1.8, "num_completion_tokens": 12, "num_total_tokens": 25, "jsonl_hash": "cccc3333"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for ex in losses:
            f.write(json.dumps(ex) + "\n")
    return path


class FakeTokenizer:
    def __init__(self):
        self.eos_token = "<eos>"
        self.vocab_size = 32000

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_tensors=None):
        text = ""
        for m in messages:
            text += m["content"] + self.eos_token
        if not tokenize:
            return text
        
        # Fake tokenization
        tokens = [1, 2, 3] * len(messages)
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens
        
    def __call__(self, text, return_tensors=None, **kwargs):
        tokens = [1, 2, 3] * min(len(text), 10)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.tensor([[1]*len(tokens)])}
        return {"input_ids": tokens}

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        if text == "<|im_start|>assistant\n":
            tokens = [100, 101, 102]
        else:
            tokens = [1, 2]
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens


class FakeModel:
    def __init__(self):
        self.device = "cpu"
        self._param = torch.tensor(1.0)

    def parameters(self):
        return iter([self._param])

    def eval(self):
        pass

    def __call__(self, input_ids, labels=None, attention_mask=None, **kwargs):
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        # Return random logits
        logits = torch.randn(batch_size, seq_len, 32000)
        # Random loss
        loss = torch.tensor(1.234)
        return SimpleNamespace(logits=logits, loss=loss)


@pytest.fixture
def fake_model():
    return FakeModel()

@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
