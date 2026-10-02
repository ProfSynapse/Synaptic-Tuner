"""chat_template_kwargs passthrough for GRPO prompt formatting."""

from datasets import Dataset

from tests.trainers._trainer_import import load_trainer_module

data_loader = load_trainer_module("grpo", "data_loader")


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return f"formatted prompt | enable_thinking={kwargs.get('enable_thinking')}"


def test_format_dataset_for_grpo_forwards_chat_template_kwargs():
    tokenizer = RecordingTokenizer()
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": "Question?"}],
                "label": "known",
            }
        ]
    )

    formatted = data_loader.format_dataset_for_grpo(
        dataset,
        tokenizer=tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert formatted[0]["prompt"] == "formatted prompt | enable_thinking=False"


def test_format_dataset_for_grpo_skips_template_for_preformatted_prompt():
    tokenizer = RecordingTokenizer()
    dataset = Dataset.from_list([{"prompt": "already formatted", "label": "known"}])

    formatted = data_loader.format_dataset_for_grpo(
        dataset,
        tokenizer=tokenizer,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert formatted[0]["prompt"] == "already formatted"
    assert tokenizer.calls == []
