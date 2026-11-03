"""Load a text-generating Transformers auto model without naming architectures."""
from __future__ import annotations


def load_text_generation_model(model_name, **kwargs):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = kwargs.pop("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(model_name, **{
            key: kwargs[key] for key in ("revision", "token", "trust_remote_code", "cache_dir", "local_files_only", "subfolder")
            if key in kwargs
        })
    loader = AutoModelForCausalLM
    if type(config) not in loader._model_mapping:
        # Text-only calls can also use registered conditional-generation models.
        # Other unsupported/custom configs retain AutoModelForCausalLM's normal
        # dispatch and refusal behavior; weight-loading errors are never retried.
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            pass
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping:
                loader = AutoModelForImageTextToText
    return loader.from_pretrained(model_name, config=config, **kwargs)
