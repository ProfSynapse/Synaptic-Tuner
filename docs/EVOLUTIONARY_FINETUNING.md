# Evolutionary Fine-Tuning

Design document for integrating evolutionary algorithms into the fine-tuning pipeline.

## Table of Contents

1. [Philosophy](#philosophy)
2. [Problem Statement](#problem-statement)
3. [Solution Overview](#solution-overview)
4. [Architecture](#architecture)
5. [Config-Driven Design](#config-driven-design)
6. [Implementation Guide](#implementation-guide)
7. [Pseudocode](#pseudocode)

---

## Philosophy

### Core Principles

1. **Config-Driven Everything**
   - Developers configure behavior through YAML, not code changes
   - New use cases = new config files, not new Python modules
   - Same config format across all systems (SynthChat, Training, Evaluation)

2. **No Code Duplication**
   - Shared infrastructure lives in `shared/`
   - Both SynthChat and Trainers import from the same source
   - Single source of truth for validation logic

3. **Maximum Flexibility**
   - No hardcoded field names or schemas
   - Validation rules are data, not code
   - Strategy parameters are passed through, not interpreted

4. **Compute Efficiency**
   - Evolutionary selection adds minimal overhead
   - Fitness evaluation uses schema validation, not LLM calls
   - Single backward pass per step, multiple cheap forward passes

---

## Problem Statement

### The Challenge with Tool Internalization

Training a model to use tools correctly involves:
- **Discrete decisions**: Which tool to call (gradient signal is noisy)
- **Structured output**: Exact JSON format required (partial credit is meaningless)
- **Pattern matching**: Intent → tool mapping (many valid paths)

Standard gradient descent struggles because:
```
Gradient says: "Adjust weights in this direction"
But we don't know if that direction actually improves tool selection
We just apply it and hope
```

### What We Want

Evolutionary selection pressure **during training**:
```
Gradient says: "Here's a direction"
We generate multiple candidates around that direction
We TEST each candidate: "Does this actually improve tool calling?"
We apply only the winning candidate
```

### Constraints

- **24GB VRAM** for a 2B model
- **No LLM judge calls** during training (too expensive)
- **Config-driven** fitness evaluation
- **Reuse existing infrastructure** (validators, rubrics)

---

## Solution Overview

### Approach: Gradient + Evolution Strategy Hybrid (C-Lite)

Instead of parallel training runs or post-hoc selection, we do **micro-evolution within each training step**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SINGLE TRAINING STEP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FORWARD + BACKWARD (standard, done once)                    │
│     Input → Model → Loss → Gradient G                          │
│                                                                 │
│  2. CANDIDATE GENERATION (new)                                  │
│     G + noise₁ → Candidate 1                                   │
│     G + noise₂ → Candidate 2                                   │
│     G × scale  → Candidate 3                                   │
│     G (pure)   → Candidate 4                                   │
│                                                                 │
│  3. FITNESS EVALUATION (new, cheap)                             │
│     For each candidate:                                         │
│       - Temporarily apply to weights                            │
│       - Forward pass on eval batch                              │
│       - Schema validation (no LLM) → fitness score              │
│       - Revert weights                                          │
│                                                                 │
│  4. SELECTION (new)                                             │
│     Apply only the highest-fitness candidate                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works

- **Backward pass**: Done once (expensive, ~3-4x forward)
- **Forward passes**: Done N times (cheap, parallelizable)
- **Fitness evaluation**: Schema validation, ~0.1ms per example
- **Total overhead**: ~3x slower per step, but better gradient quality

### Fitness Without LLM

Tool calls are structured data. Validation is deterministic:

```
Level 1: Format valid?     → Contains "tool_call:", valid JSON
Level 2: Tool valid?       → Tool name in manifest
Level 3: Schema valid?     → Required fields present, correct types
Level 4: Context aligned?  → IDs match expected values
```

All levels use existing `StructureValidator` with config-driven rules.

---

## Architecture

### Current State

```
SynthChat/
└── services/
    └── validators/           # Validation logic (stuck in SynthChat)
        ├── structure_validator.py
        ├── cross_scope_validator.py
        └── content/
            ├── json_validator.py
            └── xml_validator.py

Trainers/
└── rtx3090_sft/
    └── train_sft.py          # No evolutionary support
```

### Target State

```
shared/
├── validation/               # MOVED from SynthChat
│   ├── structure_validator.py
│   ├── cross_scope_validator.py
│   ├── content/
│   │   ├── json_validator.py
│   │   └── xml_validator.py
│   ├── fitness.py            # NEW: Thin wrapper → 0.0-1.0 score
│   └── configs/              # Validation configs (same format as rubrics)
│       └── tool_calling.yaml
│
└── evolutionary/             # NEW: Evolutionary training components
    ├── candidate_generator.py
    ├── strategies/
    │   ├── base.py
    │   ├── gradient_noise.py
    │   ├── scale_variation.py
    │   └── component_dropout.py
    └── trainer_wrapper.py

SynthChat/
└── services/
    └── validators/
        └── __init__.py       # Just: from shared.validation import *

Trainers/
└── rtx3090_sft/
    ├── train_sft.py          # Wraps trainer with evolutionary if enabled
    └── configs/
        ├── config.yaml       # Add evolutionary section
        └── fitness/
            └── tool_calling.yaml
```

### Dependency Flow

```
shared/validation/
       ↑
       ├──────────────────┐
       │                  │
SynthChat/services/   Trainers/rtx3090_sft/
       │                  │
       │                  ↓
       │           shared/evolutionary/
       │                  │
       └──────────────────┘
```

---

## Config-Driven Design

### Training Config (`config.yaml`)

Minimal additions to existing config:

```yaml
# Existing training config...
model:
  model_name: "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

training:
  learning_rate: 2e-4
  # ... existing fields ...

# NEW: Evolutionary section
evolutionary:
  enabled: false                    # Toggle on/off

  # Path to validation config (same format as rubrics)
  validation_config: "configs/fitness/tool_calling.yaml"

  # Number of candidates per step
  candidates: 4

  # Eval batch size for fitness evaluation
  eval_batch_size: 4

  # Strategy configuration (generic, passed through)
  strategy:
    type: "gradient_noise"          # Strategy name (looked up in registry)
    params:                         # Passed directly to strategy
      noise_scale: 0.1

  # Scoring configuration (generic, passed through)
  scoring:
    method: "error_count"           # binary | error_count | weighted
    params:
      base_score: 0.2
      max_errors_before_zero: 5
```

### Validation Config (Same Format as Rubrics)

```yaml
# configs/fitness/tool_calling.yaml
# Uses EXACT same format as SynthChat rubrics

name: "tool_calling_fitness"
description: "Fitness evaluation for tool internalization"
scope: response

# Standard validations array - same as rubrics
validations:
  # Format validation
  - match: "tool_call:"
    type: contains
    error: "Missing tool_call marker"

  - match: '"name":\s*"'
    type: regex
    error: "No tool name found"

  # Tool manifest validation
  - tools:
      useTools:
        _required: ["context", "calls"]
        context:
          _required: ["sessionId", "workspaceId"]
          sessionId: string
          workspaceId: string
        calls:
          _item_schema:
            _required: ["agent", "tool", "params"]
            agent: string
            tool: string
            params: object
          _subtools:
            vaultManager:
              moveNote:
                _required: ["path", "destination"]
                path: string
                destination: string
            contentManager:
              readNote:
                _required: ["path"]
                path: string

  # Cross-scope validation (IDs match system prompt)
  - cross_scope:
      from: "tool_calls[0].arguments.context.sessionId"
      to: "system_prompt"
      match_pattern: "sessionId:\\s*([\\w-]+)"
    error: "sessionId doesn't match system prompt"

# Optional: scoring weights per validation
scoring:
  method: "weighted"
  weights:
    format: 0.2
    tools: 0.5
    cross_scope: 0.3
```

### Adding a New Use Case

Developer workflow for a completely different use case (e.g., code generation):

1. **Create validation config**:
```yaml
# configs/fitness/code_generation.yaml
name: "code_generation_fitness"
scope: response

validations:
  - match: "```python"
    type: contains
    error: "No Python code block"

  - match: "def\\s+\\w+\\s*\\("
    type: regex
    error: "No function definition found"

  - field: "has_docstring"
    type: boolean
    error: "Missing docstring"
```

2. **Point training config at it**:
```yaml
evolutionary:
  enabled: true
  validation_config: "configs/fitness/code_generation.yaml"
```

3. **Run training** - no code changes needed.

---

## Implementation Guide

### Phase 1: Move Validators to Shared

**Files to move:**
```
SynthChat/services/validators/ → shared/validation/
├── structure_validator.py
├── cross_scope_validator.py
├── base.py
├── facade.py
└── content/
    ├── json_validator.py
    ├── xml_validator.py
    ├── regex_validator.py
    ├── yaml_validator.py
    ├── code_validator.py
    └── registry.py
```

**Update imports in SynthChat:**
```python
# SynthChat/services/validators/__init__.py
from shared.validation import (
    StructureValidator,
    CrossScopeValidator,
    SchemaValidator,
    # ... etc
)
```

### Phase 2: Create Fitness Wrapper

**New file:** `shared/validation/fitness.py`

Thin wrapper that:
1. Loads validation config (same format as rubrics)
2. Delegates to `StructureValidator`
3. Converts validation results to 0.0-1.0 score

### Phase 3: Create Evolutionary Components

**New directory:** `shared/evolutionary/`

```
shared/evolutionary/
├── __init__.py
├── fitness.py              # Re-export from validation
├── candidate_generator.py  # Generate weight update candidates
├── strategies/
│   ├── __init__.py
│   ├── registry.py         # Strategy lookup by name
│   ├── base.py             # Abstract strategy interface
│   ├── gradient_noise.py   # G + gaussian noise
│   ├── scale_variation.py  # G * random scales
│   └── component_dropout.py # G with random dropout
└── trainer_wrapper.py      # Wraps any HF Trainer
```

### Phase 4: Integrate into Training

**Modify:** `train_sft.py` and `train_kto.py`

```python
# At trainer initialization
if config.evolutionary.enabled:
    from shared.evolutionary import EvolutionaryTrainerWrapper
    trainer = EvolutionaryTrainerWrapper(
        base_trainer=trainer,
        config=config.evolutionary,
    )
```

---

## Pseudocode

### Fitness Evaluator

```python
# shared/validation/fitness.py

class FitnessEvaluator:
    """
    Config-driven fitness evaluation.
    Loads validation config, delegates to StructureValidator.
    """

    def __init__(self, config_path: Path):
        # Load config (same format as rubrics)
        self.config = load_yaml(config_path)
        self.validations = self.config.get("validations", [])
        self.scoring = self.config.get("scoring", {})

        # Reuse existing validators
        self.validator = StructureValidator()

    def evaluate(self, model_output: str, example: Dict) -> float:
        """
        Evaluate output against config-driven validations.
        Returns: 0.0 to 1.0
        """
        # Parse output (reuse existing parsing)
        data = self._parse_output(model_output)

        # Run validations (reuse existing validator)
        is_valid, errors = self.validator.validate(
            data=data,
            validations=self.validations,
            raw_content=model_output
        )

        # Convert to score
        return self._compute_score(is_valid, errors)

    def _compute_score(self, is_valid: bool, errors: List[str]) -> float:
        method = self.scoring.get("method", "error_count")
        params = self.scoring.get("params", {})

        if method == "binary":
            return 1.0 if is_valid else 0.0

        elif method == "error_count":
            max_errors = params.get("max_errors_before_zero", 5)
            if not errors:
                return 1.0
            return max(0.0, 1.0 - len(errors) / max_errors)

        elif method == "weighted":
            # Use per-validation weights from config
            return self._weighted_score(errors)

        return 0.5
```

### Candidate Generator

```python
# shared/evolutionary/candidate_generator.py

class CandidateGenerator:
    """
    Generate candidate weight updates from gradient.
    Strategy is config-driven.
    """

    def __init__(self, strategy_config: Dict):
        strategy_type = strategy_config.get("type", "gradient_noise")
        strategy_params = strategy_config.get("params", {})

        # Look up strategy in registry
        self.strategy = get_strategy(strategy_type, strategy_params)

    def generate(self, gradient: Dict[str, Tensor], n: int) -> List[Dict[str, Tensor]]:
        """
        Generate n candidate weight updates from gradient.
        """
        return self.strategy.generate(gradient, n)


# shared/evolutionary/strategies/gradient_noise.py

class GradientNoiseStrategy:
    """Add Gaussian noise to gradient."""

    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale

    def generate(self, gradient: Dict[str, Tensor], n: int) -> List[Dict[str, Tensor]]:
        candidates = [gradient]  # Always include pure gradient

        for _ in range(n - 1):
            noisy = {}
            for name, grad in gradient.items():
                noise = torch.randn_like(grad) * self.noise_scale * grad.abs().mean()
                noisy[name] = grad + noise
            candidates.append(noisy)

        return candidates


# shared/evolutionary/strategies/scale_variation.py

class ScaleVariationStrategy:
    """Multiply gradient by random scales."""

    def __init__(self, scales: List[float] = None):
        self.scales = scales or [0.5, 0.75, 1.0, 1.25, 1.5]

    def generate(self, gradient: Dict[str, Tensor], n: int) -> List[Dict[str, Tensor]]:
        # Select n scales (with replacement if needed)
        selected = random.choices(self.scales, k=n)

        candidates = []
        for scale in selected:
            scaled = {name: grad * scale for name, grad in gradient.items()}
            candidates.append(scaled)

        return candidates
```

### Evolutionary Trainer Wrapper

```python
# shared/evolutionary/trainer_wrapper.py

class EvolutionaryTrainerWrapper:
    """
    Wraps any HuggingFace Trainer with evolutionary selection.

    Intercepts the training step to:
    1. Capture gradient
    2. Generate candidates
    3. Evaluate fitness
    4. Apply best candidate
    """

    def __init__(
        self,
        base_trainer,
        config: Dict,
    ):
        self.trainer = base_trainer
        self.config = config

        # Initialize components from config
        self.fitness_evaluator = FitnessEvaluator(
            Path(config.get("validation_config"))
        )
        self.candidate_generator = CandidateGenerator(
            config.get("strategy", {})
        )
        self.n_candidates = config.get("candidates", 4)
        self.eval_batch_size = config.get("eval_batch_size", 4)

        # Get eval examples from dataset
        self.eval_examples = self._prepare_eval_examples()

        # Patch the training step
        self._patch_training_step()

    def _patch_training_step(self):
        """Replace trainer's training_step with evolutionary version."""
        original_step = self.trainer.training_step

        def evolutionary_step(model, inputs):
            # 1. Standard forward/backward
            loss = original_step(model, inputs)

            # 2. Capture gradient before optimizer step
            gradient = self._capture_gradient(model)

            # 3. Generate candidates
            candidates = self.candidate_generator.generate(
                gradient, self.n_candidates
            )

            # 4. Evaluate each candidate
            scores = []
            for candidate in candidates:
                score = self._evaluate_candidate(model, candidate)
                scores.append(score)

            # 5. Select best
            best_idx = scores.index(max(scores))
            best_candidate = candidates[best_idx]

            # 6. Replace gradient with best candidate
            self._apply_candidate_as_gradient(model, best_candidate)

            return loss

        self.trainer.training_step = evolutionary_step

    def _capture_gradient(self, model) -> Dict[str, Tensor]:
        """Capture current gradients from model parameters."""
        gradient = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient[name] = param.grad.clone()
        return gradient

    def _evaluate_candidate(self, model, candidate: Dict[str, Tensor]) -> float:
        """
        Evaluate a candidate weight update.

        1. Temporarily apply candidate
        2. Generate on eval examples
        3. Score with fitness evaluator
        4. Revert weights
        """
        # Store original weights
        original_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if name in candidate
        }

        try:
            # Temporarily apply candidate
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in candidate:
                        param.data -= self.trainer.args.learning_rate * candidate[name]

            # Generate and score
            scores = []
            for example in self.eval_examples[:self.eval_batch_size]:
                output = self._generate(model, example["prompt"])
                score = self.fitness_evaluator.evaluate(output, example)
                scores.append(score)

            return sum(scores) / len(scores)

        finally:
            # Revert weights
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in original_weights:
                        param.data = original_weights[name]

    def _apply_candidate_as_gradient(self, model, candidate: Dict[str, Tensor]):
        """Replace model gradients with candidate values."""
        for name, param in model.named_parameters():
            if name in candidate:
                param.grad = candidate[name]

    def _generate(self, model, prompt: str) -> str:
        """Generate model output for a prompt."""
        # Use trainer's tokenizer
        inputs = self.trainer.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
            )

        return self.trainer.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

    def train(self, *args, **kwargs):
        """Delegate to base trainer."""
        return self.trainer.train(*args, **kwargs)
```

### Integration in train_sft.py

```python
# Trainers/rtx3090_sft/train_sft.py

def main():
    # ... existing setup ...

    config = load_config()

    # ... model, tokenizer, dataset setup ...

    # Initialize base trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # ... other args ...
    )

    # Wrap with evolutionary if enabled
    if config.evolutionary.enabled:
        from shared.evolutionary import EvolutionaryTrainerWrapper

        print("=" * 60)
        print("EVOLUTIONARY TRAINING ENABLED")
        print(f"  Candidates per step: {config.evolutionary.candidates}")
        print(f"  Strategy: {config.evolutionary.strategy.type}")
        print(f"  Validation config: {config.evolutionary.validation_config}")
        print("=" * 60)

        trainer = EvolutionaryTrainerWrapper(
            base_trainer=trainer,
            config=config.evolutionary,
        )

    # Training proceeds normally
    trainer.train()
```

---

## Compute Analysis

### Per-Step Overhead

| Operation | Time | Count | Total |
|-----------|------|-------|-------|
| Backward pass | 300ms | 1 | 300ms |
| Forward pass (generate) | 100ms | 4 candidates × 4 examples | 1600ms |
| Schema validation | 0.1ms | 4 × 4 | ~0ms |
| Weight manipulation | 1ms | 4 × 2 (apply + revert) | 8ms |
| **Total** | | | **~1900ms** |

### Comparison

| Mode | Time per Step | Quality |
|------|---------------|---------|
| Standard training | ~400ms | Baseline |
| Evolutionary (N=4, eval=4) | ~1900ms | Better gradient selection |

~4-5x slower per step, but potentially fewer steps needed to converge.

### Memory Usage (2B model, 24GB VRAM)

| Component | Memory |
|-----------|--------|
| Base model (4-bit) | ~2GB |
| LoRA adapters | ~100MB |
| Training overhead | ~6GB |
| Gradient storage (N=4) | ~400MB |
| **Total** | **~8.5GB** |

Easily fits in 24GB with headroom for batch size.

---

## Future Extensions

1. **Adaptive N**: Start with more candidates, reduce as training stabilizes
2. **Fitness caching**: Skip re-evaluation if candidate is similar to previous
3. **Population across steps**: Maintain a population of weight vectors across multiple steps
4. **Latent space evolution**: Apply evolutionary pressure in hidden state space
5. **Multi-objective fitness**: Balance multiple validation criteria with Pareto selection

---

## Summary

This design integrates evolutionary selection into fine-tuning by:

1. **Reusing existing infrastructure** - Validators move to `shared/`, no duplication
2. **Config-driven fitness** - Same YAML format as rubrics
3. **Minimal code changes** - Wrapper around existing trainers
4. **Compute efficient** - Schema validation, not LLM calls
5. **Maximum flexibility** - New use cases = new YAML files

The core insight: **Tool calls are structured data. Structured data can be validated programmatically. Validation results become fitness scores. Fitness scores guide evolutionary selection.**
