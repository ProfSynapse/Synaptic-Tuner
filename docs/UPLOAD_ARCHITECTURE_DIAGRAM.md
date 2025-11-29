# Upload System Architecture Diagrams

## Current Architecture (Before Refactoring)

```
┌─────────────────────────────────────────────────────────────┐
│                      User Invocation                        │
└────────────┬─────────────────────────────┬──────────────────┘
             │                             │
             │                             │
     ┌───────▼────────┐            ┌───────▼────────┐
     │  rtx3090_sft/  │            │  rtx3090_kto/  │
     │                │            │                │
     │  upload_to_hf  │            │  upload_to_hf  │
     │  .py (1165 L)  │◄───────────┤  .py (1165 L)  │
     │                │ DUPLICATED │                │
     └───────┬────────┘            └───────┬────────┘
             │                             │
             │                             │
     ┌───────▼────────┐            ┌───────▼────────┐
     │ Windows Patches│            │ Windows Patches│
     │ GPU Memory Mgr │            │ GPU Memory Mgr │
     │ Model Saving   │◄───────────┤ Model Saving   │
     │ GGUF Creation  │ DUPLICATED │ GGUF Creation  │
     │ HF Upload      │            │ HF Upload      │
     │ Documentation  │            │ Documentation  │
     │ CLI Parsing    │            │ CLI Parsing    │
     └───────┬────────┘            └───────┬────────┘
             │                             │
             └──────────┬──────────────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │   External Services  │
             │  - HuggingFace API   │
             │  - Unsloth           │
             │  - llama.cpp         │
             └──────────────────────┘

Problems:
  ❌ 1,165 lines duplicated across 2 trainers
  ❌ 100% code duplication
  ❌ Every change requires editing 2 files
  ❌ No abstraction, hard dependencies
  ❌ 12 responsibilities in one file
  ❌ Cannot extend without modifying core
```

## Proposed Architecture (After Refactoring)

```
┌─────────────────────────────────────────────────────────────┐
│                      User Invocation                        │
└────────────┬─────────────────────────────┬──────────────────┘
             │                             │
             │                             │
     ┌───────▼────────┐            ┌───────▼────────┐
     │  rtx3090_sft/  │            │  rtx3090_kto/  │
     │  Thin Wrapper  │            │  Thin Wrapper  │
     │  (10 lines)    │            │  (10 lines)    │
     └───────┬────────┘            └───────┬────────┘
             │                             │
             └──────────┬──────────────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │  Shared Upload CLI   │
             │  (Universal Entry)   │
             └──────────┬───────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │    Upload Orchestrator        │
        │  - Coordinates workflow       │
        │  - Manages dependencies       │
        │  - Handles errors             │
        └──────┬────────────────────────┘
               │
               ├─────────────────┬──────────────┬────────────────┐
               │                 │              │                │
               ▼                 ▼              ▼                ▼
        ┌──────────┐      ┌──────────┐   ┌──────────┐   ┌──────────┐
        │Strategy  │      │Converter │   │Uploader  │   │Document  │
        │Registry  │      │Registry  │   │Registry  │   │Generator │
        └────┬─────┘      └────┬─────┘   └────┬─────┘   └────┬─────┘
             │                 │              │                │
     ┌───────┴───────┐   ┌─────┴──────┐  ┌───┴────┐      ┌────┴─────┐
     │               │   │            │  │        │      │          │
     ▼               ▼   ▼            ▼  ▼        ▼      ▼          ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  LoRA   │   │16-bit   │   │  GGUF   │   │HuggingFc│   │Manifest │
│Strategy │   │Strategy │   │Converter│   │Uploader │   │Generator│
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
     │               │           │             │             │
     ▼               ▼           ▼             ▼             ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ 4-bit   │   │ 8-bit   │   │  AWQ    │   │ Ollama  │   │ModelCard│
│Strategy │   │Strategy │   │Converter│   │Uploader │   │Generator│
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘

              ┌────────────────────────────┐
              │   Platform Utilities       │
              │  - GPU Memory Manager      │
              │  - Windows Patches         │
              │  - Filesystem Helper       │
              └────────────────────────────┘

              ┌────────────────────────────┐
              │   Model Loading Layer      │
              │  - UnslothModelLoader      │
              │  - TransformersLoader      │
              └────────────────────────────┘

              ┌────────────────────────────┐
              │   External Services        │
              │  - HuggingFace API         │
              │  - Unsloth                 │
              │  - llama.cpp               │
              └────────────────────────────┘

Benefits:
  ✅ Zero code duplication
  ✅ Single source of truth
  ✅ SOLID principles applied
  ✅ Easy to extend (add strategies/converters)
  ✅ Easy to test (dependency injection)
  ✅ Clear separation of concerns
```

## Strategy Pattern Flow

```
User Request: Upload with merged_16bit
              │
              ▼
    ┌──────────────────────┐
    │  Upload Orchestrator │
    └──────────┬───────────┘
               │
               │ 1. Get strategy
               ▼
    ┌──────────────────────┐
    │  Strategy Registry   │
    │  lookup("merged_16") │
    └──────────┬───────────┘
               │
               │ 2. Returns strategy instance
               ▼
    ┌──────────────────────┐
    │ Merged16BitStrategy  │
    └──────────┬───────────┘
               │
               │ 3. Check GPU memory
               ▼
    ┌──────────────────────┐
    │  GPU Memory Manager  │
    │  ensure_memory(14GB) │
    └──────────┬───────────┘
               │
               │ 4. Load model
               ▼
    ┌──────────────────────┐
    │   Model Loader       │
    │   (via abstraction)  │
    └──────────┬───────────┘
               │
               │ 5. Save merged model
               ▼
    ┌──────────────────────┐
    │   Output Directory   │
    │   /merged-16bit/     │
    └──────────────────────┘

Adding new strategy (e.g., 8-bit):
  1. Create Merged8BitStrategy class
  2. Register in SaveStrategyRegistry
  3. Done! No core code modified.
```

## Dependency Inversion Example

### Before: Direct Dependencies

```
┌──────────────────────┐
│  upload_to_hf.py     │
│                      │
│  from unsloth import │──────┐
│    FastLanguageModel │      │
│                      │      │ Hard dependency
│  model, tok =        │      │ Cannot swap
│    FastLanguageModel │      │
│    .from_pretrained()│      │
└──────────────────────┘      │
                              ▼
                    ┌──────────────────┐
                    │     Unsloth      │
                    │  (Concrete Impl) │
                    └──────────────────┘

Problem: Cannot test without Unsloth installed
Problem: Cannot use different model loader
Problem: Tightly coupled to one implementation
```

### After: Depend on Abstractions

```
┌──────────────────────┐
│  Merged16BitStrategy │
│                      │
│  def __init__(       │
│    model_loader:     │──────┐
│      IModelLoader    │      │ Depends on
│  )                   │      │ abstraction
│                      │      │
│  model, tok =        │      │
│    self.model_loader │      │
│    .load_model()     │      │
└──────────────────────┘      │
                              ▼
                    ┌──────────────────┐
                    │  IModelLoader    │
                    │  (Interface)     │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   Unsloth    │ │Transformers  │ │MockLoader    │
    │   Loader     │ │   Loader     │ │(for testing) │
    └──────────────┘ └──────────────┘ └──────────────┘

Benefits:
  ✅ Can swap implementations
  ✅ Easy to mock for testing
  ✅ Can add new loaders without changing strategies
  ✅ Clear contract (IModelLoader interface)
```

## Module Dependency Graph

```
                            ┌─────────────────┐
                            │  upload_cli.py  │
                            │   (Entry Point) │
                            └────────┬────────┘
                                     │
                                     ▼
                      ┌──────────────────────────┐
                      │  UploadOrchestrator      │
                      │  (Coordinates workflow)  │
                      └──────────┬───────────────┘
                                 │
        ┌────────────────────────┼─────────────────────┐
        │                        │                     │
        ▼                        ▼                     ▼
┌───────────────┐      ┌─────────────────┐    ┌──────────────┐
│  Strategies   │      │   Converters    │    │  Uploaders   │
│  - LoRA       │      │   - GGUF        │    │  - HuggingFc │
│  - 16-bit     │      │   - AWQ         │    │  - Ollama    │
│  - 4-bit      │      │   - GPTQ        │    │  - Local     │
└───────┬───────┘      └────────┬────────┘    └──────┬───────┘
        │                       │                     │
        └───────────┬───────────┴─────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Platform Utilities   │
        │  - GPU Memory         │
        │  - Windows Patches    │
        │  - Filesystem         │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Core Abstractions    │
        │  - Interfaces         │
        │  - Config Models      │
        │  - Type Definitions   │
        └───────────────────────┘

Legend:
  ───▶  Depends on
  ────  Uses/Imports
```

## File Size Comparison

```
Before Refactoring:
┌────────────────────────────────────────────────┐
│ rtx3090_sft/src/upload_to_hf.py   [████████████████████] 1,165 lines
│ rtx3090_kto/src/upload_to_hf.py   [████████████████████] 1,165 lines
│ rtx3090_sft/upload_model.sh       [███                 ]   204 lines
│ rtx3090_kto/upload_model.sh       [███                 ]   204 lines
│ rtx3090_sft/upload_model.ps1      [███                 ]   240 lines
│ rtx3090_kto/upload_model.ps1      [███                 ]   240 lines
└────────────────────────────────────────────────┘
Total: 3,218 lines (massive duplication)

After Refactoring:
┌────────────────────────────────────────────────┐
│ shared/upload/* (all modules)      [████████] 400 lines
│ rtx3090_sft/src/upload_to_hf.py   [          ]  10 lines
│ rtx3090_kto/src/upload_to_hf.py   [          ]  10 lines
│ rtx3090_sft/upload_model.sh       [          ]  20 lines
│ rtx3090_kto/upload_model.sh       [          ]  20 lines
│ rtx3090_sft/upload_model.ps1      [          ]  25 lines
│ rtx3090_kto/upload_model.ps1      [          ]  25 lines
└────────────────────────────────────────────────┘
Total: 510 lines (85% reduction)
```

## Extension Scenarios

### Scenario 1: Adding GPTQ Quantization

```
Current State:
  User wants GPTQ support
       │
       ▼
  ┌─────────────────────┐
  │ Must edit:          │
  │ 1. upload_to_hf.py  │ ◄── Edit SFT version
  │ 2. upload_to_hf.py  │ ◄── Edit KTO version
  │ 3. Both shell      │
  │ 4. Both PS scripts  │
  └─────────────────────┘
  Total changes: 6 files
  Time: 4-8 hours
  Risk: Breaking existing formats


Proposed State:
  User wants GPTQ support
       │
       ▼
  ┌─────────────────────────────┐
  │ 1. Create new file:         │
  │    strategies/gptq.py       │
  │                             │
  │ 2. Register:                │
  │    SaveStrategyRegistry     │
  │    .register("gptq", ...)   │
  └─────────────────────────────┘
  Total changes: 1 new file, 1 line added
  Time: 15-30 minutes
  Risk: Zero (doesn't touch existing code)
```

### Scenario 2: Adding Third Trainer (DPO)

```
Current State:
  New DPO trainer needs upload
       │
       ▼
  ┌─────────────────────┐
  │ Must copy:          │
  │ 1. upload_to_hf.py  │ ◄── 1,165 lines
  │ 2. upload_model.sh  │ ◄── 204 lines
  │ 3. upload_model.ps1 │ ◄── 240 lines
  │                     │
  │ Then update paths   │
  │ in all 3 files      │
  └─────────────────────┘
  Total: 1,609 lines copied
  Time: 2-3 hours
  Maintenance: 3 more files forever


Proposed State:
  New DPO trainer needs upload
       │
       ▼
  ┌──────────────────────────────┐
  │ Create thin wrappers:        │
  │                              │
  │ 1. upload_to_hf.py (10 lines)│
  │ 2. upload_model.sh (20 lines)│
  │ 3. upload_model.ps1(25 lines)│
  │                              │
  │ All inherit shared logic     │
  └──────────────────────────────┘
  Total: 55 lines (new code)
  Time: 5-10 minutes
  Maintenance: 3 tiny wrappers
```

## Testing Architecture

```
                    ┌──────────────────┐
                    │   Test Suite     │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  Unit Tests  │ │ Integration  │ │   E2E Tests  │
    │              │ │    Tests     │ │              │
    │ - Strategies │ │ - Workflow   │ │ - SFT Upload │
    │ - Converters │ │ - Registries │ │ - KTO Upload │
    │ - Uploaders  │ │ - Orchestr.  │ │ - GGUF Flow  │
    └──────────────┘ └──────────────┘ └──────────────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Test Coverage  │
                    │      >85%        │
                    └──────────────────┘

Before: ~0% test coverage (monolithic code hard to test)
After: >85% test coverage (small, focused modules easy to test)
```

---

**Visual Summary:**

The refactoring transforms a duplicated monolith into a well-architected, extensible system:

- **From:** 100% duplication → **To:** 0% duplication
- **From:** 1 giant file → **To:** Many small, focused modules
- **From:** Hard dependencies → **To:** Abstraction layers
- **From:** Modify core for changes → **To:** Extend via plugins
- **From:** Untestable → **To:** >85% test coverage

All while maintaining backwards compatibility through thin wrappers.
