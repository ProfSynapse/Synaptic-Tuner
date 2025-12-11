# Pydantic vs Current Dataclass Patterns

## Side-by-Side Comparison

### Current: Manual Dataclass with Validation

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ImprovementConfig:
    input_file: str
    output_file: str
    backend: str = "openrouter"
    batch_size: int = 10
    start_line: int = 1
    end_line: Optional[int] = None
    temperature: float = 0.3

    def validate(self) -> None:
        """Validate configuration - MUST BE CALLED MANUALLY."""
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if self.start_line < 1:
            raise ValueError("Start line must be at least 1")
        if self.end_line and self.end_line < self.start_line:
            raise ValueError("End line must be >= start line")

    def to_dict(self):
        """Manual serialization."""
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "backend": self.backend,
            "batch_size": self.batch_size,
            # ... repeat for all fields
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Manual deserialization."""
        return cls(
            input_file=data["input_file"],
            output_file=data["output_file"],
            backend=data.get("backend", "openrouter"),
            # ... repeat for all fields
        )

# Usage - easy to forget validation!
config = ImprovementConfig(
    input_file="data.jsonl",
    output_file="out.jsonl",
    batch_size=-5  # ❌ No error until validate() is called!
)
config.validate()  # ❌ Must remember to call this!
```

### With Pydantic: Automatic Validation

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ImprovementConfig(BaseModel):
    input_file: str
    output_file: str
    backend: str = "openrouter"
    batch_size: int = Field(default=10, ge=1)  # ge = greater or equal
    start_line: int = Field(default=1, ge=1)
    end_line: Optional[int] = Field(default=None, ge=1)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator('end_line')
    @classmethod
    def validate_end_line(cls, v, info):
        """Validate end_line >= start_line."""
        if v is not None and 'start_line' in info.data:
            if v < info.data['start_line']:
                raise ValueError("End line must be >= start line")
        return v

    # ✅ to_dict() and from_dict() are built-in!
    # Use: config.model_dump() and Config.model_validate(data)

# Usage - validation happens automatically!
try:
    config = ImprovementConfig(
        input_file="data.jsonl",
        output_file="out.jsonl",
        batch_size=-5  # ✅ Immediate error with helpful message!
    )
except ValidationError as e:
    print(e)
    # Output:
    # 1 validation error for ImprovementConfig
    # batch_size
    #   Input should be greater than or equal to 1 [type=greater_than_equal]

# ✅ Type coercion works automatically
config = ImprovementConfig(
    input_file="data.jsonl",
    output_file="out.jsonl",
    batch_size="10"  # ✅ Automatically converts "10" -> 10
)
```

## Environment Variable Loading

### Current: Manual Parsing

```python
# From shared/llm/config.py:31-79
@dataclass
class LLMConfig:
    provider: str
    model: str
    lmstudio_port: int = 1234

    @classmethod
    def from_env(cls, env_prefix: str = "IMPROVEMENT"):
        """Manual environment parsing with type casting."""
        _load_env_file()  # Custom .env loader

        provider = os.getenv(f"{env_prefix}_BACKEND", "openrouter")
        model = os.getenv(f"{env_prefix}_MODEL", "openai/gpt-5-mini")

        # ❌ Manual type conversion - easy to forget or get wrong
        port = int(os.getenv("LMSTUDIO_PORT", "1234"))

        return cls(
            provider=provider.lower(),
            model=model,
            lmstudio_port=port,
            # ... repeat for all fields
        )

# Usage
config = LLMConfig.from_env()
config.validate()  # ❌ Still need to validate!
```

### With Pydantic Settings

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMConfig(BaseSettings):
    """Automatically loads from environment variables."""

    # Field names automatically map to env vars
    provider: str = Field(default="openrouter", alias="IMPROVEMENT_BACKEND")
    model: str = Field(default="openai/gpt-5-mini", alias="IMPROVEMENT_MODEL")

    # ✅ Automatic type conversion and validation
    lmstudio_port: int = Field(default=1234, ge=1, le=65535)
    lmstudio_host: str = Field(default="localhost")

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
    )

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        valid = ["openrouter", "lmstudio", "ollama"]
        if v not in valid:
            raise ValueError(f"Provider must be one of {valid}")
        return v.lower()

# Usage - just instantiate!
config = LLMConfig()  # ✅ Loads from .env automatically, validates types
```

## Nested Models

### Current: Manual Nesting

```python
# From shared/upload/core/config.py:125-134
@dataclass
class FullUploadConfig:
    upload: UploadConfig
    save: SaveConfig
    conversion: Optional[ConversionConfig] = None

    # ❌ Must validate each nested config separately
    def validate_all(self):
        self.upload.validate()
        self.save.validate()
        if self.conversion:
            self.conversion.validate()
```

### With Pydantic: Automatic Nested Validation

```python
class FullUploadConfig(BaseModel):
    upload: UploadConfig  # ✅ Automatically validates nested model
    save: SaveConfig      # ✅ Recursively validates all fields
    conversion: Optional[ConversionConfig] = None

# Usage
config = FullUploadConfig(
    upload={"model_path": "/path", "repo_id": "user/model"},  # ✅ Accepts dict!
    save={"strategy_name": "invalid"}  # ✅ Validation error on nested field
)
```

## JSON/Dict Serialization

### Current: Manual to_dict/from_dict Everywhere

```python
# From improvement_engine/core/models.py:54-88
@dataclass
class Example:
    conversations: List[Dict[str, str]]
    label: Optional[bool] = None
    behavior: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """❌ 30+ lines of boilerplate per model."""
        result = {"conversations": self.conversations}
        if self.label is not None:
            result["label"] = self.label
        if self.behavior:
            result["behavior"] = self.behavior
        # ... repeat for every optional field
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """❌ Another 10+ lines of boilerplate."""
        return cls(
            conversations=data["conversations"],
            label=data.get("label"),
            behavior=data.get("behavior"),
            # ... repeat for every field
        )
```

### With Pydantic: Zero Boilerplate

```python
class Example(BaseModel):
    conversations: List[Dict[str, str]]
    label: Optional[bool] = None
    behavior: Optional[str] = None

    # ✅ That's it! Serialization is built-in.

# Usage
example = Example(conversations=[...])

# Serialize to dict
data = example.model_dump()  # ✅ Excludes None by default

# Deserialize from dict
example = Example.model_validate(data)  # ✅ Validates on load

# JSON serialization
json_str = example.model_dump_json()  # ✅ Direct to JSON
example = Example.model_validate_json(json_str)  # ✅ Direct from JSON
```

## Benefits Summary

### ✅ What You Gain

1. **Automatic Validation**: Errors on instantiation, not later
2. **Better Error Messages**: Field-level details with type info
3. **Type Coercion**: `"123"` → `123`, `"true"` → `True`
4. **Zero Boilerplate**: No more `to_dict()`/`from_dict()`
5. **Environment Variables**: `pydantic-settings` handles .env parsing
6. **Nested Validation**: Validates complex object graphs automatically
7. **IDE Support**: Better autocomplete and type checking
8. **JSON Schema**: Auto-generate schemas for API docs

### ❌ Potential Drawbacks

1. **New Dependency**: Adds ~1MB (pydantic is pure Python with Rust core)
2. **Learning Curve**: Team needs to learn decorators like `@field_validator`
3. **Different API**: `model_dump()` vs `to_dict()`, `model_validate()` vs `from_dict()`
4. **Migration Effort**: 46 dataclass files to update (but can be gradual)
5. **Performance**: Slightly slower than raw dataclasses (but negligible for config)

## Migration Strategy

### Option 1: Gradual Migration (Recommended)

Start with high-value areas:

1. **Phase 1**: Configuration files (biggest win)
   - `shared/llm/config.py` → Pydantic Settings
   - `improvement_engine/core/models.py` → BaseModel
   - `shared/upload/core/config.py` → BaseModel

2. **Phase 2**: Schema validators
   - `improvement_engine/services/schema_validator.py`
   - `Evaluator/schema_validator.py`

3. **Phase 3**: Data models (if needed)
   - Dataset models (`Example`, `ThinkingBlock`)
   - Only if serialization overhead becomes painful

### Option 2: Coexistence

Pydantic can coexist with dataclasses:

```python
# Keep existing dataclasses
@dataclass
class LegacyConfig:
    pass

# New code uses Pydantic
class NewConfig(BaseModel):
    pass

# They can even reference each other
class MixedConfig(BaseModel):
    legacy: LegacyConfig  # ✅ Pydantic validates dataclass fields too
```

### Option 3: No Migration

If the team prefers to stick with dataclasses:

1. Create a `@validated_dataclass` decorator that calls `validate()` in `__post_init__`
2. Use `dacite` library for dict ↔ dataclass conversion
3. Continue with current patterns

## Recommendation for This Codebase

**Yes, adopt Pydantic gradually**, starting with:

1. **`shared/llm/config.py`** - Environment variable loading is tedious
2. **`improvement_engine/core/models.py`** - Lots of validation logic
3. **`shared/upload/core/config.py`** - Complex nested configs

**Skip for:**
- Simple data transfer objects
- High-performance hot paths (if any)
- Models that rarely need validation

The biggest wins are:
- ✅ Environment variable parsing (no more manual `from_env()` methods)
- ✅ Validation on instantiation (no more forgetting `validate()`)
- ✅ Removing 30+ lines of `to_dict()`/`from_dict()` boilerplate per model

## Example: Before/After for LLMConfig

### Before (Current)
```python
# 128 lines in shared/llm/config.py
@dataclass
class LLMConfig:
    # ... 20+ fields

    @classmethod
    def from_env(cls, env_prefix: str = "IMPROVEMENT"):
        """50+ lines of manual parsing."""
        _load_env_file()
        # ... manual os.getenv() calls
        # ... manual type casting
        return cls(...)

    def validate(self):
        """30+ lines of validation logic."""
        if self.provider not in ["openrouter", "lmstudio", "ollama"]:
            raise ValueError(...)
        # ... more checks
```

### After (With Pydantic)
```python
# ~50 lines total
from pydantic_settings import BaseSettings

class LLMConfig(BaseSettings):
    provider: str = Field(default="openrouter")
    model: str = Field(default="openai/gpt-5-mini")
    lmstudio_port: int = Field(default=1234, ge=1, le=65535)
    # ... other fields with inline validation

    model_config = SettingsConfigDict(env_file='.env')

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        valid = ["openrouter", "lmstudio", "ollama"]
        if v not in valid:
            raise ValueError(f"Provider must be one of {valid}")
        return v.lower()

# Usage
config = LLMConfig()  # ✅ Done! Loads .env, validates, coerces types
```

**Result**: ~60% less code, automatic validation, better errors.
