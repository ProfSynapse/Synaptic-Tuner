# Tuner Utilities and CLI Layer Implementation

**Date:** 2025-11-30
**Component:** Utilities and CLI Layer (Phase 1 of tuner refactoring)
**Status:** Implemented

---

## Summary

Implemented the utilities and CLI layer for the Synaptic Tuner refactoring, following the architecture specified in `/mnt/f/Code/Toolset-Training/docs/tuner_architecture.md`.

This is Phase 1 of the migration strategy, establishing core utilities and CLI routing infrastructure that will be used by handlers (to be implemented in later phases).

---

## Files Created

### Utilities (`/mnt/f/Code/Toolset-Training/tuner/utils/`)

1. **`__init__.py`**
   - Package initialization
   - Exports all utility functions

2. **`environment.py`**
   - `detect_environment() -> str`
   - Returns "windows", "wsl", or "linux"
   - Detection logic:
     - Check `sys.platform == "win32"` → Windows
     - Check `/mnt/c` exists → WSL
     - Otherwise → Linux

3. **`conda.py`**
   - Constants: `UNSLOTH_ENV = "unsloth_latest"`, `UNSLOTH_VERSION = "2025.11.4"`
   - `get_conda_python() -> str`
   - Finds Python from unsloth_latest conda environment
   - Platform-aware search paths (Windows vs Linux/WSL)
   - Falls back to `sys.executable` with warning if not found

4. **`validation.py`**
   - `validate_repo_id(repo_id: str) -> bool` - Validates "username/model" format
   - `validate_path_exists(path: Path) -> bool` - Checks if path exists
   - `load_env_file(env_path: Path) -> bool` - Loads .env file into os.environ

### CLI Layer (`/mnt/f/Code/Toolset-Training/tuner/cli/`)

1. **`__init__.py`**
   - Package initialization
   - Exports main, parser, router

2. **`parser.py`**
   - `create_parser() -> argparse.ArgumentParser`
   - Defines command-line interface
   - Commands: train, upload, eval, pipeline (optional)
   - Includes help text and usage examples

3. **`router.py`**
   - `route_command(args: Namespace) -> int`
   - Maps commands to handlers:
     - train → TrainHandler
     - upload → UploadHandler
     - eval → EvalHandler
     - pipeline → PipelineHandler
     - (none) → MainMenuHandler
   - Gracefully handles missing handlers (not yet implemented)

4. **`main.py`**
   - `main()` entry point
   - Parses arguments via `create_parser()`
   - Routes via `route_command()`
   - Handles KeyboardInterrupt gracefully (exit code 130)
   - Handles general exceptions with traceback (exit code 1)

---

## Design Decisions

### 1. Environment Detection
- Matches existing pattern from `tuner.py` lines 162-169
- Simple, reliable detection using `sys.platform` and `/mnt/c` existence

### 2. Conda Path Finding
- Matches existing pattern from `tuner.py` lines 177-204
- Platform-specific search paths
- Fallback to current Python with warning (not hard failure)
- Preserves user-friendly error messages

### 3. CLI Argument Parsing
- Matches existing pattern from `tuner.py` lines 1033-1058
- Optional positional argument for direct command invocation
- Preserves help text and usage examples
- Compatible with both `python tuner.py` and `python -m tuner`

### 4. Command Router
- Deferred imports to avoid circular dependencies
- Graceful degradation if handlers not yet implemented
- Returns exit codes for proper shell integration
- Maps commands to handler classes (not functions)

### 5. Error Handling
- KeyboardInterrupt → exit 130 (standard Unix convention)
- General exceptions → exit 1 with traceback
- Graceful degradation during migration period

---

## Implementation Notes

### Forward Compatibility
- Router imports handlers dynamically (try/except)
- Shows helpful message if handlers not yet implemented
- This allows incremental migration without breaking existing code

### Alignment with Architecture
- Follows architecture spec Appendix A.2 (CLI Entry Point) and A.3 (Router)
- Uses same patterns as existing `Trainers/shared/upload/` framework
- Preserves exact CLI behavior from current `tuner.py`

### Code Quality
- Comprehensive docstrings with examples
- Type hints for all functions
- Clear separation of concerns
- No external dependencies beyond standard library

---

## Testing Instructions for Backend Engineer

### 1. Unit Tests for Utilities

**Test `environment.py`:**
```python
# Test environment detection
from tuner.utils.environment import detect_environment

env = detect_environment()
assert env in ["windows", "wsl", "linux"]

# On WSL, should return "wsl"
# On Linux, should return "linux"
# On Windows, should return "windows"
```

**Test `conda.py`:**
```python
# Test conda Python finding
from tuner.utils.conda import get_conda_python, UNSLOTH_ENV

python = get_conda_python()
assert python  # Should return a path
assert isinstance(python, str)

# Check constants
assert UNSLOTH_ENV == "unsloth_latest"
```

**Test `validation.py`:**
```python
from pathlib import Path
from tuner.utils.validation import validate_repo_id, validate_path_exists, load_env_file

# Test repo ID validation
assert validate_repo_id("profsynapse/model-name") == True
assert validate_repo_id("invalid") == False
assert validate_repo_id("too/many/parts") == False
assert validate_repo_id("") == False

# Test path validation
assert validate_path_exists(Path(".")) == True
assert validate_path_exists(Path("/nonexistent")) == False

# Test .env loading
env_file = Path(".env")
if env_file.exists():
    assert load_env_file(env_file) == True
else:
    assert load_env_file(env_file) == False
```

### 2. Unit Tests for CLI

**Test `parser.py`:**
```python
from tuner.cli.parser import create_parser

# Test parser creation
parser = create_parser()
assert parser is not None

# Test valid commands
args = parser.parse_args(['train'])
assert args.command == 'train'

args = parser.parse_args(['upload'])
assert args.command == 'upload'

args = parser.parse_args(['eval'])
assert args.command == 'eval'

args = parser.parse_args(['pipeline'])
assert args.command == 'pipeline'

# Test no command (interactive mode)
args = parser.parse_args([])
assert args.command is None

# Test invalid command (should raise SystemExit)
import pytest
with pytest.raises(SystemExit):
    parser.parse_args(['invalid'])
```

**Test `router.py`:**
```python
from argparse import Namespace
from tuner.cli.router import route_command

# Test router with no command (should return exit code)
args = Namespace(command=None)
exit_code = route_command(args)
# Currently returns 1 (handlers not implemented)
assert isinstance(exit_code, int)

# Note: Full router tests require handlers to be implemented
# For now, just verify it doesn't crash and returns int
```

**Test `main.py`:**
```python
import subprocess

# Test main entry point help
result = subprocess.run(
    ['python', '-m', 'tuner', '--help'],
    capture_output=True,
    text=True
)
assert result.returncode == 0
assert 'Synaptic Tuner' in result.stdout
assert 'train' in result.stdout
assert 'upload' in result.stdout

# Test invalid command
result = subprocess.run(
    ['python', '-m', 'tuner', 'invalid'],
    capture_output=True,
    text=True
)
assert result.returncode != 0
```

### 3. Integration Tests

**Test Environment Detection:**
```bash
# From repo root
cd /mnt/f/Code/Toolset-Training

# Test environment detection
python -c "from tuner.utils.environment import detect_environment; print(detect_environment())"
# Expected: "wsl" (on WSL), "linux" (on Linux), "windows" (on Windows)
```

**Test Conda Path Finding:**
```bash
# Test conda Python finding
python -c "from tuner.utils.conda import get_conda_python; print(get_conda_python())"
# Expected: Path to conda Python or current Python with warning
```

**Test CLI Help:**
```bash
# Test CLI help
python -m tuner --help
# Expected: Help text with commands (train, upload, eval, pipeline)

# Test direct command parsing
python -m tuner train
# Expected: Error message that handlers not yet implemented (graceful degradation)
```

### 4. Regression Tests

**Verify No Breaking Changes:**
```bash
# Current tuner.py should still work
python tuner.py --help
# Expected: Same help text as before

# Utilities should be importable
python -c "from tuner.utils import detect_environment, get_conda_python, validate_repo_id"
# Expected: No errors

# CLI layer should be importable
python -c "from tuner.cli import main, create_parser, route_command"
# Expected: No errors
```

### 5. Expected Failures (Before Handlers Implemented)

These commands will fail gracefully with helpful messages:

```bash
# These will show "handlers not yet implemented" message
python -m tuner train
python -m tuner upload
python -m tuner eval
python -m tuner pipeline
python -m tuner  # Interactive mode
```

Expected output:
```
Error: Handlers not yet implemented: No module named 'tuner.handlers.train_handler'
This is expected during migration. Please use tuner_legacy.py instead.
```

---

## Next Steps for Backend Engineer

### Phase 2: Test the Utilities

1. Run unit tests for each utility module
2. Verify conda Python detection works on target environment
3. Test .env file loading with sample .env
4. Verify repo ID validation with various inputs

### Phase 3: Test the CLI Layer

1. Test argument parsing with all command variations
2. Verify help text displays correctly
3. Test graceful degradation with missing handlers
4. Verify exit codes are correct

### Phase 4: Integration Testing

1. Test imports work from package
2. Verify no breaking changes to existing `tuner.py`
3. Test on WSL, Linux, and Windows (if available)
4. Verify conda environment detection

### Phase 5: Report Issues

Create test report with:
- Platform tested (WSL/Linux/Windows)
- Python version
- Test results (pass/fail for each test)
- Any errors or unexpected behavior
- Suggestions for improvements

---

## File Locations

All files created under `/mnt/f/Code/Toolset-Training/tuner/`:

```
tuner/
├── utils/
│   ├── __init__.py          # Exports utilities
│   ├── environment.py       # Environment detection
│   ├── conda.py             # Conda Python finding
│   └── validation.py        # Input validation
└── cli/
    ├── __init__.py          # Exports CLI components
    ├── main.py              # Main entry point
    ├── parser.py            # Argument parser
    └── router.py            # Command router
```

---

## Dependencies

**Standard Library Only:**
- `sys` - Platform detection, exit codes
- `os` - Environment variables
- `pathlib` - Path operations
- `argparse` - CLI argument parsing
- `traceback` - Error reporting

**No External Dependencies Required**

---

## Compatibility

**Python Version:** 3.8+
**Platforms:** WSL2, Linux, Windows
**Conda Environment:** unsloth_latest (optional, falls back gracefully)

---

## Success Criteria

- All utility functions work correctly
- CLI parser accepts all commands
- Router handles missing handlers gracefully
- No breaking changes to existing code
- Clean imports from package
- Comprehensive docstrings and comments
- Type hints for all functions

---

## Architecture Alignment

This implementation follows:
- **Architecture Spec:** `/mnt/f/Code/Toolset-Training/docs/tuner_architecture.md`
  - Section 2.3.7: Utils Layer
  - Section 2.3.1: CLI Layer
  - Appendix A.2: CLI Entry Point
  - Appendix A.3: Router
- **Migration Strategy:** Phase 1 (Core Infrastructure)
- **Design Patterns:** Single Responsibility, Separation of Concerns
- **Code Style:** Matches existing `Trainers/shared/` patterns

---

## Notes for Test Engineer

1. **Expected Behavior During Migration:**
   - Router will show "handlers not yet implemented" until Phase 4
   - This is normal and expected
   - Use `tuner_legacy.py` for actual operations during migration

2. **Test Environment Setup:**
   - Ensure conda environment exists: `conda env list | grep unsloth_latest`
   - Create sample .env file for testing load_env_file
   - Test on target platform (WSL2 preferred)

3. **Critical Tests:**
   - Environment detection must be correct (affects conda path finding)
   - Conda Python must be found or fallback gracefully
   - CLI must parse all commands without errors
   - Router must handle missing handlers without crashes

4. **Non-Critical Tests:**
   - Validation functions (used by handlers later)
   - .env loading (used by handlers later)
   - Exact error messages (may change)

---

## Documentation

All functions have:
- Clear docstrings with purpose
- Parameter descriptions
- Return value descriptions
- Usage examples
- File location and purpose in header comments

---

## Conclusion

This implementation establishes the foundation for the tuner refactoring:
- Utilities provide cross-cutting functionality
- CLI layer handles argument parsing and routing
- Clean separation of concerns
- Forward-compatible with handler implementation
- Preserves existing CLI behavior
- Easy to test in isolation

The backend engineer should focus on testing the utilities and CLI layer independently before handlers are implemented. All tests should pass except those requiring handlers (which will fail gracefully with helpful messages).
