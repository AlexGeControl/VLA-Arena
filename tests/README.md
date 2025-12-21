# VLA-Arena Test Suite

This directory contains comprehensive pytest tests for the VLA-Arena project.

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_utils.py` - Tests for utility functions
- `test_benchmark.py` - Tests for benchmark functionality
- `test_cli.py` - Tests for command-line interface
- `test_vla_arena_init.py` - Tests for initialization and path management
- `test_log_utils.py` - Tests for logging utilities
- `test_bddl_utils.py` - Tests for BDDL generation utilities
- `test_task_generation_utils.py` - Tests for task generation utilities
- `test_integration.py` - Integration tests (may require full setup)

## Running Tests

### Run all tests

```bash
pytest tests/
```

### Run specific test file

```bash
pytest tests/test_utils.py
```

### Run with coverage

```bash
pytest tests/ --cov=vla_arena --cov-report=html
```

### Run only unit tests (exclude integration tests)

```bash
pytest tests/ -m "not integration"
```

### Run only fast tests (exclude slow tests)

```bash
pytest tests/ -m "not slow"
```

### Run with verbose output

```bash
pytest tests/ -v
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (may require full setup)
- `@pytest.mark.slow` - Slow tests (may take longer to run)

## Requirements

All test dependencies should be installed:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock
```

## Notes

- Some tests may be skipped if certain dependencies are not available
- Integration tests may require proper configuration and data files
- Mock objects are used extensively to avoid requiring actual model files or environments
