# VLA-Arena Tests

This directory contains tests for VLA-Arena.

## Running Tests

Run all tests:
```bash
make test
```

Or use pytest directly:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=vla_arena --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_import.py -v
```

## Test Structure

- `test_import.py` - Basic import and package structure tests

## Adding New Tests

When adding new features or fixing bugs:

1. Create a new test file `test_<feature>.py`
2. Add appropriate test cases
3. Use fixtures from `conftest.py` when needed
4. Ensure tests are independent and can run in any order
5. Mock external dependencies when appropriate

## Test Guidelines

- Use descriptive test names: `test_<what>_<scenario>`
- Add docstrings to explain test purpose
- Keep tests focused and independent
- Use parametrize for testing multiple scenarios
- Mock external services and heavy dependencies
- Aim for high code coverage

## Continuous Integration

Tests are automatically run on GitHub Actions for:
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Multiple operating systems (Ubuntu, macOS)

See `.github/workflows/ci.yml` for CI configuration.
