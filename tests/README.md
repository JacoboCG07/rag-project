# Running Tests

This document explains how to run tests for the RAG Project.

## Prerequisites

Make sure pytest is installed:

```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

Run all tests in the project:

```bash
pytest tests/
```

### Run Tests from a Specific Folder

Run all tests in a specific folder:

```bash
# All tests in unit_tests
pytest tests/unit_tests/

# All tests in extractors
pytest tests/unit_tests/ingestion/extractors/

# All tests in txt extractor folder
pytest tests/unit_tests/ingestion/extractors/txt/
```

### Run Tests from a Specific File

Run all tests in a specific test file:

```bash
pytest tests/unit_tests/ingestion/extractors/txt/test_txt_extractor.py
```

### Run a Specific Test

Run a single specific test:

```bash
# Run a specific test method
pytest tests/unit_tests/ingestion/extractors/txt/test_txt_extractor.py::TestTXTExtractor::test_extract

# Run a specific test class
pytest tests/unit_tests/ingestion/extractors/txt/test_txt_extractor.py::TestTXTExtractor
```

## Pytest Options

### `-v` or `--verbose`

Shows detailed output for each test:

```bash
pytest tests/ -v
```

**Output example:**
```
tests/unit_tests/ingestion/extractors/txt/test_txt_extractor.py::TestTXTExtractor::test_init PASSED
tests/unit_tests/ingestion/extractors/txt/test_txt_extractor.py::TestTXTExtractor::test_get_metadata PASSED
```

### `-s` or `--capture=no`

Shows print statements and stdout output:

```bash
pytest tests/ -s
```

**Use case:** When you want to see print() statements or other output from your tests.

### `-x` or `--exitfirst`

Stop after first failure:

```bash
pytest tests/ -x
```

**Use case:** Useful when debugging, stops immediately when a test fails.

### `-k` or `--keyword`

Run tests that match a keyword expression:

```bash
# Run tests with "extract" in the name
pytest tests/ -k extract

# Run tests with "empty" in the name
pytest tests/ -k empty
```

### `--tb=short`

Show shorter traceback format:

```bash
pytest tests/ --tb=short
```

**Options:**
- `--tb=short`: Shorter traceback
- `--tb=long`: Longer traceback (default)
- `--tb=line`: One line per failure
- `--tb=no`: No traceback

### `-q` or `--quiet`

Quiet mode, less output:

```bash
pytest tests/ -q
```

### `--cov` or `--coverage`

Generate coverage report (requires pytest-cov):

```bash
# Install pytest-cov first
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=src
```

## Common Combinations

### Verbose with output

```bash
pytest tests/ -v -s
```

### Stop on first failure with verbose

```bash
pytest tests/ -v -x
```

### Run specific test with verbose and output

```bash
pytest tests/unit_tests/ingestion/extractors/txt/test_txt_extractor.py::TestTXTExtractor::test_extract -v -s
```

## Examples

### Example 1: Run all TXT extractor tests with verbose output

```bash
pytest tests/unit_tests/ingestion/extractors/txt/test_txt_extractor.py -v
```

### Example 2: Run all tests and stop on first failure

```bash
pytest tests/ -x
```

### Example 3: Run tests matching a pattern

```bash
pytest tests/ -k "test_extract" -v
```

### Example 4: Run a specific test with full output

```bash
pytest tests/unit_tests/ingestion/extractors/txt/test_txt_extractor.py::TestTXTExtractor::test_extract -v -s
```

## Directory Structure

```
tests/
├── fixtures/              # Test data files
│   ├── sample.txt
│   └── empty.txt
└── unit_tests/           # Unit tests
    └── ingestion/
        └── extractors/
            └── txt/
                └── test_txt_extractor.py
```

## Troubleshooting

### Tests not found

Make sure you're running pytest from the project root:

```bash
cd C:\Users\eliot\Desktop\rag_project
pytest tests/
```

### Import errors

Make sure the `src` directory is in the Python path. The test files already handle this, but if you encounter issues, check the imports in the test files.

### Fixture files not found

Make sure the `tests/fixtures/` directory exists and contains the necessary test files.

