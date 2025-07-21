# Bank Reconciliation AI Tool

A Python package for automating bank reconciliation processes using AI techniques.

## Installation

### Development Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/bank-rec-ai.git
cd bank-rec-ai
pip install -e .
```

### Regular Installation

```bash
pip install bank-rec-ai
```

## Usage

### As a Command Line Tool

After installation, you can run the tool directly:

```bash
bank-rec
```

### As a Python Package

```python
from app import run_reconciliation

# Example usage
result = run_reconciliation(input_file="path/to/transactions.csv")
```

## Project Structure

- `app/`: Main application code
- `data/`: Data files
- `model/`: AI model files
- `output/`: Output files

## Development

### Requirements

All requirements are listed in `requirements.txt` and will be installed automatically.

### Testing

Run tests with:

```bash
pytest
```