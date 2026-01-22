# Contributing to lif-figure

## Development Setup

```bash
git clone https://github.com/half-adder/lif-figure.git
cd lif-figure
uv sync --group dev
```

To run the tool locally:

```bash
uv run lif-figure input.lif --channels "DAPI,GFP,mCherry"
```

## Running Tests

```bash
uv run python -m pytest
```

## Linting and Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
uv run ruff check .        # Check for lint errors
uv run ruff check --fix .  # Auto-fix lint errors
uv run ruff format .       # Format code
```

## VS Code Setup

For the best development experience, install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff). The project includes `.vscode/settings.json` which enables format-on-save.

## Project Structure

```
lif-figure/
├── src/lif_figure/
│   ├── cli.py       # Command-line interface
│   ├── reader.py    # LIF file reading and metadata extraction
│   ├── figure.py    # Matplotlib figure generation
│   └── config.py    # Configuration handling
├── tests/           # Test files
└── docs/            # Documentation
```

## Documentation

- [Metadata Extraction Logic](docs/metadata-extraction.md) - How acquisition metadata is extracted from LIF files
