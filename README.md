# RAG in Python

A comprehensive RAG (Retrieval-Augmented Generation) toolkit built with LlamaIndex and FAISS, following Python community best practices.

## Features

- 🚀 **Modern Python 3.12** - Built with the latest Python features
- 📚 **LlamaIndex Integration** - Powerful document indexing and retrieval
- ⚡ **FAISS Vector Store** - High-performance similarity search
- 🔧 **CLI Interface** - Easy-to-use command-line tools  
- 🏗️ **Modular Architecture** - Clean separation of concerns
- 📦 **uv & hatchling** - Modern Python packaging and dependency management
- 🧪 **Type Safety** - Full type hints and mypy support
- 🎯 **Developer Experience** - Pre-commit hooks, linting, and testing setup

## Installation

### Requirements

- Python 3.12+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

### Install from PyPI (when published)

```bash
pip install rag-in-python
```

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/zakamai/rag-in-python.git
cd rag-in-python
```

2. Install with uv (recommended):
```bash
uv sync --dev
```

Or with pip:
```bash
pip install -e ".[dev]"
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

## Quick Start

### 1. Set up your environment

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 2. Index your documents

```bash
rag-cli index /path/to/your/documents --index-path ./my_index
```

### 3. Query your documents

```bash
rag-cli query "What is the main topic?" --index-path ./my_index
```

### 4. Interactive mode

```bash
rag-cli interactive --index-path ./my_index
```

## Python API Usage

```python
from pathlib import Path
from rag_in_python import RAGSystem

# Initialize the RAG system
rag = RAGSystem(vector_store_path=Path("./my_index"))

# Index documents
document_paths = [Path("doc1.txt"), Path("doc2.pdf")]
rag.load_and_index_files(document_paths)

# Save the index
rag.save_index()

# Query the system
result = rag.query("What is the main topic?", top_k=5)
print(result["response"])
print(f"Sources: {result['retrieved_documents']} documents")
```

## Project Structure

```
rag-in-python/
├── src/
│   └── rag_in_python/
│       ├── __init__.py          # Main package exports
│       ├── core.py              # Core RAG system orchestration
│       ├── indexing.py          # Document loading and indexing
│       ├── retrieval.py         # Document retrieval with FAISS
│       ├── generation.py        # Response generation with LLMs
│       └── cli.py               # Command-line interface
├── tests/                       # Test suite
├── docs/                        # Documentation
├── examples/                    # Usage examples
├── pyproject.toml              # Project configuration
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### Supported Document Types

- **Text files**: `.txt`, `.md`, `.rst`
- **PDFs**: `.pdf` 
- **Microsoft Office**: `.docx`, `.pptx`
- **Web content**: `.html`, `.xml`
- **Code files**: `.py`, `.js`, `.json`, etc.

## Development

### Setting up the development environment

```bash
# Clone the repository
git clone https://github.com/zakamai/rag-in-python.git
cd rag-in-python

# Install with development dependencies
uv sync --dev

# Set up pre-commit hooks
pre-commit install
```

### Running tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=rag_in_python

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "unit"      # Unit tests only
```

### Code quality

```bash
# Format code
black src/ tests/

# Lint code  
ruff src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files
```

### Building the package

```bash
# Using uv (recommended)
uv build

# Using standard tools
python -m build
```

## Architecture

### Core Components

1. **RAGSystem** (`core.py`) - Main orchestrator that coordinates all components
2. **DocumentIndexer** (`indexing.py`) - Handles document loading and FAISS indexing
3. **HybridRetriever** (`retrieval.py`) - Retrieves relevant documents using vector similarity
4. **ResponseGenerator** (`generation.py`) - Generates responses using retrieved context

### Design Principles

- **Modularity**: Each component has a single responsibility
- **Type Safety**: Full type hints for better IDE support and fewer bugs
- **Extensibility**: Easy to swap out components (e.g., different LLMs, vector stores)
- **Performance**: Batch processing and efficient vector operations
- **Observability**: Comprehensive logging throughout the system

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) - For the powerful RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) - For efficient similarity search
- [OpenAI](https://openai.com/) - For the language models and embeddings

## Support

- 📖 [Documentation](https://github.com/zakamai/rag-in-python/blob/main/README.md)
- 🐛 [Issue Tracker](https://github.com/zakamai/rag-in-python/issues)
- 💬 [Discussions](https://github.com/zakamai/rag-in-python/discussions)

---

Built with ❤️ using modern Python practices and tools.
