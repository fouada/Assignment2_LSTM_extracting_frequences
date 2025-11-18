# Contributing to LSTM Frequency Extraction System

First off, thank you for considering contributing to this project! 🎉

It's people like you that make this project a great learning resource for the deep learning community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Style Guidelines](#style-guidelines)
  - [Python Code Style](#python-code-style)
  - [Documentation Style](#documentation-style)
  - [Commit Messages](#commit-messages)
- [Development Setup](#development-setup)
- [Testing Guidelines](#testing-guidelines)
- [Community](#community)

---

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Use this template:**

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With configuration '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
 - OS: [e.g., macOS 14.0, Ubuntu 22.04]
 - Python version: [e.g., 3.10.12]
 - PyTorch version: [e.g., 2.1.0]
 - Device: [e.g., CPU, CUDA, MPS]

**Additional context**
Add any other context about the problem here.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

**Use this template:**

```markdown
**Is your feature request related to a problem?**
A clear description of the problem. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
A clear description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.

**Would you like to work on this?**
Yes/No - Let us know if you'd like to implement this yourself!
```

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue` - Issues that are good for newcomers
- `help wanted` - Issues that need assistance
- `documentation` - Documentation improvements
- `enhancement` - New features or improvements

### Pull Requests

1. **Fork the repository** and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clear, commented code
   - Follow the [Python Code Style](#python-code-style)
   - Add tests if applicable
   - Update documentation

3. **Test your changes:**
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Check code style
   black src/ tests/ --check
   flake8 src/ tests/
   
   # Type checking
   mypy src/
   ```

4. **Commit your changes:**
   ```bash
   git commit -m "feat: add amazing new feature"
   ```
   See [Commit Messages](#commit-messages) for guidelines.

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request:**
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe your changes in detail
   - Include screenshots for UI changes

**Pull Request Template:**

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues
Fixes #(issue number)

## How Has This Been Tested?
Describe the tests you ran to verify your changes.

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)
Add screenshots to help explain your changes.
```

---

## Style Guidelines

### Python Code Style

We follow **PEP 8** with some modifications. Use these tools:

#### Code Formatting
```bash
# Format code with black
black src/ tests/ main.py

# Configuration in pyproject.toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']
```

#### Linting
```bash
# Check with flake8
flake8 src/ tests/

# Configuration in .flake8
[flake8]
max-line-length = 100
exclude = .git,__pycache__,venv
ignore = E203, W503
```

#### Type Hints
- Use type hints for all function signatures
- Use `mypy` for type checking

```python
# Good ✅
def calculate_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Squared Error.
    
    Args:
        predictions: Model predictions, shape (N,)
        targets: Ground truth values, shape (N,)
        
    Returns:
        Mean squared error value
    """
    return np.mean((predictions - targets) ** 2)

# Bad ❌
def calculate_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)
```

#### Docstrings
Use **Google Style** docstrings:

```python
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int,
    device: torch.device
) -> Dict[str, List[float]]:
    """Train the LSTM model.
    
    Args:
        model: The neural network model to train
        data_loader: DataLoader providing training samples
        epochs: Number of training epochs
        device: Device to train on (cpu, cuda, mps)
        
    Returns:
        Dictionary containing training history:
            - 'loss': List of loss values per epoch
            - 'accuracy': List of accuracy values per epoch
            
    Raises:
        ValueError: If epochs < 1
        RuntimeError: If training fails
        
    Example:
        >>> model = create_model(config)
        >>> history = train_model(model, train_loader, epochs=50, device='cuda')
        >>> print(f"Final loss: {history['loss'][-1]}")
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    
    # Training implementation
    ...
```

### Documentation Style

#### Markdown
- Use clear, descriptive headings
- Include code examples
- Add links to related documentation
- Use emojis sparingly for visual guidance (✅, ❌, 🎯, 📊)

#### Code Comments
```python
# Good ✅
# Reset LSTM state at frequency boundaries to prevent
# cross-contamination between different frequency sequences
if is_first_sample:
    model.reset_state()

# Bad ❌
# Reset state
if is_first_sample:
    model.reset_state()
```

### Commit Messages

Follow **Conventional Commits** specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

#### Examples

```bash
# Good ✅
feat(model): add attention mechanism to LSTM
fix(training): resolve state management bug in batch processing
docs(readme): update installation instructions for M1 Macs
test(model): add unit tests for state reset functionality
perf(inference): optimize batch processing for 2x speedup

# Bad ❌
updated stuff
fix bug
changes
```

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/lstm-frequency-extraction.git
cd lstm-frequency-extraction

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/lstm-frequency-extraction.git
```

### 2. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n lstm-freq python=3.10
conda activate lstm-freq
```

### 3. Install Dependencies

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Or use UV (faster)
uv pip install -e .
```

### 4. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up git hooks
pre-commit install
```

This will automatically run checks before each commit.

### 5. Verify Setup

```bash
# Run tests
pytest tests/ -v

# Check code style
black src/ tests/ --check
flake8 src/ tests/

# Type checking
mypy src/

# Run a quick training
python main.py
```

---

## Testing Guidelines

### Writing Tests

We use **pytest** for testing. Place tests in `tests/` directory.

#### Test Structure

```python
"""
tests/test_model.py

Test suite for LSTM model.
"""
import pytest
import torch
from src.models.lstm_extractor import StatefulLSTMExtractor


class TestStatefulLSTM:
    """Test cases for StatefulLSTMExtractor."""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return StatefulLSTMExtractor(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model.input_size == 5
        assert model.hidden_size == 64
        assert model.num_layers == 2
        
    def test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        batch_size = 4
        seq_len = 10
        input_size = 5
        
        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 1)
        
    def test_state_management(self, model):
        """Test that state is managed correctly."""
        x = torch.randn(1, 10, 5)
        
        # First forward pass
        _ = model(x, reset_state=True)
        state_1 = model.hidden_state.clone()
        
        # Second forward pass (state should persist)
        _ = model(x, reset_state=False)
        state_2 = model.hidden_state
        
        # States should be different
        assert not torch.allclose(state_1, state_2)
```

#### Test Coverage

Aim for **80%+** code coverage:

```bash
# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

#### Test Categories

1. **Unit Tests** (`test_*.py`)
   - Test individual functions/classes
   - Fast execution
   - Isolated from dependencies

2. **Integration Tests** (`test_integration.py`)
   - Test component interactions
   - May use real data
   - Slower execution

3. **Performance Tests** (`test_performance.py`)
   - Benchmark critical paths
   - Memory profiling
   - Speed tests

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model.py -v

# Specific test function
pytest tests/test_model.py::test_forward_pass -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Parallel execution (faster)
pytest tests/ -n auto

# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s
```

---

## Project-Specific Guidelines

### Adding New Features

#### 1. New Model Architecture

```python
# src/models/your_model.py

import torch
import torch.nn as nn
from typing import Optional, Tuple


class YourNewModel(nn.Module):
    """
    Your model description.
    
    This model implements [explain architecture] for frequency extraction.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layers
        ...
        
    Example:
        >>> model = YourNewModel(input_size=5, hidden_size=128)
        >>> output = model(input_tensor)
    """
    
    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__()
        # Implementation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape (batch, seq_len, input_size)
            
        Returns:
            Output tensor, shape (batch, seq_len, 1)
        """
        # Implementation
```

Add corresponding tests:

```python
# tests/test_your_model.py

def test_your_new_model():
    model = YourNewModel(input_size=5, hidden_size=128)
    x = torch.randn(4, 10, 5)
    output = model(x)
    assert output.shape == (4, 10, 1)
```

Update documentation:

```markdown
# docs/ARCHITECTURE.md

## Your New Model

Description of your model...

### Usage

```python
from src.models.your_model import YourNewModel
...
```
```

#### 2. New Evaluation Metric

```python
# src/evaluation/metrics.py

def your_new_metric(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Calculate your new metric.
    
    Args:
        predictions: Model predictions, shape (N,)
        targets: Ground truth, shape (N,)
        
    Returns:
        Metric value
        
    Example:
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> true = np.array([1.1, 2.1, 2.9])
        >>> metric = your_new_metric(pred, true)
    """
    # Implementation
```

#### 3. New Visualization

```python
# src/visualization/plotter.py

def plot_your_new_viz(
    data: np.ndarray,
    save_path: Optional[Path] = None,
    **kwargs
) -> None:
    """
    Create your new visualization.
    
    Args:
        data: Data to visualize
        save_path: Optional path to save figure
        **kwargs: Additional plotting parameters
        
    Example:
        >>> data = np.random.randn(100)
        >>> plot_your_new_viz(data, save_path='viz.png')
    """
    # Implementation
```

---

## Community

### Getting Help

- 📖 **Documentation:** Check [docs/](docs/) first
- 🐛 **Issues:** Search existing issues or create a new one
- 💬 **Discussions:** Use GitHub Discussions for questions
- 📧 **Email:** Contact maintainers for sensitive issues

### Staying Updated

```bash
# Sync your fork with upstream
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Recognition

Contributors will be:
- Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Mentioned in release notes
- Given credit in relevant documentation

---

## Questions?

Don't hesitate to ask! We're here to help:

- Open a [GitHub Discussion](https://github.com/yourusername/lstm-frequency-extraction/discussions)
- Create an [Issue](https://github.com/yourusername/lstm-frequency-extraction/issues)
- Contact maintainers directly

---

## Thank You! 🙏

Your contributions make this project better for everyone. We appreciate your time and effort!

**Happy coding!** 🚀

---

<div align="center">

[🏠 Home](README.md) • [📋 Code of Conduct](CODE_OF_CONDUCT.md) • [🔒 Security](SECURITY.md)

</div>

