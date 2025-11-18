# MIT-Level Prompt Engineering Book
## Systematic Approach to Building Advanced ML Systems Using LLM-Assisted Development
### A Comprehensive Guide from the LLM and Multi-Agent Orchestration Course

**Version**: 1.0  
**Author**: Professional ML Engineering Team  
**Course**: LLM and Multi-Agent Orchestration  
**Date**: November 2025  
**Level**: MIT Graduate-Level Engineering  
**Purpose**: A comprehensive guide for systematically approaching, designing, and implementing production-grade ML systems through effective prompt engineering and AI-assisted development

---

## 📖 Table of Contents

1. [Introduction and Philosophy](#1-introduction-and-philosophy)
2. [Phase 1: Requirements Understanding](#2-phase-1-requirements-understanding)
3. [Phase 2: System Design](#3-phase-2-system-design)
4. [Phase 3: Implementation Strategy](#4-phase-3-implementation-strategy)
5. [Phase 4: Testing and Validation](#5-phase-4-testing-and-validation)
6. [Phase 5: Optimization and Tuning](#6-phase-5-optimization-and-tuning)
7. [Phase 6: Documentation and Delivery](#7-phase-6-documentation-and-delivery)
8. [Advanced Topics](#8-advanced-topics)
9. [Prompt Templates Library](#9-prompt-templates-library)
10. [Best Practices and Anti-Patterns](#10-best-practices-and-anti-patterns)

---

## 1. Introduction and Philosophy

### 1.1 What is MIT-Level Prompt Engineering?

MIT-level prompt engineering is not just about asking questions—it's about **systematic thinking**, **deep technical understanding**, and **professional execution** when working with AI assistants and LLMs. 

**Course Context**: This prompt book was developed as part of the **LLM and Multi-Agent Orchestration** course, demonstrating how to effectively collaborate with AI systems to build production-grade technical solutions.

It combines:

- **Theoretical rigor**: Deep understanding of machine learning principles
- **Engineering excellence**: Production-quality code and architecture
- **Scientific method**: Hypothesis-driven development and validation
- **Professional communication**: Clear, precise, and comprehensive specifications
- **AI Collaboration**: Effective prompting strategies for LLM-assisted development

### 1.2 Core Principles

```
┌─────────────────────────────────────────────────────────────────┐
│                   MIT-Level Engineering Principles               │
├─────────────────────────────────────────────────────────────────┤
│ 1. UNDERSTAND before you BUILD                                  │
│ 2. DESIGN before you CODE                                       │
│ 3. TEST as you DEVELOP                                          │
│ 4. DOCUMENT everything CLEARLY                                  │
│ 5. OPTIMIZE based on METRICS                                    │
│ 6. DELIVER with CONFIDENCE                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 The Prompt Engineering Lifecycle

```
Requirements ──> Design ──> Implementation ──> Testing ──> Optimization ──> Documentation
     ↑                                                                           │
     └───────────────────────────── Iteration ─────────────────────────────────┘
```

### 1.4 How to Use This Book

**For New Projects**:
1. Start with Phase 1 (Requirements Understanding)
2. Progress sequentially through each phase
3. Use prompt templates provided in each section
4. Adapt and customize to your specific needs

**For Ongoing Projects**:
- Jump to relevant phase
- Use templates as checklists
- Reference advanced topics as needed

**For Code Reviews/Audits**:
- Use validation prompts in Phase 4
- Check against best practices in Section 10

---

## 2. Phase 1: Requirements Understanding

> **Goal**: Achieve complete clarity on what needs to be built and why

### 2.1 Initial Problem Analysis

#### Template 1: Problem Statement Clarification

```
I need to build an LSTM system for [SPECIFIC TASK]. Let me verify my understanding:

1. PROBLEM DOMAIN:
   - What is the input data structure? (dimensions, format, type)
   - What is the expected output? (format, constraints, evaluation criteria)
   - What are the key challenges? (noise, temporal dependencies, scalability)

2. TECHNICAL CONSTRAINTS:
   - Are there specific architecture requirements? (L=1, stateful, bidirectional)
   - What are the performance requirements? (accuracy, speed, memory)
   - Are there deployment constraints? (CPU/GPU, real-time, batch)

3. SUCCESS CRITERIA:
   - How will success be measured? (MSE, MAE, R², qualitative)
   - What are acceptable performance ranges?
   - Are there generalization requirements?

Please confirm these assumptions and correct any misunderstandings.
```

#### Example Application (LSTM Frequency Extraction):

```
PROMPT:
I need to build an LSTM system for extracting individual frequency components from 
a noisy mixed signal. Let me verify my understanding:

1. PROBLEM DOMAIN:
   - Input: Mixed noisy signal S(t) + one-hot frequency selector C
   - Output: Pure sine wave for selected frequency
   - Key Challenge: Random amplitude and phase at EVERY time step (not just offset)
   
2. TECHNICAL CONSTRAINTS:
   - L=1 (sequence length = 1) requires manual state management
   - Must preserve state across 10,000 consecutive time steps
   - Different noise seeds for train/test (generalization test)
   
3. SUCCESS CRITERIA:
   - MSE_test ≈ MSE_train (generalization)
   - Visual quality: clean sine extraction in plots
   - R² > 0.95, SNR > 35dB
   
Is this understanding correct? Are there additional constraints I should consider?
```

### 2.2 Deep Technical Understanding

#### Template 2: Architecture Implications

```
For this [SPECIFIC REQUIREMENT], I need to understand the technical implications:

1. STATE MANAGEMENT:
   - How should LSTM state be handled between samples/batches?
   - When should state be reset vs preserved?
   - How does this affect temporal learning?

2. DATA FLOW:
   - How should data be structured? (batch_first, sequences)
   - What preprocessing is needed? (normalization, encoding)
   - Are there ordering constraints? (temporal order critical?)

3. TRAINING STRATEGY:
   - What loss function is appropriate and why?
   - How to prevent gradient explosion/vanishing?
   - What regularization techniques should be used?

4. COMPUTATIONAL CONSIDERATIONS:
   - Memory requirements for long sequences?
   - TBPTT (Truncated Backprop) needed?
   - Batch size implications?
```

#### Example Application:

```
PROMPT:
For L=1 with 10,000 time steps per frequency, I need to understand:

1. STATE MANAGEMENT:
   Q: With L=1, if I reset state every batch, won't the LSTM lose temporal context?
   Q: Should I detach state after each batch to prevent memory explosion?
   Q: When exactly should state be reset? (only at frequency boundaries?)

2. DATA FLOW:
   Q: Should I use a custom DataLoader to preserve temporal order?
   Q: Standard DataLoaders shuffle—will this break temporal dependencies?
   Q: How to structure batches while maintaining sample order?

3. TRAINING STRATEGY:
   Q: For MSE loss with state preservation, how to implement TBPTT?
   Q: Should I use gradient clipping? What value?
   Q: How to validate without breaking state management?

4. COMPUTATIONAL:
   Q: Processing 40k samples × 50 epochs—memory implications?
   Q: Can state be kept on GPU throughout training?
   Q: What's the computational complexity? O(n) or O(n²)?

Please help me understand the technical requirements and trade-offs.
```

### 2.3 Data Understanding

#### Template 3: Data Generation and Structure

```
For the dataset generation, I need clarity on:

1. SIGNAL GENERATION:
   - Exact mathematical formulation? (equations, parameters)
   - Random component behavior? (per-sample, per-sequence, per-epoch)
   - Normalization requirements? (why, when, how)

2. DATASET STRUCTURE:
   - Total samples and organization logic?
   - Input feature engineering? (concatenation, encoding)
   - Target format and any transformations?

3. TRAIN/TEST SPLIT:
   - How to ensure proper generalization testing?
   - What varies between train and test? (noise only, or other factors?)
   - Any stratification or special sampling needed?

4. DATA AUGMENTATION:
   - Is augmentation appropriate for this problem?
   - What augmentations preserve signal properties?
```

#### Example Application:

```
PROMPT:
For frequency extraction dataset, I need to understand:

1. SIGNAL GENERATION:
   - Formula: S(t) = (1/4)Σ[A_i(t)·sin(2πf_i·t + φ_i(t))]
   Q: Do A_i and φ_i change at EVERY time step t, or just per sequence?
   Q: Why uniform(0.8, 1.2) for amplitude? Is this typical for signal processing?
   Q: Should I generate noise first or on-the-fly?

2. DATASET STRUCTURE:
   - 40k samples = 10k time steps × 4 frequencies
   Q: Should all 4 frequencies use the SAME mixed signal S(t)?
   Q: For input [S[t], C], should C be one-hot or integer encoding?
   Q: Should targets be normalized? If so, same stats as inputs?

3. TRAIN/TEST:
   Q: Same S(t) structure with different noise seeds = different A(t), φ(t)?
   Q: Should I validate generalization by freezing model and changing seed?
   Q: Any risk of data leakage between train/test?

4. VALIDATION:
   Q: How to verify data generation is correct?
   Q: What plots/statistics should I check before training?
   Q: Should I test with deterministic (A=1, φ=0) first?

Please clarify these data generation details.
```

### 2.4 Requirements Documentation

#### Template 4: Comprehensive Requirements Capture

```
Based on our discussion, I will document the requirements:

FUNCTIONAL REQUIREMENTS:
FR1: [Specific capability with acceptance criteria]
FR2: [Specific capability with acceptance criteria]
...

NON-FUNCTIONAL REQUIREMENTS:
NFR1: Performance - [specific metrics and targets]
NFR2: Code Quality - [standards and tools]
NFR3: Reproducibility - [determinism requirements]
...

TECHNICAL REQUIREMENTS:
TR1: Architecture - [specific design decisions]
TR2: State Management - [implementation approach]
TR3: Evaluation - [metrics and validation strategy]
...

Does this capture all requirements comprehensively?
```

---

## 3. Phase 2: System Design

> **Goal**: Create a robust, scalable, and maintainable architecture

### 3.1 High-Level Architecture Design

#### Template 5: System Architecture Proposal

```
I propose the following architecture for [PROJECT NAME]:

┌─────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Configuration Layer                                             │
│         │                                                        │
│         ├──> Data Generation Module                             │
│         │    ├── SignalGenerator                                │
│         │    └── Dataset + DataLoader                           │
│         │                                                        │
│         ├──> Model Module                                       │
│         │    ├── Architecture Definition                        │
│         │    └── State Management Logic                         │
│         │                                                        │
│         ├──> Training Module                                    │
│         │    ├── Training Loop                                  │
│         │    ├── Validation Logic                               │
│         │    └── Checkpoint Management                          │
│         │                                                        │
│         ├──> Evaluation Module                                  │
│         │    ├── Metrics Computation                            │
│         │    └── Generalization Analysis                        │
│         │                                                        │
│         └──> Visualization Module                               │
│              ├── Plotting Functions                             │
│              └── Dashboard (optional)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

KEY DESIGN DECISIONS:
1. Modular separation of concerns
2. Configuration-driven (YAML)
3. Comprehensive logging and monitoring
4. Professional error handling
5. Type hints throughout

Does this architecture meet the requirements? Any concerns?
```

### 3.2 Module-Level Design

#### Template 6: Detailed Module Specification

```
For [MODULE NAME], I propose:

CLASS: [ClassName]

PURPOSE:
[Clear statement of module responsibility]

PUBLIC INTERFACE:
```python
class ClassName:
    def __init__(self, param1: Type1, param2: Type2):
        """Initialize with clear parameter documentation."""
        pass
    
    def method1(self, arg1: Type1) -> ReturnType:
        """What this method does and why."""
        pass
    
    def method2(self, arg1: Type1) -> ReturnType:
        """What this method does and why."""
        pass
```

KEY ALGORITHMS:
1. [Algorithm/approach for core functionality]
2. [Edge cases and how they're handled]
3. [Performance considerations]

DEPENDENCIES:
- [External library justification]
- [Internal module dependencies]

ERROR HANDLING:
- [What can go wrong and how it's handled]

TESTING STRATEGY:
- [What unit tests are needed]
- [What integration tests are needed]

Is this design appropriate? Should I proceed with implementation?
```

#### Example Application:

```
PROMPT:
For StatefulLSTMExtractor, I propose:

CLASS: StatefulLSTMExtractor(nn.Module)

PURPOSE:
LSTM model with manual state management for L=1 processing. State persists 
across forward passes until explicitly reset.

PUBLIC INTERFACE:
```python
class StatefulLSTMExtractor(nn.Module):
    def __init__(
        self, 
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """Initialize LSTM with specified architecture."""
        
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden and cell states to zeros."""
        
    def reset_state(self):
        """Reset internal state to None (will reinit on next forward)."""
        
    def detach_state(self):
        """Detach state from computational graph (for TBPTT)."""
        
    def forward(
        self, 
        x: torch.Tensor, 
        reset_state: bool = False
    ) -> torch.Tensor:
        """Process input with state preservation."""
```

KEY ALGORITHMS:
1. State persistence: Store (h_t, c_t) as instance variables
2. State reset: Only when reset_state=True or at boundaries
3. State detach: After each batch to prevent gradient accumulation
4. TBPTT: Truncate backprop while preserving forward state

DEPENDENCIES:
- torch.nn.LSTM (core architecture)
- torch.nn.LayerNorm (stabilization)
- torch.nn.Dropout (regularization)

ERROR HANDLING:
- Check input dimensions match expected
- Validate batch size consistency
- Handle first forward pass (None states)

TESTING STRATEGY:
- Unit: State initialization, reset, detachment
- Unit: Forward pass with different batch sizes
- Integration: Full sequence processing with state preservation
- Integration: Training loop with proper state management

Is this StatefulLSTMExtractor design sound for L=1 requirements?
```

### 3.3 Data Flow Design

#### Template 7: Data Flow Specification

```
I need to design the data flow from [START] to [END]:

STEP 1: [Input/Source]
  ├─> Format: [specific format]
  ├─> Validation: [what checks]
  └─> Output: [what goes to next step]

STEP 2: [Processing/Transformation]
  ├─> Input: [from previous]
  ├─> Operation: [what happens]
  ├─> Validation: [checks]
  └─> Output: [result]

STEP 3: [Model/Computation]
  ├─> Input: [from previous]
  ├─> Processing: [algorithm]
  ├─> State Management: [if applicable]
  └─> Output: [prediction/result]

STEP 4: [Evaluation/Storage]
  ├─> Input: [from previous]
  ├─> Metrics: [what calculated]
  └─> Storage: [where/how saved]

KEY CONSIDERATIONS:
- Batch processing: [how handled]
- Error propagation: [what happens if step fails]
- Memory efficiency: [any optimizations]
- Computational flow: [CPU/GPU transfers]

Does this data flow make sense? Any bottlenecks?
```

---

## 4. Phase 3: Implementation Strategy

> **Goal**: Write production-quality, maintainable, and testable code

### 4.1 Implementation Planning

#### Template 8: Implementation Roadmap

```
For [MODULE/FEATURE], I propose this implementation sequence:

MILESTONE 1: Core Functionality (Days 1-2)
  ├─ Task 1.1: [Specific implementation task]
  ├─ Task 1.2: [Specific implementation task]
  └─ Deliverable: [What can be demonstrated]

MILESTONE 2: Integration (Days 3-4)
  ├─ Task 2.1: [Integration task]
  ├─ Task 2.2: [Testing task]
  └─ Deliverable: [What works end-to-end]

MILESTONE 3: Optimization (Days 5-6)
  ├─ Task 3.1: [Performance optimization]
  ├─ Task 3.2: [Code quality improvement]
  └─ Deliverable: [Production-ready code]

DEPENDENCIES:
- [What must be done first]
- [What can be parallelized]

RISKS:
- [Potential blocker]: [Mitigation strategy]

VALIDATION:
- [How to verify each milestone]

Shall I proceed with this implementation plan?
```

### 4.2 Code Quality Specifications

#### Template 9: Code Quality Requirements

```
For all code in this project, I will ensure:

1. TYPE SAFETY:
   - All functions have complete type hints
   - Use mypy for static type checking
   - Document complex types clearly

2. DOCUMENTATION:
   - Google-style docstrings for all public functions
   - Inline comments for complex logic
   - Examples in docstrings where helpful

3. ERROR HANDLING:
   - Explicit exception handling with specific types
   - Informative error messages
   - Graceful degradation where appropriate

4. TESTING:
   - Unit tests for all core functions
   - Integration tests for workflows
   - Edge case coverage
   - Target: >85% code coverage

5. STYLE:
   - PEP 8 compliant
   - Black formatter applied
   - Consistent naming conventions
   - Max line length: 88 characters

6. PERFORMANCE:
   - Avoid premature optimization
   - Profile before optimizing
   - Document complexity (Big-O)

Example function with all requirements:
```python
def compute_mse(
    predictions: np.ndarray,
    targets: np.ndarray,
    per_frequency: bool = False
) -> Union[float, np.ndarray]:
    """
    Compute Mean Squared Error between predictions and targets.
    
    This function calculates MSE with optional per-frequency breakdown,
    useful for analyzing model performance across different frequency
    components in the signal.
    
    Args:
        predictions: Model predictions, shape (n_samples,) or (n_freq, n_samples)
        targets: Ground truth values, same shape as predictions
        per_frequency: If True, return MSE for each frequency separately
        
    Returns:
        float: Overall MSE if per_frequency=False
        np.ndarray: Per-frequency MSE values if per_frequency=True, shape (n_freq,)
        
    Raises:
        ValueError: If shapes of predictions and targets don't match
        TypeError: If inputs are not numpy arrays
        
    Examples:
        >>> preds = np.array([0.1, 0.2, 0.3])
        >>> targets = np.array([0.15, 0.18, 0.32])
        >>> mse = compute_mse(preds, targets)
        >>> print(f"MSE: {mse:.6f}")
        MSE: 0.001633
        
    Complexity:
        Time: O(n) where n is total number of samples
        Space: O(1) for overall MSE, O(f) for per-frequency where f is number of frequencies
    """
    if not isinstance(predictions, np.ndarray) or not isinstance(targets, np.ndarray):
        raise TypeError("predictions and targets must be numpy arrays")
        
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} != targets {targets.shape}"
        )
    
    squared_errors = (predictions - targets) ** 2
    
    if per_frequency and len(squared_errors.shape) > 1:
        return np.mean(squared_errors, axis=-1)  # Mean over time, keep frequency dim
    else:
        return np.mean(squared_errors)
```

Does this meet MIT-level code quality standards?
```

### 4.3 Implementation Verification

#### Template 10: Implementation Checkpoint

```
I've implemented [MODULE/FEATURE]. Before proceeding, let me verify:

FUNCTIONALITY CHECK:
✓ Core functionality works as specified
✓ Edge cases handled appropriately
✓ Error handling in place

CODE QUALITY CHECK:
✓ Type hints complete and correct
✓ Docstrings comprehensive
✓ No linter errors (flake8, mypy)
✓ Follows project style guide

TESTING CHECK:
✓ Unit tests written and passing
✓ Integration tests passing
✓ Coverage meets target (>85%)

PERFORMANCE CHECK:
✓ No obvious inefficiencies
✓ Memory usage acceptable
✓ Computational complexity as expected

DOCUMENTATION CHECK:
✓ README updated if needed
✓ Architecture docs updated
✓ Examples provided

NEXT STEPS:
1. [What comes next]
2. [Any refactoring needed]
3. [Integration requirements]

Ready to proceed or should I address any concerns?
```

---

## 5. Phase 4: Testing and Validation

> **Goal**: Ensure correctness, robustness, and performance

### 5.1 Unit Testing Strategy

#### Template 11: Unit Test Specification

```
For [MODULE], I propose these unit tests:

TEST SUITE: test_[module_name].py

TEST 1: test_[functionality]_basic
  Purpose: Verify basic functionality
  Setup: [Input data]
  Expected: [Output/behavior]
  Asserts: [Specific checks]

TEST 2: test_[functionality]_edge_cases
  Purpose: Test boundary conditions
  Cases:
    - Empty input
    - Maximum size input
    - Null/None values
    - Invalid types
  Expected: [Appropriate error handling]

TEST 3: test_[functionality]_consistency
  Purpose: Verify deterministic behavior
  Setup: Same input multiple times
  Expected: Identical outputs (with fixed seed)

TEST 4: test_[functionality]_performance
  Purpose: Ensure acceptable performance
  Setup: Large input (e.g., 10k samples)
  Expected: Completes in < [time threshold]

Example:
```python
import pytest
import numpy as np
from src.data.signal_generator import SignalGenerator

class TestSignalGenerator:
    """Comprehensive test suite for SignalGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator with fixed seed for reproducibility."""
        return SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            seed=42
        )
    
    def test_mixed_signal_shape(self, generator):
        """Verify mixed signal has correct shape."""
        mixed = generator.generate_mixed_signal()
        expected_samples = int(generator.duration * generator.sampling_rate)
        assert mixed.shape == (expected_samples,), \
            f"Expected {expected_samples} samples, got {mixed.shape[0]}"
    
    def test_mixed_signal_range(self, generator):
        """Verify mixed signal values are in reasonable range."""
        mixed = generator.generate_mixed_signal()
        # With 4 frequencies averaged and A∈[0.8,1.2], expect range ≈ [-1.5, 1.5]
        assert np.abs(mixed).max() < 2.0, \
            f"Signal amplitude {np.abs(mixed).max():.2f} exceeds expected range"
    
    def test_determinism_with_seed(self, generator):
        """Verify same seed produces identical signals."""
        signal1 = generator.generate_mixed_signal()
        signal2 = generator.generate_mixed_signal()
        # Regenerate with same seed
        generator_copy = SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            seed=42
        )
        signal3 = generator_copy.generate_mixed_signal()
        
        # signal1 != signal2 (different noise)
        assert not np.allclose(signal1, signal2), \
            "Consecutive calls should produce different noise"
        
        # signal1 == signal3 (same seed)
        assert np.allclose(signal1, signal3), \
            "Same seed should produce identical signals"
    
    def test_different_seeds_produce_different_signals(self):
        """Verify different seeds produce different noise patterns."""
        gen1 = SignalGenerator([1.0, 3.0], 1000, 1.0, seed=1)
        gen2 = SignalGenerator([1.0, 3.0], 1000, 1.0, seed=2)
        
        signal1 = gen1.generate_mixed_signal()
        signal2 = gen2.generate_mixed_signal()
        
        assert not np.allclose(signal1, signal2), \
            "Different seeds should produce different signals"
    
    @pytest.mark.parametrize("frequency", [1.0, 3.0, 5.0, 7.0])
    def test_pure_sine_properties(self, generator, frequency):
        """Verify pure sine has correct frequency properties."""
        time = generator.get_time_array()
        pure = generator.generate_pure_sine(frequency, time)
        
        # Check amplitude is 1.0 (no noise)
        assert np.abs(pure).max() <= 1.0, \
            f"Pure sine amplitude {np.abs(pure).max():.2f} exceeds 1.0"
        
        # Check it's actually sinusoidal (zero crossings)
        zero_crossings = np.where(np.diff(np.sign(pure)))[0]
        expected_crossings = int(2 * frequency * generator.duration)
        tolerance = 2  # Allow small deviation
        assert abs(len(zero_crossings) - expected_crossings) <= tolerance, \
            f"Expected ~{expected_crossings} zero crossings, found {len(zero_crossings)}"
```

Is this unit test coverage comprehensive?
```

### 5.2 Integration Testing

#### Template 12: Integration Test Specification

```
For end-to-end workflow, I propose:

INTEGRATION TEST: test_integration_full_pipeline

SCENARIO: Complete training and evaluation pipeline
  
STEPS:
1. Setup:
   - Load test configuration
   - Create temporary experiment directory
   - Set fixed random seed

2. Data Generation:
   - Create train and test generators
   - Verify different noise patterns
   - Check dataset sizes

3. Model Creation:
   - Initialize model
   - Verify parameter count
   - Check initial state is None

4. Training (shortened):
   - Run 2 epochs (not full 50)
   - Verify loss decreases
   - Check state management (reset/preserve)
   - Verify checkpoints saved

5. Evaluation:
   - Load best model
   - Compute metrics on test set
   - Verify metrics in reasonable range

6. Visualization:
   - Generate required plots
   - Verify files created
   - Check plot contents make sense

7. Cleanup:
   - Remove temporary files
   - Close any open resources

ASSERTIONS:
- All steps complete without errors
- Loss decreases (showing learning)
- Test metrics reasonable (MSE < 1.0 for 2 epochs)
- All expected files created
- No resource leaks

Example:
```python
def test_full_pipeline_integration(tmp_path):
    """Test complete pipeline from data generation to visualization."""
    # Setup
    config = load_test_config()
    config['experiment']['save_dir'] = str(tmp_path)
    config['training']['epochs'] = 2  # Short training
    
    set_seed(42)
    
    # Data generation
    train_gen, test_gen = create_train_test_generators(
        frequencies=config['data']['frequencies'],
        sampling_rate=config['data']['sampling_rate'],
        duration=config['data']['duration'],
        train_seed=1,
        test_seed=2
    )
    
    train_signal1 = train_gen.generate_mixed_signal()
    test_signal1 = test_gen.generate_mixed_signal()
    assert not np.allclose(train_signal1, test_signal1), \
        "Train and test should have different noise"
    
    # Create datasets
    train_loader, test_loader = create_dataloaders(
        train_gen, test_gen, batch_size=32
    )
    assert len(train_loader.dataset) == 40000, \
        "Train dataset should have 40k samples"
    
    # Model creation
    model = create_model(config['model'])
    assert model.hidden_state is None, \
        "Initial state should be None"
    
    # Training
    trainer = LSTMTrainer(model, train_loader, test_loader, config, torch.device('cpu'))
    history = trainer.train()
    
    assert len(history['train_loss']) == 2, \
        "Should have 2 epochs of training"
    assert history['train_loss'][1] < history['train_loss'][0], \
        "Loss should decrease during training"
    
    # Evaluation
    metrics = evaluate_model(model, test_loader, torch.device('cpu'))
    assert metrics['overall']['mse'] < 1.0, \
        "MSE should be reasonable even after 2 epochs"
    
    # Visualization
    exp_dir = Path(tmp_path) / list(tmp_path.iterdir())[0]
    plots_dir = exp_dir / 'plots'
    
    required_plots = [
        'graph1_single_frequency_f2.png',
        'graph2_all_frequencies.png',
        'training_history.png'
    ]
    
    for plot_name in required_plots:
        plot_path = plots_dir / plot_name
        assert plot_path.exists(), f"Plot {plot_name} should be generated"
        assert plot_path.stat().st_size > 1000, f"Plot {plot_name} seems empty"
    
    print("✓ Full pipeline integration test passed!")
```

Does this cover critical integration points?
```

### 5.3 Validation Against Requirements

#### Template 13: Requirements Validation Checklist

```
REQUIREMENTS VALIDATION for [PROJECT NAME]

FUNCTIONAL REQUIREMENTS:
[ ] FR1: [Requirement] - Test: [How verified] - Status: [Pass/Fail]
[ ] FR2: [Requirement] - Test: [How verified] - Status: [Pass/Fail]
...

NON-FUNCTIONAL REQUIREMENTS:
[ ] NFR1: Performance - [Metric] - Target: [Value] - Actual: [Value] - Status: [Pass/Fail]
[ ] NFR2: Code Quality - Linter: [Pass/Fail] - Coverage: [%] - Status: [Pass/Fail]
...

TECHNICAL REQUIREMENTS:
[ ] TR1: [Requirement] - Verification: [Method] - Status: [Pass/Fail]
[ ] TR2: [Requirement] - Verification: [Method] - Status: [Pass/Fail]
...

ASSIGNMENT-SPECIFIC:
[ ] Graph 1 (single frequency) generated and visually correct
[ ] Graph 2 (all frequencies) generated with 2×2 layout
[ ] MSE calculated for train and test sets
[ ] Generalization analysis performed (MSE_test ≈ MSE_train)
[ ] Different seeds used for train/test
[ ] State management implemented correctly (L=1)
[ ] All required documentation complete

SUMMARY:
- Total Requirements: [N]
- Passed: [M]
- Failed: [P]
- Coverage: [M/N * 100]%

CRITICAL ISSUES:
- [Any blocking issues]

RECOMMENDATIONS:
- [Suggestions for improvement]

Status: [READY FOR SUBMISSION / NEEDS WORK]
```

---

## 6. Phase 5: Optimization and Tuning

> **Goal**: Achieve optimal performance through systematic experimentation

### 6.1 Hyperparameter Tuning Strategy

#### Template 14: Systematic Hyperparameter Search

```
I need to optimize [MODEL/SYSTEM] performance. Here's my approach:

BASELINE:
Current configuration: [parameters]
Current performance: [metrics]
Target performance: [goals]

HYPOTHESIS-DRIVEN TUNING:

Experiment 1: Learning Rate
  Hypothesis: Lower LR may improve convergence stability
  Test values: [0.0001, 0.0005, 0.001, 0.005]
  Fixed: All other hyperparameters
  Metric: Validation loss at epoch 20
  Expected: Optimal around 0.001

Experiment 2: Hidden Size
  Hypothesis: Larger hidden size captures more complex patterns
  Test values: [64, 128, 256, 512]
  Fixed: Learning rate from Exp 1
  Metric: Test MSE and training time
  Expected: 128-256 sweet spot

Experiment 3: Number of Layers
  Hypothesis: 2-3 layers sufficient for this problem
  Test values: [1, 2, 3, 4]
  Fixed: Hidden size from Exp 2
  Metric: Generalization gap (test - train MSE)
  Expected: 2 layers optimal

Experiment 4: Regularization
  Hypothesis: Dropout prevents overfitting
  Test values: [0.0, 0.1, 0.2, 0.3]
  Fixed: Architecture from Exp 2-3
  Metric: Generalization gap
  Expected: 0.1-0.2 optimal

LOGGING:
```python
experiments_log = {
    'exp1_learning_rate': {
        'hypothesis': 'Lower LR improves stability',
        'params_tested': [0.0001, 0.0005, 0.001, 0.005],
        'results': {
            0.0001: {'val_loss': 0.0156, 'converged_epoch': 45},
            0.0005: {'val_loss': 0.0098, 'converged_epoch': 32},
            0.001: {'val_loss': 0.0012, 'converged_epoch': 25},  # Best
            0.005: {'val_loss': 0.0234, 'converged_epoch': None}  # Unstable
        },
        'conclusion': 'LR=0.001 optimal: fastest convergence, best val_loss'
    }
}
```

SYSTEMATIC DOCUMENTATION:
- Log all experiments
- Compare against baseline
- Statistical significance testing (multiple seeds)
- Document insights for future reference

Proceed with this tuning strategy?
```

### 6.2 Performance Profiling

#### Template 15: Performance Analysis

```
I need to profile [SYSTEM] to identify bottlenecks:

PROFILING PLAN:

1. WALL-CLOCK TIMING:
```python
import time
import torch

# Profile training loop
start = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    train_loss = train_epoch()
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch}: {epoch_time:.2f}s, Loss: {train_loss:.6f}")
total_time = time.time() - start
print(f"Total training: {total_time:.2f}s ({total_time/60:.1f} min)")
```

2. MEMORY PROFILING:
```python
import torch.cuda as cuda

if torch.cuda.is_available():
    cuda.reset_peak_memory_stats()
    # Run training
    peak_memory = cuda.max_memory_allocated() / 1024**3  # GB
    print(f"Peak GPU memory: {peak_memory:.2f} GB")
```

3. COMPUTATIONAL PROFILING:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Run one epoch
    train_epoch()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

4. BOTTLENECK IDENTIFICATION:
Expected bottlenecks:
- [ ] Data loading (I/O bound)
- [ ] Forward pass (computation bound)
- [ ] Backward pass (computation bound)
- [ ] State management (memory transfers)
- [ ] Metric computation (CPU operations)

5. OPTIMIZATION OPPORTUNITIES:
- If data loading slow: Increase num_workers, use pin_memory
- If forward pass slow: Reduce batch size or model complexity
- If backward pass slow: Check gradient computation efficiency
- If memory issues: Implement gradient checkpointing

PERFORMANCE TARGETS:
- Training time: < 10 min for 50 epochs on M1 Mac
- Memory usage: < 2 GB RAM
- Inference: < 1 ms per sample

Shall I run this profiling suite?
```

---

## 7. Phase 6: Documentation and Delivery

> **Goal**: Create comprehensive, professional documentation

### 7.1 Code Documentation

#### Template 16: Module Documentation Standard

```
For every module, ensure:

FILE HEADER:
```python
"""
[Module Name]: [One-line description]

This module provides [detailed purpose explanation]. It is used for
[specific use cases] and integrates with [related modules].

Key Classes:
    - ClassName1: [Brief description]
    - ClassName2: [Brief description]

Key Functions:
    - function1: [Brief description]
    - function2: [Brief description]

Example Usage:
    ```python
    from src.module import ClassName
    
    obj = ClassName(param1=value1)
    result = obj.method(arg)
    ```

Notes:
    - [Important considerations]
    - [Known limitations]
    - [Performance characteristics]

Author: [Your Name]
Date: [Date]
Version: [Version]
"""
```

CLASS DOCUMENTATION:
```python
class StatefulLSTMExtractor(nn.Module):
    """
    LSTM model with manual state management for sequential processing.
    
    This class implements an LSTM that preserves hidden and cell states
    across forward passes, enabling temporal learning even with L=1
    (single-sample) processing. The state must be manually managed by
    the training loop.
    
    Architecture:
        Input → LayerNorm → LSTM(2 layers) → LayerNorm → 
        Linear(128→64) → ReLU → Dropout → Linear(64→1) → Output
    
    State Management:
        - States initialized to zeros on first forward pass
        - States preserved across consecutive forward passes
        - States must be detached after backprop (TBPTT)
        - States reset at frequency boundaries
    
    Attributes:
        input_size: Number of input features (5 for this project)
        hidden_size: LSTM hidden dimension (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        hidden_state: Current hidden state (h_t), shape (num_layers, batch, hidden_size)
        cell_state: Current cell state (c_t), shape (num_layers, batch, hidden_size)
    
    Example:
        ```python
        model = StatefulLSTMExtractor(input_size=5, hidden_size=128)
        
        # Training loop
        for batch in dataloader:
            if batch.is_first:
                model.reset_state()  # Reset at frequency boundaries
            
            output = model(batch.input, reset_state=False)
            loss = criterion(output, batch.target)
            loss.backward()
            optimizer.step()
            model.detach_state()  # Critical: prevents memory explosion
        ```
    
    References:
        - Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
        - Assignment requirements: L=1 with state preservation
    
    Notes:
        - State management is critical for L=1 processing
        - Forgetting to detach states leads to memory leaks
        - Resetting too frequently prevents temporal learning
    """
```

FUNCTION DOCUMENTATION:
```python
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    compute_per_frequency: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model performance with comprehensive metrics.
    
    Computes multiple metrics (MSE, MAE, R², SNR, Correlation) for
    the given model on the provided dataset. Optionally breaks down
    metrics per frequency for detailed analysis.
    
    Args:
        model: Trained PyTorch model (must have forward method)
        dataloader: DataLoader providing (input, target, freq_idx, ...) tuples
        device: Device for computation ('cpu', 'cuda', or 'mps')
        compute_per_frequency: If True, compute metrics separately for each
            frequency component in addition to overall metrics
    
    Returns:
        Dictionary containing:
            - 'overall': Dict of overall metrics
                - 'mse': Mean Squared Error (float)
                - 'mae': Mean Absolute Error (float)
                - 'r2_score': Coefficient of determination (float)
                - 'snr_db': Signal-to-Noise Ratio in decibels (float)
                - 'correlation': Pearson correlation coefficient (float)
            - 'per_frequency': Dict mapping frequency index to metrics dict
                (only if compute_per_frequency=True)
    
    Raises:
        ValueError: If dataloader is empty
        RuntimeError: If model forward pass fails
        
    Example:
        ```python
        model = load_trained_model()
        results = evaluate_model(
            model, test_loader, torch.device('cpu'),
            compute_per_frequency=True
        )
        
        print(f"Overall MSE: {results['overall']['mse']:.6f}")
        print(f"Overall R²: {results['overall']['r2_score']:.4f}")
        
        for freq_idx, metrics in results['per_frequency'].items():
            print(f"Frequency {freq_idx}: MSE = {metrics['mse']:.6f}")
        ```
    
    Notes:
        - Model is set to eval mode automatically
        - Gradients are disabled for efficiency
        - State is properly managed for stateful models
        - Large datasets may take time; progress logging recommended
    
    Computational Complexity:
        Time: O(n) where n is number of samples
        Space: O(n) to store all predictions and targets
    """
```

Is this documentation standard comprehensive?
```

### 7.2 User Documentation

#### Template 17: User Guide Structure

```
I will create user-facing documentation:

README.md:
  ├─ Quick Start (< 5 minutes to running)
  ├─ Features Overview
  ├─ Installation Instructions
  ├─ Basic Usage Examples
  ├─ Project Structure
  ├─ Documentation Links
  ├─ Testing Instructions
  ├─ Troubleshooting
  └─ License & Citation

QUICKSTART.md:
  ├─ Prerequisites
  ├─ Installation (3 methods)
  ├─ First Run
  ├─ Expected Output
  └─ Next Steps

USAGE_GUIDE.md:
  ├─ Detailed Workflow
  ├─ Configuration Options
  ├─ Advanced Features
  ├─ Customization Examples
  ├─ CLI Arguments
  └─ FAQ

ARCHITECTURE.md:
  ├─ System Overview
  ├─ Module Descriptions
  ├─ Data Flow
  ├─ Key Algorithms
  ├─ Design Decisions
  └─ Extension Points

CONTRIBUTING.md:
  ├─ Development Setup
  ├─ Code Style Guide
  ├─ Testing Requirements
  ├─ Pull Request Process
  └─ Communication Channels

Example README structure:
```markdown
# LSTM Frequency Extraction System
## Professional Implementation with Interactive Visualization

[![Python 3.8+](badge)] [![PyTorch](badge)] [![License](badge)]

---

## Overview

A professional, production-ready implementation of an LSTM neural network for 
extracting individual frequency components from noisy mixed signals.

### Problem Statement
Extract pure frequency components (1Hz, 3Hz, 5Hz, 7Hz) from a noisy mixed 
signal using a stateful LSTM network.

---

## Quick Start

```bash
# Installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python main.py

# Expected Output
✅ Train MSE: ~0.001234
✅ Test MSE:  ~0.001256
✅ R² Score:  >0.99
```

---

## Key Features

- 🎨 Real-time interactive dashboard
- 🧠 Stateful LSTM with proper state management
- 📊 Comprehensive evaluation metrics
- ✅ Production-quality code (>85% test coverage)

---

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART](docs/QUICKSTART.md) | Get started in 5 minutes |
| [USAGE_GUIDE](docs/USAGE_GUIDE.md) | Complete reference |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Technical details |

---

## Project Structure

```
Assignment2_LSTM_extracting_frequences/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── data/           # Data generation
│   ├── models/         # LSTM architecture
│   ├── training/       # Training loop
│   ├── evaluation/     # Metrics
│   └── visualization/  # Plotting
├── tests/              # Test suite
├── experiments/        # Output directory
└── docs/               # Documentation
```

---

## License

MIT License - See LICENSE file for details
```

Should I proceed with creating all documentation?
```

### 7.3 Research Documentation

#### Template 18: Results and Analysis Documentation

```
For research/experimental documentation:

EXPERIMENTS_LOG.md:
```markdown
# Experiments Log

## Experiment 1: Baseline Architecture

**Date**: 2025-11-18  
**Hypothesis**: Standard 2-layer LSTM with hidden_size=128 sufficient  
**Configuration**:
```yaml
model:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 50
```

**Results**:
| Metric | Train | Test | Generalization |
|--------|-------|------|----------------|
| MSE | 0.001234 | 0.001267 | 2.67% |
| R² | 0.9912 | 0.9905 | - |
| Training Time | 7.2 min | - | - |

**Analysis**:
- Excellent generalization (< 3% gap)
- R² > 0.99 indicates strong performance
- Training converges by epoch 25
- State management working correctly

**Visualizations**:
![Graph 1](experiments/exp1/plots/graph1.png)
![Training History](experiments/exp1/plots/history.png)

**Conclusions**:
✅ Baseline meets all requirements
✅ No overfitting detected
✅ Proceed to optimization experiments

**Next Steps**:
- Experiment with larger hidden size (256)
- Try bidirectional LSTM
- Sensitivity analysis on learning rate

---

## Experiment 2: Hidden Size Comparison

[Similar structure for next experiment]
```

RESEARCH_FINDINGS.md:
```markdown
# Research Findings: LSTM Frequency Extraction

## Key Insights

### 1. State Management is Critical
**Finding**: Preserving state across all 10k time steps essential for learning.

**Evidence**:
- With state reset every batch: MSE = 0.234 (poor)
- With state preservation: MSE = 0.0012 (excellent)
- Difference: 195x improvement!

**Explanation**: LSTM learns frequency phase and amplitude patterns across 
time. Resetting state destroys this temporal memory.

### 2. L=1 vs L>1 Performance
**Finding**: L=10 performs similarly to L=1 but trains 3x faster.

**Results**:
| Config | Train MSE | Test MSE | Time |
|--------|-----------|----------|------|
| L=1 | 0.001234 | 0.001267 | 7.2 min |
| L=10 | 0.001189 | 0.001245 | 2.4 min |
| L=50 | 0.001201 | 0.001278 | 1.8 min |

**Recommendation**: L=10 sweet spot for this problem.

### 3. Noise Robustness Analysis
**Finding**: Model generalizes to different noise patterns (different seeds).

**Evidence**:
Tested with 10 different test seeds:
- Mean MSE: 0.001256
- Std Dev: 0.000034
- All within 5% of train MSE

**Conclusion**: Model learns frequency structure, not noise patterns.

---

## Mathematical Analysis

### Signal-to-Noise Ratio Improvement

Input SNR:
```
SNR_in = 10 * log10(P_signal / P_noise)
       ≈ 12 dB (noisy mixed signal)
```

Output SNR:
```
SNR_out ≈ 41 dB (LSTM extracted signal)
```

**Improvement**: 29 dB (noise reduced by factor of ~800)

---

## Statistical Validation

All results validated with:
- 5 independent runs (different random seeds)
- 95% confidence intervals
- Statistical significance testing (t-test, p < 0.01)

[Include detailed statistical analysis]
```

Is this research documentation comprehensive?
```

---

## 8. Advanced Topics

### 8.1 Handling Complex Architectures

#### Template 19: Advanced Architecture Design

```
For complex multi-component systems:

ARCHITECTURE: [System Name]

COMPONENTS:

1. Feature Extractor
   - Purpose: [Extract relevant features from raw input]
   - Architecture: [CNN, Transformer, custom]
   - Output: [Feature representation]

2. Temporal Processor
   - Purpose: [Model temporal dependencies]
   - Architecture: [LSTM, GRU, Temporal Conv]
   - State Management: [How handled]

3. Attention Mechanism (optional)
   - Purpose: [Focus on relevant time steps]
   - Type: [Self-attention, cross-attention]
   - Integration: [How connected to other components]

4. Output Head
   - Purpose: [Generate final predictions]
   - Architecture: [Dense layers, specific activation]
   - Loss Function: [What and why]

INTEGRATION:
```python
class AdvancedLSTMSystem(nn.Module):
    """Multi-component architecture for [task]."""
    
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = FeatureExtractor(config.features)
        self.temporal_processor = StatefulLSTM(config.lstm)
        self.attention = AttentionLayer(config.attention)
        self.output_head = OutputHead(config.output)
    
    def forward(self, x, reset_state=False):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Temporal processing with state management
        temporal_features, state = self.temporal_processor(
            features, reset_state=reset_state
        )
        
        # Attention (if applicable)
        if self.attention:
            attended = self.attention(temporal_features)
        else:
            attended = temporal_features
        
        # Output generation
        output = self.output_head(attended)
        
        return output
```

KEY DESIGN DECISIONS:
1. Why this architecture? [Justification]
2. Trade-offs? [Complexity vs performance]
3. Alternatives considered? [What and why not chosen]

Is this advanced architecture appropriate?
```

### 8.2 Production Deployment Considerations

#### Template 20: Production Readiness Checklist

```
PRODUCTION DEPLOYMENT CHECKLIST for [PROJECT]

MODEL SERVING:
[ ] Model serialization (PyTorch → ONNX/TorchScript)
[ ] Inference optimization (quantization, pruning)
[ ] Batch processing support
[ ] GPU/CPU fallback logic
[ ] Model versioning system

API DESIGN:
[ ] RESTful API (FastAPI/Flask)
[ ] Input validation and sanitization
[ ] Error handling and logging
[ ] Rate limiting
[ ] Authentication/authorization

MONITORING:
[ ] Prediction latency tracking
[ ] Model performance metrics
[ ] Data drift detection
[ ] Anomaly detection
[ ] Alert system

SCALABILITY:
[ ] Horizontal scaling strategy
[ ] Load balancing
[ ] Caching strategy
[ ] Database optimization

RELIABILITY:
[ ] Unit tests (>90% coverage)
[ ] Integration tests
[ ] Load testing
[ ] Failover mechanisms
[ ] Backup and recovery

SECURITY:
[ ] Input sanitization
[ ] SQL injection prevention
[ ] XSS protection
[ ] Rate limiting
[ ] HTTPS enforcement

DOCUMENTATION:
[ ] API documentation (Swagger/OpenAPI)
[ ] Deployment guide
[ ] Monitoring dashboard
[ ] Incident response playbook

Example Production Code:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import torch
import numpy as np
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LSTM Frequency Extraction API", version="1.0")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = torch.load("model_production.pt", map_location="cpu")
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

class PredictionRequest(BaseModel):
    """Request schema with validation."""
    signal: List[float]
    frequency_index: int
    
    @validator('signal')
    def validate_signal(cls, v):
        if len(v) != 10000:
            raise ValueError("Signal must have exactly 10000 samples")
        if not all(-10 <= x <= 10 for x in v):
            raise ValueError("Signal values must be in range [-10, 10]")
        return v
    
    @validator('frequency_index')
    def validate_frequency(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("Frequency index must be 0, 1, 2, or 3")
        return v

class PredictionResponse(BaseModel):
    """Response schema."""
    extracted_signal: List[float]
    frequency: float
    metrics: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Extract frequency component from mixed signal.
    
    Returns extracted pure sine wave for specified frequency.
    """
    try:
        # Input preparation
        signal = torch.tensor(request.signal, dtype=torch.float32)
        freq_onehot = torch.zeros(4)
        freq_onehot[request.frequency_index] = 1.0
        
        # Inference
        model.reset_state()
        predictions = []
        
        with torch.no_grad():
            for t in range(len(signal)):
                x = torch.cat([signal[t:t+1], freq_onehot]).unsqueeze(0).unsqueeze(0)
                pred = model(x, reset_state=False)
                predictions.append(pred.item())
        
        # Response
        return PredictionResponse(
            extracted_signal=predictions,
            frequency=[1.0, 3.0, 5.0, 7.0][request.frequency_index],
            metrics={
                "inference_time_ms": 123.4,  # Measure actual time
                "samples_processed": len(predictions)
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
async def get_metrics():
    """Monitoring metrics endpoint."""
    return {
        "total_predictions": 12345,  # From monitoring system
        "average_latency_ms": 98.7,
        "error_rate": 0.001
    }
```

Ready for production deployment?
```

---

## 9. Prompt Templates Library

### 9.1 Quick Reference Templates

#### Understanding Phase
```
Template 1: Problem Clarification
"I need to build [X]. Let me verify: Input is [Y], output is [Z], 
key challenge is [W]. Correct?"

Template 2: Technical Constraints
"For [requirement], what are implications for: 
1) Architecture, 2) Data flow, 3) Training strategy, 4) Evaluation?"

Template 3: Success Metrics
"How will success be measured? What are acceptable ranges for [metrics]?"
```

#### Design Phase
```
Template 4: Architecture Proposal
"I propose [architecture]. Key decisions: [list]. Concerns: [list]. 
Does this meet requirements?"

Template 5: Module Specification
"For [module], I propose [interface]. Purpose: [X]. Key methods: [Y]. 
Dependencies: [Z]. Appropriate?"

Template 6: Data Flow
"Data flows: [Input] → [Processing] → [Model] → [Output]. 
State management: [how handled]. Bottlenecks: [where]. Correct?"
```

#### Implementation Phase
```
Template 7: Implementation Plan
"Milestones: 1) [core functionality], 2) [integration], 3) [optimization]. 
Dependencies: [X]. Risks: [Y]. Proceed?"

Template 8: Code Quality
"All code will have: type hints, docstrings, tests, error handling. 
Standard: [PEP 8 / Google style]. Acceptable?"

Template 9: Implementation Checkpoint
"Completed [module]. Verified: functionality, quality, tests, docs. 
Next: [X]. Any concerns?"
```

#### Testing Phase
```
Template 10: Unit Tests
"For [module], tests: 1) basic functionality, 2) edge cases, 
3) consistency, 4) performance. Coverage: [%]. Sufficient?"

Template 11: Integration Tests
"End-to-end test: [workflow]. Steps: [list]. Assertions: [list]. 
Covers critical paths?"

Template 12: Requirements Validation
"Checklist: [all requirements]. Status: [pass/fail for each]. 
Summary: [M/N passed]. Ready?"
```

#### Optimization Phase
```
Template 13: Hyperparameter Tuning
"Tuning [parameter]. Hypothesis: [X]. Test values: [Y]. 
Metric: [Z]. Expected: [W]."

Template 14: Performance Profiling
"Profiling [system]. Timing: [X]. Memory: [Y]. Bottlenecks: [Z]. 
Optimizations: [W]."
```

#### Documentation Phase
```
Template 15: Documentation Standard
"Module docs: file header, class docs, function docs. 
Example: [show format]. Comprehensive?"

Template 16: User Guide
"User docs: README, QUICKSTART, USAGE_GUIDE, ARCHITECTURE. 
Structure: [outline]. Complete?"

Template 17: Research Docs
"Experiments log: hypothesis, config, results, analysis, conclusions. 
Format: [show]. Thorough?"
```

### 9.2 Domain-Specific Templates

#### For LSTM Projects
```
State Management Check:
"For my LSTM with [L] sequence length:
- When should state be reset?
- When should state be detached?
- How to prevent gradient explosion?
- How to validate state management is correct?"

Temporal Dependency Verification:
"To verify LSTM learns temporal patterns:
- What tests to run?
- What visualizations to create?
- How to measure temporal learning?
- What metrics indicate success?"

Sequence Processing Optimization:
"For processing [N] time steps:
- TBPTT window size?
- Batch size implications?
- Memory optimization strategies?
- Computational complexity?"
```

#### For Signal Processing Projects
```
Signal Quality Verification:
"For generated signals:
- What properties to verify? (frequency, amplitude, phase)
- What statistical tests to run?
- What visualizations to create?
- How to detect generation bugs?"

Noise Robustness Testing:
"To test noise robustness:
- What noise types to test?
- What SNR ranges to cover?
- How to verify model isn't memorizing noise?
- What generalization tests to run?"
```

#### For Production Systems
```
Production Readiness Check:
"For production deployment:
- What optimization needed? (quantization, pruning)
- What monitoring required?
- What error handling needed?
- What documentation required?
- What testing sufficient?"

API Design:
"For model serving API:
- What endpoints needed?
- What input validation required?
- What error responses appropriate?
- What monitoring metrics to track?"
```

---

## 10. Best Practices and Anti-Patterns

### 10.1 Best Practices

#### Communication Best Practices

```
✅ DO: Ask specific, well-formulated questions
Example: "For L=1 with 10k time steps, should I detach LSTM state after 
each batch to prevent gradient accumulation, or only after each frequency 
sequence?"

❌ DON'T: Ask vague questions
Example: "How do I do LSTM?"

---

✅ DO: Provide context and constraints
Example: "I need to process 40k samples with stateful LSTM. Memory constraint 
is 2GB RAM. Should I use TBPTT with window size 100, or process sample-by-sample?"

❌ DON'T: Ask without context
Example: "What batch size should I use?"

---

✅ DO: Verify understanding before proceeding
Example: "Let me confirm: state resets only at frequency boundaries (every 10k 
samples), not between batches. Correct?"

❌ DON'T: Assume and proceed
Example: "I'll just reset state every batch."

---

✅ DO: Document design decisions with rationale
Example: "Using LayerNorm instead of BatchNorm because: 1) works better with 
small batches, 2) normalizes across features not batch, 3) more stable for 
stateful processing."

❌ DON'T: Make arbitrary choices
Example: "Using LayerNorm because it's newer."
```

#### Implementation Best Practices

```
✅ DO: Test incrementally
- Write function → write test → verify → proceed
- Build complex systems from simple, tested components
- Validate at each step

❌ DON'T: Write everything then test
- Leads to difficult debugging
- Compounds errors
- Wastes time

---

✅ DO: Use type hints and validation
```python
def compute_mse(
    predictions: np.ndarray,
    targets: np.ndarray
) -> float:
    """Compute MSE with input validation."""
    if not isinstance(predictions, np.ndarray):
        raise TypeError("predictions must be numpy array")
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} != {targets.shape}")
    return float(np.mean((predictions - targets) ** 2))
```

❌ DON'T: Skip validation
```python
def compute_mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)  # What if shapes don't match?
```

---

✅ DO: Write comprehensive docstrings
```python
def train_epoch(self) -> float:
    """
    Train model for one epoch with proper state management.
    
    This method processes all training batches while preserving LSTM
    state across consecutive samples within each frequency sequence.
    State is reset at frequency boundaries and detached after each
    batch to implement Truncated Backpropagation Through Time.
    
    Returns:
        Average training loss for the epoch
        
    Raises:
        RuntimeError: If model is not in training mode
        
    Side Effects:
        - Updates model parameters
        - Modifies LSTM internal state
        - Logs metrics to tensorboard
        
    Example:
        ```python
        trainer = LSTMTrainer(model, train_loader, ...)
        for epoch in range(50):
            loss = trainer.train_epoch()
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
        ```
    """
```

❌ DON'T: Write minimal docstrings
```python
def train_epoch(self):
    """Train one epoch."""
```

---

✅ DO: Log important events and metrics
```python
logger.info(f"Epoch {epoch}/{total_epochs}")
logger.info(f"Train Loss: {train_loss:.6f}")
logger.info(f"Val Loss: {val_loss:.6f}")
logger.info(f"Learning Rate: {current_lr:.6f}")

if early_stopping:
    logger.warning(f"Early stopping triggered at epoch {epoch}")
```

❌ DON'T: Use print statements
```python
print("training")
print(train_loss)
```
```

### 10.2 Anti-Patterns to Avoid

#### Design Anti-Patterns

```
❌ ANTI-PATTERN 1: Premature Optimization
Problem: Optimizing before understanding bottlenecks
Example: "I'll use mixed precision training from the start to be faster"
Better: "Let me profile first, then optimize identified bottlenecks"

---

❌ ANTI-PATTERN 2: Overly Complex Architecture
Problem: Adding unnecessary components
Example: "I'll add attention, residual connections, and batch norm everywhere"
Better: "Start simple (2-layer LSTM), add complexity only if needed and justified"

---

❌ ANTI-PATTERN 3: Monolithic Code
Problem: Everything in one file/function
Example: 500-line train_and_evaluate_everything() function
Better: Modular design with single responsibility per module

---

❌ ANTI-PATTERN 4: Configuration in Code
Problem: Hardcoded hyperparameters
Example: hidden_size = 128  # in model code
Better: Load from config file, easy to experiment

---

❌ ANTI-PATTERN 5: Ignoring Error Handling
Problem: No validation, assumes perfect input
Example:
```python
def forward(self, x):
    return self.lstm(x)[0]  # What if x is wrong shape? Wrong device? None?
```
Better:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x is None:
        raise ValueError("Input cannot be None")
    if x.size(-1) != self.input_size:
        raise ValueError(f"Expected input size {self.input_size}, got {x.size(-1)}")
    return self.lstm(x)[0]
```
```

#### Implementation Anti-Patterns

```
❌ ANTI-PATTERN 6: Magic Numbers
Problem: Unexplained constants in code
Example: for i in range(50):  # Why 50?
Better: 
```python
MAX_EPOCHS = 50  # Standard training duration, sufficient for convergence
for epoch in range(MAX_EPOCHS):
```

---

❌ ANTI-PATTERN 7: Ignoring State Management
Problem: Resetting LSTM state incorrectly
Example:
```python
for batch in dataloader:
    model.reset_state()  # ❌ Destroys temporal learning!
    output = model(batch)
```
Better:
```python
for batch in dataloader:
    if batch.is_first_in_sequence:
        model.reset_state()  # ✅ Reset only at boundaries
    output = model(batch, reset_state=False)
    # ... backward pass ...
    model.detach_state()  # ✅ Prevent gradient accumulation
```

---

❌ ANTI-PATTERN 8: Insufficient Testing
Problem: Only testing "happy path"
Example: test_forward_pass() that only tests valid input
Better: Test valid input, invalid input, edge cases, performance

---

❌ ANTI-PATTERN 9: Poor Variable Names
Problem: Cryptic or misleading names
Example: x, y, tmp, data1, data2
Better: mixed_signal, pure_target, frequency_index, train_generator

---

❌ ANTI-PATTERN 10: Copy-Paste Programming
Problem: Duplicating code instead of abstracting
Example: Same data processing code in train(), validate(), test()
Better: Create process_batch() function, call from all three
```

### 10.3 Code Review Checklist

```
SELF-REVIEW CHECKLIST before submitting/asking for review:

FUNCTIONALITY:
[ ] Code does what it's supposed to do
[ ] Edge cases handled
[ ] Error conditions handled gracefully
[ ] No obvious bugs or logical errors

CODE QUALITY:
[ ] Type hints on all functions
[ ] Comprehensive docstrings
[ ] No linter errors (flake8, mypy)
[ ] Consistent style (black formatted)
[ ] Meaningful variable names
[ ] No magic numbers

TESTING:
[ ] Unit tests written
[ ] Integration tests written
[ ] All tests passing
[ ] Coverage > 85%
[ ] Edge cases tested

PERFORMANCE:
[ ] No obvious inefficiencies
[ ] Computational complexity documented
[ ] Memory usage reasonable
[ ] Profiling done if performance critical

ARCHITECTURE:
[ ] Modular design
[ ] Single responsibility per function/class
[ ] Appropriate abstractions
[ ] Minimal coupling, high cohesion

DOCUMENTATION:
[ ] README updated
[ ] Architecture docs updated
[ ] Example usage provided
[ ] Comments explain "why" not "what"

REPRODUCIBILITY:
[ ] Random seeds fixed where needed
[ ] Configuration externalized
[ ] Dependencies documented
[ ] Experiment tracking in place

If all checkboxes ticked → ready for review! ✅
If any unchecked → address before submitting
```

---

## 11. Conclusion and Next Steps

### 11.1 Key Takeaways

This MIT-level prompt engineering book has covered:

1. **Systematic Approach**: From requirements to delivery
2. **Professional Communication**: Clear, precise, comprehensive
3. **Engineering Excellence**: Production-quality code and architecture
4. **Best Practices**: What to do (and what to avoid)
5. **Practical Templates**: Ready-to-use for your projects

### 11.2 How to Master Prompt Engineering

```
MASTERY PATH:

Level 1: BEGINNER
- Use templates as-is
- Follow sequential phases
- Focus on clarity

Level 2: INTERMEDIATE
- Adapt templates to your needs
- Combine templates creatively
- Develop intuition for when to use which template

Level 3: ADVANCED
- Create your own templates
- Anticipate follow-up questions
- Guide conversations strategically

Level 4: EXPERT (MIT-Level)
- See the big picture immediately
- Ask questions that reveal hidden assumptions
- Design systems that are robust, scalable, maintainable
- Document for future engineers
```

### 11.3 Continuous Improvement

```
After Each Project:

1. REFLECT:
   - What prompts were most effective?
   - What questions revealed critical insights?
   - What would I ask differently next time?

2. DOCUMENT:
   - Add successful prompts to your library
   - Document anti-patterns you encountered
   - Create project-specific templates

3. ITERATE:
   - Refine your prompt templates
   - Improve your technical vocabulary
   - Deepen your domain knowledge

4. SHARE:
   - Teach others what you learned
   - Contribute to team knowledge base
   - Mentor junior engineers
```

### 11.4 Final Advice

```
🎯 REMEMBER:

"The quality of your questions determines the quality of your solutions."

✅ Ask precise, well-formulated questions
✅ Provide context and constraints
✅ Verify understanding before proceeding
✅ Document decisions with rationale
✅ Test incrementally and thoroughly
✅ Deliver production-quality work

Good luck building amazing systems! 🚀
```

---

## Appendix: Real-World Example Walkthrough

### Complete Example: LSTM Frequency Extraction Project

**This appendix shows how to apply this prompt book to the actual LSTM frequency extraction project from start to finish.**

#### Session 1: Requirements Understanding

```
PROMPT 1: Initial Problem Clarification

"I need to build an LSTM system for extracting individual frequency 
components from a noisy mixed signal. Let me verify my understanding:

1. PROBLEM DOMAIN:
   - Input: Mixed signal S(t) = (1/4)Σ[noisy sines] + one-hot selector C
   - Output: Pure sine wave for selected frequency
   - Key Challenge: Random A(t) and φ(t) at EVERY time step
   
2. TECHNICAL CONSTRAINTS:
   - L=1 (one sample at a time) requires manual state management
   - Must preserve state across 10,000 consecutive time steps
   - Different random seeds for train (seed=1) and test (seed=2)
   - 4 frequencies: 1Hz, 3Hz, 5Hz, 7Hz
   
3. SUCCESS CRITERIA:
   - MSE_test ≈ MSE_train (generalization check)
   - Visual quality in plots (clean extraction)
   - R² > 0.95, high correlation
   
Is this correct? Any additional constraints or clarifications needed?"

RESPONSE: [Confirmation and clarifications]

PROMPT 2: State Management Deep Dive

"For L=1 with 10,000 time steps per frequency, I need to understand state management:

Q1: Should I reset LSTM state at every batch, or only at frequency boundaries?
    My thinking: Reset every batch destroys temporal learning, so reset 
    only when switching frequencies?

Q2: Should I detach state after each batch?
    My thinking: Yes, to prevent backprop through entire 10k sequence (TBPTT)?

Q3: How to structure dataset to maintain temporal order?
    My thinking: Standard DataLoader shuffles, need custom loader?

Q4: During validation, same state management as training?
    My thinking: Yes, reset at boundaries, detach after batches?

Please confirm these approaches are correct."

RESPONSE: [Technical guidance on state management]

PROMPT 3: Data Generation Clarification

"For signal generation, I need to verify:

Q1: A(t) and φ(t) change at EVERY sample t, not just per sequence?
    Formula: noisy_sine[t] = A(t) * sin(2πf*t + φ(t))
    where A(t) ~ U(0.8, 1.2) and φ(t) ~ U(0, 2π) for each t?

Q2: All 4 frequencies use the SAME mixed signal S(t)?
    My thinking: Yes, efficient and makes sense - same input, different 
    targets based on selector C.

Q3: Targets are pure sines with A=1, φ=0?
    Formula: Target_i(t) = sin(2πf_i*t)

Q4: Should I normalize signals?
    My thinking: Yes, improves training stability.

Correct?"

RESPONSE: [Confirmation of data generation approach]
```

#### Session 2: System Design

```
PROMPT 4: Architecture Proposal

"Based on requirements, I propose this architecture:

MODULE STRUCTURE:
```
src/
├── data/
│   ├── signal_generator.py    # Generate S(t) and targets
│   └── dataset.py              # Dataset + StatefulDataLoader
├── models/
│   └── lstm_extractor.py       # StatefulLSTMExtractor with state management
├── training/
│   └── trainer.py              # Training loop with TBPTT
├── evaluation/
│   └── metrics.py              # MSE, R², MAE, SNR, correlation
└── visualization/
    └── plotter.py              # Required graphs + analysis plots
```

MODEL ARCHITECTURE:
```
Input(5) → LayerNorm → LSTM(128, 2 layers) → LayerNorm → 
Linear(128→64) → ReLU → Dropout → Linear(64→1) → Output(1)
```

KEY DESIGN DECISIONS:
1. Stateful LSTM with manual state management (h_t, c_t as instance vars)
2. Custom DataLoader to preserve temporal order
3. TBPTT: detach state after each batch
4. State reset only at frequency boundaries
5. Configuration-driven (YAML)

CONCERNS:
- Is 128 hidden size sufficient?
- Should I add attention mechanism?
- Is 2-layer LSTM enough?

Does this meet requirements? Any suggestions?"

RESPONSE: [Feedback on architecture]

PROMPT 5: StatefulLSTMExtractor Detailed Design

"For StatefulLSTMExtractor, I propose:

PUBLIC INTERFACE:
```python
class StatefulLSTMExtractor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.2)
    def init_hidden(self, batch_size, device)  # Init to zeros
    def reset_state(self)  # Set to None
    def detach_state(self)  # Detach from graph (TBPTT)
    def forward(self, x, reset_state=False) -> output
```

STATE MANAGEMENT LOGIC:
1. On first forward or if reset_state=True: init_hidden()
2. LSTM processes input with current (h_t, c_t)
3. Update self.hidden_state, self.cell_state with LSTM output states
4. Training loop calls detach_state() after backward pass

TRAINING PATTERN:
```python
for batch in dataloader:
    if batch.is_first_batch:
        model.reset_state()  # New frequency sequence
    output = model(batch.input, reset_state=False)
    loss.backward()
    optimizer.step()
    model.detach_state()  # Critical!
```

QUESTIONS:
- Is storing state as instance variables the right approach?
- Should detach_state() be called automatically in forward, or manually?
- How to handle variable batch sizes?

Is this design sound?"

RESPONSE: [Validation and suggestions]
```

#### Session 3: Implementation

```
PROMPT 6: Implementation Roadmap

"I propose this implementation sequence:

MILESTONE 1: Data Pipeline (Day 1)
├─ Implement SignalGenerator
│  ├─ generate_noisy_sine(f, t) with random A, φ
│  ├─ generate_pure_sine(f, t) deterministic
│  ├─ generate_mixed_signal() average of noisy sines
│  └─ Unit tests for signal properties
├─ Implement FrequencyExtractionDataset
│  ├─ 40k samples structured correctly
│  ├─ One-hot encoding for frequency selection
│  └─ Unit tests for dataset structure
└─ Implement StatefulDataLoader
   ├─ Maintain temporal order (no shuffling)
   ├─ Batch metadata (is_first, is_last flags)
   └─ Integration test with dataset

MILESTONE 2: Model Architecture (Day 2)
├─ Implement StatefulLSTMExtractor
│  ├─ LSTM layers with LayerNorm
│  ├─ State management methods
│  ├─ Forward pass
│  └─ Unit tests (initialization, forward, state management)
└─ Test with dummy data

MILESTONE 3: Training Pipeline (Day 3)
├─ Implement LSTMTrainer
│  ├─ Training loop with state management
│  ├─ Validation loop
│  ├─ Checkpoint saving
│  └─ Early stopping logic
├─ Integration test: short training run (2 epochs)
└─ Verify loss decreases

MILESTONE 4: Evaluation (Day 4)
├─ Implement metrics computation
│  ├─ MSE, MAE, R², SNR, correlation
│  ├─ Per-frequency breakdown
│  └─ Generalization analysis
└─ Unit tests for metric calculations

MILESTONE 5: Visualization (Day 5)
├─ Graph 1: Single frequency comparison
├─ Graph 2: All frequencies (2×2 grid)
├─ Training history plot
├─ Error distribution analysis
└─ Metrics comparison bar chart

MILESTONE 6: Integration & Testing (Day 6)
├─ End-to-end integration test
├─ Full training run (50 epochs)
├─ Verify all requirements met
└─ Code quality check (linting, type checking, coverage)

Proceed with this plan?"

RESPONSE: [Approval or modifications]

PROMPT 7: Implementation of SignalGenerator

"I'll implement SignalGenerator. Please review:

```python
import numpy as np
from typing import List, Tuple

class SignalGenerator:
    \"\"\"
    Generate noisy mixed signals and pure target signals.
    
    This class creates training/test data for frequency extraction.
    It generates a mixed signal S(t) composed of 4 noisy sine waves
    with random amplitude and phase at each time step.
    
    Attributes:
        frequencies: List of frequencies in Hz (e.g., [1, 3, 5, 7])
        sampling_rate: Samples per second (e.g., 1000 Hz)
        duration: Signal duration in seconds (e.g., 10.0)
        rng: NumPy random number generator with fixed seed
    \"\"\"
    
    def __init__(
        self,
        frequencies: List[float],
        sampling_rate: int,
        duration: float,
        amplitude_range: Tuple[float, float] = (0.8, 1.2),
        phase_range: Tuple[float, float] = (0, 2*np.pi),
        seed: int = 42
    ):
        self.frequencies = frequencies
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.rng = np.random.RandomState(seed)
        
    def get_time_array(self) -> np.ndarray:
        \"\"\"Get time array from 0 to duration with sampling rate.\"\"\"
        return np.arange(0, self.duration, 1/self.sampling_rate)
    
    def generate_noisy_sine(
        self, 
        frequency: float, 
        time: np.ndarray
    ) -> np.ndarray:
        \"\"\"
        Generate sine wave with random amplitude and phase at each sample.
        
        Critical: A(t) and φ(t) are resampled at EVERY time step t.
        
        Args:
            frequency: Frequency in Hz
            time: Time array
            
        Returns:
            Noisy sine wave: A(t) * sin(2πft + φ(t))
        \"\"\"
        # Generate random amplitude and phase for each time step
        amplitudes = self.rng.uniform(
            self.amplitude_range[0],
            self.amplitude_range[1],
            len(time)
        )
        phases = self.rng.uniform(
            self.phase_range[0],
            self.phase_range[1],
            len(time)
        )
        
        # Compute noisy sine
        noisy_sine = amplitudes * np.sin(2 * np.pi * frequency * time + phases)
        return noisy_sine
    
    def generate_pure_sine(
        self, 
        frequency: float, 
        time: np.ndarray
    ) -> np.ndarray:
        \"\"\"
        Generate pure sine wave (no noise).
        
        Args:
            frequency: Frequency in Hz
            time: Time array
            
        Returns:
            Pure sine wave: sin(2πft)
        \"\"\"
        return np.sin(2 * np.pi * frequency * time)
    
    def generate_mixed_signal(self) -> np.ndarray:
        \"\"\"
        Generate mixed signal from all noisy frequencies.
        
        Returns:
            Mixed signal: (1/N) * Σ noisy_sine_i(t)
        \"\"\"
        time = self.get_time_array()
        noisy_sines = [
            self.generate_noisy_sine(freq, time) 
            for freq in self.frequencies
        ]
        # Average (normalized sum)
        mixed = np.mean(noisy_sines, axis=0)
        return mixed
    
    def generate_all_targets(self) -> np.ndarray:
        \"\"\"
        Generate pure target signals for all frequencies.
        
        Returns:
            Array of shape (num_frequencies, num_time_samples)
        \"\"\"
        time = self.get_time_array()
        targets = np.array([
            self.generate_pure_sine(freq, time)
            for freq in self.frequencies
        ])
        return targets
```

QUESTIONS:
1. Is random A and φ at EVERY time step correct? (Not just per sequence?)
2. Should I add normalization here or in Dataset?
3. Any edge cases I'm missing?

Ready to proceed with unit tests?"

RESPONSE: [Review and feedback]
```

#### Session 4: Testing

```
PROMPT 8: Unit Tests for SignalGenerator

"I'll write comprehensive unit tests:

```python
import pytest
import numpy as np
from src.data.signal_generator import SignalGenerator

class TestSignalGenerator:
    \"\"\"Comprehensive test suite for SignalGenerator.\"\"\"
    
    @pytest.fixture
    def generator(self):
        \"\"\"Standard generator for most tests.\"\"\"
        return SignalGenerator(
            frequencies=[1.0, 3.0, 5.0, 7.0],
            sampling_rate=1000,
            duration=10.0,
            seed=42
        )
    
    def test_initialization(self):
        \"\"\"Test generator initializes correctly.\"\"\"
        gen = SignalGenerator([1.0], 1000, 10.0, seed=42)
        assert gen.frequencies == [1.0]
        assert gen.sampling_rate == 1000
        assert gen.duration == 10.0
        assert gen.rng is not None
    
    def test_time_array_shape(self, generator):
        \"\"\"Test time array has correct shape.\"\"\"
        time = generator.get_time_array()
        expected_samples = int(generator.duration * generator.sampling_rate)
        assert len(time) == expected_samples
        assert time[0] == 0.0
        assert np.isclose(time[-1], generator.duration - 1/generator.sampling_rate)
    
    def test_noisy_sine_shape(self, generator):
        \"\"\"Test noisy sine has correct shape.\"\"\"
        time = generator.get_time_array()
        noisy = generator.generate_noisy_sine(1.0, time)
        assert noisy.shape == time.shape
    
    def test_noisy_sine_has_randomness(self, generator):
        \"\"\"Test consecutive calls produce different noise.\"\"\"
        time = generator.get_time_array()
        noisy1 = generator.generate_noisy_sine(1.0, time)
        noisy2 = generator.generate_noisy_sine(1.0, time)
        # Should be different (different random A and φ)
        assert not np.allclose(noisy1, noisy2)
    
    def test_pure_sine_deterministic(self, generator):
        \"\"\"Test pure sine is deterministic (no randomness).\"\"\"
        time = generator.get_time_array()
        pure1 = generator.generate_pure_sine(1.0, time)
        pure2 = generator.generate_pure_sine(1.0, time)
        # Should be identical
        assert np.allclose(pure1, pure2)
    
    def test_pure_sine_amplitude(self, generator):
        \"\"\"Test pure sine has amplitude = 1.\"\"\"
        time = generator.get_time_array()
        pure = generator.generate_pure_sine(1.0, time)
        assert np.isclose(np.abs(pure).max(), 1.0, atol=0.01)
    
    def test_pure_sine_frequency(self, generator):
        \"\"\"Test pure sine has correct frequency (via zero crossings).\"\"\"
        time = generator.get_time_array()
        frequency = 1.0
        pure = generator.generate_pure_sine(frequency, time)
        
        # Count zero crossings
        zero_crossings = np.where(np.diff(np.sign(pure)))[0]
        expected_crossings = int(2 * frequency * generator.duration)
        
        assert abs(len(zero_crossings) - expected_crossings) <= 2
    
    def test_mixed_signal_shape(self, generator):
        \"\"\"Test mixed signal has correct shape.\"\"\"
        mixed = generator.generate_mixed_signal()
        expected_samples = int(generator.duration * generator.sampling_rate)
        assert len(mixed) == expected_samples
    
    def test_mixed_signal_range(self, generator):
        \"\"\"Test mixed signal values are reasonable.\"\"\"
        mixed = generator.generate_mixed_signal()
        # With 4 frequencies averaged and A∈[0.8, 1.2], expect |S| < 2
        assert np.abs(mixed).max() < 2.0
    
    def test_all_targets_shape(self, generator):
        \"\"\"Test all targets array has correct shape.\"\"\"
        targets = generator.generate_all_targets()
        assert targets.shape == (
            len(generator.frequencies),
            int(generator.duration * generator.sampling_rate)
        )
    
    def test_seed_reproducibility(self):
        \"\"\"Test same seed produces identical signals.\"\"\"
        gen1 = SignalGenerator([1.0, 3.0], 1000, 1.0, seed=42)
        gen2 = SignalGenerator([1.0, 3.0], 1000, 1.0, seed=42)
        
        signal1 = gen1.generate_mixed_signal()
        signal2 = gen2.generate_mixed_signal()
        
        assert np.allclose(signal1, signal2)
    
    def test_different_seeds_different_signals(self):
        \"\"\"Test different seeds produce different signals.\"\"\"
        gen1 = SignalGenerator([1.0, 3.0], 1000, 1.0, seed=1)
        gen2 = SignalGenerator([1.0, 3.0], 1000, 1.0, seed=2)
        
        signal1 = gen1.generate_mixed_signal()
        signal2 = gen2.generate_mixed_signal()
        
        assert not np.allclose(signal1, signal2)
    
    @pytest.mark.parametrize(\"frequency\", [1.0, 3.0, 5.0, 7.0])
    def test_each_frequency(self, frequency):
        \"\"\"Test signal generation for each frequency.\"\"\"
        gen = SignalGenerator([frequency], 1000, 1.0, seed=42)
        signal = gen.generate_mixed_signal()
        assert len(signal) == 1000
```

Does this test suite cover all important cases?"

RESPONSE: [Feedback on tests]
```

[Similar detailed walkthroughs would continue for each module through all 6 phases]

---

**End of Prompt Book**

This comprehensive guide provides a systematic, MIT-level approach to building
advanced ML systems through effective prompt engineering and LLM-assisted development.
Developed as part of the **LLM and Multi-Agent Orchestration** course, this book
demonstrates how to systematically approach complex technical projects using AI 
collaboration. Use it as a reference, adapt it to your needs, and continuously 
improve your engineering practice.

**Remember**: The quality of your questions determines the quality of your solutions.

---

**Document Information**:
- **Version**: 1.0
- **Course**: LLM and Multi-Agent Orchestration
- **Last Updated**: November 2025
- **Maintained By**: Professional ML Engineering Team
- **Location**: `docs/MIT_LEVEL_PROMPT_ENGINEERING_BOOK.md`
- **License**: MIT
- **Purpose**: Educational resource for systematic ML system development using LLM-assisted engineering

---

**For questions, clarifications, or contributions, please refer to the project
maintainers or submit an issue/PR.**

**🎓 Happy Engineering! 🚀**

