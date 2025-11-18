# Pull Request

## 📋 Description

### Summary

Brief description of what this PR does.

### Motivation and Context

Why is this change required? What problem does it solve?

Fixes # (issue)
Related to # (issue)

## 🔄 Type of Change

Please check the relevant option(s):

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 Documentation update
- [ ] 🎨 Code style update (formatting, renaming)
- [ ] ♻️ Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Test update
- [ ] 🔧 Configuration change
- [ ] 🏗️ Build/infrastructure change

## 🧪 How Has This Been Tested?

Please describe the tests you ran to verify your changes.

**Test Configuration:**
- OS: [e.g., macOS 14.0, Ubuntu 22.04]
- Python version: [e.g., 3.10.12]
- PyTorch version: [e.g., 2.1.0]
- Device: [e.g., CPU, CUDA, MPS]

**Test Cases:**
- [ ] Test case 1: Description
- [ ] Test case 2: Description

**Test Results:**

```bash
# Output of: pytest tests/ -v
```

## 📸 Screenshots (if applicable)

Add screenshots to demonstrate visual changes or new features.

| Before | After |
|--------|-------|
| (if applicable) | (if applicable) |

## 📊 Performance Impact (if applicable)

Does this change affect performance?

- [ ] No performance impact
- [ ] Performance improvement: [describe]
- [ ] Potential performance regression: [describe]

**Benchmarks (if applicable):**

```
Before: X seconds
After: Y seconds
Improvement: Z%
```

## 🚨 Breaking Changes (if applicable)

List any breaking changes and migration steps:

1. Change 1: [description and migration steps]
2. Change 2: [description and migration steps]

## 📝 Documentation

- [ ] I have updated the README (if needed)
- [ ] I have updated relevant documentation in `docs/`
- [ ] I have added/updated docstrings
- [ ] I have updated the CHANGELOG.md
- [ ] I have added usage examples (if needed)

## ✅ Checklist

### Code Quality

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My code has type hints for all functions
- [ ] I have used descriptive variable/function names

### Testing

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have added tests for edge cases
- [ ] Test coverage has not decreased

### Code Style

- [ ] My code is formatted with `black`
- [ ] My code passes `flake8` linting
- [ ] My code passes `mypy` type checking
- [ ] I have run `pytest tests/ -v` and all tests pass

```bash
# Run these commands before submitting
black src/ tests/ --check
flake8 src/ tests/
mypy src/
pytest tests/ -v
```

### Documentation

- [ ] I have updated relevant documentation
- [ ] All docstrings follow Google style
- [ ] Code examples work as expected
- [ ] Links in documentation are valid

### Version Control

- [ ] My branch is up to date with main
- [ ] I have resolved all merge conflicts
- [ ] My commits have clear, descriptive messages
- [ ] I have squashed related commits (if needed)

## 🔗 Related Issues/PRs

- Fixes #
- Related to #
- Depends on #
- Blocks #

## 📦 Dependencies

Does this PR add/update/remove dependencies?

- [ ] No dependency changes
- [ ] Added dependencies: [list]
- [ ] Updated dependencies: [list]
- [ ] Removed dependencies: [list]

If yes, why are these changes necessary?

## 🚀 Deployment Notes (if applicable)

Any special deployment considerations:

1. [Deployment step 1]
2. [Deployment step 2]

## 👥 Reviewers

**Requested reviewers:**
- @username1
- @username2

**Review focus areas:**
- Area 1: [e.g., architecture design]
- Area 2: [e.g., performance optimization]

## 📅 Timeline

**Target merge date:** [if applicable]

**Urgency:** 
- [ ] Critical - Blocking production
- [ ] High - Needed soon
- [ ] Medium - Normal priority
- [ ] Low - No rush

## 💬 Additional Notes

Add any other context, concerns, or questions about the PR here.

---

## 🙏 Thank You!

Thank you for contributing to the LSTM Frequency Extraction System! 

Your contribution helps make this project better for everyone. 🚀

---

**For Maintainers:**

- [ ] Code review completed
- [ ] Tests reviewed and passing
- [ ] Documentation reviewed
- [ ] Ready to merge

**Merge strategy:**
- [ ] Squash and merge
- [ ] Rebase and merge
- [ ] Create merge commit

