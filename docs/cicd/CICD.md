# CI/CD Pipeline Documentation

## Overview

This project now includes a comprehensive CI/CD pipeline using **GitHub Actions** for automated testing, building, and deployment.

## Table of Contents

1. [Pipeline Architecture](#pipeline-architecture)
2. [Continuous Integration](#continuous-integration)
3. [Continuous Deployment](#continuous-deployment)
4. [Docker Support](#docker-support)
5. [Configuration](#configuration)
6. [Triggering Workflows](#triggering-workflows)
7. [Secrets Management](#secrets-management)
8. [Badge Status](#badge-status)

---

## Pipeline Architecture

The CI/CD pipeline consists of two main workflows:

### 1. **CI Pipeline** (`.github/workflows/ci.yml`)
- Code Quality Checks
- Security Scanning
- Multi-platform Testing
- Integration Tests
- Performance Benchmarks
- Build Validation
- Documentation Checks
- Compliance Verification

### 2. **Deploy Pipeline** (`.github/workflows/deploy.yml`)
- PyPI Package Publishing
- Docker Image Building
- Documentation Deployment
- Release Artifact Creation

---

## Continuous Integration

### Jobs Overview

#### 1. **Lint (Code Quality)**
- **Tools**: `black`, `isort`, `flake8`, `pylint`
- **Purpose**: Ensures code follows PEP 8 and project standards
- **When**: On every push and pull request

```yaml
Checks:
- Code formatting (Black)
- Import sorting (isort)
- Style violations (Flake8)
- Code quality (Pylint)
```

#### 2. **Security**
- **Tools**: `safety`, `bandit`
- **Purpose**: Identifies security vulnerabilities
- **Outputs**: JSON reports uploaded as artifacts

```yaml
Checks:
- Dependency vulnerabilities (Safety)
- Code security issues (Bandit)
```

#### 3. **Test**
- **Matrix Strategy**: 
  - OS: Ubuntu, macOS
  - Python: 3.8, 3.9, 3.10, 3.11
- **Coverage**: Uploads to Codecov
- **Reports**: JUnit XML and HTML coverage

```bash
# Local testing
pytest tests/ --cov=src --cov-report=html -v
```

#### 4. **Integration**
- **Tests**: Full integration test suite
- **Validation**: Data generation and model creation
- **Dependencies**: Runs after unit tests pass

#### 5. **Performance**
- **Benchmarks**: Performance regression testing
- **Tools**: pytest-benchmark
- **Metrics**: Execution time, memory usage

#### 6. **Build**
- **Package**: Creates distribution packages
- **Validation**: Checks with `twine`
- **Artifacts**: Uploads to GitHub Actions

#### 7. **Compliance**
- **Checks**: ISO 9126 and quality standards
- **Tool**: `compliance_cli.py`
- **Output**: JSON compliance report

---

## Continuous Deployment

### Deployment Triggers

1. **Automatic**: On new release creation
2. **Manual**: Via workflow_dispatch with version input

### Deployment Jobs

#### 1. **PyPI Publishing**
```yaml
Environment: production
Permissions: id-token: write
Security: Uses trusted publishing (OIDC)
```

**Setup Required:**
1. Create PyPI account
2. Configure trusted publisher in PyPI
3. Add `PYPI_API_TOKEN` to GitHub secrets

#### 2. **Docker Image**
```yaml
Registry: Docker Hub
Tags: 
  - latest
  - version-specific (e.g., v1.0.0)
Cache: GitHub Actions cache
```

**Setup Required:**
1. Create Docker Hub account
2. Add secrets:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`

#### 3. **Documentation**
```yaml
Target: GitHub Pages
Tool: MkDocs
Branch: gh-pages
```

**Setup Required:**
1. Enable GitHub Pages in repository settings
2. Set source to `gh-pages` branch

#### 4. **Release Artifacts**
```yaml
Format: tar.gz
Contents: Complete source code
Upload: Automatic to GitHub Release
```

---

## Docker Support

### Building the Image

```bash
# Build locally
docker build -t lstm-frequency-extractor:latest .

# Run container
docker run -it lstm-frequency-extractor:latest

# Run with volume mounting
docker run -it \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/experiments:/app/experiments \
  lstm-frequency-extractor:latest

# Run dashboard
docker run -it -p 8050:8050 \
  lstm-frequency-extractor:latest \
  python main_with_dashboard.py
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  lstm-app:
    build: .
    image: lstm-frequency-extractor:latest
    volumes:
      - ./config:/app/config
      - ./experiments:/app/experiments
    environment:
      - PYTHONUNBUFFERED=1
    command: python main.py
```

Run with:
```bash
docker-compose up
```

---

## Configuration

### Environment Variables

```bash
# Optional environment variables
export PYTHONUNBUFFERED=1
export TORCH_HOME=/app/.torch
export MPLBACKEND=Agg  # For headless plotting
```

### GitHub Repository Settings

#### 1. **Enable Actions**
- Go to: Settings → Actions → General
- Allow all actions and reusable workflows

#### 2. **Branch Protection**
```yaml
Branch: master
Rules:
  - Require pull request reviews
  - Require status checks to pass
  - Require branches to be up to date
  - Include administrators
```

#### 3. **Environments**
Create `production` environment:
- Settings → Environments → New environment
- Add protection rules (optional)

---

## Triggering Workflows

### Automatic Triggers

1. **Push to master/main/develop**
   ```bash
   git push origin master
   ```

2. **Pull Request**
   ```bash
   git checkout -b feature-branch
   # Make changes
   git push origin feature-branch
   # Create PR on GitHub
   ```

3. **Release Creation**
   ```bash
   # Create and push tag
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   
   # Or create release on GitHub UI
   ```

### Manual Triggers

1. **Via GitHub UI**
   - Go to Actions tab
   - Select workflow
   - Click "Run workflow"
   - Enter parameters

2. **Via GitHub CLI**
   ```bash
   # Install gh CLI
   brew install gh
   
   # Trigger workflow
   gh workflow run ci.yml
   gh workflow run deploy.yml -f version=1.0.0
   ```

---

## Secrets Management

### Required Secrets

Add these in: Settings → Secrets and variables → Actions

#### For PyPI Deployment
```bash
PYPI_API_TOKEN=<your-pypi-token>
```

#### For Docker Hub
```bash
DOCKER_USERNAME=<your-docker-username>
DOCKER_PASSWORD=<your-docker-password>
```

#### For Codecov (Optional)
```bash
CODECOV_TOKEN=<your-codecov-token>
```

### Creating Secrets

```bash
# Via GitHub CLI
gh secret set PYPI_API_TOKEN < token.txt
gh secret set DOCKER_USERNAME -b"username"
gh secret set DOCKER_PASSWORD -b"password"
```

---

## Badge Status

Add badges to your README.md:

```markdown
# LSTM Frequency Extraction

![CI Pipeline](https://github.com/fouada/Assignment2_LSTM_extracting_frequences/workflows/CI%2FCD%20Pipeline/badge.svg)
![Deploy](https://github.com/fouada/Assignment2_LSTM_extracting_frequences/workflows/Deploy%20and%20Release/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Coverage](https://codecov.io/gh/fouada/Assignment2_LSTM_extracting_frequences/branch/master/graph/badge.svg)
```

---

## Monitoring and Troubleshooting

### Viewing Workflow Runs

1. **GitHub UI**
   - Go to repository
   - Click "Actions" tab
   - View workflow runs and logs

2. **GitHub CLI**
   ```bash
   # List recent runs
   gh run list
   
   # View specific run
   gh run view <run-id>
   
   # Watch run in real-time
   gh run watch <run-id>
   ```

### Common Issues

#### 1. **Test Failures**
```bash
# Run tests locally first
pytest tests/ -v

# Check specific test
pytest tests/test_model.py -v -k test_name
```

#### 2. **Build Failures**
```bash
# Validate package locally
python -m build
twine check dist/*
```

#### 3. **Docker Build Failures**
```bash
# Test Docker build locally
docker build -t test-build .

# Check with verbose output
docker build -t test-build --progress=plain .
```

#### 4. **Permission Issues**
- Check GitHub Actions permissions in repository settings
- Verify GITHUB_TOKEN has necessary scopes
- Ensure secrets are properly configured

---

## Local CI Testing

### Using Act (GitHub Actions locally)

```bash
# Install act
brew install act

# Run CI workflow locally
act -j test

# Run specific job
act -j lint

# With secrets
act -s PYPI_API_TOKEN=test-token
```

### Pre-commit Hooks

Install pre-commit hooks for local validation:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Best Practices

1. **Always test locally before pushing**
   ```bash
   pytest tests/
   black .
   isort .
   flake8 .
   ```

2. **Use feature branches**
   ```bash
   git checkout -b feature/new-feature
   # Make changes
   git push origin feature/new-feature
   # Create PR
   ```

3. **Keep workflows fast**
   - Use caching for dependencies
   - Run expensive tests in parallel
   - Use matrix strategy efficiently

4. **Monitor costs**
   - GitHub Actions is free for public repos
   - 2,000 minutes/month for private repos
   - Use self-hosted runners for unlimited minutes

5. **Security**
   - Never commit secrets
   - Use GitHub Secrets for sensitive data
   - Regularly update dependencies

---

## Extending the Pipeline

### Adding New Jobs

Edit `.github/workflows/ci.yml`:

```yaml
new-job:
  name: My New Job
  runs-on: ubuntu-latest
  needs: [test]  # Optional dependencies
  steps:
    - uses: actions/checkout@v4
    - name: Run custom script
      run: python my_script.py
```

### Adding New Workflows

Create new file in `.github/workflows/`:

```yaml
name: My Custom Workflow
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
jobs:
  custom-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Custom action
        run: echo "Running weekly job"
```

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Codecov Documentation](https://docs.codecov.io/)

---

## Support

For issues with CI/CD:
1. Check workflow logs in GitHub Actions
2. Review this documentation
3. Open an issue in the repository
4. Contact maintainers

---

## Changelog

### Version 1.0.0 (2024-11-18)
- Initial CI/CD pipeline implementation
- GitHub Actions workflows for CI and deployment
- Docker support with multi-stage builds
- Comprehensive testing matrix
- Security scanning integration
- Documentation deployment

