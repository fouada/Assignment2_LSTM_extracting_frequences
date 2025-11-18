# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

The LSTM Frequency Extraction System team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **Email** (Preferred): Contact the project maintainers
   - **Fouad Azem** (ID: 040830861) - [Fouad.Azem@gmail.com](mailto:Fouad.Azem@gmail.com)
   - **Tal Goldengorn** (ID: 207042573) - [T.goldengoren@gmail.com](mailto:T.goldengoren@gmail.com)
   - Use subject line: `[SECURITY] Brief description`
   - Include detailed information (see below)

2. **Private Security Advisory**: Use GitHub's [Private Vulnerability Reporting](https://github.com/yourusername/lstm-frequency-extraction/security/advisories/new)

### What to Include

Please include as much of the following information as possible:

* **Type of vulnerability** (e.g., code injection, XSS, authentication bypass)
* **Full paths** of source file(s) related to the vulnerability
* **Location** of the affected source code (tag/branch/commit or direct URL)
* **Step-by-step instructions** to reproduce the issue
* **Proof-of-concept** or exploit code (if possible)
* **Impact** of the vulnerability
* **Potential solutions** (if you have any)
* **Your contact information** for follow-up

### Example Report

```markdown
**Vulnerability Type**: Arbitrary Code Execution

**Affected Component**: src/data/signal_generator.py

**Description**: 
The signal generator allows arbitrary code execution through unsafe 
deserialization of YAML configuration files.

**Steps to Reproduce**:
1. Create malicious config.yaml with pickle payload
2. Run: python main.py --config malicious_config.yaml
3. Observe code execution

**Impact**: 
An attacker could execute arbitrary Python code on the system 
running the application.

**Suggested Fix**:
Use safe_load() instead of load() for YAML parsing.

**Your Name/Handle**: (optional)
**Contact**: (optional)
```

## Response Timeline

* **Initial Response**: Within 48 hours
* **Status Update**: Within 7 days
* **Fix Timeline**: Depends on severity
  - **Critical**: 1-7 days
  - **High**: 7-30 days
  - **Medium**: 30-90 days
  - **Low**: Best effort basis

## What to Expect

1. **Acknowledgment**: We'll confirm receipt of your report
2. **Assessment**: We'll evaluate the vulnerability
3. **Updates**: We'll keep you informed of our progress
4. **Resolution**: We'll develop and test a fix
5. **Disclosure**: We'll coordinate public disclosure with you

## Security Update Process

When we receive a security bug report:

1. **Confirm** the problem and determine affected versions
2. **Audit** code to find similar problems
3. **Prepare** fixes for all supported versions
4. **Release** new versions with security patches
5. **Announce** the vulnerability in release notes

## Disclosure Policy

* **Coordinated Disclosure**: We prefer coordinated disclosure
* **Public Disclosure**: After a fix is released (typically 90 days max)
* **Credit**: We'll credit the reporter unless they prefer anonymity

## Security Best Practices for Users

### Installation

```bash
# Always install from official sources
pip install -r requirements.txt

# Verify package integrity (when available)
pip install --require-hashes -r requirements.txt

# Use virtual environments
python -m venv venv
source venv/bin/activate
```

### Configuration

```yaml
# config/config.yaml
# Never commit sensitive data
# Use environment variables for secrets

data:
  # Safe: relative paths only
  data_dir: "data/"  # ✅
  
  # Unsafe: absolute paths from untrusted sources
  # data_dir: "/etc/passwd"  # ❌
```

### Running Code

```bash
# Safe: Use configuration files
python main.py

# Unsafe: Don't eval() or exec() user input
# python -c "eval(user_input)"  # ❌
```

### Data Handling

* **Never** load pickled data from untrusted sources
* **Always** validate configuration file contents
* **Use** safe YAML loading (`yaml.safe_load()`)
* **Sanitize** user inputs before processing

## Known Security Considerations

### Current Implementation

#### 1. Configuration Loading (LOW RISK)

**Component**: `config/config.yaml` loading  
**Status**: ✅ Safe - Uses `yaml.safe_load()`  
**Recommendation**: Continue using safe_load

#### 2. Model Checkpoints (MEDIUM RISK)

**Component**: PyTorch model loading  
**Status**: ⚠️ Moderate - Uses `torch.load()`  
**Mitigation**: Only load checkpoints from trusted sources  
**Recommendation**: 
```python
# Only load trusted checkpoints
checkpoint = torch.load(path, map_location=device)
# Validate checkpoint structure before use
```

#### 3. User Inputs (LOW RISK)

**Component**: Command-line arguments  
**Status**: ✅ Safe - Limited input validation  
**Current Protection**: argparse with type checking

#### 4. Dependencies (MEDIUM RISK)

**Component**: Third-party packages  
**Status**: ⚠️ Monitor regularly  
**Mitigation**: 
```bash
# Regularly update dependencies
pip install --upgrade -r requirements.txt

# Check for vulnerabilities
pip-audit
```

### Potential Attack Vectors

#### 1. Malicious Model Files

**Risk**: Arbitrary code execution via crafted `.pt` files  
**Mitigation**: 
- Only load models from trusted sources
- Implement model signature verification (future)
- Use safe loading practices

#### 2. Configuration Injection

**Risk**: YAML injection via malicious config files  
**Current Protection**: Using `yaml.safe_load()`  
**Status**: ✅ Protected

#### 3. Dependency Vulnerabilities

**Risk**: Vulnerabilities in PyTorch, NumPy, etc.  
**Mitigation**: 
- Pin dependency versions in requirements.txt
- Regularly update dependencies
- Monitor security advisories

#### 4. Path Traversal

**Risk**: Reading/writing files outside intended directories  
**Current Protection**: Relative paths in configuration  
**Recommendation**: Validate all file paths

## Security Checklist for Contributors

Before submitting code, ensure:

- [ ] No hardcoded credentials or API keys
- [ ] Input validation for all user-provided data
- [ ] Safe deserialization (no `pickle.load()` from untrusted sources)
- [ ] Path sanitization to prevent directory traversal
- [ ] Safe YAML loading (`yaml.safe_load()`)
- [ ] No use of `eval()` or `exec()` with user input
- [ ] Dependencies are up to date
- [ ] Security-sensitive changes are documented
- [ ] Tests include security scenarios

## Security Tools

We recommend using these tools:

```bash
# Dependency vulnerability scanning
pip install pip-audit
pip-audit

# Security linting
pip install bandit
bandit -r src/

# Static analysis
pip install safety
safety check

# Secret scanning
pip install detect-secrets
detect-secrets scan
```

## Hall of Fame

We recognize security researchers who have helped improve our project:

* *Your name could be here!* - [Report a vulnerability](#reporting-a-vulnerability)

## Additional Resources

* [OWASP Top 10](https://owasp.org/www-project-top-ten/)
* [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
* [PyTorch Security](https://pytorch.org/docs/stable/notes/security.html)

## Questions?

For security-related questions that are **not vulnerabilities**, please:

* Open a [GitHub Discussion](https://github.com/yourusername/lstm-frequency-extraction/discussions)
* Contact the project maintainers:
  - Fouad Azem: [Fouad.Azem@gmail.com](mailto:Fouad.Azem@gmail.com)
  - Tal Goldengorn: [T.goldengoren@gmail.com](mailto:T.goldengoren@gmail.com)

---

**Thank you for helping keep this project secure! 🔒**

Last Updated: November 2025

