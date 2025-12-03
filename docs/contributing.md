# Contributing to MAIF

Thank you for your interest in contributing to the MAIF Framework! This document provides guidelines for contributing.

## Ways to Contribute

### Reporting Issues
- **Bug Reports**: Use GitHub Issues with clear reproduction steps
- **Feature Requests**: Describe the use case and expected behavior
- **Documentation**: Report unclear or missing documentation

### Code Contributions

#### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/vineethsai/maifscratch-1.git
cd maifscratch-1

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

#### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow existing code style
   - Add tests for new features
   - Update documentation as needed

4. **Run tests and linting**:
   ```bash
   pytest tests/
   python -m pylint maif/
   ```

5. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add new feature description"
   ```

6. **Push and create Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add semantic compression algorithm
fix: correct hash chain verification
docs: update installation guide
```

## Code Style

### Python Code
- Follow PEP 8 style guide
- Use type hints where applicable
- Maximum line length: 100 characters
- Use meaningful variable names

Example:
```python
def create_artifact(
    artifact_id: str,
    enable_privacy: bool = False
) -> MAIFArtifact:
    """
    Create a new MAIF artifact.
    
    Args:
        artifact_id: Unique identifier for the artifact
        enable_privacy: Enable privacy features
        
    Returns:
        Newly created MAIF artifact
    """
    pass
```

### Documentation
- Use Markdown for documentation
- Include code examples
- Add diagrams where helpful
- Keep language clear and concise

## Testing

### Writing Tests
- Place tests in `tests/` directory
- Use pytest framework
- Name test files `test_*.py`
- Include unit and integration tests

Example:
```python
def test_artifact_creation():
    """Test basic artifact creation"""
    artifact = create_artifact("test_id")
    assert artifact.artifact_id == "test_id"
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=maif tests/
```

## Documentation Contributions

### Building Documentation
```bash
cd docs
npm install
npm run build
npm run dev  # Local preview
```

### Documentation Structure
- User guides: `docs/guide/`
- API reference: `docs/api/`
- Examples: `docs/examples/`
- Tutorials: Add to appropriate section

## Adding Examples

Great examples help users understand MAIF. To add an example:

1. **Create example file** in `examples/<category>/`
2. **Add README** explaining the example
3. **Update documentation** in `docs/examples/`
4. **Add to navigation** in `docs/.vitepress/config.js`

Example structure:
```
examples/
  your-example/
    README.md           # Usage instructions
    demo.py            # Main example code
    requirements.txt   # Dependencies
```

## Review Process

### Pull Request Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow format
- [ ] No breaking changes (or documented)
- [ ] CI/CD checks passing

### Review Timeline
- Initial review: Within 3-5 business days
- Follow-up reviews: 1-2 business days
- Merge: After approval and CI passing

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

### Getting Help
- GitHub Issues: Bug reports and feature requests
- Discussions: Questions and ideas
- Documentation: Check guides first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:
- Release notes
- GitHub contributors page
- Documentation credits

Thank you for contributing to MAIF! 🎉

