# Release Workflow Documentation

This document describes the automated release workflow for the retrieval-free context compressor project.

## Overview

The release workflow automates the entire release process including version bumping, changelog generation, testing, building, and publishing to PyPI.

## Workflow Configuration

Due to GitHub's security restrictions, this workflow configuration needs to be manually added to `.github/workflows/release.yml` by a repository administrator with `workflows` permission.

## Workflow File

```yaml
name: Release

on:
  workflow_dispatch:
    inputs:
      bump_type:
        description: 'Type of version bump'
        required: true
        default: 'patch'
        type: choice
        options:
          - patch
          - minor
          - major
      test_pypi:
        description: 'Publish to Test PyPI instead of PyPI'
        required: false
        default: false
        type: boolean

permissions:
  contents: write
  id-token: write  # For trusted publishing to PyPI

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for changelog generation
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        pip install -e ".[dev]"
    
    - name: Configure git
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
    
    - name: Run tests
      run: |
        pytest --cov=retrieval_free --cov-report=xml
    
    - name: Run linting
      run: |
        ruff check src tests
        black --check src tests
        mypy src
    
    - name: Run release script
      run: |
        python scripts/release.py ${{ inputs.bump_type }} \
          ${{ inputs.test_pypi && '--test-pypi' || '' }} \
          --skip-tests  # Tests already run above
    
    - name: Push changes
      run: |
        git push origin main
        git push origin --tags
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ steps.get_version.outputs.version }}
        release_name: Release ${{ steps.get_version.outputs.version }}
        body_path: CHANGELOG.md
        draft: false
        prerelease: false
    
    - name: Upload package artifacts
      uses: actions/upload-artifact@v3
      with:
        name: package-artifacts
        path: dist/
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      if: ${{ !inputs.test_pypi }}
      with:
        print-hash: true
    
    - name: Publish to Test PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      if: ${{ inputs.test_pypi }}
      with:
        repository-url: https://test.pypi.org/legacy/
        print-hash: true
```

## Manual Setup Required

To enable the automated release workflow:

1. **Repository Administrator** needs to create the workflow file at `.github/workflows/release.yml`
2. **Configure PyPI Publishing**:
   - Set up trusted publishing on PyPI for this repository
   - Or add `PYPI_API_TOKEN` secret to repository settings
3. **Test the Workflow**:
   - Trigger manually from GitHub Actions tab
   - Select appropriate bump type (patch/minor/major)
   - Choose PyPI or Test PyPI for publishing

## Usage

### Manual Release via GitHub Actions

1. Go to the Actions tab in the GitHub repository
2. Select "Release" workflow
3. Click "Run workflow"
4. Choose bump type and publishing target
5. Click "Run workflow" to start the release process

### Local Release via Script

```bash
# Test release (dry run)
python scripts/release.py patch --dry-run

# Patch release to Test PyPI
python scripts/release.py patch --test-pypi

# Minor release to PyPI
python scripts/release.py minor

# Major release with custom options
python scripts/release.py major --skip-tests
```

## Release Process Steps

1. **Validation**: Run tests and linting checks
2. **Version Bump**: Update version in pyproject.toml and __init__.py
3. **Changelog**: Generate changelog from git commits
4. **Build**: Create distribution packages
5. **Commit**: Commit version changes
6. **Tag**: Create git tag for the release
7. **Publish**: Upload to PyPI/Test PyPI
8. **GitHub Release**: Create GitHub release with artifacts

## Security Considerations

- Uses trusted publishing for PyPI (no API tokens needed)
- Workflow requires `contents: write` and `id-token: write` permissions
- All commits and tags are signed by GitHub Actions
- Package integrity verified with checksums

## Troubleshooting

### Common Issues

1. **Tests Fail**: Fix failing tests before releasing
2. **Linting Errors**: Run `make format` to fix code style
3. **Version Conflict**: Ensure version doesn't already exist on PyPI
4. **Permission Denied**: Check repository permissions and secrets

### Support

For issues with the release workflow:
- Check GitHub Actions logs for detailed error messages
- Verify all required secrets are configured
- Ensure branch protection rules allow the release process