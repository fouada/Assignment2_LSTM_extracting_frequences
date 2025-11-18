"""
Setup script for LSTM Frequency Extraction System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="lstm-frequency-extraction",
    version="1.0.0",
    author="Professional ML Engineering Team",
    author_email="ml-team@example.com",
    description="Professional LSTM implementation for frequency extraction from mixed signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lstm-frequency-extraction",
    packages=find_packages(exclude=["tests", "experiments", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Signal Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lstm-freq-extract=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

