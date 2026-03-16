from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="etdacvo",
    version="1.0.0",
    author="Indrakumar K",
    description="ETDACVO: Enhanced Tasmanian Devil Anti-Conservative Variable Optimization for Medical Image Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision",
        "numpy",
        "scikit-image",
        "scikit-learn",
        "lpips",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "pandas",
        "nibabel",
        "SimpleITK",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)