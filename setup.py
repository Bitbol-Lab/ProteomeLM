#!/usr/bin/env python3
"""Setup script for ProteomeLM."""

from setuptools import setup, find_packages
import os


# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


# Get version from package
def get_version():
    version_file = os.path.join("proteomelm", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


setup(
    name="proteomelm",
    version=get_version(),
    author="Cyril Malbranke, Gionata Paolo Zalaffi and Anne-Florence Bitbol",
    author_email="cyril.malbranke@epfl.ch",
    description="A proteome-scale language model for fast prediction of protein-protein interactions and gene essentiality across taxa",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Bitbol-Lab/ProteomeLM",
    project_urls={
        "Bug Tracker": "https://github.com/Bitbol-Lab/ProteomeLM/issues",
        "Source Code": "https://github.com/Bitbol-Lab/ProteomeLM",
        "Paper": "https://arxiv.org/abs/placeholder",
    },
    packages=find_packages(exclude=["experiments", "notebooks"]),
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "gpu": ["flash-attn>=2.0.0"],
        "deepspeed": ["deepspeed>=0.9.0"],
    },
    entry_points={
        "console_scripts": [
            "proteomelm-train=proteomelm.cli:train_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "proteomelm": ["configs/*.yaml"],
    },
    keywords=[
        "protein language model",
        "protein-protein interactions",
        "gene essentiality",
        "proteomics",
        "bioinformatics",
        "transformer",
        "deep learning",
        "computational biology",
    ],
)
