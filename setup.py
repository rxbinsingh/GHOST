from setuptools import setup, find_packages

setup(
    name="ghost-supply-chain",
    version="1.0.0",
    author="Robin Singh",
    author_email="robinsingh4889@gmail.com",
    description="GHOST: Graph-based Hierarchical On-the-fly Self-correcting Threat Detector",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rxbinsingh/GHOST",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "networkx>=2.8",
        "numpy>=1.21",
        "scipy>=1.9",
        "scikit-learn>=1.1",
        "pandas>=1.4",
        "matplotlib>=3.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
