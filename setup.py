"""
Setup script for Apeiron Tools
Infinite AI Tool Orchestration Network on Bittensor
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Apeiron Tools - Access Every Knowledge The World Has"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("sqlite3"):
                    requirements.append(line)
    return requirements

setup(
    name="apeiron-tools",
    version="0.1.0",
    description="Infinite AI Tool Orchestration Network on Bittensor Subnet #122",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Apeiron Tools Team",
    author_email="contact@apeiron.tools",
    url="https://github.com/apeiron-tools/apeiron-tools",
    license="MIT",
    
    packages=find_packages(),
    include_package_data=True,
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    
    keywords=[
        "bittensor",
        "blockchain", 
        "machine-learning",
        "ai-tools",
        "mcp",
        "model-context-protocol",
        "tool-orchestration",
        "distributed-computing",
        "decentralized-ai"
    ],
    
    entry_points={
        "console_scripts": [
            "apeiron-miner=neurons.miner:main",
            "apeiron-validator=neurons.validator:main",
        ],
    },
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    
    project_urls={
        "Documentation": "https://docs.apeiron.tools",
        "Source": "https://github.com/apeiron-tools/apeiron-tools",
        "Tracker": "https://github.com/apeiron-tools/apeiron-tools/issues",
        "Discord": "https://discord.gg/apeiron-tools",
        "Twitter": "https://twitter.com/ApeironTools",
    },
    
    zip_safe=False,
)