from setuptools import setup, find_packages

setup(
    name="wavevibes",
    version="0.1.0",
    description="A prototyping environment for real-time audio algorithms",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "wavevibes=main:main",
        ],
    },
)