# setup.py
from setuptools import setup, find_packages

setup(
    name="pdf_outline",
    version="0.1.0",
    description="Heading‑outline extractor (Donut INT8 + rule‑based cluster)",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(exclude=[
        "tests*",
        "sample_dataset*",
        "models*",
        "wheelhouse*",
        "pdf_env*",
    ]),
    include_package_data=True,
    install_requires=[],   # we install all deps via requirements.txt
    entry_points={
        "console_scripts": [
            "extract_outline=pdf_outline.cli:main",
        ],
    },
)
