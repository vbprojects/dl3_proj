from setuptools import setup, find_packages

setup(
    name="lvm_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "Pillow",
        "numpy",
        "pyarrow",
        "pandas"
    ],
    description="LVM utility functions and caching",
)
