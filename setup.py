from setuptools import setup, find_packages

setup(
    name="lvm_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "accelerate",
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "bitsandbytes",
        "Pillow",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "python-dotenv",
        "tensorboard",
        "safetensors",
        "pyarrow",
        "datasets"
    ],
    description="LVM utility functions and caching",
)
