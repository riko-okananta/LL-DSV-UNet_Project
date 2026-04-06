from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name            = "lldsvunet",
    version         = "1.0.0",
    author          = "Your Name",
    author_email    = "your.email@example.com",
    description     = "LL-DSV-UNet: Low-Light Image Enhancement with Multi-Level Fusion Deep Supervision",
    long_description= long_description,
    long_description_content_type = "text/markdown",
    url             = "https://github.com/yourusername/LL-DSV-UNet",
    packages        = find_packages(),
    python_requires = ">=3.9",
    install_requires= requirements,
    classifiers     = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords = [
        "low-light image enhancement", "deep learning", "U-Net",
        "deep supervision", "attention mechanism", "tensorflow",
        "image restoration", "computer vision"
    ],
)
