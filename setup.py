from setuptools import setup, find_packages

setup(
    name="translation_site_predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'numpy>=1.19.2',
        'biopython>=1.78',
        'scikit-learn>=0.24.2',
        'tqdm>=4.50.0',
    ],
    author="wuzengding",
    author_email="wuzengding@therarna.cn",
    description="RNA translation site prediction tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/translation_site_predictor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)