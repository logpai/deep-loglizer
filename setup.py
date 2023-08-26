from setuptools import setup


setup(
    name="inw-deeploglizer",
    version="1.6.0",
    description="Fork of Deep learning-based log analysis toolkit for automated anomaly detection.",
    author="LOGPAI",
    maintainer="luismavs",
    author_email="info@logpai.com",
    maintainer_email="luismavseabra@gmail.com",
    install_requires=['pandas', 'torch>=1.10', 'tqdm', 'numpy', 'scikit-learn', 'ordered-set','loguru'],
    packages=['deeploglizer', 'deeploglizer.models', 'deeploglizer.common'],
    python_requires=">=3.8",
    zip_safe=False,
)
