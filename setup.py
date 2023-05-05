from setuptools import setup


setup(
    name="inw-deeploglizer",
    version="1.4.1",
    description="Fork of Deep learning-based log analysis toolkit for automated anomaly detection.",
    author="LOGPAI & luismavs",
    author_email="info@logpai.com",
    install_requires=['pandas', 'torch>=1.10', 'tqdm', 'numpy', 'scikit-learn'],
    packages=['deeploglizer', 'deeploglizer.models', 'deeploglizer.common'],
    python_requires=">=3.8",
    zip_safe=False,
)
