from setuptools import setup


setup(
    name="deeploglizer",
    version="1.0.0",
    description="Fork of Deep learning-based log analysis toolkit for automated anomaly detection.",
    author="LOGPAI",
    author_email="info@logpai.com",
    install_requires=['pandas', 'torch>=1.10', 'tqdm', 'numpy', 'scikit-learn'],
    packages=['deeploglizer'],
    python_requires=">=3.8",
    zip_safe=False,
)
