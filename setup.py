from setuptools import setup, find_packages

setup(
    name="neutrosophic-stock-forecast",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "tensorflow>=2.15.0",
        "yfinance>=0.2.36",
        "statsmodels>=0.14.1",
        "pmdarima>=2.0.4",
        "fastapi>=0.109.0",
        "streamlit>=1.31.0",
    ],
)
