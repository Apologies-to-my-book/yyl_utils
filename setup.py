from setuptools import setup, find_packages

setup(
    name="yyl_utils",
    version="0.1.0",
    description="yyl 的个人工具库，包含 LFP 分析、绘图等工具",
    author="Apologies-to-my-book",
    author_email="yyl1216466507@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scipy>=1.5.0",
        "matplotlib>=3.0.0",
        "seaborn>=0.10.0",
    ],
    python_requires=">=3.6",
)