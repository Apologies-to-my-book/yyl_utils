from setuptools import setup, find_packages

setup(
    name="yyl_utils",
    version="0.1.0",
    description="yyl 的个人工具库",
    author="Apologies-to-my-book",
    author_email="yyl1216466507@gmail.com",
    packages=find_packages(),

    package_data={
        "yyl_utils": [
            "*.pyi",
            "py.typed",
        ],
    },
    include_package_data=True,
    # 只放必要的核心依赖
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scipy>=1.5.0",
        "matplotlib>=3.0.0",
        "seaborn>=0.10.0",
    ],
    # 可选依赖（大型包，用户按需安装）
    extras_require={
        "full": [
            "spikeinterface[full]>=0.98.0",
            "torch>=2.0.0",
        ],
        "spike": ["spikeinterface[full]>=0.98.0"],
        "ml": ["torch>=2.0.0", "scikit-learn>=1.0.0"],
    },
    python_requires=">=3.6",
)