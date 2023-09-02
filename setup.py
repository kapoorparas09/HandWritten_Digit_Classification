from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name ="src",
    version="0.0.2",
    author="Paras Kapoor",
    description="ANN implementation on mnist dataset",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/kapoorparas09/HandWritten_Digit_Classification",
    author_email="kapoorparas0001@gmail.com",
    packages=["src"],
    python_requires= ">=3.9",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas"
    ]
)