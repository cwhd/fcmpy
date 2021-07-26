from setuptools import setup


def requirements():
    with open("requirements.txt") as f:
        req = f.readlines()
    return req


def readme():
    with open("README.md") as f:
        r = f.read()
    return r


setup(
    name="fcmpy",
    packages=["fcmpy"],
    version="0.0.1",
    license="MIT",
    description="A very useful Python package",
    long_description=readme(),
    author="Davis 36",
    author_email="davis36@gmail.com",
    install_requires=requirements(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ]
)