import os
import sys

import setuptools

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 8

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)

TEST_REQUIRES = ["pytest", "pytest-cov"]

DEV_REQUIRES = TEST_REQUIRES + ["black", "flake8", "isort"]

root_dir = os.path.dirname(__file__)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="autosbo",
    version="0.0.1",
    author="Sami Alabed",
    author_email="sa894@cam.ac.uk",
    description="A package for structured optimising of computer system",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    long_description=read("README.md"),
    test_suite="setup.my_test_suite",
    keywords=[
        "Bayesian optimization",
        "Structured optimization",
        "Computer System optimization",
    ],
    packages=setuptools.find_packages(),
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
    },
)
