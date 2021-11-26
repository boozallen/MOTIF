#!/usr/bin/env python3

import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="UTF-8") as f:
    readme = f.read()

requires = open("requirements.txt", "r").read().strip().split()
package_data = {}
setup(
    name="ember",
    version="1.0",
    description="MOTIF Dataset",
    long_description=readme,
    packages=["EMBER", "MalConv2", "benchmarks"],
    package_data=package_data,
    install_requires=requires,
    author_email="joyce8@umbc.edu"
)
