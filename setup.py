#!/usr/bin/env python3

from setuptools import setup

setup(
    name="ECC2025",
    version="0.1.1",
    description="ECC2025",
    url="https://github.com/LucianoPereiraValenzuela/ECC_2025_testing",
    author="Luciano Pereira",
    author_email="luciano.pereira.valenzuela@gmail.com",
    license="Apache 2.0",
    install_requires=["numpy", "scipy", "qiskit", "qiskit_ibm_runtime", "qiskit_aer", "pylatexenc"],
)