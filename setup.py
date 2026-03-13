#setup.py: Setup script for the healthrisk package

from setuptools import setup, find_packages

setup(
    name        = 'healthrisk',
    version     = '0.1.0',
    description = 'A package for health risk assessment and management',
    author      = 'Niral Patel',
    author_email= 'nir64.au@gmail.com',
    packages    = find_packages(),
    install_requires=[
    ]
)