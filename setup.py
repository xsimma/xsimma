# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='xsimma'
    version='0.0.1',
    description='The engine of matter simulations',
    long_description=readme,
    author='X-SimMa contributors',
    author_email='xipinggong@umass.edu',
    url='https://github.com/xsimma/xsimma',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

