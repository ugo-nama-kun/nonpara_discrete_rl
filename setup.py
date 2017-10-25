# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nprl',
    version='0.2.3',
    description='Example code of the TD learning, Q learning and Model-based RL',
    long_description=readme,
    author='Naoto Yoshida',
    author_email='yossy0157@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests'))
)

