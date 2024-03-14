from setuptools import setup

with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='STR Bachelor Project',
    description='Semantic Text Relatedness in Low-Resource Languages',
    install_requires=requirements,
)
