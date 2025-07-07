from setuptools import setup, find_packages

setup(
    name='SBP',               # package name
    version='0.1.0',
    description='DG utilities: Legendre, mesh, SBP operators, etc.',
    author='Abdulrahman Taher',
    author_email='abdulrahman.taher@kaust.edu.sa',
    url='https://github.com/AbdulrahmanKT/SBP-SAT',  # if you have a repo
    packages=find_packages(),    # finds dg_lib
    install_requires=[
        'numpy>=1.18',
        'matplotlib',
        'sympy',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)

